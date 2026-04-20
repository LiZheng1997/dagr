"""DAGR IPC binary protocol helpers (see PROTOCOL.md).

Little-endian throughout, no Python pickling, fixed-layout structs so C++
and Python clients can share one wire format.
"""
import socket
import struct
from typing import Tuple

import numpy as np

SOCKET_PATH = "/tmp/dagr_infer.sock"

MAGIC = 0x44414752  # 'DAGR' as LE u32
VERSION = 1
MSG_INFER_REQ = 1
MSG_INFER_REPLY = 2
MSG_PING = 3
MSG_PONG = 4

HEADER_FMT = "<IBBHII"  # magic, version, msg_type, reserved, req_id, payload_len
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 16, HEADER_SIZE

# 16-byte packed Event entry
EVENT_DTYPE = np.dtype(
    [("x", "<u2"), ("y", "<u2"), ("t_ns", "<i8"), ("p", "u1"), ("pad", "V3")],
    align=False,
)
assert EVENT_DTYPE.itemsize == 16

# 24-byte packed Detection entry
DET_DTYPE = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("w", "<f4"), ("h", "<f4"),
     ("class_id", "<u4"), ("conf", "<f4")],
    align=False,
)
assert DET_DTYPE.itemsize == 24


# ---------- low-level socket helpers -----------------------------------------

def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError(f"peer closed after {len(buf)}/{n} bytes")
        buf.extend(chunk)
    return bytes(buf)


def recv_header(conn: socket.socket) -> Tuple[int, int, int, int]:
    """Return (msg_type, req_id, payload_len, version)."""
    raw = _recv_exact(conn, HEADER_SIZE)
    magic, version, msg_type, reserved, req_id, payload_len = struct.unpack(HEADER_FMT, raw)
    if magic != MAGIC:
        raise ValueError(f"bad magic 0x{magic:08x}, expected 0x{MAGIC:08x}")
    if version != VERSION:
        raise ValueError(f"bad version {version}, expected {VERSION}")
    return msg_type, req_id, payload_len, version


def pack_header(msg_type: int, req_id: int, payload_len: int) -> bytes:
    return struct.pack(HEADER_FMT, MAGIC, VERSION, msg_type, 0, req_id, payload_len)


# ---------- INFER_REQ --------------------------------------------------------

def encode_infer_req(req_id: int, t_ns: int, image_bgr: np.ndarray,
                     events: dict) -> bytes:
    """Serialize an inference request.

    events dict must contain numpy arrays for keys 'x', 'y', 't_ns', 'p'
    (any numeric dtypes; will be cast).
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.dtype != np.uint8:
        raise ValueError(f"image must be (H,W,3) uint8 BGR, got {image_bgr.shape} {image_bgr.dtype}")
    if not image_bgr.flags["C_CONTIGUOUS"]:
        image_bgr = np.ascontiguousarray(image_bgr)

    h, w, c = image_bgr.shape

    n = int(len(events["x"]))
    if n > 0:
        ev = np.zeros(n, dtype=EVENT_DTYPE)
        ev["x"] = np.asarray(events["x"], dtype=np.uint16)
        ev["y"] = np.asarray(events["y"], dtype=np.uint16)
        ev["t_ns"] = np.asarray(events["t_ns"], dtype=np.int64)
        ev["p"] = np.asarray(events["p"], dtype=np.uint8)
        ev_bytes = ev.tobytes()
    else:
        ev_bytes = b""

    img_bytes = image_bgr.tobytes()
    header_payload = struct.pack("<qIII", int(t_ns), h, w, c)
    tail = struct.pack("<I", n) + ev_bytes

    payload_len = len(header_payload) + len(img_bytes) + len(tail)
    hdr = pack_header(MSG_INFER_REQ, req_id, payload_len)
    return b"".join([hdr, header_payload, img_bytes, tail])


def decode_infer_req(payload: bytes) -> Tuple[int, np.ndarray, dict]:
    t_ns, h, w, c = struct.unpack_from("<qIII", payload, 0)
    off = 8 + 4 + 4 + 4
    img_size = h * w * c
    img = np.frombuffer(payload, dtype=np.uint8, count=img_size, offset=off).reshape(h, w, c)
    off += img_size
    (n,) = struct.unpack_from("<I", payload, off)
    off += 4
    if n > 0:
        ev = np.frombuffer(payload, dtype=EVENT_DTYPE, count=n, offset=off)
        events = {
            "x": ev["x"].copy(),
            "y": ev["y"].copy(),
            "t_ns": ev["t_ns"].copy(),
            "p": ev["p"].copy(),
        }
    else:
        events = {
            "x": np.empty(0, np.uint16),
            "y": np.empty(0, np.uint16),
            "t_ns": np.empty(0, np.int64),
            "p": np.empty(0, np.uint8),
        }
    return int(t_ns), img, events


# ---------- INFER_REPLY ------------------------------------------------------

def encode_infer_reply_ok(req_id: int, detections: np.ndarray, elapsed_ms: float) -> bytes:
    """detections: structured array with DET_DTYPE fields."""
    if detections.dtype != DET_DTYPE:
        conv = np.zeros(len(detections), dtype=DET_DTYPE)
        for key in ("x", "y", "w", "h", "class_id", "conf"):
            src_key = "class_confidence" if key == "conf" and "class_confidence" in detections.dtype.names else key
            if src_key in detections.dtype.names:
                conv[key] = detections[src_key]
        detections = conv

    n = len(detections)
    head = struct.pack("<IfI", 0, float(elapsed_ms), n)
    tail = detections.tobytes() if n > 0 else b""
    payload_len = len(head) + len(tail)
    hdr = pack_header(MSG_INFER_REPLY, req_id, payload_len)
    return hdr + head + tail


def encode_infer_reply_err(req_id: int, err: str, elapsed_ms: float = 0.0) -> bytes:
    msg = err.encode("utf-8")
    head = struct.pack("<IfII", 1, float(elapsed_ms), 0, len(msg))
    payload_len = len(head) + len(msg)
    hdr = pack_header(MSG_INFER_REPLY, req_id, payload_len)
    return hdr + head + msg


def decode_infer_reply(payload: bytes) -> Tuple[int, float, np.ndarray]:
    """Return (status, elapsed_ms, detections_or_errmsg_bytes)."""
    status, elapsed_ms, n = struct.unpack_from("<IfI", payload, 0)
    if status == 0:
        off = 12
        if n > 0:
            dets = np.frombuffer(payload, dtype=DET_DTYPE, count=n, offset=off).copy()
        else:
            dets = np.zeros(0, dtype=DET_DTYPE)
        return status, elapsed_ms, dets
    # error branch
    _, _, _, err_len = struct.unpack_from("<IfII", payload, 0)
    err_bytes = payload[16:16 + err_len]
    err = err_bytes.decode("utf-8", errors="replace")
    raise RuntimeError(f"server error: {err}")


# ---------- high-level client (Python) --------------------------------------

class IpcClient:
    def __init__(self, socket_path: str = SOCKET_PATH, timeout: float = 10.0):
        self.socket_path = socket_path
        self.timeout = timeout
        self._sock = None
        self._req_id = 0

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self.socket_path)
        self._sock = s

    def close(self):
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def _ensure(self):
        if self._sock is None:
            self.connect()

    def ping(self) -> bool:
        try:
            self._ensure()
            self._sock.sendall(pack_header(MSG_PING, 0, 0))
            msg_type, _, plen, _ = recv_header(self._sock)
            if plen > 0:
                _recv_exact(self._sock, plen)
            return msg_type == MSG_PONG
        except (OSError, ConnectionError, ValueError):
            self.close()
            return False

    def infer(self, t_ns: int, image_bgr: np.ndarray, events: dict):
        self._ensure()
        self._req_id += 1
        req_id = self._req_id
        data = encode_infer_req(req_id, t_ns, image_bgr, events)
        try:
            self._sock.sendall(data)
            msg_type, reply_id, plen, _ = recv_header(self._sock)
            if msg_type != MSG_INFER_REPLY:
                raise RuntimeError(f"unexpected msg_type {msg_type}")
            payload = _recv_exact(self._sock, plen)
            status, elapsed_ms, dets = decode_infer_reply(payload)
            return dets, elapsed_ms
        except (OSError, ConnectionError, ValueError) as e:
            self.close()
            raise RuntimeError(f"IPC error: {e}") from e
