# DAGR IPC binary protocol (v1)

Unix Domain Socket, SOCK_STREAM, little-endian throughout.

All integer fields are little-endian. Floats are IEEE 754 binary32.
Structures are packed (no alignment padding unless specified).

## Envelope header — 16 bytes, fixed

| offset | size | type | field            | notes                         |
|--------|------|------|------------------|-------------------------------|
| 0      | 4    | u32  | magic            | `0x44414752` ('DAGR' as LE)   |
| 4      | 1    | u8   | version          | `1`                            |
| 5      | 1    | u8   | msg_type         | 1=INFER_REQ, 2=INFER_REPLY, 3=PING, 4=PONG |
| 6      | 2    | u16  | reserved         | `0`                            |
| 8      | 4    | u32  | req_id           | client-chosen monotonic       |
| 12     | 4    | u32  | payload_len      | bytes following this header   |

`payload_len` does **not** include the 16-byte header itself.

Total message on the wire = 16 + payload_len.

### Ping / Pong

`msg_type=3`, `payload_len=0`. Reply is `msg_type=4`, `payload_len=0`.

## INFER_REQ payload (msg_type=1)

| offset | size                  | type        | field          |
|--------|-----------------------|-------------|----------------|
| 0      | 8                     | i64         | t_ns           |
| 8      | 4                     | u32         | img_h          |
| 12     | 4                     | u32         | img_w          |
| 16     | 4                     | u32         | img_c          |
| 20     | img_h*img_w*img_c     | u8[]        | image (BGR)    |
| ...    | 4                     | u32         | num_events     |
| ...    | num_events * 16       | Event[]     | events         |

### Event — 16 bytes

| offset | size | type | field   |
|--------|------|------|---------|
| 0      | 2    | u16  | x       |
| 2      | 2    | u16  | y       |
| 4      | 8    | i64  | t_ns    |
| 12     | 1    | u8   | polarity|
| 13     | 3    | u8[3]| pad     |

`num_events` == 0 is legal (no events window).

## INFER_REPLY payload (msg_type=2)

| offset | size                   | type         | field             |
|--------|------------------------|--------------|-------------------|
| 0      | 4                      | u32          | status (0=ok)     |
| 4      | 4                      | f32          | elapsed_ms        |
| 8      | 4                      | u32          | num_detections    |
| 12     | num_detections * 24    | Detection[]  | detections        |

If `status != 0`, Detection[] is omitted and the remainder is:
| offset | size       | type | field    |
|--------|------------|------|----------|
| 12     | 4          | u32  | err_len  |
| 16     | err_len    | u8[] | err_utf8 |

### Detection — 24 bytes

| offset | size | type | field      |
|--------|------|------|------------|
| 0      | 4    | f32  | x          |  top-left, in original sensor pixel coords
| 4      | 4    | f32  | y          |
| 8      | 4    | f32  | w          |
| 12     | 4    | f32  | h          |
| 16     | 4    | u32  | class_id   |  0=car, 1=pedestrian
| 20     | 4    | f32  | confidence |

## Python reference

`ipc/protocol.py` provides `encode_infer_req`, `decode_header`, `decode_infer_reply`.

## C++ reference

`dagr_ros_cpp` provides `dagr::ipc::Client` using POSIX socket + `::send/::recv`.
