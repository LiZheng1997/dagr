"""Sensor-frame <-> model-frame affine fit for live inference.

The DAGR DSEC checkpoints expect a 320x215 input. DSEC frames are 640x480 and
are top-cropped to 430 rows then halved, which is what the training loader does.
A GenX320 is 320x320, so any "crop the top 215 rows" shortcut discards the
bottom third of the field of view. For a vehicle-mounted camera that is exactly
where the nearest road users appear, so the default here keeps the whole frame.

All transforms are axis-aligned affines:

    model_x = sensor_x * sx + ox
    model_y = sensor_y * sy + oy

and are inverted exactly when boxes are mapped back to the sensor frame. Only
numpy is imported so the module can be unit tested without torch.
"""
from dataclasses import dataclass

import numpy as np

FIT_MODES = ("letterbox", "squash", "crop_top", "dsec")


@dataclass(frozen=True)
class FitTransform:
    sensor_w: int
    sensor_h: int
    model_w: int
    model_h: int
    sx: float
    sy: float
    ox: float
    oy: float
    mode: str

    # --- forward (sensor -> model) ---------------------------------------
    def fwd_xy(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return x * self.sx + self.ox, y * self.sy + self.oy

    def event_mask_and_coords(self, x, y):
        """Map event coordinates and drop those outside the model frame.

        Returns (mask, mx, my) where mx/my are int16 model-frame pixels and
        mask selects the surviving events in the *input* order.
        """
        mx, my = self.fwd_xy(x, y)
        mx = np.floor(mx).astype(np.int32)
        my = np.floor(my).astype(np.int32)
        mask = (mx >= 0) & (mx < self.model_w) & (my >= 0) & (my < self.model_h)
        return mask, mx[mask].astype(np.int16), my[mask].astype(np.int16)

    # --- inverse (model -> sensor) ----------------------------------------
    def inv_box_xywh(self, boxes_xyxy):
        """Map Nx4 model-frame [x0,y0,x1,y1] to sensor-frame [x,y,w,h]."""
        b = np.asarray(boxes_xyxy, dtype=np.float64).reshape(-1, 4)
        x0 = (b[:, 0] - self.ox) / self.sx
        x1 = (b[:, 2] - self.ox) / self.sx
        y0 = (b[:, 1] - self.oy) / self.sy
        y1 = (b[:, 3] - self.oy) / self.sy
        # Clip to the sensor frame so letterbox padding never yields boxes
        # that lie outside the image the operator sees.
        x0 = np.clip(x0, 0, self.sensor_w); x1 = np.clip(x1, 0, self.sensor_w)
        y0 = np.clip(y0, 0, self.sensor_h); y1 = np.clip(y1, 0, self.sensor_h)
        return np.stack([x0, y0, x1 - x0, y1 - y0], axis=1)

    # --- image ------------------------------------------------------------
    def apply_image(self, bgr):
        """Resample a sensor-frame BGR image into the model frame.

        Pixels of the model frame not covered by the sensor image are zero
        (black), which is what the training data shows at image borders after
        undistortion.
        """
        import cv2
        h, w = bgr.shape[:2]
        if (w, h) != (self.sensor_w, self.sensor_h):
            raise ValueError(
                f"image is {w}x{h} but transform was built for "
                f"{self.sensor_w}x{self.sensor_h}")
        if self.mode == "dsec":
            # Training-identical path: top crop then 2x downsample.
            crop_h = int(round(self.model_h / self.sy))
            if h < crop_h:
                pad = np.zeros((crop_h - h, w, 3), dtype=bgr.dtype)
                bgr = np.vstack([bgr, pad])
            bgr = bgr[:crop_h]
            return cv2.resize(bgr, (self.model_w, self.model_h),
                              interpolation=cv2.INTER_CUBIC)

        out_w = int(round(w * self.sx))
        out_h = int(round(h * self.sy))
        resized = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA
                             if self.sx < 1.0 else cv2.INTER_CUBIC)
        canvas = np.zeros((self.model_h, self.model_w, 3), dtype=bgr.dtype)
        ox = int(round(self.ox)); oy = int(round(self.oy))
        # Source window inside `resized` that lands inside the canvas.
        src_x0 = max(0, -ox); src_y0 = max(0, -oy)
        dst_x0 = max(0, ox);  dst_y0 = max(0, oy)
        cw = min(out_w - src_x0, self.model_w - dst_x0)
        ch = min(out_h - src_y0, self.model_h - dst_y0)
        if cw > 0 and ch > 0:
            canvas[dst_y0:dst_y0 + ch, dst_x0:dst_x0 + cw] = \
                resized[src_y0:src_y0 + ch, src_x0:src_x0 + cw]
        return canvas


def make_fit(sensor_w: int, sensor_h: int, model_w: int, model_h: int,
             mode: str) -> FitTransform:
    """Build the sensor->model transform for one of FIT_MODES.

    letterbox : uniform scale so the whole sensor frame fits, centred, padded.
    squash    : independent x/y scale, whole frame, aspect ratio not kept.
    crop_top  : uniform scale on width, keep the top rows (legacy behaviour).
    dsec      : training-identical 640x480 -> crop 430 rows -> 320x215.
    """
    if mode not in FIT_MODES:
        raise ValueError(f"unknown fit mode {mode!r}, expected one of {FIT_MODES}")
    if mode == "dsec":
        s = model_w / sensor_w          # 0.5 for 640 -> 320
        return FitTransform(sensor_w, sensor_h, model_w, model_h,
                            sx=s, sy=s, ox=0.0, oy=0.0, mode=mode)
    if mode == "letterbox":
        s = min(model_w / sensor_w, model_h / sensor_h)
        ox = (model_w - sensor_w * s) / 2.0
        oy = (model_h - sensor_h * s) / 2.0
        return FitTransform(sensor_w, sensor_h, model_w, model_h,
                            sx=s, sy=s, ox=ox, oy=oy, mode=mode)
    if mode == "squash":
        return FitTransform(sensor_w, sensor_h, model_w, model_h,
                            sx=model_w / sensor_w, sy=model_h / sensor_h,
                            ox=0.0, oy=0.0, mode=mode)
    # crop_top
    s = model_w / sensor_w
    return FitTransform(sensor_w, sensor_h, model_w, model_h,
                        sx=s, sy=s, ox=0.0, oy=0.0, mode=mode)
