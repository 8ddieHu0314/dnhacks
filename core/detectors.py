"""Object detectors for PPE compliance checking.

Two sources feed the pipeline:
- SH17Detector: local YOLOv8 model trained on the SH17 PPE dataset (17 classes).
- ToolsOnnxDetector (alias RoboflowDetector): Roboflow-trained tools model run
  locally from an ONNX file with onnxruntime. No hosted calls, no key at runtime.

Both return a common Detection type so downstream rules do not care which
detector produced a box.
"""

import os
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    class_name: str
    conf: float
    box: tuple  # (x1, y1, x2, y2) in pixel coordinates
    source: str  # "sh17" | "roboflow"


class SH17Detector:
    """Local YOLOv8 detector trained on the SH17 PPE dataset."""

    def __init__(self, weights_path: str, device: str = "mps", conf: float = 0.4):
        from ultralytics import YOLO

        self.weights_path = weights_path
        self.conf = conf
        self.device = device
        self.model = YOLO(weights_path)
        self.names = self.model.names

    def detect(self, frame_bgr: np.ndarray) -> list:
        try:
            results = self.model.predict(
                frame_bgr, conf=self.conf, device=self.device, verbose=False
            )
        except Exception:
            # mps can raise on some ops/hardware; fall back to cpu.
            self.device = "cpu"
            results = self.model.predict(
                frame_bgr, conf=self.conf, device=self.device, verbose=False
            )

        detections = []
        if not results:
            return detections
        result = results[0]
        boxes = result.boxes
        if boxes is None:
            return detections
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            class_name = self.names.get(cls_id, str(cls_id)) if isinstance(self.names, dict) else self.names[cls_id]
            detections.append(
                Detection(
                    class_name=class_name,
                    conf=conf,
                    box=(x1, y1, x2, y2),
                    source="sh17",
                )
            )
        return detections


class ToolsOnnxDetector:
    """Roboflow-trained tools model (mechanical-tools-10000/3, YOLOv5 nano)
    run with plain onnxruntime from a local ONNX file. No Roboflow runtime,
    no hosted calls, no API key at runtime.

    The ONNX plus class list were fetched once from Roboflow's model cache
    (see docs/ARCHITECTURE.md, "Roboflow tools model") into ./weights/.
    Preprocessing follows the bundle's inference_config.json: stretch resize
    to 640x640, RGB, scale by 1/255. Output is raw YOLOv5 rows
    [cx, cy, w, h, obj, cls...] at 640 scale, so NMS is done here.

    Fails soft: any error is caught, stored on self.last_error, and detect()
    returns [] so the alert path keeps working without tools detection.
    """

    def __init__(
        self,
        onnx_path: str = "weights/tools_yolov5n.onnx",
        classes_path: str = "weights/tools_yolov5n.classes.txt",
        conf: float = 0.4,
        iou: float = 0.45,
        input_size: int = 640,
    ):
        self.onnx_path = onnx_path
        self.classes_path = classes_path
        self.conf = conf
        self.iou = iou
        self.input_size = input_size
        self.last_error = None
        self.last_latency_ms = None
        self._session = None
        self._input_name = None
        self.names = []

    def _get_session(self):
        if self._session is not None:
            return self._session
        import onnxruntime as ort

        with open(self.classes_path) as f:
            self.names = [line.strip() for line in f if line.strip()]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        # CoreML partitions this graph poorly (one op unsupported), so CPU is
        # both simpler and, measured on this Mac, about 6 ms per frame.
        self._session = ort.InferenceSession(self.onnx_path, opts, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        return self._session

    def detect(self, frame_bgr: np.ndarray) -> list:
        try:
            session = self._get_session()
            t0 = time.perf_counter()
            h0, w0 = frame_bgr.shape[:2]
            s = self.input_size
            img = cv2.resize(frame_bgr, (s, s), interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = np.transpose(img, (2, 0, 1))[None]
            out = session.run(None, {self._input_name: blob})[0]
            rows = out[0] if out.ndim == 3 else out  # (N, 5 + num_classes)

            obj = rows[:, 4]
            cls_scores = rows[:, 5:]
            cls_idx = cls_scores.argmax(axis=1)
            scores = obj * cls_scores[np.arange(len(rows)), cls_idx]
            keep = scores >= self.conf
            rows, scores, cls_idx = rows[keep], scores[keep], cls_idx[keep]

            detections = []
            if len(rows):
                sx, sy = w0 / s, h0 / s
                cx, cy, w, h = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
                x1 = (cx - w / 2) * sx
                y1 = (cy - h / 2) * sy
                bw = w * sx
                bh = h * sy
                boxes_xywh = np.stack([x1, y1, bw, bh], axis=1).tolist()
                idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), self.conf, self.iou)
                idxs = np.array(idxs).reshape(-1) if len(idxs) else []
                for i in idxs:
                    name = self.names[int(cls_idx[i])] if int(cls_idx[i]) < len(self.names) else "tool"
                    detections.append(
                        Detection(
                            class_name=name,
                            conf=float(scores[i]),
                            box=(float(x1[i]), float(y1[i]), float(x1[i] + bw[i]), float(y1[i] + bh[i])),
                            source="roboflow",
                        )
                    )
            self.last_latency_ms = (time.perf_counter() - t0) * 1000.0
            self.last_error = None
            return detections
        except Exception as exc:
            self.last_error = str(exc)
            return []


# Backward compatible name used by spine, app, stream_demo, server.
RoboflowDetector = ToolsOnnxDetector


def is_tool(det: Detection) -> bool:
    # Class name casing varies by weights file (e.g. "Tools" vs "tool"),
    # so compare case-insensitively and accept either singular or plural.
    return det.source == "roboflow" or det.class_name.lower() in ("tool", "tools")
