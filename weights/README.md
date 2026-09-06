# Model weights

Only the small component detector is committed. Everything else in this folder (SH17 and the
gloves/helmet/vest fine-tunes, the Roboflow tools ONNX) is gitignored and fetched or trained
locally; see `services/relay_receiver/README.md` and the `label/` tooling on `feat/gabe-dev`.

## components_yolov8.onnx (12 MB) + components_yolov8.classes.txt

YOLOv8 nano, 640x640, 14 Arduino-kit classes. Used by `services/relay_receiver/detect.py` to
draw boxes on the dashboard and pre-trigger the Claude identification.

Source: Roboflow Universe project **arduino-lcxdx** v3 by workspace `gab-qwrl3`,
https://universe.roboflow.com/gab-qwrl3/arduino-lcxdx , licensed **CC BY 4.0**
(https://creativecommons.org/licenses/by/4.0/). Exported unchanged from Roboflow's inference
model cache on 2026-09-05. Attribution to the dataset author is required if this model or
anything derived from it is redistributed.
