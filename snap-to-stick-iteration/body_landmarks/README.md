# Body landmarks (prototype)

Automatic whole-body points for the Wrap add-on, found the way the face
landmarks are: the mesh is rendered from 36 directions, a 2D keypoint model
marks each picture, and the views are combined in 3D (RANSAC over pairs of
rays, then least squares on the views that agree).

- **Model:** RTMW-x 384x288 (OpenMMLab, Apache-2.0), 133 COCO-WholeBody points:
  body 17, feet 6, face 68, two hands of 21. Run through `rtmlib` on ONNX
  Runtime, CPU only. No person detector: the whole render is the box.
- **Joints** (shoulders, elbows, wrists, hips, knees, ankles, finger joints) are
  where the rays meet, inside the body. **Surface points** (nose, eyes, ears,
  toes, heels, fingertips, face) are also moved onto the nearest surface point.
- About 14 s per mesh (36 renders and model runs).

## Setup

`.venv` holds Python 3.14 with `rtmlib`, `onnxruntime`, `opencv-python-headless`,
`vtk` and `numpy`. The model (219 MB, downloaded once from download.openmmlab.com)
is kept in `models\hub\checkpoints\`.

## By hand

```
.venv\Scripts\python.exe body_landmarks.py body.obj landmarks.json --views 36 --debug DIR
```

The mesh is in metres, Y up, facing +Z. `--debug` saves every render with its points.

## First test (2026-10-14, eve and the MetaHuman-style template)

All 133 points found on both. Template points carried onto eve through a
wrap, against the points found on eve (median / max): fingertips 3.4 / 9 mm,
finger joints and wrists 4.5 / 8 mm, face 3.7 / 10 mm, toes and heels 10 / 36 mm,
body joints 11 / 56 mm (hips and knees are the least sure).
