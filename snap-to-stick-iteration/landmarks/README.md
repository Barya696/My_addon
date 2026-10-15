# Face landmarks

Automatic face point pairs for the Wrap add-on. The add-on's **Face Landmarks**
button exports the head of the source and of the target, runs
`face_landmarks.py` here on each, and turns the landmarks into point pairs.

- **Model:** [Deep-MVLM](https://github.com/RasmusRPaulsen/Deep-MVLM) (Paulsen et al.,
  "Multi-view Consensus CNN for 3D Facial Landmark Placement"), DTU-3D set,
  `geometry+depth` model, so it works on untextured scans. The model file is in
  `Deep-MVLM/saved/trained/`.
- **Landmarks used:** 1-62 (brows, eyes, pupils, nose, lips). The jaw line
  (63-73) is left out: it is the least precise part.
- **Runs on the CPU**, outside Blender, in its own Python environment (`.venv`).

## Setup

Run `setup_windows.bat` once (it installs PyTorch for CPU and VTK into `.venv`).
Then, in Blender, open the Wrap add-on preferences and set **Face Landmarks** to
this folder.

## Changes to Deep-MVLM

The 2019 code is patched to run on current libraries:

- `logger/__init__.py`, `base/__init__.py`: TensorBoard and the trainer are
  optional (prediction does not need them)
- `utils3d/render3d.py`: random view angles as scalars (NumPy 2)
- `utils3d/utils3d.py`: points passed to VTK as three numbers (VTK 9)

## By hand

```
.venv\Scripts\python.exe face_landmarks.py head.obj landmarks.json --views 32
```

The mesh must be in millimetres, centred, face along +Z, Y up.
