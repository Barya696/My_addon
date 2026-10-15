# My addon

Wrap for Blender (Blender 5.2, Windows): fit a template mesh onto a scanned
body (Init → Snap → Attract → Stick, then rig Fit), with a Python daemon that
replaces the original C++ wrapDaemon.

## Layout

| Path | What it is |
| --- | --- |
| `add-on-wrap-for-blender/` | The add-on (extension `teca3d_wrap_for_blender`): point pairs, ring markers, face and body landmarks, rig Fit. `client/` talks to the daemon over TCP. |
| `snap-to-stick-iteration/daemon/` | The Python wrapDaemon: Snap and Attract are embedded deformation on a mirror-symmetric tet cage (Gauss-Newton, PARDISO), Stick is surface ARAP. Solver version `33-mirrored-cage`. |
| `snap-to-stick-iteration/landmarks/` | Face landmarks (Deep-MVLM, CPU) for the Face Landmarks button. |
| `snap-to-stick-iteration/body_landmarks/` | Whole-body landmarks (RTMW-x, 36 views) for the Body Landmarks button. |
| `snap-to-stick-iteration/addon_patches/` | Older small patches to the add-on. |
| `snap-to-stick-iteration/app/`, `components/`, … | Empty Next.js placeholder, unused. |
| `cpp_probe/` | Black-box probes of the original C++ daemon: scripts, recordings (`ground_truth/`) and `REPORT.md`. |
| `fit_rig_to_shape.py`, `swap_def_retarget.py` | Standalone Blender scripts for fitting the rig. |
| `eve_wrap_demo.blend` | Demo scene with hand-picked point pairs. |
| `eve_wrap_demo_fresh.blend` | The same scene without hand-picked pairs. |
| `*_world.npz` | World-space results of the body-landmark wrap tests. |

## Not in the repo, and how to get it back

- **Daemon build:** run `snap-to-stick-iteration\daemon\build_windows.bat`, then
  copy `dist\wrapDaemon\` (`wrapDaemon.exe` and `_internal\`) into the add-on's `bin\`.
- **Face detector:** run `snap-to-stick-iteration\landmarks\setup_windows.bat`
  (creates `.venv`). The weights (71 MB) come from the URL in
  `Deep-MVLM/deepmvlm/api.py` and go in `Deep-MVLM/saved/trained/`.
- **Body detector:** `.venv` with Python 3.14, rtmlib, onnxruntime,
  opencv-python-headless, vtk and numpy. The model (219 MB, from
  download.openmmlab.com) goes in `body_landmarks/models/hub/checkpoints/`;
  see `body_landmarks/README.md`.
- **Installing the add-on:** copy `add-on-wrap-for-blender\` with its `bin\` to
  `%APPDATA%\Blender Foundation\Blender\5.2\extensions\user_default\teca3d_wrap_for_blender\`,
  then set the two detector folders in the add-on's preferences.
- Also left out: backups, test scenes, zips, research papers and the original
  TECA 3D daemon.

## Status (2026-10-15)

- Daemon solver 33 is built and installed.
- Renaming a ring no longer freezes Blender (fixed 2026-10-15). The cause: the
  per-vertex positions of each stage were stored as lists of 54k items, which
  Blender searched item by item after any edit in a UI list. They are now flat
  arrays (`core/vertex_store.py`). Files saved earlier are converted on load.
- Body Landmarks works but gave no better wraps on eve. It is optional.
- Accepted as is: a small gap next to held vertices in Stick (test S2).
- `daemon/README.md` still describes the first Stick ("strong surface
  projection"). Stick is now surface ARAP (`stick_solver.py`).

## Working rules

- Test on copies, never on `eve_wrap_demo.blend` itself. Save each approach
  as a new `.blend` and never overwrite existing files.
- Back up before replacing installed files.
- Never type the TECA 3D licence email or password.
- Stop test daemons by process ID, not by name.
