# Body landmarks: whole-body points on an untextured human mesh, found the
# way face_landmarks.py finds face points: render the mesh from many
# directions, run a 2D keypoint model on every picture, and combine the views
# in 3D.
#
# Model: RTMW-x (OpenMMLab, Apache-2.0), 133 COCO-WholeBody points (body 17,
# feet 6, face 68, two hands of 21), run through rtmlib on ONNX Runtime (CPU).
# Every view gives each point a ray from the camera; a point's 3D position is
# where the rays of the views that agree meet (RANSAC over pairs of rays, then
# least squares on the agreeing views). Joints (shoulders, elbows, knees...)
# stay where the rays meet, inside the body; surface points (toes, heels,
# fingertips, nose, eyes, ears) are also moved onto the nearest surface point.
#
#   .venv\Scripts\python.exe body_landmarks.py body.obj landmarks.json [--views 36] [--debug DIR]
#
# The mesh is in metres, Y up, facing +Z. The model file is kept in models\.
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('TORCH_HOME', os.path.join(HERE, 'models'))
MODEL_URL = ('https://download.openmmlab.com/mmpose/v1/projects/rtmw/onnx_sdk/'
             'rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip')

BODY = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
FEET = ['left_big_toe', 'left_small_toe', 'left_heel', 'right_big_toe', 'right_small_toe', 'right_heel']
HAND = ['wrist', 'thumb_1', 'thumb_2', 'thumb_3', 'thumb_tip', 'index_1', 'index_2', 'index_3', 'index_tip',
        'middle_1', 'middle_2', 'middle_3', 'middle_tip', 'ring_1', 'ring_2', 'ring_3', 'ring_tip',
        'pinky_1', 'pinky_2', 'pinky_3', 'pinky_tip']
NAMES = BODY + FEET + [f'face_{i}' for i in range(68)] + [f'left_{h}' for h in HAND] + [f'right_{h}' for h in HAND]
SURFACE = set(range(0, 5)) | set(range(17, 23)) | set(range(23, 91)) | {91 + 4 * k for k in range(1, 6)} | {112 + 4 * k for k in range(1, 6)}


def render_views(mesh_path, views, size):
  """Offscreen VTK renders around the mesh. Returns the images (top row
  first), per view a function pixel -> world ray, and the mesh locator."""
  import vtk
  from vtk.util.numpy_support import vtk_to_numpy
  reader = vtk.vtkOBJReader(); reader.SetFileName(mesh_path); reader.Update()
  poly = reader.GetOutput()
  normals = vtk.vtkPolyDataNormals(); normals.SetInputData(poly); normals.SplittingOff(); normals.Update()
  mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(normals.GetOutputPort())
  actor = vtk.vtkActor(); actor.SetMapper(mapper)
  pr = actor.GetProperty(); pr.SetColor(0.78, 0.76, 0.74); pr.SetDiffuse(0.85); pr.SetSpecular(0.15); pr.SetAmbient(0.15)
  ren = vtk.vtkRenderer(); ren.AddActor(actor); ren.SetBackground(0.35, 0.37, 0.40)
  kit = vtk.vtkLightKit(); kit.AddLightsToRenderer(ren)
  W, H = size
  win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.AddRenderer(ren); win.SetSize(W, H)
  b = np.array(poly.GetBounds()).reshape(3, 2); c = b.mean(1); height = b[1, 1] - b[1, 0]
  cam = ren.GetActiveCamera(); cam.SetViewAngle(30.0)
  dist = 0.62 * max(height, b[0, 1] - b[0, 0]) / np.tan(np.radians(15.0))
  locator = vtk.vtkCellLocator(); locator.SetDataSet(poly); locator.BuildLocator()
  # azimuths all round at three heights, the front and back more densely
  els = [-15.0, 5.0, 25.0]
  azs = np.linspace(0.0, 360.0, max(views // len(els), 4), endpoint=False)
  out = []
  for el in els:
    for az in azs:
      a, e = np.radians(az), np.radians(el)
      eye = c + dist * np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
      cam.SetFocalPoint(*c); cam.SetPosition(*eye); cam.SetViewUp(0, 1, 0)
      ren.ResetCameraClippingRange(); win.Render()
      w2i = vtk.vtkWindowToImageFilter(); w2i.SetInput(win); w2i.ReadFrontBufferOff(); w2i.Update()
      img = vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars()).reshape(H, W, -1)[::-1, :, :3].copy()
      frame = cam_frame(eye, c, H, W)
      if not out:   # once: the pinhole frame must agree with VTK's own projection
        worst = 0.0
        for u, v in ((0, 0), (W - 1, H - 1), (W // 3, 2 * H // 3)):
          ren.SetDisplayPoint(u + 0.5, H - 1 - v + 0.5, 1.0); ren.DisplayToWorld(); p = np.array(ren.GetWorldPoint())
          d = p[:3] / p[3] - eye; d /= np.linalg.norm(d)
          worst = max(worst, np.degrees(np.arccos(np.clip(d @ frame(u, v)[1], -1, 1))))
        print(f'camera check: pinhole vs VTK rays differ by {worst:.4f} deg at most', flush=True)
      out.append(dict(img=img, eye=eye.copy(), az=az, el=el, cam=frame))
  return out, locator, height, c


def cam_frame(eye, centre, H, W, fov=30.0):
  """Pinhole frame of a camera at eye looking at centre with Y up: maps a
  pixel (u right, v down) to a unit ray direction."""
  f = centre - eye; f /= np.linalg.norm(f)
  r = np.cross(f, [0.0, 1.0, 0.0]); r /= np.linalg.norm(r)
  up = np.cross(r, f)
  t = np.tan(np.radians(fov / 2.0))          # VTK's view angle is vertical
  def ray(u, v):
    x = (2.0 * (u + 0.5) / W - 1.0) * t * W / H
    y = (1.0 - 2.0 * (v + 0.5) / H) * t
    d = f + x * r + y * up
    return eye, d / np.linalg.norm(d)
  return ray


def triangulate(rays, tol):
  """rays: list of (origin, direction, weight). RANSAC over pairs, then
  weighted least squares on the inliers. Returns (point, inlier count, rms)."""
  if len(rays) < 2:
    return None, 0, None
  O = np.array([r[0] for r in rays]); D = np.array([r[1] for r in rays]); w = np.array([r[2] for r in rays])
  def solve(idx):
    A = np.zeros((3, 3)); b = np.zeros(3)
    for i in idx:
      P = np.eye(3) - np.outer(D[i], D[i]); A += w[i] * P; b += w[i] * P @ O[i]
    return np.linalg.lstsq(A, b, rcond=None)[0]
  def dist(X):
    v = X - O; return np.linalg.norm(v - (v * D).sum(1)[:, None] * D, axis=1)
  best = None
  n = len(rays)
  for i in range(n):
    for j in range(i + 1, n):
      if abs(D[i] @ D[j]) > 0.995:          # nearly parallel views tell nothing about depth
        continue
      X = solve([i, j]); inl = np.nonzero(dist(X) < tol)[0]
      score = w[inl].sum()
      if best is None or score > best[0]:
        best = (score, inl)
  if best is None or len(best[1]) < 3:
    return None, 0 if best is None else len(best[1]), None
  inl = best[1]
  for _ in range(3):
    X = solve(inl); inl = np.nonzero(dist(X) < tol)[0]
  X = solve(inl)
  return X, len(inl), float(np.sqrt((dist(X)[inl] ** 2).mean()))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('mesh'); ap.add_argument('out')
  ap.add_argument('--views', type=int, default=36)
  ap.add_argument('--size', type=int, nargs=2, default=[640, 960])
  ap.add_argument('--min-score', type=float, default=0.3)
  ap.add_argument('--debug', default='')
  args = ap.parse_args()
  t0 = time.time()
  from rtmlib import RTMPose
  model = RTMPose(onnx_model=MODEL_URL, model_input_size=(288, 384), backend='onnxruntime', device='cpu')
  views, locator, height, centre = render_views(args.mesh, args.views, tuple(args.size))
  W, H = args.size
  rays = [[] for _ in NAMES]
  per_view = []
  for vi, v in enumerate(views):
    img = v['img']
    fg = np.abs(img.astype(int) - img[0, 0].astype(int)).sum(2) > 12
    ys, xs = np.nonzero(fg)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pw, ph = 0.15 * (x1 - x0), 0.08 * (y1 - y0)
    bbox = [max(0, x0 - pw), max(0, y0 - ph), min(W - 1, x1 + pw), min(H - 1, y1 + ph)]
    k, s = model(img[:, :, ::-1].copy(), bboxes=[bbox])   # rtmlib expects BGR
    k, s = k[0], s[0]
    per_view.append((k, s))
    for i in range(len(NAMES)):
      if s[i] >= args.min_score:
        o, d = v['cam'](k[i, 0], k[i, 1]); rays[i].append((o, d, float(s[i])))
    if args.debug:
      import cv2
      os.makedirs(args.debug, exist_ok=True)
      dbg = img[:, :, ::-1].copy()
      for i in range(len(NAMES)):
        if s[i] >= args.min_score:
          col = (0, 200, 0) if i < 23 else ((0, 160, 255) if i < 91 else (255, 80, 0))
          cv2.circle(dbg, (int(k[i, 0]), int(k[i, 1])), 3, col, -1)
      cv2.imwrite(os.path.join(args.debug, f'view_{vi:02d}_az{int(v["az"]):03d}_el{int(v["el"]):+03d}.png'), dbg)
  results = []
  for i, name in enumerate(NAMES):
    tol = (0.012 if i >= 91 or 17 <= i < 23 else 0.025) * height / 1.7      # hands and feet tighter
    X, n, rms = triangulate(rays[i], tol)
    rec = dict(index=i, name=name, kind='surface' if i in SURFACE else 'joint', views=n, seen=len(rays[i]))
    if X is not None:
      rec['point'] = X.tolist(); rec['rms_m'] = rms
      if i in SURFACE:
        cp = [0.0, 0.0, 0.0]; cid = __import__('vtk').reference(0); sub = __import__('vtk').reference(0); d2 = __import__('vtk').reference(0.0)
        locator.FindClosestPoint(list(X), cp, cid, sub, d2)
        rec['surface_point'] = list(cp); rec['surface_dist_m'] = float(np.sqrt(float(d2)))
    results.append(rec)
  json.dump(dict(model='RTMW-x 384x288 (COCO-WholeBody 133)', views=len(views), height_m=height, seconds=time.time() - t0, points=results),
            open(args.out, 'w'), indent=1)
  found = sum(1 for r in results if 'point' in r)
  print(f'{found}/{len(results)} points found from {len(views)} views in {time.time() - t0:.0f} s')


if __name__ == '__main__':
  main()
