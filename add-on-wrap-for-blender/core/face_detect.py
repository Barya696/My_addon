# Face detection
#
# Exports the head of a mesh for the face landmark detector (Deep-MVLM, run
# by snap-to-stick-iteration/landmarks/face_landmarks.py in its own Python),
# runs it, and maps the landmarks back to world space.
#
# The detector wants millimetres, the face looking along +Z and Y up; Blender
# meshes face -Y with Z up, at real scale in metres.

import json
import os
import subprocess

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from .face_landmarks import EAR_POINTS

HEAD_DEPTH = 0.28        # metres from the top of the head kept for detection
FACE_LANDMARKS = 62      # DTU-3D 1-62 (brows, eyes, pupils, nose, lips); the jaw is left out
EAR_ATTACHMENTS = (('right', 63, -1.0), ('left', 73, 1.0))  # DTU-3D, faces look -Y
EAR_THIN = 0.02          # an ear flap is thinner than this, a skull is ~0.15 m
SKIN_REACH = 0.012       # a ray out from open skin meets nothing within this (an eyelid's inner rim meets the lid within ~10 mm)
SKIN_SEARCH = 0.02       # how far from a landmark to look for open skin

# Blender (x, y, z), face toward -Y, Z up -> (x, z, -y): face toward +Z, Y up
R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def mesh_triangles(obj):
  """
  World-space vertices and triangles of the mesh as shown
  """
  dg = bpy.context.evaluated_depsgraph_get()
  eo = obj.evaluated_get(dg)
  me = eo.to_mesh()
  V = np.empty(len(me.vertices) * 3)
  me.vertices.foreach_get('co', V)
  M = np.array(obj.matrix_world)
  V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
  me.calc_loop_triangles()
  T = np.empty(len(me.loop_triangles) * 3, dtype=np.int64)
  me.loop_triangles.foreach_get('vertices', T)
  eo.to_mesh_clear()
  return V, T.reshape(-1, 3)


def export_head(V, T, path):
  """
  Write the head (the top HEAD_DEPTH of the mesh, or all of a head-only mesh)
  as OBJ for the detector. Returns the centre used, in world space.
  """
  top = V[:, 2].max()
  keep = V[:, 2] >= top - HEAD_DEPTH
  tris = T[keep[T].all(axis=1)]
  used = np.unique(tris)
  remap = -np.ones(len(V), dtype=np.int64)
  remap[used] = np.arange(len(used))
  P = V[used]
  center = (P.min(axis=0) + P.max(axis=0)) / 2.0
  Q = (P - center) @ R.T * 1000.0
  with open(path, 'w') as f:
    f.write(''.join(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in Q))
    f.write(''.join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in remap[tris]))
  return center


def detector_paths(folder):
  """
  (python, runner) of the landmarks folder, or None if it is not set up
  """
  if not folder:
    return None
  folder = bpy.path.abspath(folder)
  python = os.path.join(folder, '.venv', 'Scripts', 'python.exe')
  runner = os.path.join(folder, 'face_landmarks.py')
  if os.path.isfile(python) and os.path.isfile(runner):
    return python, runner
  return None


def run_detector(paths, mesh_path, out_path, views):
  """
  Run the detector on one exported head (blocking, no console window).
  Returns the landmarks (n, 3) in the exported space and the seconds taken.
  """
  python, runner = paths
  flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
  done = subprocess.run([python, runner, mesh_path, out_path, '--views', str(views)],
                        capture_output=True, text=True, creationflags=flags)
  if done.returncode != 0 or not os.path.isfile(out_path):
    lines = (done.stderr or done.stdout or '').strip().splitlines()
    raise RuntimeError(lines[-1] if lines else f"exit code {done.returncode}")
  with open(out_path) as f:
    data = json.load(f)
  return np.array(data['landmarks']), data.get('seconds', 0.0)


def to_world(landmarks, center):
  """
  Exported space (millimetres, face along +Z) -> world space
  """
  return (landmarks / 1000.0) @ R + center


def vertex_normals(V, T):
  face_normals = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
  N = np.zeros_like(V)
  for k in range(3):
    np.add.at(N, T[:, k], face_normals)
  return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


class Skin:
  """
  Which vertices are on the open skin: a ray out along the normal does not
  meet the mesh again. The inner rim of an eyelid, the inside of the mouth or
  of a nostril do: a landmark must not be moved onto those.
  """

  def __init__(self, V, T):
    self.V = V
    self.N = vertex_normals(V, T)
    self.bvh = BVHTree.FromPolygons([Vector(v) for v in V], T.tolist())
    self.known = {}

  def open(self, i):
    if i not in self.known:
      hit = self.bvh.ray_cast(Vector(self.V[i] + self.N[i] * 1e-4),
                              Vector(self.N[i]), SKIN_REACH)
      self.known[i] = hit[0] is None
    return self.known[i]

  def nearest(self, point, among=None):
    """
    The nearest open-skin vertex to `point` (of the vertex indices `among`,
    default all), or the nearest vertex if there is no open skin nearby
    """
    cand = np.arange(len(self.V)) if among is None else np.asarray(among)
    d = np.linalg.norm(self.V[cand] - point, axis=1)
    order = np.argsort(d)
    for k in order:
      if d[k] > SKIN_SEARCH:
        break
      if self.open(int(cand[k])):
        return int(cand[k])
    return int(cand[order[0]])


def find_ears(V, T, landmarks):
  """
  The ears next to the ear attachment landmarks, from their shape: the thin
  part of the surface (an inward ray leaves the mesh within EAR_THIN), as one
  connected piece. In it: the top of the helix, the back edge at mid-height,
  the bottom of the lobe and the front top, where the helix meets the head.
  Returns {'right' / 'left': {name: vertex index} or None if no flap is found}
  """
  N = vertex_normals(V, T)
  bvh = BVHTree.FromPolygons([Vector(v) for v in V], T.tolist())

  out = {}
  for side, number, outward in EAR_ATTACHMENTS:
    d = V - landmarks[number - 1]
    near = np.where((np.linalg.norm(d, axis=1) < 0.075) & (d[:, 0] * outward > -0.01))[0]
    thin = set()
    for i in near.tolist():
      hit = bvh.ray_cast(Vector(V[i] - N[i] * 1e-4), Vector(-N[i]), EAR_THIN)
      if hit[0] is not None:
        thin.add(i)
    out[side] = None
    if not thin:
      continue

    # The connected thin piece nearest the attachment
    adj = {}
    for a, b, c in T[np.isin(T, near).all(axis=1)].tolist():
      for x, y in ((a, b), (b, c), (c, a)):
        if x in thin and y in thin:
          adj.setdefault(x, set()).add(y)
          adj.setdefault(y, set()).add(x)
    start = min(thin, key=lambda i: np.linalg.norm(d[i]))
    piece, stack = {start}, [start]
    while stack:
      for j in adj.get(stack.pop(), ()):
        if j not in piece:
          piece.add(j)
          stack.append(j)
    E = np.array(sorted(piece))
    if len(E) < 30:
      continue

    up, back = d[E, 2], d[E, 1]
    h0, h1 = up.min(), up.max()
    mid = (up > h0 + 0.35 * (h1 - h0)) & (up < h0 + 0.65 * (h1 - h0))
    upper = (up > h0 + 0.55 * (h1 - h0)) & (up < h0 + 0.9 * (h1 - h0))
    if not mid.any() or not upper.any():
      continue
    found = {'helix_top': E[np.argmax(up)],
             'back_edge': E[mid][np.argmax(back[mid])],
             'lobe_bottom': E[np.argmin(up)],
             'front_top': E[upper][np.argmin(back[upper])]}
    out[side] = {name: int(found[name]) for name in EAR_POINTS}
  return out
