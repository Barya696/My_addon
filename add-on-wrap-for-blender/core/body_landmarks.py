# Body landmarks
#
# Whole-body points from RTMW (COCO-WholeBody, 133 points), found by
# snap-to-stick-iteration/body_landmarks/body_landmarks.py in its own Python
# on the whole source and target mesh. Two uses:
#   - point pairs on the surface points the detector finds well: the
#     fingertips, the toes and the heels. They carry the landmark number
#     BODY_FIRST + their COCO index, and are shown numbered after the face
#     landmarks (63 on), so the Exclude field works for them too.
#   - ring markers placed from the joints: the upper arms, elbows, wrists and
#     ankles. Hips and knees are left out: the detector is least sure there.
#
# The detector wants metres, the body facing +Z and Y up; Blender meshes face
# -Y with Z up (the same turn as the face detector's, face_detect.R).

import json
import os
import subprocess

import numpy as np

BODY_FIRST = 200            # stored landmark number = BODY_FIRST + COCO index
LABEL_FIRST = 63            # shown numbers continue after the face's 1-62

# COCO-WholeBody indices used here
COCO = {
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
    'left_big_toe': 17, 'left_small_toe': 18, 'left_heel': 19,
    'right_big_toe': 20, 'right_small_toe': 21, 'right_heel': 22,
}
_FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
for _side, _first in (('left', 91), ('right', 112)):
  COCO[f'{_side}_hand_wrist'] = _first
  for _k, _f in enumerate(_FINGERS):
    COCO[f'{_side}_{_f}_1'] = _first + 1 + 4 * _k
    COCO[f'{_side}_{_f}_2'] = _first + 2 + 4 * _k
    COCO[f'{_side}_{_f}_3'] = _first + 3 + 4 * _k
    COCO[f'{_side}_{_f}_tip'] = _first + 4 + 4 * _k

# The points made into point pairs, in the order they are numbered on screen
PAIR_POINTS = ([f'left_{f}_tip' for f in _FINGERS] + [f'right_{f}_tip' for f in _FINGERS] +
               ['left_big_toe', 'left_small_toe', 'left_heel',
                'right_big_toe', 'right_small_toe', 'right_heel'])

# The character's right side and its mirror on the left (for a symmetric source)
MIRROR = {COCO[n]: COCO[n.replace('right_', 'left_')] for n in PAIR_POINTS if n.startswith('right_')}

_LABEL = {BODY_FIRST + COCO[n]: LABEL_FIRST + k for k, n in enumerate(PAIR_POINTS)}
_FEATURE = {BODY_FIRST + COCO[n]: ('hands' if '_tip' in n else 'feet') for n in PAIR_POINTS}
FEATURE_COLORS = {
    'hands': (0.35, 0.55, 1.0, 1.0),
    'feet': (0.4, 1.0, 0.8, 1.0),
}


def is_body(number):
  return number >= BODY_FIRST


def shown_number(number):
  return _LABEL.get(number, number)


def feature_color(number):
  return FEATURE_COLORS.get(_FEATURE.get(number), (1.0, 1.0, 1.0, 1.0))


############################################################
# Rings from the joints

# (ring number, name, kind); the number is kept on the ring marker so a new
# run replaces the rings it placed and leaves rings added by hand alone
RINGS = ((1, 'left upper arm', 'arm'), (2, 'left elbow', 'arm'), (3, 'left wrist', 'arm'), (4, 'left ankle', 'ankle'),
         (5, 'right upper arm', 'arm'), (6, 'right elbow', 'arm'), (7, 'right wrist', 'arm'), (8, 'right ankle', 'ankle'))
_FINGER_RINGS = []          # per hand: thumb's first bone, the other fingers' first and second bones
for _side, _base in (('left', 20), ('right', 40)):
  _FINGER_RINGS.append((_base, f'{_side} thumb', 'finger', f'{_side}_thumb_2', f'{_side}_thumb_3'))
  for _k, _f in enumerate(_FINGERS[1:]):
    _FINGER_RINGS.append((_base + 1 + 2 * _k, f'{_side} {_f} 1', 'finger', f'{_side}_{_f}_1', f'{_side}_{_f}_2'))
    _FINGER_RINGS.append((_base + 2 + 2 * _k, f'{_side} {_f} 2', 'finger', f'{_side}_{_f}_2', f'{_side}_{_f}_3'))
_FOOT_RINGS = ((60, 'left mid-foot', 'foot', 'left', 0.55), (61, 'left fore-foot', 'foot', 'left', 0.8),
               (62, 'right mid-foot', 'foot', 'right', 0.55), (63, 'right fore-foot', 'foot', 'right', 0.8))
RINGS = RINGS + tuple(r[:3] for r in _FINGER_RINGS) + tuple(r[:3] for r in _FOOT_RINGS)
RING_KIND = {number: kind for number, _, kind in RINGS}

# A ring is kept only if its loop is this size at most (metres, mean radius)
# and its middle this near where it was placed: a ring that slipped onto the
# next finger or around a whole hand is left out
RING_LIMITS = {'arm': (0.09, 0.06), 'ankle': (0.08, 0.05), 'finger': (0.016, 0.012), 'foot': (0.07, 0.04)}

UPPER_ARM = 0.35            # the upper-arm ring sits this far from the shoulder to the elbow
SEAM_REACH = 0.1            # metres: the seam hint lies this far out from the center


def _unit(v):
  n = np.linalg.norm(v)
  return v / n if n > 1e-12 else None


def _across(direction, normal):
  """direction with its part along normal removed, as a unit vector"""
  return _unit(direction - (direction @ normal) * normal)


def ring_params(points):
  """
  points: {COCO index: world point}. Returns {ring number: (center, normal,
  seam hint)}: each ring's plane cuts its limb across, and its seam starts
  toward the thumb side (arms) or the toes (ankles), on any mesh alike
  """
  P = {name: np.asarray(points[i]) for name, i in COCO.items() if i in points}
  out = {}
  for side, first in (('left', 1), ('right', 5)):
    s, e, w = P.get(f'{side}_shoulder'), P.get(f'{side}_elbow'), P.get(f'{side}_wrist')
    thumb, pinky = P.get(f'{side}_thumb_1'), P.get(f'{side}_pinky_1')
    radial = (thumb - pinky) if thumb is not None and pinky is not None else None
    if s is not None and e is not None:
      n = _unit(e - s)
      if n is not None:
        out[first] = (s + UPPER_ARM * (e - s), n, radial)
    if s is not None and e is not None and w is not None:
      a, b = _unit(e - s), _unit(w - e)
      n = _unit(a + b) if a is not None and b is not None else None
      if n is not None:
        out[first + 1] = (e, n, radial)
    if e is not None and w is not None:
      n = _unit(w - e)
      if n is not None:
        out[first + 2] = (w, n, radial)
    k, an = P.get(f'{side}_knee'), P.get(f'{side}_ankle')
    toe, heel = P.get(f'{side}_big_toe'), P.get(f'{side}_heel')
    if k is not None and an is not None:
      n = _unit(an - k)
      if n is not None:
        out[first + 3] = (an, n, (toe - heel) if toe is not None and heel is not None else None)

  # Fingers: across each bone between two finger joints; seam toward the
  # same side of the hand on any mesh (the hand's plane normal)
  for number, _, _, a_name, b_name in _FINGER_RINGS:
    side = a_name.split('_')[0]
    a, b = P.get(a_name), P.get(b_name)
    wr, ix, pk = P.get(f'{side}_hand_wrist'), P.get(f'{side}_index_1'), P.get(f'{side}_pinky_1')
    if a is None or b is None:
      continue
    n = _unit(b - a)
    toward = np.cross(ix - wr, pk - wr) if wr is not None and ix is not None and pk is not None else None
    if n is not None:
      out[number] = (0.5 * (a + b), n, toward)

  # Feet: across the foot from the heel to the toes, at a fraction of the
  # way; seam toward the top of the foot
  for number, _, _, side, f in _FOOT_RINGS:
    heel, big, small = P.get(f'{side}_heel'), P.get(f'{side}_big_toe'), P.get(f'{side}_small_toe')
    if heel is None or big is None or small is None:
      continue
    toes = 0.5 * (big + small)
    n = _unit(toes - heel)
    if n is not None:
      out[number] = (heel + f * (toes - heel), n, np.array([0.0, 0.0, 1.0]))
  # seam hint: a point out from the center, across the limb, toward the
  # thumb side / the toes
  done = {}
  for number, (center, normal, toward) in out.items():
    d = _across(toward, normal) if toward is not None else None
    if d is None:
      d = _across(np.array([0.0, -1.0, 0.0]), normal)       # the front
    if d is None:
      d = _across(np.array([1.0, 0.0, 0.0]), normal)
    done[number] = (center, normal, center + SEAM_REACH * d)
  return done


############################################################
# The detector

def detector_paths(folder, face_folder=''):
  """
  (python, runner) of the body landmarks folder, or None if it is not set
  up. Without a folder set: body_landmarks next to the face landmarks folder.
  """
  import bpy
  candidates = []
  if folder:
    candidates.append(bpy.path.abspath(folder))
  if face_folder:
    candidates.append(os.path.join(os.path.dirname(os.path.normpath(bpy.path.abspath(face_folder))), 'body_landmarks'))
  for folder in candidates:
    python = os.path.join(folder, '.venv', 'Scripts', 'python.exe')
    runner = os.path.join(folder, 'body_landmarks.py')
    if os.path.isfile(python) and os.path.isfile(runner):
      return python, runner
  return None


# Blender (x, y, z), body toward -Y, Z up -> (x, z, -y): toward +Z, Y up
R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def export_body(V, T, path):
  """
  Write the whole mesh (world space, metres) as OBJ for the detector
  """
  Q = V @ R.T
  with open(path, 'w') as f:
    f.write(''.join(f"v {x:.6f} {y:.6f} {z:.6f}\n" for x, y, z in Q))
    f.write(''.join(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in T))


def run_detector(paths, mesh_path, out_path, views):
  """
  Run the detector on one exported mesh (blocking, no console window).
  Returns {COCO index: world point} (surface points on the surface, joints
  where the views meet) and the seconds taken.
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
  points = {}
  for p in data['points']:
    q = p.get('surface_point') or p.get('point')
    if q is not None:
      points[int(p['index'])] = (np.array(q) @ R).tolist()
  return points, data.get('seconds', 0.0)
