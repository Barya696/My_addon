# Face landmarks
#
# The DTU-3D set of 73 facial landmarks (Deep-MVLM): which ones outline which
# feature, so point pairs that carry a landmark number can be drawn as a face.
# Numbers are 1-based, as in docs/DTU-3D_landmark_info.txt. Ear points found
# from the ear's shape are numbered from 101.

from . import body_landmarks as body

# Around each ear, in drawing order
EAR_POINTS = ('helix_top', 'back_edge', 'lobe_bottom', 'front_top')
EAR_FIRST = {'right': 101, 'left': 105}


def ear_number(side, name):
  return EAR_FIRST[side] + EAR_POINTS.index(name)


# Found but not made into point pairs: the pupils sit in an open eye socket
# on a template, so their nearest vertex is on the lid edge and the pair pulls
# the lid into the eye; the inner lips sit on the lip contact line, and
# pinning them folds the lower lip into a flap
DTU3D_LEFT_OUT = (33, 34, 57, 58, 59, 60, 61, 62)

# The face's right side (the character's right) and its mirror on the left,
# and the landmarks on the middle of the face
DTU3D_MIRROR = {
    1: 9, 2: 16, 3: 15, 4: 14, 5: 13, 6: 12, 7: 11, 8: 10,          # brows
    17: 25, 18: 32, 19: 31, 20: 30, 21: 29, 22: 28, 23: 27, 24: 26,  # eyes
    41: 39, 42: 38, 43: 37, 44: 36, 45: 35,                          # nose
    47: 53, 48: 52, 49: 51, 56: 54,                                  # lips
}
DTU3D_MIRROR.update({EAR_FIRST['right'] + k: EAR_FIRST['left'] + k
                     for k in range(len(EAR_POINTS))})               # ears
DTU3D_MIDLINE = (40, 46, 50, 55)


# (feature, landmark numbers in drawing order, closed loop)
DTU3D_OUTLINES = (
    ('brows', (1, 2, 3, 4, 5, 6, 7, 8), True),             # right eyebrow
    ('brows', (9, 10, 11, 12, 13, 14, 15, 16), True),      # left eyebrow
    ('eyes', (17, 18, 19, 20, 21, 22, 23, 24), True),      # right eye
    ('eyes', (25, 26, 27, 28, 29, 30, 31, 32), True),      # left eye
    ('nose', (35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45), False),  # bridge, wings, columella
    ('nose', (40, 46), False),                             # columella to the tip
    ('lips', (47, 48, 49, 50, 51, 52, 53, 54, 55, 56), True),  # outer lips
    ('jaw', (63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73), False),  # ear to ear
    ('ears', (101, 102, 103, 104), True),                  # right ear
    ('ears', (105, 106, 107, 108), True),                  # left ear
)

FEATURE_COLORS = {
    'brows': (1.0, 0.85, 0.25, 1.0),
    'eyes': (0.3, 0.85, 1.0, 1.0),
    'nose': (1.0, 0.55, 0.2, 1.0),
    'lips': (1.0, 0.35, 0.6, 1.0),
    'jaw': (0.45, 0.95, 0.45, 1.0),
    'ears': (0.7, 0.5, 1.0, 1.0),
}


_FEATURE = {n: feature for feature, numbers, _ in DTU3D_OUTLINES for n in numbers}


# On-screen numbers: the landmarks made into point pairs counted 1 to 62 in
# order (brows, eyes, nose, lips, ears), without the gaps of the left-out ones
_LABEL = {n: i for i, n in enumerate(
    [n for n in range(1, 63) if n not in DTU3D_LEFT_OUT] +
    list(range(EAR_FIRST['right'], EAR_FIRST['left'] + len(EAR_POINTS))), 1)}


def shown_number(number):
  """
  The number shown next to a landmark point (face or body)
  """
  if body.is_body(number):
    return body.shown_number(number)
  return _LABEL.get(number, number)


def landmark_label(number):
  return str(shown_number(number))


def parse_numbers(text):
  """
  '31-34, 50' -> {31, 32, 33, 34, 50}; anything else is ignored
  """
  out = set()
  for item in text.split(','):
    a, _, b = (s.strip() for s in item.partition('-'))
    if a.isdigit() and (b.isdigit() or not b):
      lo, hi = sorted((int(a), int(b or a)))
      out.update(range(lo, hi + 1))
  return out


def landmark_filter(wrap_options):
  """
  A function telling whether a face landmark (by its stored number) is used:
  not when its on-screen number is in the Exclude list
  """
  excluded = parse_numbers(wrap_options.landmark_exclude)

  def used(number):
    return shown_number(number) not in excluded
  return used


def feature_color(number):
  """
  Color of the feature a landmark belongs to (face or body)
  """
  if body.is_body(number):
    return body.feature_color(number)
  return FEATURE_COLORS.get(_FEATURE.get(number), (1.0, 1.0, 1.0, 1.0))


def outlines(positions):
  """
  positions: {landmark number: point}. Returns [(color, points, closed)] for
  every outline, skipping landmarks that are missing
  """
  out = []
  for feature, numbers, closed in DTU3D_OUTLINES:
    pts = [positions[n] for n in numbers if n in positions]
    if len(pts) >= 2:
      out.append((FEATURE_COLORS[feature], pts, closed and len(pts) > 2))
  return out
