# Source vertex positions of each stage
#
# The source mesh's world-space vertex positions at each stage (original,
# snap, attract, stick, fit), one flat float array per stage in the scene's
# wrap options. Plain arrays, not lists of items: whenever a field inside a
# list (a ring's name, its eye) is edited in the UI, Blender looks for that
# list item through every list of the scene item by item, and 54k-vertex
# lists made one rename take minutes.
#
# Files saved by earlier versions keep the positions in the old lists
# (source_<stage>_vertices); they are moved into the arrays on load.

import numpy as np


STAGES = ('original', 'snap', 'attract', 'stick', 'fit')


def _key(stage):
  return f'source_{stage}_coords'


def _old(wrap_options, stage):
  return getattr(wrap_options, f'source_{stage}_vertices')


def get(wrap_options, stage):
  """
  World-space positions (n, 3) float32, empty when the stage has none
  """
  flat = wrap_options.get(_key(stage))
  if flat is not None and len(flat):
    return np.array(flat, dtype=np.float32).reshape(-1, 3)   # a copy: undo may free the original
  old = _old(wrap_options, stage)
  if len(old):
    # Not moved yet (where data cannot be written): read the old list
    coords = np.empty(len(old) * 3, dtype=np.float32)
    old.foreach_get('coord', coords)
    return coords.reshape(-1, 3)
  return np.zeros((0, 3), dtype=np.float32)


def count(wrap_options, stage):
  flat = wrap_options.get(_key(stage))
  if flat is not None and len(flat):
    return len(flat) // 3
  return len(_old(wrap_options, stage))


def store(wrap_options, stage, coords):
  """
  Keep the positions of `stage` (anything (n, 3)); empty clears them
  """
  flat = np.asarray(coords, dtype=np.float32).reshape(-1)
  if len(flat):
    wrap_options[_key(stage)] = flat
  elif wrap_options.get(_key(stage)) is not None:
    del wrap_options[_key(stage)]
  old = _old(wrap_options, stage)
  if len(old):
    old.clear()


def clear(wrap_options, stage):
  store(wrap_options, stage, ())


def move_old_lists(wrap_options):
  """
  Positions saved by earlier versions: into the arrays, the old lists emptied
  """
  for stage in STAGES:
    if len(_old(wrap_options, stage)):
      store(wrap_options, stage, get(wrap_options, stage))
