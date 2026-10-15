# Operator for body landmarks

import json
import os
import tempfile
import threading

import bpy
import numpy as np
from mathutils import Vector

from .timer_operator import Timer_Operator, redraw_all_views
from ..preferences.preferences import get_wrap_prefs
from ..core.face_detect import mesh_triangles, Skin
from ..core.body_landmarks import (
    BODY_FIRST,
    COCO,
    PAIR_POINTS,
    MIRROR,
    RINGS,
    RING_KIND,
    RING_LIMITS,
    ring_params,
    detector_paths,
    export_body,
    run_detector
)
from ..core.ring_markers import (
    new_marker,
    set_side_params,
    ring,
    settle,
    clear_geometry,
    update_ring_pairs
)


############################################################
# Source points are kept: the template does not change

CACHE_VERSION = 1


def cached_source(wrap_options, obj):
  """
  {COCO index: world point} found on the source, or None
  """
  try:
    data = json.loads(wrap_options.body_landmark_source or '{}')
  except ValueError:
    return None
  if data.get('key') == [obj.name, len(obj.data.vertices)] and data.get('version') == CACHE_VERSION:
    return {int(k): v for k, v in data['points'].items()}
  return None


def store_source(wrap_options, obj, points):
  wrap_options.body_landmark_source = json.dumps(
      {'key': [obj.name, len(obj.data.vertices)], 'version': CACHE_VERSION,
       'points': {str(k): v for k, v in points.items()}})


############################################################
# Points found -> point pairs and rings

def source_vertices(wrap_options, points, V, T):
  """
  {COCO index: source vertex} of the pair points, on open skin; made
  symmetric right to left when the source is symmetrized
  """
  skin = Skin(V, T)
  out = {COCO[n]: skin.nearest(np.array(points[COCO[n]])) for n in PAIR_POINTS if COCO[n] in points}
  if wrap_options.source_symmetrized:
    sym = np.fromstring(wrap_options.source_vertex_sym_idxs, np.int64, sep=' ')
    if len(sym) == len(V):
      for right, left in MIRROR.items():
        if right in out:
          out[left] = int(sym[out[right]])
        elif left in out:
          out[right] = int(sym[out[left]])
  return out


def apply_body_landmarks(context, source_points, target_points, source_mesh,
                         make_pairs=True, ring_kinds=None):
  """
  Replace the pairs and rings Body Landmarks placed before with new ones.
  source_points / target_points: {COCO index: world point}; source_mesh: the
  source's (V, T) the points were found on. make_pairs: add the fingertip,
  toe and heel pairs; ring_kinds: the kinds of rings to place ('arm',
  'ankle', 'finger', 'foot'; None: all). Returns (pairs added, rings added).
  """
  wrap_options = context.scene.wrap_options
  target = bpy.data.objects.get(wrap_options.target_mesh_name)

  # Point pairs: fingertips, toes, heels. Earlier body pairs go; a pair on
  # the same source vertex picked by hand or from the face wins
  point_pairs = wrap_options.point_pairs
  for i in reversed(range(len(point_pairs))):
    if point_pairs[i].landmark >= BODY_FIRST:
      point_pairs.remove(i)
  taken = {p.source_vertex_index for p in point_pairs}

  src = source_vertices(wrap_options, source_points, *source_mesh)
  M = target.matrix_world
  M_inv = M.inverted()
  pairs = 0
  for name in (PAIR_POINTS if make_pairs else ()):
    i = COCO[name]
    if i not in src or i not in target_points or src[i] in taken:
      continue
    ok, loc, normal, face = target.closest_point_on_mesh(M_inv @ Vector(target_points[i]))
    if not ok:
      continue
    item = point_pairs.add()
    item.source_vertex_index = src[i]
    item.target_pos = M @ loc
    item.target_normal = (M.to_3x3() @ normal).normalized()
    item.target_face_index = face
    item.landmark = BODY_FIRST + i
    taken.add(src[i])
    pairs += 1

  # Rings: upper arms, elbows, wrists, ankles. Earlier rings from here go;
  # rings added by hand stay. A ring whose plane misses either mesh is left out.
  markers = wrap_options.ring_markers
  for i in reversed(range(len(markers))):
    if markers[i].landmark > 0:
      markers.remove(i)
  clear_geometry()
  s_rings, t_rings = ring_params(source_points), ring_params(target_points)
  rings = 0
  for number, name, kind in RINGS:
    if ring_kinds is not None and kind not in ring_kinds:
      continue
    if number not in s_rings or number not in t_rings:
      continue
    m = new_marker(wrap_options, name)
    m.landmark = number
    ok = True
    max_radius, max_shift = RING_LIMITS[RING_KIND[number]]
    for side, (center, normal, seam) in (('source', s_rings[number]), ('target', t_rings[number])):
      set_side_params(m, side, center=Vector(center), normal=Vector(normal), seam=Vector(seam))
      loop = ring(context, m, side)
      if loop is None or loop.radius() > max_radius or               np.linalg.norm(loop.centroid() - np.asarray(center)) > max_shift:
        ok = False
        break
      settle(context, m, side)
    if not ok:
      markers.remove(len(markers) - 1)
      continue
    rings += 1
  update_ring_pairs(context)
  if rings:
    wrap_options.draw_rings = True
  return pairs, rings


############################################################
# Body landmarks

class TECA3D_VIEW3D_OT_body_landmarks(Timer_Operator):
  bl_idname = 'view3d.body_landmarks'
  bl_label = 'Body Landmarks'
  bl_description = 'Find the body on the source mesh and the target mesh (joints, fingertips, toes, heels) and add point pairs for the fingertips, toes and heels and rings at the upper arms, elbows, wrists and ankles (runs the body landmark detector on the CPU, about 15 s per body)'

  total_time = 15.0

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options
    source = bpy.data.objects.get(wrap_options.source_mesh_name)
    target = bpy.data.objects.get(wrap_options.target_mesh_name)
    if not source or not target:
      self.report({'WARNING'}, "The source mesh and target mesh have not yet been set")
      return {'CANCELLED'}
    if wrap_options.source_mesh_mode != 'ORIGINAL':
      self.report({'WARNING'}, 'To do this, you must be in original mode')
      return {'CANCELLED'}

    prefs = get_wrap_prefs(context)
    paths = detector_paths(prefs.body_landmarks_dir, prefs.face_landmarks_dir)
    if not paths:
      self.report({'WARNING'}, "Set the Body Landmarks folder in the add-on preferences "
                  "(snap-to-stick-iteration/body_landmarks, with its .venv)")
      return {'CANCELLED'}

    # Export the meshes here; the detector runs in the background
    folder = tempfile.mkdtemp(prefix='wrap_body_')
    self.jobs = []
    self.source_points = cached_source(wrap_options, source)
    self.source_mesh = mesh_triangles(source)
    if self.source_points is None:
      path = os.path.join(folder, 'source.obj')
      export_body(*self.source_mesh, path)
      self.jobs.append(('source', path))
    V, T = mesh_triangles(target)
    path = os.path.join(folder, 'target.obj')
    export_body(V, T, path)
    self.jobs.append(('target', path))

    self.total_time = 15.0 * len(self.jobs)
    self.results, self.error, self.seconds = {}, None, 0.0
    views = prefs.body_landmark_views

    def work():
      try:
        for side, path in self.jobs:
          points, seconds = run_detector(paths, path, path[:-4] + '.json', views)
          self.results[side] = points
          self.seconds += seconds
      except Exception as e:
        self.error = str(e)
      self.is_running = False

    result = Timer_Operator.invoke(self, context, event)
    threading.Thread(target=work, daemon=True).start()
    return result

  def on_stop(self, context):
    wrap_options = context.scene.wrap_options
    if self.error or 'target' not in self.results:
      self.report({'WARNING'}, f"Body landmarks failed: {self.error}")
      return
    if self.source_points is None:
      self.source_points = self.results['source']
      store_source(wrap_options, bpy.data.objects.get(wrap_options.source_mesh_name), self.source_points)

    pairs, rings = apply_body_landmarks(context, self.source_points, self.results['target'], self.source_mesh)
    self.draw_points = True   # show them when the progress ends
    redraw_all_views(context)
    self.report({'INFO'}, f"{pairs} body point pairs and {rings} rings added ({self.seconds:.0f} s)")


#############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_body_landmarks,)
