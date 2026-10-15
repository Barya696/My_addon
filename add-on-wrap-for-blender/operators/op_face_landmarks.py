# Operator for face landmarks

import json
import os
import tempfile
import threading

import bpy
import numpy as np
from mathutils import Vector

from .timer_operator import Timer_Operator, redraw_all_views
from ..preferences.preferences import get_wrap_prefs
from ..core.face_detect import (
    FACE_LANDMARKS,
    mesh_triangles,
    export_head,
    detector_paths,
    run_detector,
    to_world,
    find_ears,
    Skin
)
from ..core.face_landmarks import (
    EAR_POINTS,
    ear_number,
    DTU3D_LEFT_OUT,
    DTU3D_MIRROR,
    DTU3D_MIDLINE
)
from ..core.body_landmarks import BODY_FIRST


############################################################
# Source landmarks are kept: the template does not change

CACHE_VERSION = 2   # 2: landmarks on open skin only


def cached_source(wrap_options, obj):
  """
  {'vertices': face landmark vertices, 'ears': {side: {name: vertex}}}
  """
  try:
    data = json.loads(wrap_options.face_landmark_source or '{}')
  except ValueError:
    return None
  if data.get('key') == [obj.name, len(obj.data.vertices)]           and data.get('version') == CACHE_VERSION:
    return data
  return None


def store_source(wrap_options, obj, vertices, ears):
  wrap_options.face_landmark_source = json.dumps(
      {'key': [obj.name, len(obj.data.vertices)], 'version': CACHE_VERSION,
       'vertices': vertices, 'ears': ears})


############################################################
# The source is symmetric, the target is not

def symmetrize_source(wrap_options, vertices, skin):
  """
  {landmark number: source vertex} made symmetric, right to left: each left
  landmark becomes the mirror vertex of its right one, and the landmarks on
  the middle of the face an open-skin vertex on the symmetry plane
  """
  if not wrap_options.source_symmetrized:
    return vertices
  sym = np.fromstring(wrap_options.source_vertex_sym_idxs, np.int64, sep=' ')
  if not len(sym):
    return vertices
  out = dict(vertices)
  for right, left in DTU3D_MIRROR.items():
    if right in out:
      out[left] = int(sym[out[right]])
    elif left in out:            # the right one was not found (an ear)
      out[right] = int(sym[out[left]])

  on_plane = np.nonzero(sym == np.arange(len(sym)))[0]
  if len(on_plane) and len(skin.V) == len(sym):
    for n in DTU3D_MIDLINE:
      if n in out:
        out[n] = skin.nearest(skin.V[out[n]], among=on_plane)
  return out


############################################################
# Face landmarks

class TECA3D_VIEW3D_OT_face_landmarks(Timer_Operator):
  bl_idname = 'view3d.face_landmarks'
  bl_label = 'Face Landmarks'
  bl_description = 'Find the face (brows, eyes, nose, lips) on the source mesh and the target mesh and add them as point pairs (runs the face landmark detector on the CPU, about a minute per face)'

  total_time = 60.0

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
    paths = detector_paths(prefs.face_landmarks_dir)
    if not paths:
      self.report({'WARNING'}, "Set the Face Landmarks folder in the add-on preferences "
                  "(snap-to-stick-iteration/landmarks, after running its setup_windows.bat)")
      return {'CANCELLED'}

    # Export the heads here; the detector runs in the background
    folder = tempfile.mkdtemp(prefix='wrap_face_')
    self.jobs = []
    self.source = cached_source(wrap_options, source)
    V, T = mesh_triangles(source)
    self.source_mesh = (V, T)
    if self.source is None:
      path = os.path.join(folder, 'source.obj')
      self.jobs.append(('source', path, export_head(V, T, path)))
    V, T = mesh_triangles(target)
    self.target_mesh = (V, T)
    path = os.path.join(folder, 'target.obj')
    self.jobs.append(('target', path, export_head(V, T, path)))

    self.total_time = 60.0 * len(self.jobs)
    self.results, self.error, self.seconds = {}, None, 0.0
    views = prefs.face_landmark_views

    def work():
      try:
        for side, path, center in self.jobs:
          landmarks, seconds = run_detector(paths, path, path[:-4] + '.json', views)
          self.results[side] = to_world(landmarks, center)
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
      self.report({'WARNING'}, f"Face landmarks failed: {self.error}")
      return

    source = bpy.data.objects.get(wrap_options.source_mesh_name)
    target = bpy.data.objects.get(wrap_options.target_mesh_name)

    # Source: the template vertex nearest to each landmark, and its ears (kept
    # for next time)
    V, T = self.source_mesh
    skin = Skin(V, T)
    if self.source is None:
      vertices = [skin.nearest(p) for p in self.results['source'][:FACE_LANDMARKS]]
      ears = find_ears(V, T, self.results['source'])
      store_source(wrap_options, source, vertices, ears)
      self.source = {'vertices': vertices, 'ears': ears}
    V_t, T_t = self.target_mesh
    target_ears = find_ears(V_t, T_t, self.results['target'])

    # Source vertices by landmark number, made symmetric (right to left)
    src = {n: v for n, v in enumerate(self.source['vertices'], 1)
           if n not in DTU3D_LEFT_OUT}
    for side, names in self.source['ears'].items():
      if names:
        src.update({ear_number(side, name): v for name, v in names.items()})
    src = symmetrize_source(wrap_options, src, skin)

    # Target points by landmark number, as found on each side
    tgt = {n: p for n, p in enumerate(self.results['target'][:FACE_LANDMARKS], 1)}
    for side, names in target_ears.items():
      if names:
        tgt.update({ear_number(side, name): V_t[v] for name, v in names.items()})

    # (landmark number, source vertex, target point)
    wanted = [(n, src[n], tgt[n]) for n in sorted(src) if n in tgt]
    ears_found = sum(1 for side in ('right', 'left')
                     if all(ear_number(side, name) in tgt for name in EAR_POINTS))

    # Replace earlier face pairs (not the body landmarks); a pair picked by
    # hand on the same vertex wins
    point_pairs = wrap_options.point_pairs
    for i in reversed(range(len(point_pairs))):
      if 0 < point_pairs[i].landmark < BODY_FIRST:
        point_pairs.remove(i)
    taken = {p.source_vertex_index for p in point_pairs}

    M = target.matrix_world
    M_inv = M.inverted()
    added = 0
    for n, vertex, p in wanted:
      if vertex in taken:
        continue
      ok, loc, normal, face = target.closest_point_on_mesh(M_inv @ Vector(p))
      if not ok:
        continue
      item = point_pairs.add()
      item.source_vertex_index = vertex
      item.target_pos = M @ loc
      item.target_normal = (M.to_3x3() @ normal).normalized()
      item.target_face_index = face
      item.landmark = n
      taken.add(vertex)
      added += 1

    self.draw_points = True   # show them when the progress ends
    redraw_all_views(context)
    self.report({'INFO'}, f"{added} face point pairs added, {ears_found} of 2 ears ({self.seconds:.0f} s)")


#############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_face_landmarks,)
