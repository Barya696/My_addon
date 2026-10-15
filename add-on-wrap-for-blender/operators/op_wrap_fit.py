# Operator for the wrap fit stage

import json
import time

import bpy
import numpy as np
from bpy.types import Operator
from mathutils import Matrix

from .timer_operator import redraw_all_views
from ..core import rig_swap
from ..core.mesh_process import (
    source_mesh_mode_update,
    build_fit_shape_key,
    FIT_ACTIVE
)
from ..core.rig_fit import Fitter
from ..core import vertex_store


############################################################
# Helpers

def get_base_world(obj):
  """
  The mesh's own vertex positions (no modifiers), world space
  """
  co = np.empty(len(obj.data.vertices) * 3)
  obj.data.vertices.foreach_get('co', co)
  M = np.array(obj.matrix_world)
  return co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]


def get_armature(obj):
  for mod in obj.modifiers:
    if mod.type == 'ARMATURE' and mod.object and mod.show_viewport:
      return mod.object
  return None


def abort_swap(rig):
  """
  Undo a swap without retargeting (used when the fit fails half way)
  """
  state = rig.get(rig_swap.STATE_KEY)
  if not state:
    return
  state = json.loads(state)
  rig_swap.remove_bindings(rig, state)
  for name, m in state.get('basis', {}).items():
    if name in rig.pose.bones:
      rig.pose.bones[name].matrix_basis = Matrix([m[0:4], m[4:8], m[8:12], m[12:16]])
  rig_swap.unmute(rig, state)


############################################################
# Wrap fit

class TECA3D_VIEW3D_OT_wrap_fit(Operator):
  bl_idname = 'view3d.wrap_fit'
  bl_label = 'Wrap Fit'
  bl_description = 'Pose the rig of the source mesh so that, on the original shape, it reproduces the stick result exactly; what the rig cannot reach goes into a "Wrap Fit" shape key that only acts in the Fit stage while the rig is in Pose Position (takes about a minute)'
  bl_options = {'REGISTER'}

  def execute(self, context):
    wrap_options = context.scene.wrap_options

    obj = bpy.data.objects.get(wrap_options.source_mesh_name)
    if not obj:
      self.report({'WARNING'}, "The source mesh has not been set")
      return {'CANCELLED'}

    rig = get_armature(obj)
    if not rig:
      self.report(
          {'WARNING'}, "The source mesh has no armature modifier with an armature")
      return {'CANCELLED'}

    n = len(obj.data.vertices)
    if wrap_options.wrap_process_stage < 4 \
            or vertex_store.count(wrap_options, 'stick') != n \
            or vertex_store.count(wrap_options, 'original') != n:
      self.report({'WARNING'}, "Run Wrap Stick before Wrap Fit")
      return {'CANCELLED'}

    start = time.time()

    # Original shape, rig at rest
    if wrap_options.source_mesh_mode != 'ORIGINAL':
      wrap_options.source_mesh_mode = 'ORIGINAL'
      source_mesh_mode_update(wrap_options, context)
    for pose_bone in rig.pose.bones:
      pose_bone.matrix_basis = Matrix.Identity(4)
    rig.data.pose_position = 'POSE'  # the fit poses the rig

    rest = vertex_store.get(wrap_options, 'original').astype(np.float64)
    target = vertex_store.get(wrap_options, 'stick').astype(np.float64)
    fitter = Fitter(obj, rig, rest, target)
    fitter.set_base_world(rest)

    # 1. Pose the deform bones, then hand the pose to the main controls
    rig_swap.swap(rig, quiet=True)
    try:
      R, t = fitter.fit_pose()
      fitter.place(R, t)
    except Exception:
      abort_swap(rig)
      raise
    rig_swap.restore(rig)
    pose_error = np.linalg.norm(fitter.mesh_world() - target, axis=1)

    # 2. What posing cannot reach becomes the Fit shape key; the base goes
    #    back to the original shape
    error = fitter.bake()
    fit_world = get_base_world(obj)
    fitter.set_base_world(rest)
    obj[FIT_ACTIVE] = 0
    build_fit_shape_key(obj, rig, fit_world.tolist())

    # 3. Store the stage: the Fit shape (to rebuild the key) and the pose
    vertex_store.store(wrap_options, 'fit', fit_world)
    wrap_options.wrap_fit_pose = json.dumps({
        'rig': rig.name,
        'bones': {pb.name: [v for row in pb.matrix_basis for v in row]
                  for pb in rig.pose.bones
                  if pb.matrix_basis != Matrix.Identity(4)}
    })
    wrap_options.wrap_process_stage = 5
    wrap_options.source_mesh_mode = 'FIT'
    source_mesh_mode_update(wrap_options, context)

    self.report(
        {'INFO'},
        f"Wrap Fit done in {time.time() - start:.0f} s: pose alone {pose_error.mean() * 1000:.1f} mm "
        f"from Stick (mean), with the Fit shape {error.max() * 1000:.3f} mm (max)")

    redraw_all_views(context)
    return {'FINISHED'}


#############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_wrap_fit,)
