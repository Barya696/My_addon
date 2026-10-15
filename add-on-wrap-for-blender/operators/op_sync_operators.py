# Synchronize 3d views

import bpy
from bpy.types import Operator

from .op_auto_master import reset_active_area


############################################################
# Sync views

def sync_views(context):
  scene = context.scene
  wrap_options = scene.wrap_options

  view3d_areas = [
      area for window in context.window_manager.windows
      for area in window.screen.areas if area.type == 'VIEW_3D']

  master_index = wrap_options.master_view_index
  if master_index >= len(view3d_areas):
    return

  master_area = view3d_areas[master_index]
  master_space = master_area.spaces.active
  master_region = master_space.region_3d

  for i, area in enumerate(view3d_areas):
    if i != master_index:
      space = area.spaces.active
      region = space.region_3d

      # Sync properties
      region.view_distance = master_region.view_distance * \
          wrap_options.sync_view_distance_adjust

      region.view_location = master_region.view_location
      region.view_rotation = master_region.view_rotation

      region.view_camera_zoom = master_region.view_camera_zoom
      region.view_camera_offset = master_region.view_camera_offset

      region.view_perspective = master_region.view_perspective

      space.clip_start = master_space.clip_start
      space.clip_end = master_space.clip_end

      space.lens = master_space.lens

      # Force view update
      region.update()


class TECA3D_VIEW3D_OT_sync_views(Operator):
  bl_idname = "view3d.sync_views"
  bl_label = "Sync Views"
  bl_description = "Synchronize source view and target view"

  def execute(self, context):
    sync_views(context)
    return {'FINISHED'}


############################################################
# Real-time Sync

last_master_view_state = None


def real_time_sync_timer():
  global last_master_view_state

  if not hasattr(bpy.context.scene, 'wrap_options'):
    last_master_view_state = None
    return None

  context = bpy.context
  scene = context.scene
  wrap_options = scene.wrap_options

  if not wrap_options.real_time_sync:
    last_master_view_state = None
    return None

  view3d_areas = [
      area for window in context.window_manager.windows
      for area in window.screen.areas if area.type == 'VIEW_3D']

  master_index = wrap_options.master_view_index
  if master_index >= len(view3d_areas):
    return wrap_options.sync_refresh_rate

  master_area = view3d_areas[master_index]
  master_space = master_area.spaces.active
  master_region = master_space.region_3d

  current_state = (
      master_region.view_location.copy(),
      master_region.view_rotation.copy(),
      master_region.view_distance,
      wrap_options.sync_view_distance_adjust,

      master_region.view_camera_zoom,
      tuple(master_region.view_camera_offset),
      master_region.view_perspective,

      master_space.clip_start,
      master_space.clip_end,
      master_space.lens,
  )

  if last_master_view_state != current_state:
    bpy.ops.view3d.sync_views()
    last_master_view_state = current_state

  return wrap_options.sync_refresh_rate


def real_time_sync_update(self, context):
  if self.real_time_sync:
    # Set an invalid value for master_view_index to prevent
    # synchronization operations until the auto master operator
    # determines the correct value for master_view_index
    #
    # Please note that the minimum value for the master_view_index
    # property is 0. Therefore, do not set it to -1, as it will
    # automatically be constrained to 0, which is clearly not the
    # desired outcome
    wrap_options = context.scene.wrap_options
    wrap_options.master_view_index = 100

    reset_active_area()

    if not bpy.app.timers.is_registered(real_time_sync_timer):
      bpy.app.timers.register(real_time_sync_timer)

    bpy.ops.view3d.auto_master_modal('INVOKE_DEFAULT')
  else:
    if bpy.app.timers.is_registered(real_time_sync_timer):
      bpy.app.timers.unregister(real_time_sync_timer)


############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_sync_views,)
