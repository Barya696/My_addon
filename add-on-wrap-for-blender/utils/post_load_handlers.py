# Post load handlers

import bpy
from bpy.app.handlers import persistent

from ..core import vertex_store
from ..core.ring_markers import state as ring_state, clear_geometry


############################################################
# Init

@persistent
def post_load_init_wrap(_):
  if not hasattr(bpy.context.scene, 'wrap_options'):
    return

  move_old_vertex_lists()

  scene = bpy.context.scene
  wrap_options = scene.wrap_options

  wrap_options.draw_points = False
  wrap_options.draw_labels = False
  wrap_options.draw_rings = False
  ring_state['editing'] = False
  ring_state['hover'] = None
  clear_geometry()

  wrap_options.source_pick_index = -1
  wrap_options.target_pick_index = -1
  wrap_options.target_pick_index_mir = -1

  wrap_options.draw_free = False
  wrap_options.draw_fixed = False

  wrap_options.wrap_server_running = False

  wrap_options.real_time_sync = False

  wrap_options.enable_clean_view = False

  wrap_options.progress_value = 0.0
  wrap_options.progress_time = 0.0
  wrap_options.progress_iter_num = 1


def move_old_vertex_lists():
  """
  Source positions saved by earlier versions as lists: into flat arrays
  """
  for scene in bpy.data.scenes:
    wrap_options = getattr(scene, 'wrap_options', None)
    if wrap_options is not None:
      vertex_store.move_old_lists(wrap_options)


############################################################
# Post handler

@persistent
def my_post_load_handler(dummy):
  print("--- Running my_post_load_handler ---")

  # 1. Find a 3D View area and a WINDOW region in the current screen
  area = next((a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'), None)
  if area is None:
    print("Error: No 3D View area found.")
    return

  region = next((r for r in area.regions if r.type == 'WINDOW'), None)
  if region is None:
    print("Error: No WINDOW region found in 3D View area.")
    return

  # 2. Use context override to execute an operator with the 3D view context
  with bpy.context.temp_override(area=area, region=region):
    # Example: Add a cube to the scene
    # This operation requires the 3D View context to work correctly
    bpy.ops.mesh.primitive_cube_add(size=2, location=(3, 0, 0))
    print("Added a cube in the 3D view context.")

  print("--- my_post_load_handler finished ---")


############################################################
# Setup

def register():
  # bpy.app.handlers.load_post.clear()

  bpy.app.handlers.load_post.append(post_load_init_wrap)
  # bpy.app.handlers.load_post.append(my_post_load_handler)

  # A file already open when the add-on is enabled
  if not bpy.app.timers.is_registered(move_old_vertex_lists):
    bpy.app.timers.register(move_old_vertex_lists, first_interval=0.1)


def unregister():
  bpy.app.handlers.load_post.remove(post_load_init_wrap)
  # bpy.app.handlers.load_post.remove(my_post_load_handler)
