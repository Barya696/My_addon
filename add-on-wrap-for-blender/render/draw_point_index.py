# Draw point index

import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader

from ..operators.timer_operator import redraw_all_views
from ..utils.view_helper import (
    get_all_view3d_areas,
    get_source_target_areas
)
from ..utils.point_pair_helper import get_source_points


############################################################
# Draw point index

_draw_point_index_handlers = []


def remove_draw_point_index_handlers():
  """
  Remove all draw handlers
  """
  global _draw_point_index_handlers

  for handler in _draw_point_index_handlers:
    try:
      bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')
    except:
      # Handler might have been already removed
      pass

  _draw_point_index_handlers.clear()

  # Force redraw all areas
  redraw_all_views(bpy.context)


def setup_draw_point_index_handlers():
  """
  Setup draw handlers for all 3D viewports
  """
  global _draw_point_index_handlers

  # Clear any existing handlers first
  remove_draw_point_index_handlers()

  # Add a single draw handler that will work for all viewports
  # The handler will automatically be called for each viewport when they redraw
  handler = bpy.types.SpaceView3D.draw_handler_add(
      draw_point_index_overlay, (), 'WINDOW', 'POST_VIEW')

  _draw_point_index_handlers.append(handler)

  # Force redraw all 3D viewports to trigger the handler
  redraw_all_views(bpy.context)


############################################################
# Enable

def draw_point_index_update(self, context):
  if self.draw_free or self.draw_fixed:
    setup_draw_point_index_handlers()
  else:
    remove_draw_point_index_handlers()


############################################################
# Draw points

def draw_point_index_overlay():
  if not hasattr(bpy.context.scene, 'wrap_options'):
    return

  context = bpy.context
  area = context.area
  wrap_options = context.scene.wrap_options

  draw_free = wrap_options.draw_free
  draw_fixed = wrap_options.draw_fixed
  point_size = wrap_options.point_size

  if not draw_free and not draw_fixed:
    return

  # Check area
  window_area_pairs = get_all_view3d_areas(context)
  source_area, _ = get_source_target_areas(window_area_pairs)

  if area != source_area:
    return

  # Get points
  free_points = None
  fixed_points = None

  if draw_free:
    indices = np.fromstring(wrap_options.free_index, np.int32, sep=' ')
    free_points = get_source_points(context, indices)

  if draw_fixed:
    indices = np.fromstring(wrap_options.fixed_index, np.int32, sep=' ')
    fixed_points = get_source_points(context, indices)

  if not free_points and not fixed_points:
    return

  # Batch
  shader = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')

  batch_free, batch_fixed = None, None

  if free_points:
    batch_free = batch_for_shader(shader, 'POINTS', {"pos": free_points})

  if fixed_points:
    batch_fixed = batch_for_shader(shader, 'POINTS', {"pos": fixed_points})

  # Enable Depth Testing (Hides points behind objects)
  # depth_test = gpu.state.depth_test_get()
  # gpu.state.depth_test_set('LESS_EQUAL')

  # Draw points
  if context.mode == 'EDIT_MESH':
    point_size *= 1.5

  gpu.state.point_size_set(point_size)

  if batch_free:
    shader.uniform_float("color", (1.0, 0.0, 1.0, 1.0))
    batch_free.draw(shader)

  if batch_fixed:
    shader.uniform_float("color", (0.0, 0.0, 1.0, 1.0))
    batch_fixed.draw(shader)

  # Restore state
  # gpu.state.depth_test_set(depth_test)
