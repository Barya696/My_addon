# Draw point pairs

import bpy
import blf
import gpu
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils
from mathutils import Vector

from ..operators.timer_operator import redraw_all_views
from ..utils.view_helper import (
    get_all_view3d_areas,
    get_source_target_areas
)
from ..utils.point_pair_helper import (
    get_source_indices,
    get_source_points,
    get_target_points,
    get_source_pick_stick,
    get_target_pick,
    get_center_indices
)
from ..core.face_landmarks import outlines, feature_color, landmark_label, landmark_filter


############################################################
# Draw Points

_draw_points_handlers = []


def remove_draw_points_handlers():
  """
  Remove all draw handlers
  """
  global _draw_points_handlers

  for handler in _draw_points_handlers:
    try:
      bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')
    except:
      # Handler might have been already removed
      pass

  _draw_points_handlers.clear()

  # Force redraw all areas
  redraw_all_views(bpy.context)


def setup_draw_points_handlers():
  """
  Setup draw handlers for all 3D viewports
  """
  global _draw_points_handlers

  # Clear any existing handlers first
  remove_draw_points_handlers()

  # Add a single draw handler that will work for all viewports
  # The handler will automatically be called for each viewport when they redraw
  handler = bpy.types.SpaceView3D.draw_handler_add(
      draw_points_overlay, (), 'WINDOW', 'POST_VIEW')

  _draw_points_handlers.append(handler)

  # Force redraw all 3D viewports to trigger the handler
  redraw_all_views(bpy.context)


############################################################
# Draw labels

_draw_labels_handlers = []


def remove_draw_labels_handlers():
  """
  Remove all draw handlers
  """
  global _draw_labels_handlers

  for handler in _draw_labels_handlers:
    try:
      bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')
    except:
      # Handler might have been already removed
      pass

  _draw_labels_handlers.clear()

  # Force redraw all areas
  redraw_all_views(bpy.context)


def setup_draw_labels_handlers():
  """
  Setup draw handlers for all 3D viewports
  """
  global _draw_labels_handlers

  # Clear any existing handlers first
  remove_draw_labels_handlers()

  # Add a single draw handler that will work for all viewports
  # The handler will automatically be called for each viewport when they redraw
  handler = bpy.types.SpaceView3D.draw_handler_add(
      draw_labels_overlay, (), 'WINDOW', 'POST_PIXEL')

  _draw_labels_handlers.append(handler)

  # Force redraw all 3D viewports to trigger the handler
  redraw_all_views(bpy.context)


############################################################
# Enable

def draw_points_update(self, context):
  # Labels follow the points: face landmarks always show their number, the
  # other pairs only with Labels on
  if self.draw_points:
    setup_draw_points_handlers()
    setup_draw_labels_handlers()
  else:
    remove_draw_points_handlers()
    remove_draw_labels_handlers()


def draw_labels_update(self, context):
  redraw_all_views(context)


############################################################
# Draw points

def draw_points_overlay():
  if not hasattr(bpy.context.scene, 'wrap_options'):
    return

  context = bpy.context
  area = context.area
  wrap_options = context.scene.wrap_options

  draw_lines = wrap_options.draw_lines
  point_size = wrap_options.point_size
  line_width = wrap_options.line_width
  pick_index = wrap_options.source_pick_index

  if not wrap_options.draw_points:
    return

  # Get points
  window_area_pairs = get_all_view3d_areas(context)
  source_area, target_area = get_source_target_areas(window_area_pairs)

  points = None
  goals = None
  pick = None
  pick_stick = None
  centers = None
  if area == source_area:
    indices = get_source_indices(context)
    points = get_source_points(context, indices)
    if draw_lines:
      goals = get_target_points(context)
    if pick_index != -1:
      pick = get_source_points(context, [pick_index])
    center_indices = get_center_indices(context)
    if center_indices:
      centers = get_source_points(context, center_indices)
  elif area == target_area:
    points = get_target_points(context)
    pick = get_target_pick(context)
    pick_stick = get_source_pick_stick(context)

  if not points and not pick and not centers:
    return

  # Draw
  shader = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')

  batch_points, batch_goals, batch_lines = None, None, None
  batch_pick, batch_pick_mir, batch_pick_stick = None, None, None
  batch_centers, batch_center_line = None, None

  # Pairs picked by hand in red; face landmarks are drawn with their outlines
  plain = [p for p, pair in zip(points or [], wrap_options.point_pairs)
           if pair.landmark <= 0]
  if plain:
    batch_points = batch_for_shader(shader, 'POINTS', {"pos": plain})

  if goals:
    # Not for the face landmarks in the Exclude list
    used = landmark_filter(wrap_options)
    shown = [pair.landmark <= 0 or used(pair.landmark) for pair in wrap_options.point_pairs]
    starts = [p for p, s in zip(points, shown) if s]
    goals = [g for g, s in zip(goals, shown) if s]

  if goals:
    batch_goals = batch_for_shader(shader, 'POINTS', {"pos": goals})

    line_coords = starts + goals
    line_indices = [(i, i + len(starts)) for i in range(len(starts))]
    batch_lines = batch_for_shader(
        shader, 'LINES', {"pos": line_coords}, indices=line_indices)

  if pick:
    batch_pick = batch_for_shader(shader, 'POINTS', {"pos": pick[:1]})
    if len(pick) > 1:
      batch_pick_mir = batch_for_shader(shader, 'POINTS', {"pos": pick[1:2]})

  if pick_stick:
    batch_pick_stick = batch_for_shader(shader, 'POINTS', {"pos": pick_stick})

  if centers:
    batch_centers = batch_for_shader(shader, 'POINTS', {"pos": centers})

    if len(centers) == 2:
      batch_center_line = batch_for_shader(
          shader, 'LINES', {"pos": centers}, indices=[(0, 1)])

  # Draw goals and lines
  if batch_goals:
    shader.uniform_float("color", (1.0, 0.2, 1.0, 1.0))
    gpu.state.point_size_set(point_size * 0.6)
    batch_goals.draw(shader)

    if batch_lines:
      shader.uniform_float("color", (0.2, 1.0, 1.0, 1.0))
      gpu.state.line_width_set(line_width)
      batch_lines.draw(shader)

  # Enable Depth Testing (Hides points behind objects)
  depth_test = gpu.state.depth_test_get()
  gpu.state.depth_test_set('LESS_EQUAL')

  # Draw centers and line
  if batch_centers:
    shader.uniform_float("color", (1.0, 1.0, 0.2, 1.0))
    gpu.state.point_size_set(point_size * 1.2)
    batch_centers.draw(shader)

    if batch_center_line:
      shader.uniform_float("color", (1.0, 1.0, 0.2, 1.0))
      gpu.state.line_width_set(line_width)
      batch_center_line.draw(shader)

  # Draw points
  if batch_points:
    shader.uniform_float("color", (1.0, 0.0, 0.0, 1.0))
    gpu.state.point_size_set(point_size)
    batch_points.draw(shader)

  # Face outlines between the pairs that carry a landmark number
  draw_face_outlines(context, points, line_width, point_size)

  # Restore state
  gpu.state.depth_test_set(depth_test)

  # Draw pick
  if batch_pick_stick:
    shader.uniform_float("color", (1.0, 0.2, 1.0, 1.0))
    gpu.state.point_size_set(point_size)
    batch_pick_stick.draw(shader)

  if batch_pick:
    if batch_pick_mir:
      shader.uniform_float("color", (0.0, 0.0, 1.0, 1.0))
      batch_pick_mir.draw(shader)

    shader.uniform_float("color", (0.0, 1.0, 0.0, 1.0))
    gpu.state.point_size_set(point_size)
    batch_pick.draw(shader)


def draw_face_outlines(context, points, line_width, point_size):
  """
  Lines through the face landmarks (eyebrows, eyes, nose, lips, jaw), in the
  view's own positions of the pairs, faint where the mesh hides them, and the
  landmark points in their feature's color; not the ones in the Exclude list
  """
  if not points:
    return
  wrap_options = context.scene.wrap_options
  used = landmark_filter(wrap_options)
  positions = {p.landmark: pos for p, pos in zip(wrap_options.point_pairs, points)
               if p.landmark > 0 and used(p.landmark)}
  if not positions:
    return

  # A little toward the viewer, so points lying on the surface are not hidden
  # by the surface itself
  rv3d = context.region_data
  eye = rv3d.view_matrix.inverted().translation
  toward = rv3d.view_rotation @ Vector((0.0, 0.0, 1.0))
  for n, pos in positions.items():
    pos = Vector(pos)
    if rv3d.is_perspective:
      positions[n] = pos + (eye - pos) * 0.003
    else:
      positions[n] = pos + toward * (0.003 * rv3d.view_distance)

  region = context.region
  shader = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
  shader.uniform_float("viewportSize", (region.width, region.height))
  width = max(2.0, float(line_width))
  gpu.state.blend_set('ALPHA')

  for color, pts, closed in outlines(positions):
    coords = [tuple(p) for p in pts] + ([tuple(pts[0])] if closed else [])
    batch = batch_for_shader(shader, 'LINE_STRIP', {"pos": coords})
    for depth, alpha in (('NONE', 0.3), ('LESS_EQUAL', 1.0)):
      gpu.state.depth_test_set(depth)
      shader.uniform_float("lineWidth", width + 2.0)
      shader.uniform_float("color", (0.0, 0.0, 0.0, 0.6 * alpha))
      batch.draw(shader)
      shader.uniform_float("lineWidth", width)
      shader.uniform_float("color", color[:3] + (alpha,))
      batch.draw(shader)

  dots = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')
  gpu.state.depth_test_set('LESS_EQUAL')
  by_color = {}
  for n, pos in positions.items():
    by_color.setdefault(feature_color(n), []).append(pos)
  for color, pts in by_color.items():
    batch = batch_for_shader(dots, 'POINTS', {"pos": pts})
    gpu.state.point_size_set(point_size + 3)
    dots.uniform_float("color", (0.0, 0.0, 0.0, 0.8))
    batch.draw(dots)
    gpu.state.point_size_set(point_size)
    dots.uniform_float("color", color)
    batch.draw(dots)

  gpu.state.blend_set('NONE')


############################################################
# Draw labels

def draw_labels_overlay():
  if not hasattr(bpy.context.scene, 'wrap_options'):
    return

  context = bpy.context
  area = context.area

  wrap_options = context.scene.wrap_options
  label_size = wrap_options.label_size

  if not wrap_options.draw_points:
    return
  label_all = wrap_options.draw_labels

  # Get points
  window_area_pairs = get_all_view3d_areas(context)
  source_area, target_area = get_source_target_areas(window_area_pairs)

  points = None
  if area == source_area:
    indices = get_source_indices(context)
    points = get_source_points(context, indices)
  elif area == target_area:
    points = get_target_points(context)

  if not points:
    return

  # Draw
  region = context.region
  rv3d = context.region_data

  points_2d = [
      view3d_utils.location_3d_to_region_2d(region, rv3d, p)
      for p in points]

  labels = [str(i + 1) for i in range(len(points_2d))]
  # Face landmarks: their landmark number, in their feature's color
  colors = [feature_color(p.landmark) if p.landmark > 0 else (1.0, 0.0, 0.0, 1.0)
            for p in wrap_options.point_pairs]

  font_id = 0
  blf.size(font_id, label_size)

  used = landmark_filter(wrap_options)
  for pos, text, color, pair in zip(points_2d, labels, colors, wrap_options.point_pairs):
    if pos and (pair.landmark > 0 and used(pair.landmark) or pair.landmark <= 0 and label_all):
      blf.color(font_id, *color)
      blf.position(font_id, pos[0], pos[1], 0)
      blf.draw(font_id, landmark_label(pair.landmark) if pair.landmark > 0 else text)
