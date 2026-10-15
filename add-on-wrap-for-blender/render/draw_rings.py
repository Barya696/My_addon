# Draw ring markers

import colorsys
import math

import bpy
import gpu
import numpy as np
from bpy_extras import view3d_utils
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from ..operators.timer_operator import redraw_all_views
from ..utils.view_helper import (
    get_all_view3d_areas,
    get_source_target_areas
)
from ..core.ring_markers import (
    ring_visible,
    state,
    ring,
    ring_display_points,
    handles,
    editable
)


def ring_color(uid, alpha=1.0):
  """
  Every ring keeps its own color, the same in both views
  """
  r, g, b = colorsys.hsv_to_rgb((uid * 0.618034) % 1.0, 0.75, 1.0)
  return (r, g, b, alpha)


############################################################
# Handlers

_handlers = []


def remove_draw_rings_handlers():
  for handler in _handlers:
    try:
      bpy.types.SpaceView3D.draw_handler_remove(handler, 'WINDOW')
    except:
      pass
  _handlers.clear()
  redraw_all_views(bpy.context)


def setup_draw_rings_handlers():
  remove_draw_rings_handlers()
  _handlers.append(bpy.types.SpaceView3D.draw_handler_add(
      draw_rings_overlay, (), 'WINDOW', 'POST_VIEW'))
  redraw_all_views(bpy.context)


def draw_rings_update(self, context):
  if self.draw_rings:
    setup_draw_rings_handlers()
  else:
    remove_draw_rings_handlers()


############################################################
# Draw

def _lift(pts, rv3d, amount):
  """
  Move points a little toward the viewer so a ring on the surface is not
  hidden by the surface itself
  """
  if rv3d.is_perspective:
    eye = np.array(rv3d.view_matrix.inverted().translation)
    toward = eye - pts
    toward /= np.maximum(np.linalg.norm(toward, axis=1, keepdims=True), 1e-12)
  else:
    toward = np.array(rv3d.view_matrix.inverted().to_3x3().col[2])[None, :]
  return pts + toward * amount


def draw_rings_overlay():
  if not hasattr(bpy.context.scene, 'wrap_options'):
    return

  context = bpy.context
  wrap_options = context.scene.wrap_options
  if not wrap_options.draw_rings or not len(wrap_options.ring_markers):
    return

  source_area, target_area = get_source_target_areas(
      get_all_view3d_areas(context))
  if context.area == source_area:
    side = 'source'
  elif context.area == target_area:
    side = 'target'
  else:
    return

  region = context.region
  rv3d = context.region_data
  line = gpu.shader.from_builtin('POLYLINE_UNIFORM_COLOR')
  flat = gpu.shader.from_builtin('UNIFORM_COLOR')
  right = rv3d.view_rotation @ Vector((1.0, 0.0, 0.0))
  up = rv3d.view_rotation @ Vector((0.0, 1.0, 0.0))
  width = max(2.0, float(wrap_options.line_width) + 1.0)
  size = float(wrap_options.point_size)
  show_handles = state['editing'] and editable(context, side)   # old adjust mode only
  hover = state['hover']

  depth_test = gpu.state.depth_test_get()
  gpu.state.blend_set('ALPHA')

  def polyline(coords, color, depth, px=width):
    batch = batch_for_shader(line, 'LINE_STRIP', {"pos": coords})
    line.uniform_float("viewportSize", (region.width, region.height))
    gpu.state.depth_test_set(depth)
    # Dark edge under the color, so a ring reads on any surface
    line.uniform_float("lineWidth", px + 2.0)
    line.uniform_float("color", (0.0, 0.0, 0.0, 0.6 * color[3]))
    batch.draw(line)
    line.uniform_float("lineWidth", px)
    line.uniform_float("color", color)
    batch.draw(line)

  def dot(pos, color, px):
    # A round disc of `px` pixels across, facing the viewer
    p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, pos)
    if p2 is None:
      return
    edge = view3d_utils.region_2d_to_location_3d(
        region, rv3d, (p2.x + px / 2.0, p2.y), pos)
    r = (edge - pos).length
    rim = [pos + (right * math.cos(a) + up * math.sin(a)) * r
           for a in (i * math.tau / 16 for i in range(16))]
    tris = [(tuple(pos), tuple(rim[i]), tuple(rim[(i + 1) % 16])) for i in range(16)]
    batch = batch_for_shader(flat, 'TRIS', {"pos": [v for t in tris for v in t]})
    flat.uniform_float("color", color)
    batch.draw(flat)

  active = wrap_options.ring_markers[wrap_options.ring_index].uid \
      if 0 <= wrap_options.ring_index < len(wrap_options.ring_markers) else None
  for marker in wrap_options.ring_markers:
    if not ring_visible(wrap_options, marker):
      continue   # hidden in the ring list
    loop = ring(context, marker, side)
    if loop is None:
      continue
    # A mirrored pair shares one color
    color = ring_color(min(marker.uid, marker.mirror_uid) if marker.mirror_uid else marker.uid)
    px = width * 1.8 if marker.uid == active else width   # the ring selected in the list
    pts = _lift(ring_display_points(context, marker, side, loop), rv3d,
                loop.radius() * 0.02)
    coords = [tuple(p) for p in pts]
    if loop.closed:
      coords.append(coords[0])

    # Faint where the mesh hides it, solid where it is visible
    polyline(coords, color[:3] + (0.25,), 'NONE', px)
    polyline(coords, color, 'LESS_EQUAL', px)

    if marker.uid != active or not editable(context, side):
      continue   # the picked ring's axis line; its handles are gizmos

    gpu.state.depth_test_set('NONE')
    h = handles(marker, side, loop)
    polyline([tuple(h['center']), tuple(h['axis'])], color, 'NONE', 2.0)
    if not show_handles:
      continue   # (the old adjust mode, from F3, still draws its dots)
    for name, pos in h.items():
      hot = hover == (marker.uid, side, name)
      px = size * (1.6 if hot else 1.2) if name == 'center' else size * (1.4 if hot else 1.0)
      dot(pos, (0.0, 0.0, 0.0, 0.8), px + 4.0)
      fill = (1.0, 1.0, 1.0, 1.0) if hot else \
          ((0.95, 0.95, 0.95, 1.0) if name == 'center' else color)
      dot(pos, fill, px)

  gpu.state.depth_test_set(depth_test)
  gpu.state.blend_set('NONE')
