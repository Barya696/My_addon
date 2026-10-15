# Ring handles as gizmos
#
# The ring picked in the ring list shows three handles in the source and the
# target view: the center (slide along the ring's axis, Shift: move freely),
# the axis tip (tilt) and the seam (where the ring starts). They are Blender
# gizmos: nothing runs between drags, so renaming a ring, Ctrl+Z, G / R / S and
# every other tool work as usual while the handles are shown. Each drag is one
# undo step. On the source a mirrored ring follows live (mirrored); on the
# target each side is adjusted on its own.

import math

import bpy
from bpy.types import Gizmo, GizmoGroup
from bpy_extras import view3d_utils
from mathutils import Matrix, Vector

from ..core.ring_markers import (
    ring,
    handles,
    editable,
    side_params,
    set_side_params,
    settle,
    find_marker,
    active_marker,
    ring_visible,
    mirror_into,
    sync_mirror,
    update_ring_pairs
)
from ..utils.view_helper import get_all_view3d_areas, get_source_target_areas


def view_side(context):
  """'source' / 'target' for the two wrap views, else None"""
  source_area, target_area = get_source_target_areas(get_all_view3d_areas(context))
  if context.area is not None and context.area == source_area:
    return 'source'
  if context.area is not None and context.area == target_area:
    return 'target'
  return None


def _disc(segments=20):
  tris = []
  for i in range(segments):
    a0, a1 = i * math.tau / segments, (i + 1) * math.tau / segments
    tris += [(0.0, 0.0, 0.0), (math.cos(a0), math.sin(a0), 0.0), (math.cos(a1), math.sin(a1), 0.0)]
  return tris


############################################################
# One handle

class TECA3D_GT_ring_handle(Gizmo):
  bl_idname = 'TECA3D_GT_ring_handle'

  def setup(self):
    if not hasattr(self, 'shape'):
      self.shape = self.new_custom_shape('TRIS', _disc())
      self.role = 'center'
      self.side = 'source'
      self.uid = 0

  def draw(self, context):
    # a dark rim under the disc, so the handle reads on any surface
    color = tuple(self.color)
    self.color = (0.0, 0.0, 0.0)
    self.draw_custom_shape(self.shape, matrix=self.matrix_world @ Matrix.Scale(1.45, 4))
    self.color = color
    self.draw_custom_shape(self.shape)

  def draw_select(self, context, select_id):
    self.draw_custom_shape(self.shape, select_id=select_id)

  def invoke(self, context, event):
    wo = context.scene.wrap_options
    _, marker = find_marker(wo, self.uid)
    if marker is None:
      return {'CANCELLED'}
    loop = ring(context, marker, self.side)
    if loop is None:
      return {'CANCELLED'}
    self.start = tuple(v.copy() for v in side_params(marker, self.side))
    self.tip = handles(marker, self.side, loop)['axis']
    return {'RUNNING_MODAL'}

  def modal(self, context, event, tweak):
    wo = context.scene.wrap_options
    _, marker = find_marker(wo, self.uid)
    if marker is None:
      return {'CANCELLED'}
    region, rv3d = context.region, context.region_data
    coord = (event.mouse_region_x, event.mouse_region_y)
    side = self.side
    center, normal, _ = side_params(marker, side)

    if self.role == 'center':
      if event.shift:
        # Free move, e.g. onto another part of the body
        new = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, self.start[0])
        set_side_params(marker, side, center=new)
      else:
        # Slide: to the point of the axis nearest the mouse ray, then onto the
        # middle of the ring there, so it follows the limb
        axis = normal.normalized()
        o = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        ray = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord).normalized()
        b = axis.dot(ray)
        if 1.0 - b * b > 1e-6:
          w = center - o
          t = (b * ray.dot(w) - axis.dot(w)) / (1.0 - b * b)
          set_side_params(marker, side, center=center + axis * t)
          loop = ring(context, marker, side)
          if loop is not None:
            set_side_params(marker, side, center=Vector(loop.centroid()))

    elif self.role == 'axis':
      tip = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, self.tip)
      if (tip - center).length > 1e-6:
        set_side_params(marker, side, normal=tip - center)

    else:
      # The ring point under the mouse; where front and back overlap on
      # screen, the one facing the viewer
      loop = ring(context, marker, side)
      if loop is not None:
        mouse = Vector(coord)
        eye = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
        ray = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord).normalized()
        best = None
        for p in loop.pts:
          p = Vector(p)
          p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, p)
          if p2 is not None:
            key = (round((p2 - mouse).length / 6.0), (p - eye).dot(ray))
            if best is None or key < best[0]:
              best = (key, p)
        if best:
          set_side_params(marker, side, seam=best[1])

    # The mirrored ring follows live on the source
    if side == 'source' and marker.mirror_uid:
      _, partner = find_marker(wo, marker.mirror_uid)
      if partner is not None:
        mirror_into(context, marker, partner, 'source', settle_after=False)
    for area in context.screen.areas:
      if area.type == 'VIEW_3D':
        area.tag_redraw()
    return {'RUNNING_MODAL'}

  def exit(self, context, cancel):
    wo = context.scene.wrap_options
    _, marker = find_marker(wo, self.uid)
    if marker is None:
      return
    if cancel:
      c, n, s = self.start
      set_side_params(marker, self.side, center=c, normal=n, seam=s)
    elif self.role != 'seam':
      settle(context, marker, self.side)
    if self.side == 'source':
      sync_mirror(context, marker, ('source',))
    if not cancel:
      update_ring_pairs(context)
      bpy.ops.ed.undo_push(message="Adjust Ring")
    for area in context.screen.areas:
      if area.type == 'VIEW_3D':
        area.tag_redraw()


############################################################
# The picked ring's handles in each wrap view

class TECA3D_GGT_ring_handles(GizmoGroup):
  bl_idname = 'TECA3D_GGT_ring_handles'
  bl_label = 'Ring Handles'
  bl_space_type = 'VIEW_3D'
  bl_region_type = 'WINDOW'
  bl_options = {'3D', 'PERSISTENT'}

  @classmethod
  def poll(cls, context):
    wo = getattr(context.scene, 'wrap_options', None)
    if wo is None or not wo.draw_rings:
      return False
    marker = active_marker(wo)
    if marker is None or not ring_visible(wo, marker):
      return False
    side = view_side(context)
    return side is not None and editable(context, side)

  def setup(self, context):
    self.handles = {}
    for role in ('center', 'axis', 'seam'):
      gz = self.gizmos.new(TECA3D_GT_ring_handle.bl_idname)
      gz.role = role
      gz.scale_basis = 1.0
      gz.use_draw_scale = False   # sized in pixels in draw_prepare
      gz.use_draw_modal = True
      gz.alpha = 0.95
      gz.alpha_highlight = 1.0
      gz.color_highlight = (1.0, 1.0, 1.0)
      self.handles[role] = gz

  def draw_prepare(self, context):
    from ..render.draw_rings import ring_color
    wo = context.scene.wrap_options
    marker = active_marker(wo)
    side = view_side(context)
    loop = ring(context, marker, side) if marker is not None and side else None
    for gz in self.handles.values():
      gz.hide = loop is None
    if loop is None:
      return
    color = ring_color(min(marker.uid, marker.mirror_uid) if marker.mirror_uid else marker.uid)[:3]
    region, rv3d = context.region, context.region_data
    face = rv3d.view_rotation.to_matrix().to_4x4()
    size = float(wo.point_size)
    for role, pos in handles(marker, side, loop).items():
      gz = self.handles[role]
      gz.uid, gz.side = marker.uid, side
      # a disc facing the viewer, its radius in screen pixels as the old dots
      px = size * (0.8 if role == 'center' else 0.65)
      p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, pos)
      r = 1e-3
      if p2 is not None:
        edge = view3d_utils.region_2d_to_location_3d(region, rv3d, (p2.x + px, p2.y), pos)
        r = max((edge - pos).length, 1e-6)
      gz.matrix_basis = Matrix.Translation(pos) @ face @ Matrix.Scale(r, 4)
      gz.color = (0.95, 0.95, 0.95) if role == 'center' else color


#############################################################
# All classes

classes = (
    TECA3D_GT_ring_handle,
    TECA3D_GGT_ring_handles)
