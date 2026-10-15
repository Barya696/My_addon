# Operators for ring markers

import bpy
from bpy.props import EnumProperty, IntProperty
from bpy.types import Menu, Operator, UIList
from bpy_extras import view3d_utils
from mathutils import Vector

from .timer_operator import redraw_all_views
from ..utils.view_helper import (
    get_all_view3d_areas,
    get_source_target_areas
)
from ..core.ring_markers import (
    SIDES,
    state,
    ring,
    handles,
    editable,
    side_params,
    set_side_params,
    settle,
    place_at_view_center,
    new_marker,
    clear_markers,
    clear_geometry,
    update_ring_pairs,
    find_marker,
    remove_marker,
    add_mirror,
    mirror_into,
    sync_mirror,
    ring_visible,
    active_marker
)


HINT = ("Ring: drag white dot = slide (Shift: move freely), axis tip = tilt, "
        "ring dot = start  |  click another ring to pick it  |  X delete  |  Esc done")


############################################################
# Helpers

def view_areas(context):
  """
  (side, area, WINDOW region, region_3d) of the source and target views
  """
  source_area, target_area = get_source_target_areas(
      get_all_view3d_areas(context))
  out = []
  for side, area in zip(SIDES, (source_area, target_area)):
    if area:
      region = next((r for r in area.regions if r.type == 'WINDOW'), None)
      if region:
        out.append((side, area, region, area.spaces.active.region_3d))
  return out


def check_ready(self, context):
  wrap_options = context.scene.wrap_options
  if not wrap_options.source_mesh_name or not wrap_options.target_mesh_name:
    self.report({'WARNING'},
                "The source mesh and target mesh have not yet been set")
    return None
  views = view_areas(context)
  if len(views) != 2:
    self.report({'WARNING'}, "There should be two side-by-side 3D views")
    return None
  return views


############################################################
# Add marker

class TECA3D_VIEW3D_OT_add_ring_marker(Operator):
  bl_idname = 'view3d.add_ring_marker'
  bl_label = 'Add Ring Marker'
  bl_description = 'Add a ring at the center of the source view and of the target view, then adjust both by hand'
  bl_options = {'REGISTER', 'UNDO'}

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options
    views = check_ready(self, context)
    if not views:
      return {'CANCELLED'}
    if wrap_options.source_mesh_mode != 'ORIGINAL':
      self.report({'WARNING'}, 'To do this, you must be in original mode')
      return {'CANCELLED'}

    clear_geometry()
    marker = new_marker(wrap_options)
    for side, area, region, rv3d in views:
      if not place_at_view_center(context, marker, side, region, rv3d):
        wrap_options.ring_markers.remove(len(wrap_options.ring_markers) - 1)
        self.report({'WARNING'}, f"Center the {side} view on its mesh first")
        return {'CANCELLED'}

    update_ring_pairs(context)
    wrap_options.draw_rings = True   # its handles show: it is the picked ring
    redraw_all_views(context)
    return {'FINISHED'}


############################################################
# Adjust markers

class TECA3D_VIEW3D_OT_edit_ring_markers(Operator):
  bl_idname = 'view3d.edit_ring_markers'
  bl_label = 'Adjust Ring Markers'
  bl_description = 'Drag the ring handles in both views: the center to move a ring, the axis tip to tilt it, the colored dot to set where it starts'

  def invoke(self, context, event):
    if state['editing']:
      return {'CANCELLED'}
    views = check_ready(self, context)
    if not views:
      return {'CANCELLED'}

    clear_geometry()
    self.drag = None
    state['editing'] = True
    state['hover'] = None
    context.scene.wrap_options.draw_rings = True
    for _, area, _, _ in views:
      area.header_text_set(HINT)
    context.window_manager.modal_handler_add(self)
    redraw_all_views(context)
    return {'RUNNING_MODAL'}

  # Mouse

  def view_under_mouse(self, context, event):
    for side, area, region, rv3d in view_areas(context):
      if region.x <= event.mouse_x < region.x + region.width and \
              region.y <= event.mouse_y < region.y + region.height:
        return side, region, rv3d
    return None

  def handle_under_mouse(self, context, event):
    found = self.view_under_mouse(context, event)
    if not found:
      return None
    side, region, rv3d = found
    if not editable(context, side):
      return None
    mouse = Vector((event.mouse_x - region.x, event.mouse_y - region.y))
    reach = max(12.0, context.scene.wrap_options.point_size * 1.5)
    best = None
    picked = active_marker(context.scene.wrap_options)
    for marker in ([picked] if picked is not None else []):
      if not ring_visible(context.scene.wrap_options, marker):
        continue
      loop = ring(context, marker, side)
      if loop is None:
        continue
      for name, pos in handles(marker, side, loop).items():
        p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, pos)
        if p2 is None:
          continue
        d = (p2 - mouse).length
        if d <= reach and (best is None or d < best[0]):
          best = (d, (marker.uid, side, name))
    return best[1] if best else None

  def ring_under_mouse(self, context, event):
    """Index of the shown ring whose line passes under the mouse, or None"""
    found = self.view_under_mouse(context, event)
    if not found:
      return None
    side, region, rv3d = found
    wrap_options = context.scene.wrap_options
    mouse = Vector((event.mouse_x - region.x, event.mouse_y - region.y))
    reach = max(10.0, wrap_options.line_width * 3.0)
    best = None
    for i, marker in enumerate(wrap_options.ring_markers):
      if not ring_visible(wrap_options, marker):
        continue
      loop = ring(context, marker, side)
      if loop is None:
        continue
      for p in loop.pts[::max(1, len(loop.pts) // 64)]:
        p2 = view3d_utils.location_3d_to_region_2d(region, rv3d, Vector(p))
        if p2 is not None:
          d = (p2 - mouse).length
          if d <= reach and (best is None or d < best[0]):
            best = (d, i)
    return best[1] if best else None

  # Drag

  def start_drag(self, context, hover):
    uid, side, name = hover
    _, marker = find_marker(context.scene.wrap_options, uid)
    region, rv3d = next((r, v) for s, _, r, v in view_areas(context) if s == side)
    loop = ring(context, marker, side)
    self.drag = {
        'uid': uid, 'side': side, 'name': name,
        'region': region, 'rv3d': rv3d,
        'start': side_params(marker, side),
        'tip': handles(marker, side, loop)['axis'],
    }

  def update_drag(self, context, event):
    self._update_drag(context, event)
    # A mirrored ring's partner follows live on the template (settled when
    # the drag ends); on the target each side is adjusted on its own
    if self.drag and self.drag['side'] == 'source':
      _, marker = find_marker(context.scene.wrap_options, self.drag['uid'])
      if marker is not None and marker.mirror_uid:
        _, partner = find_marker(context.scene.wrap_options, marker.mirror_uid)
        if partner is not None:
          mirror_into(context, marker, partner, self.drag['side'], settle_after=False)

  def _update_drag(self, context, event):
    d = self.drag
    _, marker = find_marker(context.scene.wrap_options, d['uid'])
    if marker is None:
      self.drag = None
      return
    region, rv3d, side = d['region'], d['rv3d'], d['side']
    coord = (event.mouse_x - region.x, event.mouse_y - region.y)
    center, normal, _ = side_params(marker, side)

    if d['name'] == 'center':
      if event.shift:
        # Free move, e.g. onto another part of the body
        new = view3d_utils.region_2d_to_location_3d(
            region, rv3d, coord, d['start'][0])
        set_side_params(marker, side, center=new)
        return
      # Slide: to the point of the ring's axis nearest to the mouse ray, then
      # onto the middle of the ring there, so the ring follows the limb even
      # where the axis is not quite along it
      axis = normal.normalized()
      o = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
      ray = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord).normalized()
      b = axis.dot(ray)
      if 1.0 - b * b < 1e-6:
        return  # looking straight down the axis
      w = center - o
      t = (b * ray.dot(w) - axis.dot(w)) / (1.0 - b * b)
      set_side_params(marker, side, center=center + axis * t)
      loop = ring(context, marker, side)
      if loop is not None:
        set_side_params(marker, side, center=Vector(loop.centroid()))

    elif d['name'] == 'axis':
      tip = view3d_utils.region_2d_to_location_3d(region, rv3d, coord, d['tip'])
      if (tip - center).length > 1e-6:
        set_side_params(marker, side, normal=tip - center)

    else:
      # The ring point under the mouse; where the front and back of the ring
      # overlap on screen, the one facing the viewer
      loop = ring(context, marker, side)
      if loop is None:
        return
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

  def end_drag(self, context, cancel=False):
    d = self.drag
    self.drag = None
    _, marker = find_marker(context.scene.wrap_options, d['uid'])
    if marker is None:
      return
    if cancel:
      c, n, s = d['start']
      set_side_params(marker, d['side'], center=c, normal=n, seam=s)
      if d['side'] == 'source':
        sync_mirror(context, marker, ('source',))
      return
    if d['name'] != 'seam':
      settle(context, marker, d['side'])
    if d['side'] == 'source':
      sync_mirror(context, marker, ('source',))
    update_ring_pairs(context)
    bpy.ops.ed.undo_push(message="Adjust Ring Marker")

  # Modal

  def finish(self, context):
    state['editing'] = False
    state['hover'] = None
    for _, area, _, _ in view_areas(context):
      area.header_text_set(None)
    redraw_all_views(context)

  def modal(self, context, event):
    wrap_options = context.scene.wrap_options
    if not state['editing']:
      self.finish(context)
      return {'CANCELLED'}

    if event.type == 'MOUSEMOVE':
      if self.drag:
        self.update_drag(context, event)
        redraw_all_views(context)
        return {'RUNNING_MODAL'}
      hover = self.handle_under_mouse(context, event)
      if hover != state['hover']:
        state['hover'] = hover
        redraw_all_views(context)
      return {'PASS_THROUGH'}

    if event.type == 'LEFTMOUSE':
      if event.value == 'PRESS' and state['hover'] and not self.drag:
        self.start_drag(context, state['hover'])
        return {'RUNNING_MODAL'}
      if event.value == 'PRESS' and not self.drag:
        # A click on another ring's line picks that ring
        i = self.ring_under_mouse(context, event)
        if i is not None and i != wrap_options.ring_index:
          wrap_options.ring_index = i
          state['hover'] = None
          redraw_all_views(context)
          return {'RUNNING_MODAL'}
      if event.value == 'RELEASE' and self.drag:
        self.end_drag(context)
        redraw_all_views(context)
        return {'RUNNING_MODAL'}
      return {'PASS_THROUGH'}

    if self.drag and event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
      self.end_drag(context, cancel=True)
      redraw_all_views(context)
      return {'RUNNING_MODAL'}

    if event.type in {'X', 'DEL'} and event.value == 'PRESS' and state['hover']:
      i, _ = find_marker(wrap_options, state['hover'][0])
      if i is not None:
        remove_marker(context, i)
        state['hover'] = None
        bpy.ops.ed.undo_push(message="Delete Ring Marker")
        redraw_all_views(context)
      return {'RUNNING_MODAL'}

    if event.type in {'ESC', 'RIGHTMOUSE', 'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
      self.finish(context)
      return {'FINISHED'}

    return {'PASS_THROUGH'}


############################################################
# Ring list: rename, mirror, show, solo; remove, move

class TECA3D_UL_ring_markers(UIList):
  bl_idname = 'TECA3D_UL_ring_markers'

  def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
    wrap_options = context.scene.wrap_options
    row = layout.row(align=True)
    row.prop(item, "name", text="", emboss=False, placeholder=f"Ring {item.uid}")
    sub = row.row(align=True)
    sub.active = item.mirror_uid > 0
    sub.operator("view3d.mirror_ring_marker", text="", icon='MOD_MIRROR', emboss=False).uid = item.uid
    row.prop(item, "show", text="", icon='HIDE_OFF' if item.show else 'HIDE_ON', emboss=False)


class TECA3D_VIEW3D_OT_remove_ring_marker(Operator):
  bl_idname = 'view3d.remove_ring_marker'
  bl_label = 'Remove Ring'
  bl_description = 'Delete the selected ring (its mirrored ring stays)'
  bl_options = {'REGISTER', 'UNDO'}

  @classmethod
  def poll(cls, context):
    wo = context.scene.wrap_options
    return 0 <= wo.ring_index < len(wo.ring_markers)

  def execute(self, context):
    remove_marker(context, context.scene.wrap_options.ring_index)
    state['hover'] = None
    redraw_all_views(context)
    return {'FINISHED'}


class TECA3D_VIEW3D_OT_move_ring_marker(Operator):
  bl_idname = 'view3d.move_ring_marker'
  bl_label = 'Move Ring'
  bl_description = 'Move the selected ring up or down the list'
  bl_options = {'REGISTER', 'UNDO'}

  direction: EnumProperty(items=(('UP', 'Up', ''), ('DOWN', 'Down', '')))  # type: ignore

  @classmethod
  def poll(cls, context):
    return len(context.scene.wrap_options.ring_markers) > 1

  def execute(self, context):
    wo = context.scene.wrap_options
    i = wo.ring_index
    j = i - 1 if self.direction == 'UP' else i + 1
    if 0 <= i < len(wo.ring_markers) and 0 <= j < len(wo.ring_markers):
      wo.ring_markers.move(i, j)
      wo.ring_index = j
    return {'FINISHED'}


class TECA3D_VIEW3D_OT_mirror_ring_marker(Operator):
  bl_idname = 'view3d.mirror_ring_marker'
  bl_label = 'Mirror Ring'
  bl_description = ('Mirror the ring onto the other side of the body (named left / right or .L / .R). '
                    'On the template the two stay mirrored when either is adjusted; on the target the mirrored '
                    'ring starts mirrored and is then adjusted on its own. Click again to delete the mirrored ring')
  bl_options = {'REGISTER', 'UNDO'}

  uid: IntProperty()  # type: ignore

  def execute(self, context):
    wo = context.scene.wrap_options
    _, marker = find_marker(wo, self.uid)
    if marker is None:
      return {'CANCELLED'}
    if wo.source_mesh_mode != 'ORIGINAL':
      self.report({'WARNING'}, 'To do this, you must be in original mode')
      return {'CANCELLED'}
    clear_geometry()
    j, partner = find_marker(wo, marker.mirror_uid)
    if partner is not None:
      remove_marker(context, j)
    elif add_mirror(context, marker) is None:
      self.report({'WARNING'}, "The mirrored ring does not land on the mesh")
      return {'CANCELLED'}
    i, _ = find_marker(wo, self.uid)
    if i is not None:
      wo.ring_index = i
    redraw_all_views(context)
    return {'FINISHED'}


class TECA3D_VIEW3D_OT_mirror_all_ring_markers(Operator):
  bl_idname = 'view3d.mirror_all_ring_markers'
  bl_label = 'Mirror All Rings'
  bl_description = 'Mirror every ring that has no mirrored ring yet'
  bl_options = {'REGISTER', 'UNDO'}

  def execute(self, context):
    wo = context.scene.wrap_options
    if wo.source_mesh_mode != 'ORIGINAL':
      self.report({'WARNING'}, 'To do this, you must be in original mode')
      return {'CANCELLED'}
    clear_geometry()
    made = 0
    for uid in [m.uid for m in wo.ring_markers if not m.mirror_uid]:
      _, m = find_marker(wo, uid)
      if m is not None and not m.mirror_uid and add_mirror(context, m) is not None:
        made += 1
    self.report({'INFO'}, f"{made} rings mirrored")
    redraw_all_views(context)
    return {'FINISHED'}


class TECA3D_VIEW3D_OT_show_all_ring_markers(Operator):
  bl_idname = 'view3d.show_all_ring_markers'
  bl_label = 'Show All Rings'
  bl_description = 'Show every ring again'
  bl_options = {'REGISTER', 'UNDO'}

  def execute(self, context):
    wo = context.scene.wrap_options
    for m in wo.ring_markers:
      if not m.show:
        m.show = True
    redraw_all_views(context)
    return {'FINISHED'}


class TECA3D_MT_ring_markers(Menu):
  bl_idname = 'TECA3D_MT_ring_markers'
  bl_label = 'Rings'

  def draw(self, context):
    layout = self.layout
    layout.operator("view3d.mirror_all_ring_markers", icon='MOD_MIRROR')
    layout.operator("view3d.show_all_ring_markers", icon='HIDE_OFF')
    layout.separator()
    layout.operator("view3d.clear_ring_markers", text="Delete All Rings", icon='TRASH')


############################################################
# Clear markers

class TECA3D_VIEW3D_OT_clear_ring_markers(Operator):
  bl_idname = 'view3d.clear_ring_markers'
  bl_label = 'Clear Ring Markers'
  bl_description = 'Delete all ring markers'
  bl_options = {'REGISTER', 'UNDO'}

  def execute(self, context):
    clear_markers(context)
    state['hover'] = None
    redraw_all_views(context)
    return {'FINISHED'}

  def invoke(self, context, event):
    if not len(context.scene.wrap_options.ring_markers):
      return {'CANCELLED'}
    return context.window_manager.invoke_props_dialog(
        self, title="All ring markers will be deleted")


#############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_add_ring_marker,
    TECA3D_VIEW3D_OT_edit_ring_markers,
    TECA3D_VIEW3D_OT_clear_ring_markers,
    TECA3D_UL_ring_markers,
    TECA3D_VIEW3D_OT_remove_ring_marker,
    TECA3D_VIEW3D_OT_move_ring_marker,
    TECA3D_VIEW3D_OT_mirror_ring_marker,
    TECA3D_VIEW3D_OT_mirror_all_ring_markers,
    TECA3D_VIEW3D_OT_show_all_ring_markers,
    TECA3D_MT_ring_markers)
