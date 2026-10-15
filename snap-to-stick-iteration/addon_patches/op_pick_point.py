# Pick a location on mesh


import bpy
import gpu
import numpy as np
from mathutils import Vector
from gpu_extras.batch import batch_for_shader
from bpy_extras import view3d_utils

from ..operators.timer_operator import redraw_all_views
from ..utils.mesh_helper import (
    get_mesh_vertices,
    get_face_vertices,
    find_nearest_vertex,
    find_nearest_vertex_global
)
from ..utils.view_helper import (
    get_all_view3d_areas,
    get_source_target_areas
)


############################################################
# Draw

def draw_3d(self, context):
  area = context.area

  if area != self.source_area and area != self.target_area:
    return

  wrap_options = context.scene.wrap_options

  hit_pos = []
  mir_pos = []
  if self.is_source_area:
    if area == self.source_area:
      if self.vert_index is not None:
        hit_pos = [self.vert_world]
        if len(self.vert_sym_idxs) > 0 and self.vert_index < len(self.vert_sym_idxs):
          mir_index = self.vert_sym_idxs[self.vert_index]
          if mir_index != self.vert_index and mir_index < len(self.vertices):
            mir_pos = [self.vertices[mir_index]]
    else:
      if self.vert_index is not None:
        if len(self.stick_vertices) > 0 and self.vert_index < len(self.stick_vertices):
          hit_pos = [self.stick_vertices[self.vert_index]]
  elif self.is_target_area:
    if area == self.target_area:
      if self.hit_world:
        hit_pos = [self.hit_world]
      if self.hit_world_mir:
        mir_pos = [self.hit_world_mir]

  if not hit_pos:
    return

  point_size = wrap_options.point_size

  shader = gpu.shader.from_builtin('POINT_UNIFORM_COLOR')

  if mir_pos:
    batch_mir = batch_for_shader(shader, 'POINTS', {"pos": mir_pos})
    shader.uniform_float("color", (1.0, 0.2, 1.0, 1.0))
    gpu.state.point_size_set(point_size)
    batch_mir.draw(shader)

  batch = batch_for_shader(shader, 'POINTS', {"pos": hit_pos})
  shader.uniform_float("color", (1.0, 1.0, 0.0, 1.0))
  gpu.state.point_size_set(point_size)
  batch.draw(shader)


############################################################
# Ray cast

def get_hit_pos(context, event, obj, mirror: bool) -> list:
  """
  Get hit position
  """
  result = []

  area = context.area
  mouse_x, mouse_y = event.mouse_x, event.mouse_y

  in_area = area.x <= mouse_x < area.x + area.width and \
      area.y <= mouse_y < area.y + area.height

  if not in_area:
    return result

  region = context.region
  rv3d = context.region_data
  coord = event.mouse_region_x, event.mouse_region_y

  view_vector = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
  ray_origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

  ray_target = ray_origin + view_vector

  def obj_ray_cast(obj, origin_pos, target_pos):
    """
    Wrapper for ray casting that moves the ray into object space
    """
    # get the ray relative to the object
    matrix_inv = obj.matrix_world.inverted()
    ray_origin_obj = matrix_inv @ origin_pos
    ray_target_obj = matrix_inv @ target_pos
    ray_direction_obj = ray_target_obj - ray_origin_obj

    # cast the ray
    success, location, normal, face_index = obj.ray_cast(
        ray_origin_obj, ray_direction_obj)

    if success:
      return location, normal, face_index
    else:
      return None, None, None

  # cast rays and find the closest vertex
  if obj.type == 'MESH':
    hit, normal, face_index = obj_ray_cast(obj, ray_origin, ray_target)

    if hit:
      hit_world = obj.matrix_world @ hit
      normal_world = obj.matrix_world.to_3x3() @ normal
      result = [hit_world, normal_world, face_index]

      # Get mirror
      if mirror:
        ray_origin_mir = Vector((-ray_origin.x, ray_origin.y, ray_origin.z))
        ray_target_mir = Vector((-ray_target.x, ray_target.y, ray_target.z))

        hit_mir, normal_mir, face_index_mir = obj_ray_cast(
            obj, ray_origin_mir, ray_target_mir)

        if hit_mir:
          hit_world_mir = obj.matrix_world @ hit_mir
          normal_world_mir = obj.matrix_world.to_3x3() @ normal_mir
          result.extend([hit_world_mir, normal_world_mir, face_index_mir])

  return result


def get_nearest_vertex(obj, hit_world, face_index):
  """
  Get nearest vertex of hit position

  PATCHED: obj.ray_cast() returns a face index of the EVALUATED mesh
  (with modifiers applied). If the object has topology-changing
  modifiers (Mirror, Subsurf, ...), that index may not exist on the
  base mesh and get_face_vertices() returns an empty list. In that
  case we fall back to a KD-tree search over ALL base-mesh vertices,
  so picking still works and never raises.

  Returns (vert_index, vert_world) or (None, None) if nothing found.
  """
  face_vertices = get_face_vertices(obj, face_index)
  nearest_vertice = find_nearest_vertex(hit_world, face_vertices)

  if nearest_vertice is None:
    # Fallback: nearest vertex on the whole base mesh
    nearest_vertice = find_nearest_vertex_global(obj, hit_world)

  if nearest_vertice is None:
    return None, None

  vert_index = nearest_vertice[0]
  vert_world = nearest_vertice[2]

  return vert_index, vert_world


############################################################
# Operator

class TECA3D_VIEW3D_OT_pick_location(bpy.types.Operator):
  """
  Select a location on the mesh
  """
  bl_idname = "view3d.pick_location"
  bl_label = "Operator for selecting a location on a mesh"

  def modal(self, context, event):
    wrap_options = context.scene.wrap_options

    mirror = wrap_options.source_symmetrized

    # allow navigation
    if event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
      return {'PASS_THROUGH'}

    redraw_all_views(context)

    if event.type == 'MOUSEMOVE':
      if self.obj:
        self.hit_world = None
        self.hit_normal = None
        self.face_index = None

        self.hit_world_mir = None
        self.hit_normal_mir = None
        self.face_index_mir = None

        self.vert_index = None
        self.vert_world = None

        result = []
        if self.is_source_area:
          result = get_hit_pos(context, event, self.obj, False)
        else:
          result = get_hit_pos(context, event, self.obj, mirror)

        if len(result) >= 3:
          self.hit_world, self.hit_normal, self.face_index = result[:3]
          if len(result) >= 6:
            self.hit_world_mir, self.hit_normal_mir, self.face_index_mir = result[3:6]

        if self.hit_world and self.is_source_area:
          self.vert_index, self.vert_world = get_nearest_vertex(
              self.obj, self.hit_world, self.face_index)

      # allow auto master
      return {'PASS_THROUGH'}

    elif event.type == 'LEFTMOUSE':
      if self.is_source_area:
        if self.vert_world:
          wrap_options.source_pick_index = self.vert_index
      elif self.is_target_area:
        if self.hit_world and (not mirror or self.hit_world_mir):
          wrap_options.target_pick_pos = self.hit_world
          wrap_options.target_pick_normal = self.hit_normal
          wrap_options.target_pick_index = self.face_index

          if mirror:
            wrap_options.target_pick_pos_mir = self.hit_world_mir
            wrap_options.target_pick_normal_mir = self.hit_normal_mir
            wrap_options.target_pick_index_mir = self.face_index_mir

      bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
      return {'FINISHED'}

    elif event.type in {'RIGHTMOUSE', 'ESC'}:
      bpy.types.SpaceView3D.draw_handler_remove(self._handle, 'WINDOW')
      return {'CANCELLED'}

    return {'RUNNING_MODAL'}

  def invoke(self, context, event):
    if context.area.type == 'VIEW_3D':
      self.hit_world = None
      self.hit_normal = None
      self.face_index = None

      self.hit_world_mir = None
      self.hit_normal_mir = None
      self.face_index_mir = None

      self.vert_index = None
      self.vert_world = None

      wrap_options = context.scene.wrap_options

      # Single object in local view mode
      self.obj = None

      if context.area.spaces.active.local_view:
        visible_objects = context.visible_objects
        if len(visible_objects) == 1:
          self.obj = visible_objects[0]

      if not self.obj:
        self.report({'WARNING'},
                    "This 3D view must be in local view mode for a single object")
        return {'CANCELLED'}

      self.vertices = get_mesh_vertices(self.obj)

      self.vert_sym_idxs = np.fromstring(
          wrap_options.source_vertex_sym_idxs, np.int32, sep=' ')

      # Area
      self.area = context.area

      self.source_area = None
      self.target_area = None

      self.is_source_area = False
      self.is_target_area = False

      window_area_pairs = get_all_view3d_areas(context)
      if len(window_area_pairs) == 2:
        self.source_area, self.target_area = \
            get_source_target_areas(window_area_pairs)
        self.is_source_area = self.area == self.source_area
        self.is_target_area = self.area == self.target_area

      # Stick positions
      self.stick_vertices = []

      if wrap_options.wrap_process_stage == 4:
        self.stick_vertices = [
            v.coord for v in wrap_options.source_stick_vertices]

      # Check
      if self.is_source_area or self.is_target_area:
        wrap_options.draw_points = True

        args = (self, context)
        self._handle = bpy.types.SpaceView3D.draw_handler_add(
            draw_3d, args, 'WINDOW', 'POST_VIEW')

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}
      else:
        self.report({'WARNING'},
                    "This 3D view must be either the source view or the target view")
        return {'CANCELLED'}
    else:
      self.report({'WARNING'}, "View3D not found, cannot run operator")
      return {'CANCELLED'}


#############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_pick_location,)


############################################################
# Setup

def menu_func(self, context):
  self.layout.operator(TECA3D_VIEW3D_OT_pick_location.bl_idname,
                       text="Pick Location")


# Register and add to the "view" menu (required to also use F3 search "Modal Draw Operator" for quick access).
def register():
  bpy.types.VIEW3D_MT_view.append(menu_func)


def unregister():
  bpy.types.VIEW3D_MT_view.remove(menu_func)
