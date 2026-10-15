# View3d helpers

import bpy
from mathutils import Matrix

from ..utils.point_pair_helper import clear_pick_points
from ..core.ring_markers import clear_markers, clear_geometry
from ..utils.mesh_helper import (
    preserve_source_vertices,
    clear_source_vertices_reserve,
    get_source_mesh_diagonal_length
)


############################################################
# Set mesh

def set_mesh_from_selected(self, context):
  """
  Set source or target mesh from selected object
  """
  active_object = context.active_object
  wrap_options = context.scene.wrap_options

  # Check active object
  if not active_object or active_object.type != 'MESH':
    self.report({'WARNING'}, "Please select exactly one mesh object")
    return {'CANCELLED'}

  bpy.ops.object.select_all(action='DESELECT')
  active_object.select_set(True)

  # Ensure that two 3D views are present
  window_area_pairs = get_all_view3d_areas(context)
  if len(window_area_pairs) != 2:
    self.report({'WARNING'}, "There should be two side-by-side 3D views")
    return {'CANCELLED'}

  # Check source mesh mode
  if wrap_options.source_mesh_mode != 'ORIGINAL':
    self.report({'WARNING'}, 'To do this, you must be in original mode')
    return {'CANCELLED'}

  # Check mesh names to avoid conflicts
  area = context.area
  source_area, target_area = get_source_target_areas(window_area_pairs)

  mesh_name = active_object.name
  need_clear = False

  if area == source_area:
    if mesh_name == wrap_options.target_mesh_name:
      disable_local_view(context, target_area)
      reset_target_mesh_options(context)
      need_clear = True
  elif area == target_area:
    if mesh_name == wrap_options.source_mesh_name:
      disable_local_view(context, source_area)
      reset_source_mesh_options(context)
      need_clear = True
  else:
    self.report({'WARNING'},
                "This region is neither the source mesh view nor the target mesh view")
    return {'CANCELLED'}

  # Local view the source mesh in the 3d view
  local_view_selected(context, area)

  # Save mesh name to options
  if area == source_area:
    if wrap_options.source_mesh_name != mesh_name:
      reset_source_mesh_options(context)

      wrap_options.source_mesh_name = mesh_name
      wrap_options.source_diagonal_length = \
          get_source_mesh_diagonal_length(context)

      preserve_source_vertices(context)
      # active_object.matrix_world = Matrix.Identity(4)

      need_clear = True
      self.report({'INFO'},
                  f'The mesh {mesh_name} has been set as the source mesh')

  elif area == target_area:
    if wrap_options.target_mesh_name != mesh_name:
      reset_target_mesh_options(context)

      wrap_options.target_mesh_name = mesh_name

      need_clear = True
      self.report({'INFO'},
                  f'The mesh {mesh_name} has been set as the target mesh')

  # Clear point pairs and pick points
  if need_clear:
    wrap_options.point_pairs.clear()
    clear_pick_points(context)
    clear_markers(context)
    clear_geometry()

  return {'FINISHED'}


############################################################
# Reset mesh options

def reset_source_mesh_options(context):
  """
  Reset all options related to the source mesh
  """
  wrap_options = context.scene.wrap_options

  # Name
  wrap_options.source_mesh_name = ''

  # Diagonal
  wrap_options.source_diagonal_length = 0

  # Symmetric
  wrap_options.center_point_index = (-1, -1)
  wrap_options.source_face_sym_idxs = ''
  wrap_options.source_vertex_sym_idxs = ''

  # Free
  wrap_options.free_index = ''

  # Fixed
  wrap_options.fixed_index = ''

  # Wrap
  wrap_options.wrap_process_stage = 0

  wrap_options.wrap_init_data = ''

  # Source mesh mode
  clear_source_vertices_reserve(context)

  wrap_options.source_mesh_mode = 'ORIGINAL'


def reset_target_mesh_options(context):
  """
  Reset all options related to the target mesh
  """
  wrap_options = context.scene.wrap_options

  # Name
  wrap_options.target_mesh_name = ''

  # Wrap
  wrap_options.wrap_process_stage = 0

  # Source mesh mode
  wrap_options.source_mesh_mode = 'ORIGINAL'


############################################################
# Local view

def get_all_view3d_areas(context):
  return [(window, area)
          for window in context.window_manager.windows
          for area in window.screen.areas
          if area.type == 'VIEW_3D']


def get_source_target_areas(window_area_pairs):
  if len(window_area_pairs) != 2:
    return None, None

  a1 = window_area_pairs[0][1]
  a2 = window_area_pairs[1][1]

  if a1.y != a2.y:
    return None, None

  if a1.x < a2.x:
    return a1, a2
  else:
    return a2, a1


def disable_local_view(context, area):
  override = context.copy()
  override['area'] = area
  override['region'] = area.regions[-1]

  with context.temp_override(**override):
    # Toggle out of local view if the area is already in local view
    if area.spaces.active.local_view:
      bpy.ops.view3d.localview()


def local_view_selected(context, area):
  override = context.copy()
  override['area'] = area
  override['region'] = area.regions[-1]

  with context.temp_override(**override):
    # Toggle out of local view if the area is already in local view
    if area.spaces.active.local_view:
      bpy.ops.view3d.localview()

    # Enter local view mode for this area
    bpy.ops.view3d.localview()
