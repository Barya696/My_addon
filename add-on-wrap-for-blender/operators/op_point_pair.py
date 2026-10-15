# Operators for point pair

import os
import json
from datetime import datetime
from bpy.types import Operator
from bpy.props import StringProperty
from mathutils import Vector

from ..operators.timer_operator import redraw_all_views
from ..utils.view_helper import (
    get_all_view3d_areas,
    get_source_target_areas
)
from ..utils.point_pair_helper import (
    get_source_indices,
    clear_pick_points
)
from ..utils.file_helper import (
    get_file_directory,
    point_pairs_to_data,
    data_to_point_pairs
)


############################################################
# Load point pairs

class TECA3D_VIEW3D_OT_load_point_pairs(Operator):
  bl_idname = 'view3d.load_point_pairs'
  bl_label = 'Load Point Pairs'
  bl_description = 'Load all point pairs from a JSON file'

  filepath: StringProperty(subtype="FILE_PATH")  # type: ignore

  def execute(self, context):
    # Load from file
    with open(self.filepath, 'r') as f:
      data = json.load(f)
      data_to_point_pairs(context, data)
      self.report({"INFO"}, "Point pairs have been loaded")
    return {'FINISHED'}

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options

    file_directory = get_file_directory()
    if not file_directory:
      self.report({'WARNING'},
                  "The current .blend file has not been saved yet")
      return {'CANCELLED'}

    source_name = wrap_options.source_mesh_name
    target_name = wrap_options.target_mesh_name

    if not source_name or not target_name:
      self.report({'WARNING'},
                  "The source mesh and target mesh have not yet been set")
      return {'CANCELLED'}

    file_name = f'{source_name}-{target_name}-pp.json'
    self.filepath = os.path.join(file_directory, file_name)

    context.window_manager.fileselect_add(self)
    return {'RUNNING_MODAL'}


############################################################
# Save point pairs

class TECA3D_VIEW3D_OT_save_point_pairs(Operator):
  bl_idname = 'view3d.save_point_pairs'
  bl_label = 'Save Point Pairs'
  bl_description = 'Save all point pairs to a JSON file'

  filepath: StringProperty(subtype="FILE_PATH")  # type: ignore

  def execute(self, context):
    data = point_pairs_to_data(context)

    with open(self.filepath, 'w') as f:
      json.dump(data, f, indent=2)
      self.report({"INFO"}, "Point pairs have been saved")

    return {'FINISHED'}

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options

    file_directory = get_file_directory()
    if not file_directory:
      self.report({'WARNING'},
                  "The current .blend file has not been saved yet")
      return {'CANCELLED'}

    point_pairs = wrap_options.point_pairs
    if len(point_pairs) == 0 and len(wrap_options.ring_markers) == 0:
      self.report({'WARNING'}, "No point pairs or ring markers available for saving")
      return {'CANCELLED'}

    source_name = wrap_options.source_mesh_name
    target_name = wrap_options.target_mesh_name

    if not source_name or not target_name:
      self.report({'WARNING'},
                  "The source mesh and target mesh have not yet been set")
      return {'CANCELLED'}

    stamp = datetime.now().strftime('%y%m%d_%H%M%S')

    file_name = f'{source_name}-{target_name}-pp-{stamp}.json'
    self.filepath = os.path.join(file_directory, file_name)

    context.window_manager.fileselect_add(self)
    return {'RUNNING_MODAL'}


############################################################
# Add point pair

class TECA3D_VIEW3D_OT_add_point_pair(Operator):
  bl_idname = 'view3d.add_point_pair'
  bl_label = 'Add Point Pair'
  bl_description = 'Add a point-pair between source mesh and target mesh'

  def execute(self, context):
    wrap_options = context.scene.wrap_options

    # Check
    source_index = wrap_options.source_pick_index
    target_index = wrap_options.target_pick_index

    if source_index == -1 or target_index == -1:
      self.report(
          {'WARNING'}, "Please select one point on the source mesh and one point on the target mesh")
      return {'CANCELLED'}

    indices = get_source_indices(context)
    if source_index in indices:
      self.report(
          {'WARNING'}, "The selected point already exists in the point pair list")
      return {'CANCELLED'}

    # Add point pair
    item = wrap_options.point_pairs.add()

    item.source_vertex_index = wrap_options.source_pick_index
    item.target_pos = wrap_options.target_pick_pos
    item.target_normal = wrap_options.target_pick_normal
    item.target_face_index = wrap_options.target_pick_index

    # No mirrored pair: its target point would only be a guess, a posed
    # target is not symmetric. The other side is picked by hand.

    # Clear pick
    clear_pick_points(context)

    redraw_all_views(context)
    return {'FINISHED'}


############################################################
# Remove point pair

class TECA3D_VIEW3D_OT_remove_point_pair(Operator):
  bl_idname = 'view3d.remove_point_pair'
  bl_label = 'Remove Point Pair'
  bl_description = 'Remove a point pair between source mesh and target mesh'

  def execute(self, context):
    wrap_options = context.scene.wrap_options

    # Check
    source_index = wrap_options.source_pick_index

    if source_index == -1:
      self.report({'WARNING'}, "Please select a point on the source mesh")
      return {'CANCELLED'}

    indices = get_source_indices(context)
    if source_index not in indices:
      self.report(
          {'WARNING'}, "The selected point is not in the point pair list")
      return {'CANCELLED'}

    # Remove point pair (only this one: the target side of a pair is never
    # mirrored, so pairs do not come in mirrored couples)
    loc = indices.index(source_index)
    wrap_options.point_pairs.remove(loc)

    # Clear pick
    clear_pick_points(context)

    redraw_all_views(context)
    return {'FINISHED'}


############################################################
# Edit point pair

class TECA3D_VIEW3D_OT_edit_point_pair(Operator):
  bl_idname = 'view3d.edit_point_pair'
  bl_label = 'Edit Point Pair'
  bl_description = 'edit a point-pair between source mesh and target mesh'

  def execute(self, context):
    wrap_options = context.scene.wrap_options

    # Check
    source_index = wrap_options.source_pick_index
    target_index = wrap_options.target_pick_index

    if source_index == -1 or target_index == -1:
      self.report(
          {'WARNING'}, "Please select one point on the source mesh and one point on the target mesh")
      return {'CANCELLED'}

    indices = get_source_indices(context)
    if source_index not in indices:
      self.report(
          {'WARNING'}, "The selected point is not in the point pair list")
      return {'CANCELLED'}

    # Edit
    loc = indices.index(source_index)
    item = wrap_options.point_pairs[loc]

    item.target_pos = wrap_options.target_pick_pos
    item.target_normal = wrap_options.target_pick_normal
    item.target_face_index = wrap_options.target_pick_index

    # Clear pick
    clear_pick_points(context)

    redraw_all_views(context)
    return {'FINISHED'}


############################################################
# Clear point pairs

class TECA3D_VIEW3D_OT_clear_point_pairs(Operator):
  bl_idname = 'view3d.clear_point_pairs'
  bl_label = 'Clear Point Pairs'
  bl_description = 'Clear All Point Pairs'

  def execute(self, context):
    wrap_options = context.scene.wrap_options
    wrap_options.point_pairs.clear()
    redraw_all_views(context)
    return {'FINISHED'}

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options
    if len(wrap_options.point_pairs) == 0:
      return {'CANCELLED'}

    return context.window_manager.invoke_props_dialog(
        self, title="All point pairs will be deleted and cannot be recovered. Please save them beforehand.")

  def cancel(self, context):
    return


############################################################
# Clear pick points

class TECA3D_VIEW3D_OT_clear_pick_point(Operator):
  bl_idname = 'view3d.clear_pick_point'
  bl_label = 'Clear Selected'
  bl_description = 'Clear selected point'

  def execute(self, context):
    area = context.area
    wrap_options = context.scene.wrap_options

    window_area_pairs = get_all_view3d_areas(context)
    source_area, target_area = get_source_target_areas(window_area_pairs)

    if area == source_area:
      wrap_options.source_pick_index = -1
    elif area == target_area:
      wrap_options.target_pick_pos = Vector()
      wrap_options.target_pick_normal = Vector()
      wrap_options.target_pick_index = -1

    redraw_all_views(context)
    return {'FINISHED'}


#############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_load_point_pairs,
    TECA3D_VIEW3D_OT_save_point_pairs,
    TECA3D_VIEW3D_OT_add_point_pair,
    TECA3D_VIEW3D_OT_remove_point_pair,
    TECA3D_VIEW3D_OT_edit_point_pair,
    TECA3D_VIEW3D_OT_clear_point_pairs,
    TECA3D_VIEW3D_OT_clear_pick_point)
