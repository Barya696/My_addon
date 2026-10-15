# Operators for point index

import os
import bmesh
import numpy as np
from datetime import datetime
from bpy.types import Operator
from bpy.props import StringProperty

from ..operators.timer_operator import redraw_all_views
from ..utils.file_helper import get_file_directory


############################################################
# Selected

def get_selected_vertices(context):
  # Get the mesh from the active object
  obj = context.active_object
  bm = bmesh.from_edit_mesh(obj.data)

  # Extract indices of selected vertices
  indices = [v.index for v in bm.verts if v.select]
  return indices


def deselect_all_vertices(context):
  # Get the mesh from the active object
  obj = context.active_object
  bm = bmesh.from_edit_mesh(obj.data)

  for v in bm.verts:
    v.select = False

  for e in bm.edges:
    e.select = False

  for f in bm.faces:
    f.select = False

  # Update the mesh to reflect changes in the viewport
  bmesh.update_edit_mesh(obj.data)


############################################################
# Add point index

class TECA3D_VIEW3D_OT_add_point_index(Operator):
  bl_idname = 'view3d.add_point_index'
  bl_label = 'Add Point Index'
  bl_description = 'Add the selected vertices to the list of free vertices or the list of fixed vertices'

  target: StringProperty()  # type: ignore

  def execute(self, context):
    wrap_options = context.scene.wrap_options

    # 'OBJECT', 'EDIT_MESH', 'EDIT_ARMATURE', etc
    if context.mode != 'EDIT_MESH':
      self.report({'WARNING'}, "You must do this in Edit Mesh mode")
      return {'CANCELLED'}

    indices = get_selected_vertices(context)
    if not indices:
      self.report({'WARNING'}, "No vertices have been selected yet")
      return {'CANCELLED'}

    if self.target == 'Free Vertices':
      free_index = np.fromstring(wrap_options.free_index, np.int32, sep=' ')
      prev_number = len(free_index)

      free_index = np.unique(np.append(free_index, indices))

      wrap_options.free_index = ' '.join(free_index.astype(str))
      wrap_options.free_index_number = len(free_index)
      added_number = len(free_index) - prev_number

      deselect_all_vertices(context)
      self.report(
          {"INFO"}, f"{added_number} vertices have been added as free points")

    elif self.target == 'Fixed Vertices':
      fixed_index = np.fromstring(wrap_options.fixed_index, np.int32, sep=' ')
      prev_number = len(fixed_index)

      fixed_index = np.unique(np.append(fixed_index, indices))

      wrap_options.fixed_index = ' '.join(fixed_index.astype(str))
      wrap_options.fixed_index_number = len(fixed_index)
      added_number = len(fixed_index) - prev_number

      deselect_all_vertices(context)
      self.report(
          {"INFO"}, f"{added_number} vertices have been added as fixed points")

    redraw_all_views(context)
    return {'FINISHED'}


############################################################
# Remove point index

class TECA3D_VIEW3D_OT_remove_point_index(Operator):
  bl_idname = 'view3d.remove_point_index'
  bl_label = 'Remove Point Index'
  bl_description = 'Remove the selected vertices from the list of free vertices or the list of fixed vertices.'

  target: StringProperty()  # type: ignore

  def execute(self, context):
    wrap_options = context.scene.wrap_options

    if context.mode != 'EDIT_MESH':
      self.report({'WARNING'}, "You must do this in Edit Mesh mode")
      return {'CANCELLED'}

    indices = get_selected_vertices(context)
    if not indices:
      self.report({'WARNING'}, "No vertices have been selected yet")
      return {'CANCELLED'}

    if self.target == 'Free Vertices':
      free_index = np.fromstring(wrap_options.free_index, np.int32, sep=' ')
      prev_number = len(free_index)

      free_index = free_index[~np.isin(free_index, indices)]

      wrap_options.free_index = ' '.join(free_index.astype(str))
      wrap_options.free_index_number = len(free_index)
      removed_number = prev_number - len(free_index)

      deselect_all_vertices(context)
      self.report(
          {"INFO"}, f"{removed_number} vertices have been removed from the list of free vertices")

    elif self.target == 'Fixed Vertices':
      fixed_index = np.fromstring(wrap_options.fixed_index, np.int32, sep=' ')
      prev_number = len(fixed_index)

      fixed_index = fixed_index[~np.isin(fixed_index, indices)]

      wrap_options.fixed_index = ' '.join(fixed_index.astype(str))
      wrap_options.fixed_index_number = len(fixed_index)
      removed_number = prev_number - len(fixed_index)

      deselect_all_vertices(context)
      self.report(
          {"INFO"}, f"{removed_number} vertices have been removed from the list of fixed vertices")

    redraw_all_views(context)
    return {'FINISHED'}


############################################################
# Load point index

class TECA3D_VIEW3D_OT_load_point_index(Operator):
  bl_idname = 'view3d.load_point_index'
  bl_label = 'Load Point Index'
  bl_description = 'Load the point index from a text file'

  filepath: StringProperty(subtype="FILE_PATH")  # type: ignore
  target: StringProperty()  # type: ignore

  def execute(self, context):
    wrap_options = context.scene.wrap_options

    try:
      data = np.loadtxt(self.filepath, np.int32, delimiter=' ')
      dataStr = ' '.join(data.astype(str))

      if len(data) == 0:
        self.report({"WARNING"}, "The file is empty; no action will be taken")
        return {'CANCELLED'}

      if self.target == 'Free Vertices':
        wrap_options.free_index = dataStr
        wrap_options.free_index_number = len(data)
        self.report({"INFO"}, f"{len(data)} free vertices have been loaded")
      elif self.target == 'Fixed Vertices':
        wrap_options.fixed_index = dataStr
        wrap_options.fixed_index_number = len(data)
        self.report({"INFO"}, f"{len(data)} fixed vertices have been loaded")
      else:
        print(f"Error: Invalid target, {self.target}")

    except FileNotFoundError:
      self.report({"WARNING"},
                  f"The file '{self.filepath}' was not found")
    except ValueError as e:
      self.report({"WARNING"},
                  f"Value error occurred - {e}")
    except Exception as e:
      self.report({"WARNING"},
                  f"An unexpected error occurred: {e}")

    redraw_all_views(context)
    return {'FINISHED'}

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options

    file_directory = get_file_directory()
    if not file_directory:
      self.report({'WARNING'},
                  "The current .blend file has not been saved yet")
      return {'CANCELLED'}

    source_name = wrap_options.source_mesh_name
    if not source_name:
      self.report({'WARNING'},
                  "The source mesh have not yet been set")
      return {'CANCELLED'}

    label = self.target.replace(' ', '-').lower()

    file_name = f'{source_name}-{label}-*.txt'
    self.filepath = os.path.join(file_directory, file_name)

    context.window_manager.fileselect_add(self)
    return {'RUNNING_MODAL'}


############################################################
# Save point index

class TECA3D_VIEW3D_OT_save_point_index(Operator):
  bl_idname = 'view3d.save_point_index'
  bl_label = 'Save Point Index'
  bl_description = 'Save the point index to a text file'

  filepath: StringProperty(subtype="FILE_PATH")  # type: ignore
  source: StringProperty()  # type: ignore

  dataStr = ''

  def execute(self, context):
    with open(self.filepath, 'w') as f:
      f.write(self.dataStr)
      if self.source == 'Free Vertices':
        self.report(
            {"INFO"}, f"The free vertices have been saved to the file {self.filepath}")
      elif self.source == 'Fixed Vertices':
        self.report(
            {"INFO"}, f"The fixed vertices have been saved to the file {self.filepath}")

    return {'FINISHED'}

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options

    file_directory = get_file_directory()
    if not file_directory:
      self.report({'WARNING'},
                  "The current .blend file has not been saved yet")
      return {'CANCELLED'}

    if self.source == 'Free Vertices':
      self.dataStr = wrap_options.free_index
    elif self.source == 'Fixed Vertices':
      self.dataStr = wrap_options.fixed_index
    else:
      print(f"Error: Invalid source, {self.source}")

    if not self.dataStr:
      self.report({'WARNING'}, "There are no vertices to save")
      return {'CANCELLED'}

    source_name = wrap_options.source_mesh_name
    if not source_name:
      self.report({'WARNING'}, "The source mesh have not yet been set")
      return {'CANCELLED'}

    stamp = datetime.now().strftime('%y%m%d_%H%M%S')

    label = self.source.replace(' ', '-').lower()

    file_name = f'{source_name}-{label}-{stamp}.txt'
    self.filepath = os.path.join(file_directory, file_name)

    context.window_manager.fileselect_add(self)
    return {'RUNNING_MODAL'}


############################################################
# Clear point pairs

class TECA3D_VIEW3D_OT_clear_point_index(Operator):
  bl_idname = 'view3d.clear_point_index'
  bl_label = 'Clear Point Index'
  bl_description = 'Clear the point index'

  target: StringProperty()  # type: ignore

  def execute(self, context):
    wrap_options = context.scene.wrap_options

    if self.target == 'Free Vertices':
      wrap_options.free_index = ''
      self.report({'INFO'}, "All free vertices have been cleared")
    elif self.target == 'Fixed Vertices':
      wrap_options.fixed_index = ''
      self.report({'INFO'}, "All fixed vertices have been cleared")

    redraw_all_views(context)
    return {'FINISHED'}

  def draw(self, context):
    pass

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options

    dataStr = ''
    if self.target == 'Free Vertices':
      dataStr = wrap_options.free_index
    elif self.target == 'Fixed Vertices':
      dataStr = wrap_options.fixed_index

    if not dataStr:
      self.report({'INFO'}, "There are no vertices to clear")
      return {'CANCELLED'}

    return context.window_manager.invoke_props_dialog(
        self, title="Are you sure you want to clear all vertices?")

  def cancel(self, context):
    return


#############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_add_point_index,
    TECA3D_VIEW3D_OT_remove_point_index,
    TECA3D_VIEW3D_OT_load_point_index,
    TECA3D_VIEW3D_OT_save_point_index,
    TECA3D_VIEW3D_OT_clear_point_index)
