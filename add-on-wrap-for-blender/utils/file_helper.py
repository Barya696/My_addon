# File helper

import os
import bpy
import mathutils

from ..core.ring_markers import markers_to_data, data_to_markers


############################################################
# Point pairs

def point_pairs_to_data(context):
  """
  Convert point pairs to data
  """
  point_pairs = context.scene.wrap_options.point_pairs

  return {
      'point_pairs': [
          (v.source_vertex_index,
           v.target_face_index,
           list(v.target_pos),
           list(v.target_normal),
           v.weight,
           v.landmark) for v in point_pairs],
      'ring_markers': markers_to_data(context.scene.wrap_options),
  }


def data_to_point_pairs(context, data):
  """
  Convert data to point pairs
  """
  wrap_options = context.scene.wrap_options

  # Disable drawing points to avoid access conflicts,
  # otherwise it will cause severe performance issues
  draw_points = wrap_options.draw_points
  draw_labels = wrap_options.draw_labels

  wrap_options.draw_points = False
  wrap_options.draw_labels = False

  # Set point pairs
  point_pairs = wrap_options.point_pairs
  point_pairs.clear()

  for d in data['point_pairs']:
    item = point_pairs.add()

    item.source_vertex_index = d[0]
    item.target_face_index = d[1]
    item.target_pos = mathutils.Vector(d[2])
    item.target_normal = mathutils.Vector(d[3])
    item.weight = d[4]
    item.landmark = d[5] if len(d) > 5 else 0

  # Ring markers (files saved before rings have none)
  if 'ring_markers' in data:
    data_to_markers(context, data['ring_markers'])

  # Restore drawing point settings
  wrap_options.draw_points = draw_points
  wrap_options.draw_labels = draw_labels


############################################################
# Path

def get_file_directory():
  # It might be an empty string if the file hasn't been saved yet
  filepath = bpy.data.filepath

  if filepath:
    # Convert the Blender-specific path (which might use the // prefix)
    # to a full absolute system path
    abs_filepath = bpy.path.abspath(filepath)

    # To get only the directory containing the file
    directory = os.path.dirname(abs_filepath)
    return directory
  else:
    return None


############################################################
# Json

def blender_data_encoder(obj):
  # Convert mathutils types to lists
  if isinstance(obj, mathutils.Vector):
    return list(obj)
  if isinstance(obj, mathutils.Euler):
    return list(obj)
  if isinstance(obj, mathutils.Quaternion):
    return list(obj)
  if isinstance(obj, mathutils.Matrix):
    return [list(row) for row in obj]

  # Handle bpy_prop_collection or bpy_struct by converting to dictionary
  if hasattr(obj, "bl_rna"):
    return obj.path_resolve()
    # return dict(obj.items())

  raise TypeError(
      f"Object of type {type(obj).__name__} is not JSON serializable")
