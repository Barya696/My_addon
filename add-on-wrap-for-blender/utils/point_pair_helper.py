# Point pair helper

import bpy
from mathutils import Vector

from ..core import vertex_store
from ..core.ring_markers import ring_pairs
from ..core.face_landmarks import landmark_filter


############################################################
# Point pair

def get_source_indices(context):
  point_pairs = context.scene.wrap_options.point_pairs
  indices = [v.source_vertex_index for v in point_pairs]
  return indices


def get_source_points(context, indices):
  wrap_options = context.scene.wrap_options
  mesh_name = wrap_options.source_mesh_name

  obj = bpy.data.objects.get(mesh_name)
  if not obj or obj.type != 'MESH':
    return []

  # if bpy.context.mode != 'OBJECT':
  #   bpy.ops.object.mode_set(mode='OBJECT')

  vertices = obj.data.vertices
  matrix = obj.matrix_world

  points = [matrix @ vertices[i].co for i in indices]
  return points


def get_target_points(context):
  point_pairs = context.scene.wrap_options.point_pairs
  points = [Vector(v.target_pos) for v in point_pairs]
  return points


def get_point_pair_index_pos(context):
  """
  Get the source index and target position of the point pair used for the wrap operation
  """
  wrap_options = context.scene.wrap_options

  # Disable drawing points to avoid access conflicts,
  # otherwise it will cause severe performance issues
  draw_points = wrap_options.draw_points
  draw_labels = wrap_options.draw_labels

  wrap_options.draw_points = False
  wrap_options.draw_labels = False

  # Get the source index and target position: pairs picked by hand, and the
  # face landmarks not in the Exclude list
  used = landmark_filter(wrap_options)
  point_pairs = [p for p in wrap_options.point_pairs
                 if p.landmark <= 0 or used(p.landmark)]

  source_index = [p.source_vertex_index for p in point_pairs]
  target_pos = [list(p.target_pos) for p in point_pairs]

  # Ring markers, as point pairs (only while there are rings: never stale
  # ones); a point picked by hand wins
  taken = set(source_index)
  rings = ring_pairs(wrap_options) if len(wrap_options.ring_markers) else ([], [])
  for i, pos in zip(*rings):
    if i not in taken:
      taken.add(i)
      source_index.append(i)
      target_pos.append(pos)

  # Restore drawing point settings
  wrap_options.draw_points = draw_points
  wrap_options.draw_labels = draw_labels

  return source_index, target_pos


############################################################
# Pick

def get_source_pick_stick(context):
  wrap_options = context.scene.wrap_options

  pick_index = wrap_options.source_pick_index
  stick_verts = vertex_store.get(wrap_options, 'stick')

  if pick_index != -1 and len(stick_verts) > 0:
    return [tuple(stick_verts[pick_index])]
  else:
    return None


def get_target_pick(context):
  wrap_options = context.scene.wrap_options
  if wrap_options.target_pick_index != -1:
    if wrap_options.target_pick_index_mir != -1:
      return [Vector(wrap_options.target_pick_pos),
              Vector(wrap_options.target_pick_pos_mir)]
    else:
      return [Vector(wrap_options.target_pick_pos)]
  else:
    return None


############################################################
# Center points

def get_center_indices(context):
  wrap_options = context.scene.wrap_options
  center_point_index = wrap_options.center_point_index

  indices = [v for v in center_point_index if v != -1]
  return indices


############################################################
# Pick points

def clear_pick_points(context):
  wrap_options = context.scene.wrap_options

  wrap_options.source_pick_index = -1

  wrap_options.target_pick_pos = Vector()
  wrap_options.target_pick_normal = Vector()
  wrap_options.target_pick_index = -1

  wrap_options.target_pick_pos_mir = Vector()
  wrap_options.target_pick_normal_mir = Vector()
  wrap_options.target_pick_index_mir = -1
