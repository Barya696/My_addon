# Operators for mesh symmetry

import threading

import bpy
import numpy as np
from bpy.types import Operator

from .timer_operator import Timer_Operator, redraw_all_views
from ..core import vertex_store
from ..utils.point_pair_helper import clear_pick_points
from ..core.mesh_symmetry import (
    get_topo_symmetry,
    symmetrize_vertex_coords
)


############################################################
# Center points

class TECA3D_VIEW3D_OT_add_center_point(Operator):
  bl_idname = 'view3d.add_center_point'
  bl_label = 'Add Center Point'
  bl_description = 'Add a point on the central plane as the seed point for the symmetry operation'

  def execute(self, context):
    wrap_options = context.scene.wrap_options

    center_point_index = wrap_options.center_point_index
    source_pick_index = wrap_options.source_pick_index

    if source_pick_index == -1:
      self.report({'WARNING'},
                  "To do this, you must first select a vertex on the source mesh")
      return {'CANCELLED'}

    current = 0
    if center_point_index[0] == -1:
      current = 0
    elif center_point_index[1] == -1:
      current = 1
    else:
      center_point_index[1] = -1

    center_point_index[current] = source_pick_index

    self.report(
        {"INFO"}, f"The center point {current + 1} has been set: {source_pick_index}")

    clear_pick_points(context)

    redraw_all_views(context)
    return {'FINISHED'}


class TECA3D_VIEW3D_OT_remove_center_point(Operator):
  bl_idname = 'view3d.remove_center_point'
  bl_label = 'Remove Center Point'
  bl_description = 'Remove a point on the central plane'

  def execute(self, context):
    wrap_options = context.scene.wrap_options

    center_point_index = wrap_options.center_point_index

    current = 0
    if center_point_index[0] == -1:
      center_point_index[1] = -1
      self.report({'WARNING'}, "No center point has been set")
      return {'CANCELLED'}
    else:
      if center_point_index[1] != -1:
        current = 1

    center_point_index[current] = -1

    self.report(
        {"INFO"}, f"Center Point {current + 1} has been removed")

    redraw_all_views(context)
    return {'FINISHED'}


############################################################
# Symmetrization

def do_symmetrize_source_mesh(self):
  """
  Symmetrize the source mesh
  """
  # Symmetrize
  if self.obj and self.first != -1 and self.second != -1:
    self.face_idxs, self.vertex_idxs = get_topo_symmetry(
        self.obj, self.first, self.second)

  self.is_running = False


class TECA3D_VIEW3D_OT_symmetrize_source_mesh(Timer_Operator):
  bl_idname = 'view3d.symmetrize_source_mesh'
  bl_label = 'Symmetrize Source Mesh'
  bl_description = 'Calculate the indices of the symmetric vertices and faces of the source mesh, and eliminate asymmetry in the vertex coordinates—that is, the x-coordinates of symmetric vertices are opposite, while their y and z coordinates are equal'

  total_time = 10.0

  obj = None
  first, second = -1, -1
  face_idxs, vertex_idxs = [], []

  def modal(self, context, event):
    # Timer
    if event.type == "TIMER" and not self.is_running:
      wrap_options = context.scene.wrap_options

      if len(self.face_idxs) > 0 and len(self.vertex_idxs) > 0:
        wrap_options.source_face_sym_idxs = ' '.join(
            self.face_idxs.astype(str))
        wrap_options.source_vertex_sym_idxs = ' '.join(
            self.vertex_idxs.astype(str))

        # Coordinates
        vertices = vertex_store.get(wrap_options, 'original').astype(np.float64)

        verts_sym = symmetrize_vertex_coords(vertices, self.vertex_idxs)

        vertex_store.store(wrap_options, 'original', verts_sym)

        self.report({"INFO"}, f"Source mesh symmetrization successful")
      else:
        self.report(
            {'WARNING'}, "Symmetrization failed. Please ensure that the two center points are indeed located on the center plane and that they are adjacent to each other")

    return Timer_Operator.modal(self, context, event)

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options

    # Check source mesh
    self.obj = bpy.data.objects.get(wrap_options.source_mesh_name)

    if not self.obj:
      self.report({'WARNING'}, "The source mesh has not been set up yet")
      return {'CANCELLED'}

    # Check center points
    self.first, self.second = wrap_options.center_point_index

    if self.first == -1 or self.second == -1:
      self.report({'WARNING'},
                  "You must select two center points as seeds to perform the symmetrization operation")
      return {'CANCELLED'}

    # Run thread
    thread = threading.Thread(
        target=do_symmetrize_source_mesh,
        args=(self,))
    thread.daemon = True
    thread.start()

    return Timer_Operator.invoke(self, context, event)


############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_add_center_point,
    TECA3D_VIEW3D_OT_remove_center_point,
    TECA3D_VIEW3D_OT_symmetrize_source_mesh)
