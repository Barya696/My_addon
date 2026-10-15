# Operators for set mesh

from bpy.types import Operator

from ..utils.view_helper import set_mesh_from_selected


############################################################
# Set source mesh operator

class TECA3D_VIEW3D_OT_set_source_mesh(Operator):
  bl_idname = 'view3d.set_source_mesh'
  bl_label = 'Set Source Mesh'
  bl_description = 'Set the source mesh (template), Note: All point pairs will be deleted. If you need to reuse them, please save them before proceeding'

  def execute(self, context):
    return set_mesh_from_selected(self, context)

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options
    if not wrap_options.source_mesh_name:
      return self.execute(context)

    return context.window_manager.invoke_props_dialog(
        self, title="Are you sure you want to change the source mesh? All point pairs will be deleted.")

  def cancel(self, context):
    return


############################################################
# Set target mesh operator

class TECA3D_VIEW3D_OT_set_target_mesh(Operator):
  bl_idname = 'view3d.set_target_mesh'
  bl_label = 'Set Target Mesh'
  bl_description = 'Set the target mesh (destination), Note: All point pairs will be deleted. If you need to reuse them, please save them before proceeding'

  def execute(self, context):
    return set_mesh_from_selected(self, context)

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options
    if not wrap_options.target_mesh_name:
      return self.execute(context)

    return context.window_manager.invoke_props_dialog(
        self, title="Are you sure you want to change the target mesh? All point pairs will be deleted.")

  def cancel(self, context):
    return


############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_set_source_mesh,
    TECA3D_VIEW3D_OT_set_target_mesh)
