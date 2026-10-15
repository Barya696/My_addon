# Create workspace

from bpy.types import Operator

from ..utils.workspace_helper import create_workspace, split_view3d


############################################################
# Create workspace

class TECA3D_VIEW3D_OT_create_workspace(Operator):
  bl_idname = 'view3d.create_workspace'
  bl_label = 'Create Workspace'
  bl_description = 'Create a workspace for wrap'

  def execute(self, context):
    if create_workspace('Wrap'):
      self.report({'INFO'}, "The Wrap workspace has been successfully created")
    else:
      self.report({'WARNING'}, "Creating Workspace Wrap Failed")

    return {'FINISHED'}


############################################################
# Create 3d views

class TECA3D_VIEW3D_OT_create_view3d(Operator):
  bl_idname = 'view3d.create_view3d'
  bl_label = 'Create 3D Views'
  bl_description = 'Create two side-by-side 3D views'

  def execute(self, context):
    if split_view3d(context):
      self.report({'INFO'},
                  "Two side-by-side views have been successfully created")
    else:
      self.report({'WARNING'}, "Failed to create side-by-side views")

    return {'FINISHED'}


#############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_create_workspace,
    TECA3D_VIEW3D_OT_create_view3d)
