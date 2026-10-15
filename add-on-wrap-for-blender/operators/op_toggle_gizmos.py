# Toggle gizmos

import bpy


############################################################
# Toggle gizmos

class TECA3D_VIEW3D_OT_toggle_gizmos(bpy.types.Operator):
  """
  Toggle the display/hide status of some UI elements to keep the 3D view clean and help you stay focused on your work
  """
  bl_idname = "view3d.toggle_gizmos"
  bl_label = "Toggle Gizmos and Overlays"
  bl_options = {'REGISTER', 'UNDO'}

  def execute(self, context):
    wrap_options = context.scene.wrap_options
    show_gizmos = not wrap_options.enable_clean_view

    spaces = [area.spaces.active
              for window in context.window_manager.windows
              for area in window.screen.areas
              if area.type == 'VIEW_3D']

    for space in spaces:
      space.show_gizmo_navigate = show_gizmos
      space.show_gizmo_tool = show_gizmos
      space.show_gizmo_context = show_gizmos
      space.show_gizmo_empty_image = show_gizmos
      space.show_gizmo_empty_force_field = show_gizmos
      space.show_gizmo_light_size = show_gizmos
      space.show_gizmo_light_look_at = show_gizmos
      space.show_gizmo_camera_lens = show_gizmos
      space.show_gizmo_camera_dof_distance = show_gizmos

      space.overlay.show_overlays = show_gizmos

      space.show_region_header = show_gizmos

      # Toggle tool header (Tool Settings)
      # space.show_region_tool_settings = True

      # Toggle Toolbar (T-Panel)
      space.show_region_toolbar = show_gizmos

      # Toggle Sidebar (N-Panel)
      # space.show_region_ui = False

    return {'FINISHED'}


############################################################
# Update

def enable_clean_view_update(self, context):
  bpy.ops.view3d.toggle_gizmos('INVOKE_DEFAULT')


#############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_toggle_gizmos,)
