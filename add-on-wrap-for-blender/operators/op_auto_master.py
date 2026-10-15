# Auto set master index

from bpy.types import Operator


############################################################
# Auto master

active_area = None
active_viewport_id = "None"


class TECA3D_VIEW3D_OT_auto_master_modal(Operator):
  bl_idname = "view3d.auto_master_modal"
  bl_label = "Auto Master Modal"
  bl_description = "Monitor active viewport for Auto Master"

  timer = None

  def modal(self, context, event):
    global active_area, active_viewport_id

    wrap_options = context.scene.wrap_options

    # If the current workspace is not a Wrap (e.g., Layout), ignore this
    # operation to prevent setting active_area to an incorrect area
    # outside the Wrap workspace. Otherwise, be_inside would always be
    # true, preventing the master index from being updated correctly.

    if context.workspace.name != 'Wrap':
      return {'PASS_THROUGH'}

    # Only terminate the operator within timer events to avoid
    # consuming mouse events (such as mouse left-button release),
    # which would result in incomplete UI operations
    #
    # For example, when a user clicks the sync button to stop
    # synchronizing the view, the real_time_sync property will be
    # set to False.
    # If the mouse left-click release event is then passed to the
    # exec function, this function will return {“CANCELLED”},
    # causing the mouse left-click release event to be consumed.
    # The click operation thus remains incomplete, and the Blender
    # interface will remain frozen until the user clicks the mouse
    # left button again.

    if event.type == "TIMER":
      # Stop tracking master view index
      if not wrap_options.real_time_sync:
        if self.timer:
          context.window_manager.event_timer_remove(self.timer)
          self.timer = None
        return {'CANCELLED'}

    # Update mouse position
    if event.type == 'MOUSEMOVE':
      mouse_x, mouse_y = event.mouse_x, event.mouse_y

      be_inside = active_area and \
          active_area.x <= mouse_x < active_area.x + active_area.width and \
          active_area.y <= mouse_y < active_area.y + active_area.height

      # Into another area
      if not be_inside:
        window_area_pairs = [
            (window, area)
            for window in context.window_manager.windows
            for area in window.screen.areas if area.type == 'VIEW_3D']

        found = [
            (i, w, a)
            for i, (w, a) in enumerate(window_area_pairs)
            if a.x <= mouse_x < a.x + a.width and
            a.y <= mouse_y < a.y + a.height]

        # If we found a new active area
        if len(found) == 1:
          active_area = found[0][2]

          # Update active viewport ID with window information
          active_viewport_id = f"View ID: {found[0][2].as_pointer()} (Window: {found[0][1].screen.name})"

          # Update master view index
          wrap_options.master_view_index = found[0][0]

    return {'PASS_THROUGH'}

  def invoke(self, context, event):
    if not context.scene.wrap_options.real_time_sync:
      return {'CANCELLED'}

    wm = context.window_manager
    self.timer = wm.event_timer_add(0.1, window=context.window)
    wm.modal_handler_add(self)

    return {'RUNNING_MODAL'}


def reset_active_area():
  global active_area, active_viewport_id

  active_area = None
  active_viewport_id = "None"


############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_auto_master_modal,)
