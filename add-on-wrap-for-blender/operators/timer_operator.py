# Modal operator with a timer

import time
import threading

from bpy.types import Operator


############################################################
# Base operator

class Timer_Operator(Operator):
  """
  A modal operator with a timer and progress indicator
  """
  timer = None
  start_time = 0.0
  is_running = False

  total_time = 3.0
  mutex = threading.Lock()

  draw_points = False
  draw_labels = False

  def on_timer(self, context):
    """
    Execute when a timer event is received
    """
    pass

  def on_stop(self, context):
    """
    Execute when the run flag is turned off
    """
    pass

  def modal(self, context, event):
    # Timer
    if event.type == "TIMER":
      wrap_options = context.scene.wrap_options

      redraw_all_views(context)

      # Check running flag
      if not self.is_running:
        # The thread can deliver its last result and finish between two
        # timer events: apply it before stopping, or it is lost
        self.on_timer(context)
        self.on_stop(context)

        if self.timer:
          context.window_manager.event_timer_remove(self.timer)
          self.timer = None

        wrap_options.progress_value = 0.0
        wrap_options.progress_iter_num = 1

        restore_draw_points(self, context)
        return {'FINISHED'}

      self.on_timer(context)

      # Update progress
      with self.mutex:
        elapsed_seconds = time.perf_counter() - self.start_time

      wrap_options.progress_value = elapsed_seconds / self.total_time
      wrap_options.progress_time = elapsed_seconds

    # Allow auto master
    elif event.type == 'MOUSEMOVE':
      return {'PASS_THROUGH'}

    # Allow navigation
    elif event.type in {'MIDDLEMOUSE', 'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
      return {'PASS_THROUGH'}

    # Other
    elif event.type in {"LEFTMOUSE", "RIGHTMOUSE"}:
      self.report(
          {'WARNING'}, "The operation is in progress; please do not perform any other operations")

    return {'RUNNING_MODAL'}

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options

    disable_draw_points(self, context)

    # Progress
    wrap_options.progress_value = 0.0
    wrap_options.progress_iter_num = 1

    # Timer
    wm = context.window_manager
    self.timer = wm.event_timer_add(0.1, window=context.window)
    wm.modal_handler_add(self)

    self.start_time = time.perf_counter()
    self.is_running = True

    return {'RUNNING_MODAL'}


############################################################
# Draw

def redraw_all_views(context):
  for window in context.window_manager.windows:
    for area in window.screen.areas:
      if area.type == 'VIEW_3D':
        area.tag_redraw()


def disable_draw_points(self, context):
  wrap_options = context.scene.wrap_options

  self.draw_points = wrap_options.draw_points
  self.draw_labels = wrap_options.draw_labels

  wrap_options.draw_points = False
  wrap_options.draw_labels = False


def restore_draw_points(self, context):
  wrap_options = context.scene.wrap_options

  wrap_options.draw_points = self.draw_points
  wrap_options.draw_labels = self.draw_labels
