# Wrap operators

import time
import threading
from operator import itemgetter

from bpy.types import Operator

from .timer_operator import Timer_Operator, redraw_all_views
from ..client import wrap_client
from ..client.socket_manager import receive_data_block
from ..utils.system_helper import (
    start_process,
    get_process_pid,
    get_addon_dir,
    global_wrap_server_name
)
from ..utils.mesh_helper import (
    set_source_mesh_vertices,
    preserve_source_vertices
)
from ..core.mesh_process import source_mesh_mode_update
from ..core.ring_markers import update_ring_pairs
from ..preferences.preferences import get_wrap_prefs


############################################################
# Wrap start

def do_wrap_start(self, context):
  wrap_options = context.scene.wrap_options

  addon_dir = get_addon_dir()
  process_name = global_wrap_server_name

  # Start the packaging server; if it is already running, do nothing
  pid = get_process_pid(process_name)

  if pid == -1:
    wrap_prefs = get_wrap_prefs(context)
    server_port = wrap_prefs.server_port

    start_process(f"'{addon_dir}\\bin\\{process_name}.exe' {server_port}")

    for _ in range(3):
      time.sleep(1)
      pid = get_process_pid(process_name)
      if pid != -1:
        break

    if pid == -1:
      self.report({'ERROR'}, "Failed to start the Wrap server")
    else:
      self.report({'INFO'}, "The Wrap server has started")
  else:
    self.report({'INFO'}, "The Wrap server is already running")

  if pid != -1:
    wrap_options.wrap_server_running = True
    wrap_options.wrap_process_stage = 0

    if wrap_options.source_mesh_mode != 'ORIGINAL':
      wrap_options.source_mesh_mode = 'ORIGINAL'
      source_mesh_mode_update(wrap_options, context)

    redraw_all_views(context)


class TECA3D_VIEW3D_OT_wrap_start(Operator):
  bl_idname = 'view3d.wrap_start'
  bl_label = 'Wrap Start'
  bl_description = 'Start the Wrap server'

  def execute(self, context):
    do_wrap_start(self, context)
    return {'FINISHED'}

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options
    if wrap_options.wrap_process_stage <= 1:
      return self.execute(context)

    return context.window_manager.invoke_props_dialog(
        self, title="Starting the wrap server will delete all previous wrap results. Are you sure?")

  def cancel(self, context):
    return


############################################################
# Init

def do_wrap_init(self, context):
  """
  Initialize wrap
  """
  self.response = wrap_client.wrap_init(context)
  self.is_running = False


class TECA3D_VIEW3D_OT_wrap_init(Timer_Operator):
  bl_idname = 'view3d.wrap_init'
  bl_label = 'Wrap Init'
  bl_description = 'Initialize wrap computation'

  total_time = 60.0

  response = None

  def on_stop(self, context):
    if self.response:
      if 'code' in self.response:
        code, status, message = itemgetter(
            'code', 'status', 'message')(self.response)

        if code == 0:
          self.report({"INFO"}, "Wrap initialization successful")
          if message:
            wrap_client.set_wrap_init_data(context, message)
            self.report({"INFO"}, "Initialization data has been saved")

          # Original mode
          wrap_options = context.scene.wrap_options
          if wrap_options.source_mesh_mode != 'ORIGINAL':
            wrap_options.source_mesh_mode = 'ORIGINAL'
            source_mesh_mode_update(wrap_options, context)

          # Done
          wrap_options.wrap_process_stage = 1
        else:
          self.report(
              {'WARNING'}, f"Failed to initialize wrap: {code}, {status}")
      else:
        self.report(
            {'WARNING'}, "Failed to initialize wrap: invalid response")
    else:
      self.report(
          {'WARNING'}, "Failed to initialize wrap: no response. Aren't you running the Wrap server?")

  # def modal(self, context, event):
  #   # Timer
  #   if event.type == "TIMER" and not self.is_running:
  #     if self.response:
  #       if 'code' in self.response:
  #         code, status, message = itemgetter(
  #             'code', 'status', 'message')(self.response)

  #         if code == 0:
  #           self.report({"INFO"}, "Wrap initialization successful")
  #           if message:
  #             wrap_client.set_wrap_init_data(context, message)
  #             self.report({"INFO"}, "Initialization data has been saved")

  #           # Original mode
  #           wrap_options = context.scene.wrap_options
  #           if wrap_options.source_mesh_mode != 'ORIGINAL':
  #             wrap_options.source_mesh_mode = 'ORIGINAL'
  #             source_mesh_mode_update(wrap_options, context)

  #           # Done
  #           wrap_options.wrap_process_stage = 1
  #         else:
  #           self.report(
  #               {'WARNING'}, f"Failed to initialize wrap: {code}, {status}")
  #       else:
  #         self.report(
  #             {'WARNING'}, "Failed to initialize wrap: invalid response")
  #     else:
  #       self.report(
  #           {'WARNING'}, "Failed to initialize wrap: no response. Aren't you running the Wrap server?")

  #   return Timer_Operator.modal(self, context, event)

  def invoke(self, context, event):
    wrap_options = context.scene.wrap_options

    # Check source mesh mode
    if wrap_options.source_mesh_mode != 'ORIGINAL':
      self.report({'WARNING'}, 'To do this, you must be in original mode')
      return {'CANCELLED'}

    # Check symmetric
    if wrap_options.source_is_symmetrical and \
            not wrap_options.source_symmetrized:
      self.report(
          {'WARNING'}, "You have enabled the symmetry option, but the source mesh has not yet been made symmetric")
      return {'CANCELLED'}

    # Set source original positions
    preserve_source_vertices(context)

    # Running Thread
    thread = threading.Thread(
        target=do_wrap_init,
        args=(self, context))
    thread.daemon = True
    thread.start()

    return Timer_Operator.invoke(self, context, event)


############################################################
# Base operator

class Wrap_Operator(Timer_Operator):
  """
  The base operator for all wrap operators
  """
  mesh_data = None
  mesh_changed = False

  def on_timer(self, context):
    wrap_options = context.scene.wrap_options

    with self.mutex:
      # Update mesh vertices
      if self.mesh_data:
        set_source_mesh_vertices(context, self.mesh_data)
        self.mesh_data = None
        self.mesh_changed = True
        wrap_options.progress_iter_num += 1

  def on_stop(self, context):
    # Set mesh vertices
    if self.mesh_changed:
      preserve_source_vertices(context)

  # def modal(self, context, event):
  #   # Timer
  #   if event.type == "TIMER":
  #     wrap_options = context.scene.wrap_options

  #     # Check running flag
  #     if not self.is_running:
  #       # Set mesh vertices
  #       if self.mesh_changed:
  #         preserve_source_vertices(context)

  #     else:
  #       with self.mutex:
  #         # Update mesh vertices
  #         if self.mesh_data:
  #           set_source_mesh_vertices(context, self.mesh_data)
  #           self.mesh_data = None
  #           self.mesh_changed = True
  #           wrap_options.progress_iter_num += 1

  #   return Timer_Operator.modal(self, context, event)


############################################################
# Snap

def recv_feedback(self):
  """
  Receive feedback (vertex positions)
  """
  buffer = b''

  while True:
    buffer, block = receive_data_block(buffer)

    if not block:
      break

    # Update
    with self.mutex:
      self.start_time = time.perf_counter()
      self.mesh_data = block


def do_wrap_snap(self, context):
  """
  Snap source mesh to target mesh
  """
  response = wrap_client.wrap_snap(context)

  if response:
    if 'code' in response:
      code, status = itemgetter('code', 'status')(response)

      if code == 0:
        self.report({"INFO"}, "The wrap snap operation has started")

        # Snap mode
        wrap_options = context.scene.wrap_options
        if wrap_options.source_mesh_mode != 'SNAP':
          wrap_options.source_mesh_mode = 'SNAP'
          source_mesh_mode_update(wrap_options, context)

        # Feedback
        recv_feedback(self)

        # Done
        wrap_options.wrap_process_stage = 2
        self.report({"INFO"}, "The wrap snap operation is complete")
      else:
        self.report(
            {'WARNING'}, f"Failed to execute wrap snap: {code}, {status}")
    else:
      self.report(
          {'WARNING'}, "Failed to execute wrap snap: invalid response")
  else:
    self.report(
        {'WARNING'}, "Failed to execute wrap snap: no response")

  self.is_running = False


class TECA3D_VIEW3D_OT_wrap_snap(Wrap_Operator):
  bl_idname = 'view3d.wrap_snap'
  bl_label = 'Wrap Snap'
  bl_description = 'Roughly wrap the source mesh onto the target mesh by snapping point pairs'

  total_time = 10.0

  def invoke(self, context, event):
    if len(context.scene.wrap_options.ring_markers):
      update_ring_pairs(context)

    thread = threading.Thread(
        target=do_wrap_snap,
        args=(self, context))
    thread.daemon = True
    thread.start()

    return Timer_Operator.invoke(self, context, event)


############################################################
# Attract

def do_wrap_attract(self, context):
  """
  Attract source mesh to target mesh
  """
  response = wrap_client.wrap_attract(context)

  if response:
    if 'code' in response:
      code, status = itemgetter('code', 'status')(response)

      if code == 0:
        self.report({"INFO"}, "The wrap attract operation has started")

        # Snap mode
        wrap_options = context.scene.wrap_options
        if wrap_options.source_mesh_mode != 'ATTRACT':
          wrap_options.source_mesh_mode = 'ATTRACT'
          source_mesh_mode_update(wrap_options, context)

        # Feedback
        recv_feedback(self)

        # Done
        wrap_options.wrap_process_stage = 3
        self.report({"INFO"}, "The wrap attract operation is complete")
      else:
        self.report(
            {'WARNING'}, f"Failed to execute wrap attract: {code}, {status}")
    else:
      self.report(
          {'WARNING'}, "Failed to execute wrap attract: invalid response")
  else:
    self.report(
        {'WARNING'}, "Failed to execute wrap attract: no response")

  self.is_running = False


class TECA3D_VIEW3D_OT_wrap_attract(Wrap_Operator):
  bl_idname = 'view3d.wrap_attract'
  bl_label = 'Wrap Attract'
  bl_description = 'Use the target mesh to attract the source mesh, bringing the vertices of the source mesh close to the surface of the target mesh'

  total_time = 10.0

  def invoke(self, context, event):
    thread = threading.Thread(
        target=do_wrap_attract,
        args=(self, context))
    thread.daemon = True
    thread.start()

    return Timer_Operator.invoke(self, context, event)


############################################################
# Stick

def do_wrap_stick(self, context):
  """
  Stick source mesh to target mesh
  """
  response = wrap_client.wrap_stick(context)

  if response:
    if 'code' in response:
      code, status = itemgetter('code', 'status')(response)

      if code == 0:
        self.report({"INFO"}, "The wrap stick operation has started")

        # Snap mode
        wrap_options = context.scene.wrap_options
        if wrap_options.source_mesh_mode != 'STICK':
          wrap_options.source_mesh_mode = 'STICK'
          source_mesh_mode_update(wrap_options, context)

        # Feedback
        recv_feedback(self)

        # Done
        wrap_options.wrap_process_stage = 4
        self.report({"INFO"}, "The wrap stick operation is complete")
      else:
        self.report(
            {'WARNING'}, f"Failed to execute wrap stick: {code}, {status}")
    else:
      self.report(
          {'WARNING'}, "Failed to execute wrap stick: invalid response")
  else:
    self.report(
        {'WARNING'}, "Failed to execute wrap stick: no response")

  self.is_running = False


class TECA3D_VIEW3D_OT_wrap_stick(Wrap_Operator):
  bl_idname = 'view3d.wrap_stick'
  bl_label = 'Wrap Stick'
  bl_description = 'Make all vertices of the source mesh stick to the surface of the target mesh'

  total_time = 10.0

  def invoke(self, context, event):
    if len(context.scene.wrap_options.ring_markers):
      update_ring_pairs(context)

    thread = threading.Thread(
        target=do_wrap_stick,
        args=(self, context))
    thread.daemon = True
    thread.start()

    return Timer_Operator.invoke(self, context, event)


#############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_wrap_start,
    TECA3D_VIEW3D_OT_wrap_init,
    TECA3D_VIEW3D_OT_wrap_snap,
    TECA3D_VIEW3D_OT_wrap_attract,
    TECA3D_VIEW3D_OT_wrap_stick)
