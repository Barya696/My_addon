# Wrap for blender

import bpy

# Panels
from .panels.main_panel import TECA3D_VIEW3D_PT_Wrap_MainPanel
from .panels.settings_panel import TECA3D_VIEW3D_PT_Wrap_SettingsPanel

# Operators
from .operators import classes as operators
from .operators.op_sync_operators import real_time_sync_timer
from .render.draw_rings import remove_draw_rings_handlers

# Properties
from .properties import properties

# Preferences
from .preferences import preferences

# Post handler
from .utils import post_load_handlers
from .utils.system_helper import (
    get_process_pid,
    kill_process,
    global_wrap_server_name
)


############################################################
# Setup

classes = (
    *operators,
    TECA3D_VIEW3D_PT_Wrap_MainPanel,
    TECA3D_VIEW3D_PT_Wrap_SettingsPanel)


def register():
  for cls in classes:
    bpy.utils.register_class(cls)

  properties.register()

  preferences.register()

  post_load_handlers.register()


def unregister():
  if bpy.app.timers.is_registered(real_time_sync_timer):
    bpy.app.timers.unregister(real_time_sync_timer)

  post_load_handlers.unregister()

  remove_draw_rings_handlers()

  preferences.unregister()

  properties.unregister()

  for cls in reversed(classes):
    bpy.utils.unregister_class(cls)

  pid = get_process_pid(global_wrap_server_name)
  if pid != -1:
    kill_process(pid)


############################################################
# Script

if __name__ == "__main__":
  register()
