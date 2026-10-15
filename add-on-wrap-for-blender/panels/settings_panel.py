# Settings panel

import bpy
from bpy.types import Panel

from ..preferences.preferences import get_wrap_prefs


############################################################
# Settings

class TECA3D_VIEW3D_PT_Wrap_SettingsPanel(Panel):
  bl_idname = 'TECA3D_VIEW3D_PT_Wrap_SettingsPanel'
  bl_space_type = 'VIEW_3D'
  bl_region_type = 'WINDOW'
  bl_label = "Wrap Settings"
  bl_options = {'HIDE_HEADER'}

  def draw_header(self, context):
    layout = self.layout
    layout.label(text="", icon="OPTIONS")

  def draw(self, context):
    layout = self.layout
    wrap_prefs = get_wrap_prefs(context)

    col = layout.column(align=True)

    # Account
    row = col.row(align=True)
    split = row.split(factor=0.3)
    split.label(text="Email:")
    split.prop(wrap_prefs, "user_email", text="")

    col.separator(factor=0.5)

    row = col.row(align=True)
    split = row.split(factor=0.3)
    split.label(text="Password:")
    split.prop(wrap_prefs, "user_password", text="")

    # Server
    col.separator()

    row = col.row(align=True)
    split = row.split(factor=0.3)
    split.label(text="Port:")
    split.prop(wrap_prefs, "server_port", text="")
