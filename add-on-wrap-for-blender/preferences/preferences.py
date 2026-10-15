# User preferences

import bpy
from bpy.types import AddonPreferences
from bpy.props import StringProperty, IntProperty


############################################################
# Getter

def get_main_package_name():
  return '.'.join(__package__.split('.')[:-1])


def get_wrap_prefs(context):
  name = get_main_package_name()
  return context.preferences.addons[name].preferences


############################################################
# Preferences

class TECA3D_Wrap_Preferences(AddonPreferences):
  # This must match the add-on's module name
  # often __name__ if it's a single file addon
  # or __package__ if it's a multi-file addon

  # The ID name of the add-on's preference class must match the
  # add-on's package name; otherwise, the preference class will not
  # register correctly, will not appear in Blender's add-on settings,
  # and will not be accessible.

  bl_idname = get_main_package_name()

  # Account
  user_email: StringProperty(
      name='Email',
      description='The email address you used to register your account',
      default=''
  )  # type: ignore

  user_password: StringProperty(
      name='Password',
      description='The password you used to register your account',
      default=''
  )  # type: ignore

  # Client
  server_port: IntProperty(
      name="Server Port",
      description="The port number that the Wrap server listens on",
      default=31008,
      min=30000,
      max=60000
  )  # type: ignore

  # Face landmarks
  face_landmarks_dir: StringProperty(
      name='Face Landmarks',
      description='The landmarks folder of snap-to-stick-iteration (with its .venv, made by setup_windows.bat)',
      subtype='DIR_PATH',
      default=''
  )  # type: ignore

  face_landmark_views: IntProperty(
      name="Views",
      description="Views the face detector looks from: more is steadier, fewer is faster (32 takes about a minute per face)",
      default=32,
      min=8,
      max=96
  )  # type: ignore

  # Body landmarks
  body_landmarks_dir: StringProperty(
      name='Body Landmarks',
      description='The body_landmarks folder of snap-to-stick-iteration (with its .venv). Empty: body_landmarks next to the Face Landmarks folder',
      subtype='DIR_PATH',
      default=''
  )  # type: ignore

  body_landmark_views: IntProperty(
      name="Views",
      description="Views the body detector looks from: more is steadier, fewer is faster (36 takes about 15 s per body)",
      default=36,
      min=12,
      max=96
  )  # type: ignore

  def draw(self, context):
    layout = self.layout

    col = layout.column(align=True)

    # Account
    row = col.row(align=True)
    row.enabled = False
    row.label(text="Account:")

    row = col.row(align=True)
    row.prop(self, "user_email", text="Email")
    row.separator(factor=2)
    row.prop(self, "user_password", text="Password")

    # Server
    col.separator()

    row = col.row(align=True)
    row.enabled = False
    row.label(text="Server:")

    row = col.row(align=True)
    row.prop(self, "server_port", text="Port")
    row.separator(factor=2)
    row.label(text="")

    # Face landmarks
    col.separator()

    row = col.row(align=True)
    row.enabled = False
    row.label(text="Face Landmarks:")

    row = col.row(align=True)
    row.prop(self, "face_landmarks_dir", text="Folder")
    row.separator(factor=2)
    row.prop(self, "face_landmark_views", text="Views")

    # Body landmarks
    col.separator()

    row = col.row(align=True)
    row.enabled = False
    row.label(text="Body Landmarks:")

    row = col.row(align=True)
    row.prop(self, "body_landmarks_dir", text="Folder")
    row.separator(factor=2)
    row.prop(self, "body_landmark_views", text="Views")


############################################################
# Setup

classes = (
    TECA3D_Wrap_Preferences,)


def register():
  for cls in classes:
    bpy.utils.register_class(cls)


def unregister():
  for cls in reversed(classes):
    bpy.utils.unregister_class(cls)
