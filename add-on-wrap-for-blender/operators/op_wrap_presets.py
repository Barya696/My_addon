# Wrap presets

import bpy
from bpy.types import Operator
from bpy.props import StringProperty


############################################################
# Global variables

global_wrap_presets = {
    'Low': {
        "wrap_init_flex": 50,
        "wrap_init_accuracy": 4,
        "wrap_snap_force": 6,
        "wrap_snap_softness": 2,
        "wrap_attract_distance": 1,
        "wrap_attract_angle": 30,
        "wrap_stick_distance": 0.5,
        "wrap_stick_angle": 50
    },
    'Medium': {
        "wrap_init_flex": 100,
        "wrap_init_accuracy": 3,
        "wrap_snap_force": 7,
        "wrap_snap_softness": 2,
        "wrap_attract_distance": 1,
        "wrap_attract_angle": 30,
        "wrap_stick_distance": 0.5,
        "wrap_stick_angle": 50
    },
    'High': {
        "wrap_init_flex": 150,
        "wrap_init_accuracy": 2.5,
        "wrap_snap_force": 8,
        "wrap_snap_softness": 2,
        "wrap_attract_distance": 1,
        "wrap_attract_angle": 30,
        "wrap_stick_distance": 0.5,
        "wrap_stick_angle": 50
    },
}


############################################################
# Operations

class TECA3D_VIEW3D_OT_load_wrap_preset(Operator):
  bl_idname = 'view3d.load_wrap_preset'
  bl_label = 'Load Wrap Preset'
  bl_description = 'Load the preset used for all wrap operations'

  preset_name: StringProperty(
      name="Preset Name", default="High")  # type: ignore

  def execute(self, context):
    global global_wrap_presets

    preset = global_wrap_presets[self.preset_name]

    wrap_options = context.scene.wrap_options

    wrap_options.wrap_init_flex = preset["wrap_init_flex"]
    wrap_options.wrap_init_accuracy = preset["wrap_init_accuracy"]

    wrap_options.wrap_snap_force = preset["wrap_snap_force"]
    wrap_options.wrap_snap_softness = preset["wrap_snap_softness"]

    wrap_options.wrap_attract_distance = preset["wrap_attract_distance"]
    wrap_options.wrap_attract_angle = preset["wrap_attract_angle"]

    wrap_options.wrap_stick_distance = preset["wrap_stick_distance"]
    wrap_options.wrap_stick_angle = preset["wrap_stick_angle"]

    return {'FINISHED'}


############################################################
# Update

def wrap_preset_update(self, context):
  wrap_preset = self.wrap_preset

  name = None
  if wrap_preset == 'LOW':
    name = 'Low'
  elif wrap_preset == 'MEDIUM':
    name = 'Medium'
  elif wrap_preset == 'HIGH':
    name = 'High'

  if name:
    bpy.ops.view3d.load_wrap_preset(preset_name=name)


#############################################################
# All classes

classes = (
    TECA3D_VIEW3D_OT_load_wrap_preset,)
