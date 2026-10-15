# Workspace helpers

import bpy

from .view_helper import get_all_view3d_areas


############################################################
# Create workspace

def create_workspace(name):
  """
  Create a workspace
  """
  if name in bpy.data.workspaces:
    print(f'Error: Workspace {name} already exists')
    return False

  bpy.ops.workspace.append_activate(
      idname="Layout", filepath="<startup.blend>")

  workspace_name = 'Layout.001'
  if workspace_name not in bpy.data.workspaces:
    print(f'Error: Workspace {workspace_name} not found')
    return False

  bpy.data.workspaces[workspace_name].name = name

  return True


############################################################
# Split area

def split_view3d(context, factor=0.5):
  """
  Split the 3d view area horizontally
  """
  window_area_pairs = get_all_view3d_areas(context)

  if len(window_area_pairs) != 1:
    print(
        f'Error: The workspace contains {len(window_area_pairs)} 3D view areas, but only one is expected')
    return False

  window, area = window_area_pairs[0]

  override = context.copy()
  override['window'] = window
  override['screen'] = window.screen
  override['area'] = area
  override['region'] = area.regions[-1]

  with context.temp_override(**override):
    bpy.ops.screen.area_split(
        direction='VERTICAL',
        factor=factor)

  return True
