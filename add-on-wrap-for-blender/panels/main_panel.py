# Wrap panel

import bpy
from bpy.types import Panel

from ..utils.view_helper import (
    get_all_view3d_areas,
    get_source_target_areas
)
from ..utils.point_pair_helper import (
    get_source_indices
)
from ..core.ring_markers import state as ring_state


############################################################
# Main panel

class TECA3D_VIEW3D_PT_Wrap_MainPanel(Panel):
  bl_idname = 'TECA3D_VIEW3D_PT_Wrap_MainPanel'
  bl_space_type = 'VIEW_3D'
  bl_region_type = 'UI'
  bl_category = "Wrap"
  bl_label = "Wrap for Blender"
  bl_options = {"DEFAULT_CLOSED"}

  def draw_header(self, context):
    layout = self.layout
    layout.label(text="", icon="EVENT_W")

  def draw(self, context):
    layout = self.layout
    area = context.area
    wrap_options = context.scene.wrap_options

    col = layout.column(align=True)

    # Create workspace
    if 'Wrap' not in bpy.data.workspaces:
      row = col.row(align=True)
      row.operator("view3d.create_workspace", text="Create Workspace")
      return

    if context.window.workspace.name != 'Wrap':
      return

    # Get all areas
    window_area_pairs = get_all_view3d_areas(context)
    if len(window_area_pairs) == 1:
      row = col.row(align=True)
      row.operator("view3d.create_view3d", text="Create Side-by-Side View")
      return

    if len(window_area_pairs) != 2:
      return

    source_area, target_area = get_source_target_areas(window_area_pairs)

    if not source_area or not target_area:
      return

    # Check local view
    # row = col.row(align=False)
    # row.scale_y = 0.32
    # row.alert = True
    # row.alignment = "CENTER"
    # row.label(text='———————————')
    # row.alert = False
    # col.separator()

    process_stage = wrap_options.wrap_process_stage

    ########################################################
    # Source mesh and target mesh

    if area == source_area:
      row = col.row(align=True)
      row.enabled = False
      row.prop(wrap_options, "source_mesh_name", placeholder="Source", text='')

      row = col.row(align=True)
      row.operator("view3d.set_source_mesh", text="Set")
      row.prop(wrap_options, "real_time_sync", text="Sync", toggle=True)

      sub_row = row.row(align=True)
      sub_row.enabled = False
      sub_row.prop(wrap_options, "source_diagonal_length", text="")

    elif area == target_area:
      row = col.row(align=True)
      row.enabled = False
      row.prop(wrap_options, "target_mesh_name", placeholder="Target", text='')

      row = col.row(align=True)
      row.operator("view3d.set_target_mesh", text="Set")
      row.prop(wrap_options, "real_time_sync", text="Sync", toggle=True)
    else:
      return

    ########################################################
    # Free

    if area == source_area:
      col.separator()

      row = col.row(align=True)
      row.prop(wrap_options, "draw_free", text="Free", toggle=True)
      sub_row = row.row(align=True)
      sub_row.enabled = False
      sub_row.prop(wrap_options, "free_index_number", text='')

      if wrap_options.draw_free:
        op = row.operator("view3d.add_point_index", text="Add")
        op.target = 'Free Vertices'
        op = row.operator("view3d.remove_point_index", text="Remove")
        op.target = 'Free Vertices'

      # Show, save, load fixed index
      if wrap_options.draw_free:
        row = col.row(align=True)
        op = row.operator("view3d.load_point_index", text="Load")
        op.target = 'Free Vertices'
        op = row.operator("view3d.save_point_index", text="Save")
        op.source = 'Free Vertices'
        op = row.operator("view3d.clear_point_index", text='Clear')
        op.target = 'Free Vertices'

    ########################################################
    # Fixed

    if area == source_area:
      col.separator()

      row = col.row(align=True)
      row.prop(wrap_options, "draw_fixed", text="Fixed", toggle=True)
      sub_row = row.row(align=True)
      sub_row.enabled = False
      sub_row.prop(wrap_options, "fixed_index_number", text='')

      if wrap_options.draw_fixed:
        op = row.operator("view3d.add_point_index", text="Add")
        op.target = 'Fixed Vertices'
        op = row.operator("view3d.remove_point_index", text="Remove")
        op.target = 'Fixed Vertices'

      # Show, save, load fixed index
      if wrap_options.draw_fixed:
        row = col.row(align=True)
        op = row.operator("view3d.load_point_index", text="Load")
        op.target = 'Fixed Vertices'
        op = row.operator("view3d.save_point_index", text="Save")
        op.source = 'Fixed Vertices'
        op = row.operator("view3d.clear_point_index", text='Clear')
        op.target = 'Fixed Vertices'

    ########################################################
    # Point pairs

    if area == target_area:
      col.separator()

      row = col.row(align=True)
      row.prop(wrap_options, "draw_points", text="Points", toggle=True)
      sub_row = row.row(align=True)
      sub_row.enabled = False
      sub_row.prop(wrap_options, "point_pairs_number", text='')

      if wrap_options.draw_points:
        row.prop(wrap_options, "point_size", text="")

        row = col.row(align=True)
        row.prop(wrap_options, "draw_lines", text="Lines", toggle=True)
        row.prop(wrap_options, "line_width", text="")
        row.prop(wrap_options, "draw_labels", text="Labels", toggle=True)
        row.prop(wrap_options, "label_size", text="")

        # Save, load point pairs
        row = col.row(align=True)
        row.operator("view3d.load_point_pairs", text="Load")
        row.operator("view3d.save_point_pairs", text="Save")
        row.operator("view3d.clear_point_pairs", text='Clear')

      # Face point pairs found automatically
      row = col.row(align=True)
      row.operator("view3d.face_landmarks", text="Face Landmarks", icon='USER')
      row.operator("view3d.body_landmarks", text="Body Landmarks", icon='ARMATURE_DATA')

      # Face and body landmark numbers left out of the wrap (and hidden)
      row = col.row(align=True)
      row.prop(wrap_options, "landmark_exclude", placeholder="e.g. 31-34, 50")

    ########################################################
    # Ring markers

    col.separator()

    row = col.row(align=True)
    row.prop(wrap_options, "draw_rings", text="Rings", toggle=True)
    sub_row = row.row(align=True)
    sub_row.enabled = False
    sub_row.prop(wrap_options, "ring_markers_number", text='')

    if wrap_options.draw_rings:
      # The ring list, laid out like Blender's bone collections: name,
      # mirror, show (both views, and the wrap); add, remove, menu, move.
      # Picking a ring adjusts it (its handles show in both views)
      row = col.row()
      row.template_list("TECA3D_UL_ring_markers", "", wrap_options, "ring_markers",
                        wrap_options, "ring_index", rows=4)
      side = row.column(align=True)
      side.operator("view3d.add_ring_marker", text="", icon='ADD')
      side.operator("view3d.remove_ring_marker", text="", icon='REMOVE')
      side.separator()
      side.menu("TECA3D_MT_ring_markers", text="", icon='DOWNARROW_HLT')
      side.separator()
      side.operator("view3d.move_ring_marker", text="", icon='TRIA_UP').direction = 'UP'
      side.operator("view3d.move_ring_marker", text="", icon='TRIA_DOWN').direction = 'DOWN'

    ########################################################
    # Symmetric

    if area == source_area:
      col.separator()

      row = col.row(align=True)
      row.prop(wrap_options, "source_is_symmetrical",
               text="Is Symmetrical", toggle=True)
      sub_row = row.row(align=True)
      sub_row.enabled = False
      sub_row.prop(wrap_options, "source_symmetrized",
                   text="Symmetrized", toggle=True)

      if wrap_options.source_is_symmetrical and \
              not wrap_options.source_symmetrized:
        row = col.row(align=True)
        row.operator("view3d.add_center_point", text="Add")
        row.operator('view3d.remove_center_point', text='Remove')

        first, second = wrap_options.center_point_index
        sub_row = row.row(align=True)
        sub_row.enabled = first != -1 and second != -1
        sub_row.operator('view3d.symmetrize_source_mesh', text='Symmetrize')

    ########################################################
    # Selected point

    col.separator()

    # if area == source_area:
    #   row = col.row(align=True)
    #   row.enabled = False
    #   row.label(text='Point Pair:')

    row = col.row(align=True)
    row.operator("view3d.pick_location", text="Select")
    row.operator('view3d.clear_pick_point', text='Unselect')

    # Add, remove, edit
    source_index = wrap_options.source_pick_index
    target_index = wrap_options.target_pick_index

    if source_index != -1:
      indices = get_source_indices(context)

      # Select a point pair
      if source_index in indices:
        if target_index != -1:
          row.operator("view3d.edit_point_pair", text="Change")
        else:
          row.operator("view3d.remove_point_pair", text="Remove")

      # Select a new point
      elif target_index != -1:
        row.operator("view3d.add_point_pair", text="Add")

    ########################################################
    # Source mesh mode

    if area == source_area:
      col.separator()

      row = col.row(align=True)
      items = wrap_options.bl_rna.properties["source_mesh_mode"].enum_items_static
      for i, item in enumerate(items):
        identifier = item.identifier
        sub_row = row.row(align=True)
        sub_row.prop_enum(wrap_options, "source_mesh_mode", identifier)
        sub_row.enabled = i == 0 or i < process_stage

    ########################################################
    # Progress bar

    if area == source_area:
      progress_value = wrap_options.progress_value
      progress_time = wrap_options.progress_time
      progress_iter_num = wrap_options.progress_iter_num

      if progress_value > 0.0 or progress_iter_num > 1:
        col.separator(factor=1.5)

        row = col.row(align=True)
        row.scale_y = 0.5

        split = row.split(factor=0.8)

        left = split.row(align=True)
        left.progress(factor=progress_value, type='BAR')

        right = split.row(align=True)
        right.label(text=f' {progress_time: 4.1f} {progress_iter_num: 2d}')

    ########################################################
    # Wrap

    if area == target_area:
      col.separator()

      # row = col.row(align=True)
      # row.enabled = False
      # row.label(text='Wrap:')

      # Start
      if not wrap_options.wrap_server_running:
        row = col.row(align=True)
        row.operator("view3d.wrap_start", text="Start")

      # Wrap
      else:
        row = col.row(align=True)
        row.prop(wrap_options, "wrap_preset", expand=True)

        col.separator(factor=0.5)

        is_custom_preset = wrap_options.wrap_preset == "CUSTOM"

        row = col.row(align=True)
        row.operator("view3d.wrap_init", text="Init")
        sub_row = row.row(align=True)
        sub_row.enabled = is_custom_preset
        sub_row.prop(wrap_options, "wrap_init_flex", text="")
        sub_row.prop(wrap_options, "wrap_init_accuracy", text="")

        if process_stage >= 1:
          row = col.row(align=True)
          row.operator("view3d.wrap_snap", text="Snap")
          sub_row = row.row(align=True)
          sub_row.enabled = is_custom_preset
          sub_row.prop(wrap_options, "wrap_snap_force", text="")
          sub_row.prop(wrap_options, "wrap_snap_softness", text="")

        if process_stage >= 2:
          row = col.row(align=True)
          row.operator("view3d.wrap_attract", text="Attract")
          sub_row = row.row(align=True)
          sub_row.enabled = is_custom_preset
          sub_row.prop(wrap_options, "wrap_attract_distance", text="")
          sub_row.prop(wrap_options, "wrap_attract_angle", text="")

        if process_stage >= 3:
          row = col.row(align=True)
          row.operator("view3d.wrap_stick", text="Stick")
          sub_row = row.row(align=True)
          sub_row.enabled = is_custom_preset
          sub_row.prop(wrap_options, "wrap_stick_distance", text="")
          sub_row.prop(wrap_options, "wrap_stick_angle", text="")

        if process_stage >= 4:
          row = col.row(align=True)
          row.operator("view3d.wrap_fit", text="Fit")

    ########################################################
    # UI

    if area == target_area:
      col.separator()

      # row = col.row(align=True)
      # row.enabled = False
      # row.label(text='UI:')

      row = col.row(align=True)
      row.prop(wrap_options, "enable_clean_view",
               text="Clean View", toggle=True)
      row.operator("wm.window_fullscreen_toggle", text="Fullscreen")

    ########################################################
    # Settings

    if area == target_area:
      col.separator()

      row = col.row()
      row.popover(panel="TECA3D_VIEW3D_PT_Wrap_SettingsPanel", text="Settings")
