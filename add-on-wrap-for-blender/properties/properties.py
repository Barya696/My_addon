import bpy
from bpy.props import (
    IntProperty,
    FloatProperty,
    EnumProperty,
    BoolProperty,
    StringProperty,
    FloatVectorProperty,
    IntVectorProperty,
    CollectionProperty
)
from bpy.types import PropertyGroup

from ..operators.op_sync_operators import real_time_sync_update
from ..operators.op_toggle_gizmos import enable_clean_view_update
from ..operators.op_wrap_presets import wrap_preset_update
from ..render.draw_point_pairs import draw_points_update, draw_labels_update
from ..render.draw_point_index import draw_point_index_update
from ..render.draw_rings import draw_rings_update
from ..core.mesh_process import source_mesh_mode_update


def ring_show_update(self, context):
  # A hidden ring is left out of the wrap too: refresh the ring point pairs
  from ..core.ring_markers import update_ring_pairs
  update_ring_pairs(context)
  for area in context.screen.areas if context.screen else ():
    if area.type == 'VIEW_3D':
      area.tag_redraw()


def ring_view_update(self, context):
  for area in context.screen.areas if context.screen else ():
    if area.type == 'VIEW_3D':
      area.tag_redraw()


def ring_index_update(self, context):
  # Picking a ring in the list shows its handles (gizmos) in both views
  ring_view_update(self, context)


############################################################
# Vertex

class TECA3D_Vertex(PropertyGroup):
  # Source
  coord: FloatVectorProperty(
      name='Vertex Coordinate',
      description='Vertex coordinate'
  )  # type: ignore


############################################################
# Point pair

class TECA3D_PointPair(PropertyGroup):
  # Source
  source_vertex_index: IntProperty(
      name='Source Vertex Index',
      description='Index of the selected vertex on the source mesh',
      default=-1
  )  # type: ignore

  # Target
  target_pos: FloatVectorProperty(
      name='Target Position',
      description='Target point position of point pair'
  )  # type: ignore

  target_normal: FloatVectorProperty(
      name='Target Normal',
      description='Target point normal of point pair'
  )  # type: ignore

  target_face_index: IntProperty(
      name='Target Face Index',
      description='Index of the selected face on the target mesh',
      default=-1
  )  # type: ignore

  # Weight
  weight: FloatProperty(
      name="Weight",
      description="Weight of point pair",
      default=1.0,
      min=0.01,
      max=100.0
  )  # type: ignore

  # Face landmark number (DTU-3D, 1-73) of an automatically placed pair, 0
  # for a pair picked by hand
  landmark: IntProperty(
      name="Landmark",
      description="Face landmark number of the point pair (0: picked by hand)",
      default=0
  )  # type: ignore


############################################################
# Ring marker

class TECA3D_RingMarker(PropertyGroup):
  uid: IntProperty(
      name='Ring ID',
      description='Number of the ring marker, it keeps its color'
  )  # type: ignore

  # Each side: the plane cutting the mesh (center, normal) and the seam,
  # where the ring starts
  source_center: FloatVectorProperty(name='Source Center', size=3)  # type: ignore
  source_normal: FloatVectorProperty(name='Source Axis', size=3)  # type: ignore
  source_seam: FloatVectorProperty(name='Source Seam', size=3)  # type: ignore
  target_center: FloatVectorProperty(name='Target Center', size=3)  # type: ignore
  target_normal: FloatVectorProperty(name='Target Axis', size=3)  # type: ignore
  target_seam: FloatVectorProperty(name='Target Seam', size=3)  # type: ignore

  # Placed by Body Landmarks (its ring number, body_landmarks.RINGS), 0 if
  # added by hand: a new Body Landmarks run replaces only its own rings
  landmark: IntProperty(
      name='Body Landmark Ring',
      description='Ring placed by Body Landmarks (0: added by hand)',
      default=0
  )  # type: ignore

  # Shown in both views and used in the wrap; off: hidden and left out
  show: BoolProperty(
      name='Show',
      description='Show the ring in both views and use it in the wrap (off: hidden and left out of the wrap)',
      default=True,
      update=ring_show_update
  )  # type: ignore

  # The ring mirrored from this one on the other side of the body (its uid)
  mirror_uid: IntProperty(
      name='Mirror',
      description='The ring mirrored from this one on the other side of the body (0: none)',
      default=0
  )  # type: ignore


############################################################
# Wrap options

class TECA3D_WrapOptions(PropertyGroup):
  # Meshes
  source_mesh_name: StringProperty(
      name='Source Mesh Name',
      description='The name of the source mesh (template)',
      default=''
  )  # type: ignore

  source_diagonal_length: FloatProperty(
      name='Source Diagonal Length',
      description='The diagonal length of the source mesh bounding box',
      precision=3,
      default=0.0
  )  # type: ignore

  target_mesh_name: StringProperty(
      name='Target Mesh Name',
      description='The name of the target mesh (destination)',
      default=''
  )  # type: ignore

  # Point pairs
  point_pairs: CollectionProperty(
      type=TECA3D_PointPair,
      name='Point Pairs',
      description='Point pairs between the source mesh and target mesh used for wrapping'
  )  # type: ignore

  point_pairs_number: IntProperty(
      name='Point Pairs Number',
      description='Number of point pairs',
      get=lambda self: len(self.point_pairs)
  )  # type: ignore

  draw_points: BoolProperty(
      name="Draw Points",
      description="Enable drawing point pairs",
      default=False,
      update=draw_points_update
  )  # type: ignore

  draw_lines: BoolProperty(
      name="Draw Lines",
      description="Enable drawing lines between point pairs",
      default=False
  )  # type: ignore

  draw_labels: BoolProperty(
      name="Draw Labels",
      description="Enable drawing labels for point pairs",
      default=False,
      update=draw_labels_update
  )  # type: ignore

  point_size: IntProperty(
      name='Point Size',
      description='Point size (in pixels)',
      default=8
  )  # type: ignore

  line_width: IntProperty(
      name='Line Width',
      description='Line width',
      default=1
  )  # type: ignore

  label_size: IntProperty(
      name='Label Size',
      description='Label size (in pixels)',
      default=22
  )  # type: ignore

  # Ring markers
  ring_markers: CollectionProperty(
      type=TECA3D_RingMarker,
      name='Ring Markers',
      description='Rings around the same part of the source mesh and the target mesh, used as point pairs for wrapping'
  )  # type: ignore

  ring_markers_number: IntProperty(
      name='Ring Markers Number',
      description='Number of ring markers',
      get=lambda self: len(self.ring_markers)
  )  # type: ignore

  ring_next_uid: IntProperty(
      name='Next Ring ID',
      default=0
  )  # type: ignore

  ring_pairs_data: StringProperty(
      name='Ring Point Pairs',
      description='The point pairs the ring markers expand into (JSON)',
      default=''
  )  # type: ignore

  ring_index: IntProperty(
      name='Active Ring',
      description='The ring picked in the ring list: its handles show in both views to adjust it',
      default=0,
      update=ring_index_update
  )  # type: ignore

  face_landmark_source: StringProperty(
      name='Source Face Landmarks',
      description='The source vertices found as face landmarks, kept so the source face is only detected once (JSON)',
      default=''
  )  # type: ignore

  body_landmark_source: StringProperty(
      name='Source Body Landmarks',
      description='The points found on the source body, kept so the source is only detected once (JSON)',
      default=''
  )  # type: ignore

  landmark_exclude: StringProperty(
      name='Exclude',
      description='Face and body landmark numbers left out of the wrap and hidden, as shown next to the points: separated by commas, with a dash for a range (e.g. 31-34, 50). Clear it to bring them back',
      default='',
      update=draw_labels_update
  )  # type: ignore

  draw_rings: BoolProperty(
      name="Draw Rings",
      description="Show the ring markers",
      default=False,
      update=draw_rings_update
  )  # type: ignore

  # Symmetric
  source_is_symmetrical: BoolProperty(
      name="Source Is Symmetrical",
      description="Turn this on if the source mesh is symmetrical (e.g., a human or animal model). If this option is enabled, the vertex coordinates of the source mesh must be symmetrized before any wrapping operations are performed",
      default=True
  )  # type: ignore

  center_point_index: IntVectorProperty(
      name='Center Point Index',
      description='Index of center points',
      size=2,
      default=(-1, -1)
  )  # type: ignore

  source_symmetrized: BoolProperty(
      name="Source Symmetrized",
      description="Indicates that the source mesh has been symmetrized",
      get=lambda self: self.source_vertex_sym_idxs != ''
  )  # type: ignore

  source_face_sym_idxs: StringProperty(
      name='Source Face Symmetry Index',
      description='Indexes of faces symmetric to the yz plane',
      default=''
  )  # type: ignore

  source_vertex_sym_idxs: StringProperty(
      name='Source Vertex Symmetry Index',
      description='Indexes of vertices symmetric to the yz plane',
      default=''
  )  # type: ignore

  # Source pick
  source_pick_index: IntProperty(
      name='Source Pick Index',
      description='Index of the selected vertex on the source mesh',
      default=-1
  )  # type: ignore

  # Target pick
  target_pick_pos: FloatVectorProperty(
      name='Target Pick Position',
      description='Position of the selected point on the target mesh',
  )  # type: ignore

  target_pick_normal: FloatVectorProperty(
      name='Target Pick normal',
      description='Normal of the selected point on the target mesh',
  )  # type: ignore

  target_pick_index: IntProperty(
      name='Target Pick Index',
      description='Index of the selected face on the target mesh',
      default=-1
  )  # type: ignore

  target_pick_pos_mir: FloatVectorProperty(
      name='Target Pick Position',
      description='Position of the selected point on the target mesh',
  )  # type: ignore

  target_pick_normal_mir: FloatVectorProperty(
      name='Target Pick normal',
      description='Normal of the selected point on the target mesh',
  )  # type: ignore

  target_pick_index_mir: IntProperty(
      name='Target Pick Index',
      description='Index of the selected face on the target mesh',
      default=-1
  )  # type: ignore

  # Free
  free_index: StringProperty(
      name='Free Index',
      description='Indexes of free vertices in the source mesh',
      default=''
  )  # type: ignore

  free_index_number: IntProperty(
      name='Free Index Number',
      description='Number of free vertices in the source mesh',
      default=0
  )  # type: ignore

  draw_free: BoolProperty(
      name="Draw Free Points",
      description="Draw the free vertices in the source mesh",
      default=False,
      update=draw_point_index_update
  )  # type: ignore

  # Fixed
  fixed_index: StringProperty(
      name='Fixed Index',
      description='Indexes of fixed vertices in the source mesh',
      default=''
  )  # type: ignore

  fixed_index_number: IntProperty(
      name='Fixed Index Number',
      description='Number of fixed vertices in the source mesh',
      default=0
  )  # type: ignore

  draw_fixed: BoolProperty(
      name="Draw Fixed Points",
      description="Draw the fixed vertices in the source mesh",
      default=False,
      update=draw_point_index_update
  )  # type: ignore

  # Wrap start
  wrap_server_running: BoolProperty(
      name="Wrap Server Running",
      description="Indicates whether the wrap server is running",
      default=False
  )  # type: ignore

  wrap_process_stage: IntProperty(
      name="Wrap Process Stage",
      description="The stage number for wrap processing",
      default=0,
      min=0,
      max=5
  )  # type: ignore

  # Wrap presets
  wrap_preset: EnumProperty(
      name='Wrap Presets',
      description='Select a preset for the wrap operation',
      items=[
          ('LOW', 'Low',
           'The Low preset offers the fastest speed and is suitable for simple models', '', 0),
          ('MEDIUM', 'Medium',
           'The Medium preset is suitable for complex models, such as the human head, while still maintaining fast processing speeds', '', 1),
          ('HIGH', 'High',
           'The High preset can handle most complex models, such as the entire human body', '', 2),
          ('CUSTOM', 'Custom',
           'You can customize the parameters of the wrap operation', '', 3),
      ],
      default='MEDIUM',
      update=wrap_preset_update,
  )  # type: ignore

  # Wrap init
  wrap_init_data: StringProperty(
      name='Wrap Init Data',
      description='Data used to initialize the Wrap server',
      default=''
      # subtype='BYTE_STRING'
  )  # type: ignore

  wrap_init_flex: FloatProperty(
      name='Wrap Flexibility',
      description='Flexibility of the wrap transformation',
      default=100.0,
      min=50.0,
      max=200.0
  )  # type: ignore

  wrap_init_accuracy: FloatProperty(
      name='Wrap Accuracy',
      description='Accuracy of the wrap transformation',
      default=3.0,
      min=1.0,
      max=5.0
  )  # type: ignore

  # Wrap snap
  wrap_snap_force: FloatProperty(
      name='Wrap Snap Force',
      description="The pulling force of the wrap-snap operation",
      default=7.0,
      min=5.0,
      max=9.0
  )  # type: ignore

  wrap_snap_softness: FloatProperty(
      name='Wrap Snap Softness',
      description="Smoothness when the source mesh aligns with the target point",
      default=2.0,
      min=0.0,
      max=3.0
  )  # type: ignore

  # Wrap attract
  wrap_attract_distance: FloatProperty(
      name='Wrap Attract Distance',
      description="The maximum allowed distance at which vertices on the source mesh are attracted to the target mesh, expressed as a percentage of the source mesh's diagonal length",
      default=1.0,
      min=0.0,
      max=10.0
  )  # type: ignore

  wrap_attract_angle: FloatProperty(
      name='Wrap Attract Angle',
      description='A vertex on the source mesh will only be attracted if the angle between its normal and the normal of the nearest point on the target mesh is less than this value',
      default=30.0,
      min=0.0,
      max=90.0
  )  # type: ignore

  # Wrap stick
  wrap_stick_distance: FloatProperty(
      name='Wrap Stick Distance',
      description="The maximum allowable distance between a vertex on the source mesh and the surface of the target mesh, expressed as a percentage of the source mesh's diagonal length",
      default=0.5,
      min=0.0,
      max=5.0
  )  # type: ignore

  wrap_stick_angle: FloatProperty(
      name='Wrap Stick Angle',
      description='The maximum allowed angle between the normal of a source mesh vertex and the normal of its nearest point on the target mesh',
      default=50.0,
      min=0.0,
      max=90.0
  )  # type: ignore

  # Source vertices: kept by core/vertex_store.py as flat arrays; these
  # lists only hold positions saved by earlier versions (moved on load)
  source_original_vertices: CollectionProperty(
      type=TECA3D_Vertex,
      name='Source Original Vertices',
      description='Original vertices of the source mesh'
  )  # type: ignore

  source_snap_vertices: CollectionProperty(
      type=TECA3D_Vertex,
      name='Source Snap Vertices',
      description='Vertices of the source mesh after wrap snap operation'
  )  # type: ignore

  source_attract_vertices: CollectionProperty(
      type=TECA3D_Vertex,
      name='Source Attract Vertices',
      description='Vertices of the source mesh after wrap attract operation'
  )  # type: ignore

  source_stick_vertices: CollectionProperty(
      type=TECA3D_Vertex,
      name='Source Stick Vertices',
      description='Vertices of the source mesh after wrap stick operation'
  )  # type: ignore

  source_fit_vertices: CollectionProperty(
      type=TECA3D_Vertex,
      name='Source Fit Vertices',
      description='Vertices of the source mesh for the wrap fit stage: the original shape plus what the posed rig cannot reach'
  )  # type: ignore

  wrap_fit_pose: StringProperty(
      name='Wrap Fit Pose',
      description='Pose of the rig for the wrap fit stage',
      default=''
  )  # type: ignore

  source_mesh_mode: EnumProperty(
      name='Source Mesh Mode',
      description='Select the display mode for source mesh vertices',
      items=[
          ('ORIGINAL', 'Original',
           'Display the original positions of mesh vertices', '', 0),
          ('SNAP', 'Snap', 'Display the position of mesh vertices after the wrap snap operation', '', 1),
          ('ATTRACT', 'Attract',
           'Display the position of mesh vertices after the wrap attract operation', '', 2),
          ('STICK', 'Stick', 'Display the position of mesh vertices after the wrap stick operation', '', 3),
          ('FIT', 'Fit', 'Display the original shape deformed by the rig posed to the wrap stick result', '', 4),
      ],
      default='ORIGINAL',
      update=source_mesh_mode_update,
  )  # type: ignore

  # Views
  real_time_sync: BoolProperty(
      name="Real-time Sync",
      description="Enable real-time synchronization",
      default=False,
      update=real_time_sync_update
  )  # type: ignore

  master_view_index: IntProperty(
      name="Master View Index",
      description="Index of the master View3D",
      default=0,
      min=0
  )  # type: ignore

  # Sync
  sync_refresh_rate: FloatProperty(
      name="Refresh Rate",
      description="Refresh rate for real-time synchronization (in seconds)",
      default=0.01,
      min=0.01,
      max=1.0
  )  # type: ignore

  sync_view_distance_adjust: FloatProperty(
      name="Adjust View Distance:",
      default=1.0,
      min=0.01,
      max=5.0,
      description="Fine adjustment (Multiplier) for Sync View Distance"
  )   # type: ignore

  # UI
  enable_clean_view: BoolProperty(
      name="Enable Clean View",
      description="Achieve a clean view by hiding certain UI elements",
      default=False,
      update=enable_clean_view_update
  )  # type: ignore

  progress_value: FloatProperty(
      name="Progress Bar Value",
      subtype='PERCENTAGE',
      default=0.0,
      min=0.0,
      max=1.0,
      description="The value displayed on the progress bar that shows the progress of time-consuming operations"
  )   # type: ignore

  progress_time: FloatProperty(
      name="Progress Time",
      default=0.0,
      description="Time spent on progress (in seconds)"
  )   # type: ignore

  progress_iter_num: IntProperty(
      name="Progress iteration number",
      default=0,
      description="Number of iterations of the progress bar"
  )   # type: ignore


############################################################
# Setup

classes = (
    TECA3D_Vertex,
    TECA3D_PointPair,
    TECA3D_RingMarker,
    TECA3D_WrapOptions)


def register():
  for cls in classes:
    bpy.utils.register_class(cls)

  bpy.types.Scene.wrap_options = bpy.props.PointerProperty(
      type=TECA3D_WrapOptions)


def unregister():
  del bpy.types.Scene.wrap_options

  for cls in reversed(classes):
    bpy.utils.unregister_class(cls)


############################################################
# Script

if __name__ == "__main__":
  register()
