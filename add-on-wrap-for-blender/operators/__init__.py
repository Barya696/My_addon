# Operators

from .op_create_workspace import classes as create_workspace
from .op_sync_operators import classes as sync_operators
from .op_auto_master import classes as auto_master
from .op_mesh_operators import classes as mesh_operators
from .op_mesh_symmetry import classes as mesh_symmetry
from .op_point_index import classes as point_index
from .op_point_pair import classes as point_pair
from .op_pick_point import classes as pick_point
from .op_wrap_operators import classes as wrap_operators
from .op_wrap_presets import classes as wrap_presets
from .op_toggle_gizmos import classes as toggle_gizmos
from .op_wrap_fit import classes as wrap_fit
from .op_ring_markers import classes as ring_markers
from .op_ring_gizmos import classes as ring_gizmos
from .op_face_landmarks import classes as face_landmarks
from .op_body_landmarks import classes as body_landmarks

classes = (
    *create_workspace,
    *sync_operators,
    *auto_master,
    *mesh_operators,
    *mesh_symmetry,
    *point_index,
    *point_pair,
    *pick_point,
    *wrap_operators,
    *wrap_presets,
    *toggle_gizmos,
    *wrap_fit,
    *ring_markers,
    *ring_gizmos,
    *face_landmarks,
    *body_landmarks)
