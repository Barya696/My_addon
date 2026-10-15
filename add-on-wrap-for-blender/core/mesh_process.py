# Mesh process

import json

import bpy
from mathutils import Matrix, Vector

from . import vertex_store
from ..utils.mesh_helper import set_mesh_vertices


############################################################
# Source mesh mode

def source_mesh_mode_update(self, context):
  obj = bpy.data.objects.get(self.source_mesh_name)
  if not obj:
    return

  mode = self.source_mesh_mode
  vertices = []

  if mode == 'ORIGINAL':
    vertices = [Vector(v) for v in vertex_store.get(self, 'original')]
    if len(vertices) == 0:
      print("Error: The original vertex list is empty")

  elif mode == 'SNAP':
    vertices = [Vector(v) for v in vertex_store.get(self, 'snap')]
    if len(vertices) == 0:
      print("Warning: The snap vertex list is empty")

  elif mode == 'ATTRACT':
    vertices = [Vector(v) for v in vertex_store.get(self, 'attract')]
    if len(vertices) == 0:
      print("Warning: The attract vertex list is empty")

  elif mode == 'STICK':
    vertices = [Vector(v) for v in vertex_store.get(self, 'stick')]
    if len(vertices) == 0:
      print("Warning: The stick vertex list is empty")

  elif mode == 'FIT':
    # The original shape: the Fit shape key and the rig's pose do the rest
    vertices = [Vector(v) for v in vertex_store.get(self, 'original')]
    if len(vertices) == 0:
      print("Error: The original vertex list is empty")

  if len(vertices) > 0:
    set_mesh_vertices(obj, vertices)

  # The fitted pose and the Fit shape key belong to the Fit stage only: every
  # other stage shows its vertices with the rig at rest
  set_fit_pose(self, mode == 'FIT')


############################################################
# Fit stage

FIT_KEY = 'Wrap Fit'
FIT_ACTIVE = 'wrap_fit_active'


def set_fit_pose(wrap_options, apply):
  """
  Put the rig in the stored fit pose and switch the Fit shape key on (apply),
  or put the rig back at rest and switch the key off
  """
  if not wrap_options.wrap_fit_pose:
    return

  try:
    data = json.loads(wrap_options.wrap_fit_pose)
  except json.JSONDecodeError:
    print("Warning: The wrap fit pose could not be read")
    return

  rig = bpy.data.objects.get(data.get('rig', ''))
  if not rig or rig.type != 'ARMATURE':
    return

  for name, m in data.get('bones', {}).items():
    pose_bone = rig.pose.bones.get(name)
    if pose_bone:
      pose_bone.matrix_basis = (Matrix([m[0:4], m[4:8], m[8:12], m[12:16]])
                                if apply else Matrix.Identity(4))

  obj = bpy.data.objects.get(wrap_options.source_mesh_name)
  if not obj:
    return

  keys = obj.data.shape_keys
  if apply and (not keys or FIT_KEY not in keys.key_blocks) \
          and vertex_store.count(wrap_options, 'fit') == len(obj.data.vertices):
    build_fit_shape_key(obj, rig, vertex_store.get(wrap_options, 'fit').tolist())

  obj[FIT_ACTIVE] = 1 if apply else 0
  obj.update_tag()


def build_fit_shape_key(obj, rig, coords_world):
  """
  Corrective shape key with what the posed rig cannot reach. A driver makes
  it act only in the Fit stage while the rig is in Pose Position:
      value = fit * (1 - rest)
  fit is the mesh object's wrap_fit_active property (set by the stage
  switch), rest is the armature's Pose / Rest Position switch
  """
  if FIT_ACTIVE not in obj:
    obj[FIT_ACTIVE] = 0

  mesh = obj.data
  if not mesh.shape_keys:
    obj.shape_key_add(name='Basis', from_mix=False)
  keys = mesh.shape_keys

  key = keys.key_blocks.get(FIT_KEY)
  if not key:
    key = obj.shape_key_add(name=FIT_KEY, from_mix=False)
  key.relative_key = keys.reference_key
  key.slider_min = 0.0
  key.slider_max = 1.0

  matrix = obj.matrix_world.inverted()
  key.data.foreach_set('co', [c for v in coords_world for c in matrix @ Vector(v)])

  key.driver_remove('value')
  driver = key.driver_add('value').driver
  driver.type = 'SCRIPTED'
  for name, id_type, owner, path in (
          ('fit', 'OBJECT', obj, f'["{FIT_ACTIVE}"]'),
          ('rest', 'ARMATURE', rig.data, 'pose_position')):
    var = driver.variables.new()
    var.name = name
    var.type = 'SINGLE_PROP'
    var.targets[0].id_type = id_type
    var.targets[0].id = owner
    var.targets[0].data_path = path
  driver.expression = 'fit * (1 - rest)'
  mesh.update()
