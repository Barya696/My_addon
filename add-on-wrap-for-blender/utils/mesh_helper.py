# Mesh helper

from operator import itemgetter

import bpy
import bmesh
from mathutils import Vector, kdtree
import numpy as np

from ..core import vertex_store


############################################################
# Mesh vertices

def get_mesh_vertices(obj, is_global=True):
  """
  Get mesh vertices
  """
  if not obj or obj.type != 'MESH':
    return []

  mesh = obj.data

  if is_global:
    coords = [obj.matrix_world @ v.co for v in mesh.vertices]
  else:
    coords = [v.co.copy() for v in mesh.vertices]

  # # Ensure we are in Object Mode to access the mesh data directly
  # if bpy.context.mode != 'OBJECT':
  #   bpy.ops.object.mode_set(mode='OBJECT')

  # # Get a BMesh representation and fill it in from the mesh
  # bm = bmesh.new()
  # bm.from_mesh(obj.data)

  # coords = []
  # if is_global:
  #   coords = [obj.matrix_world @ v.co for v in bm.verts]
  # else:
  #   coords = [v.co.copy() for v in bm.verts]

  # # Free and prevent further access
  # bm.free()

  return coords


def set_mesh_vertices(obj, vertices, is_global=True) -> bool:
  """
  Set mesh vertices
  """
  if not obj or obj.type != 'MESH':
    return False

  mesh = obj.data

  # Ensure the number of new positions matches the number of vertices
  if len(vertices) != len(mesh.vertices):
    print(
        f'Error: Number of positions provided ({len(vertices)}) does not match vertex count ({len(mesh.vertices)}).')
    return False

  # Iterate through the vertices and assign new coordinates
  if is_global:
    matrix = obj.matrix_world.inverted()
    for a, b in zip(mesh.vertices, vertices):
      a.co = matrix @ Vector(b)
  else:
    for a, b in zip(mesh.vertices, vertices):
      a.co = b

  sync_basis_key(mesh)
  return True


def sync_basis_key(mesh):
  """
  With shape keys, Blender evaluates the reference key instead of the mesh's
  own vertex positions: copy them into it so that a vertex write still shows
  """
  keys = mesh.shape_keys
  if not keys or not keys.reference_key:
    return

  coords = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
  mesh.vertices.foreach_get('co', coords)
  keys.reference_key.data.foreach_set('co', coords)
  mesh.update()


############################################################
# Source and target mesh

def get_source_vertices(context):
  """
  Get source mesh vertices
  """
  wrap_options = context.scene.wrap_options
  mesh_name = wrap_options.source_mesh_name

  obj = bpy.data.objects.get(mesh_name)
  return get_mesh_vertices(obj)


def get_target_vertices(context):
  """
  Get target mesh vertices
  """
  wrap_options = context.scene.wrap_options
  mesh_name = wrap_options.target_mesh_name

  obj = bpy.data.objects.get(mesh_name)
  return get_mesh_vertices(obj)


############################################################
# Source mesh mode

def preserve_source_vertices(context):
  """
  Set source mesh original vertices
  """
  coords_world = get_source_vertices(context)

  wrap_options = context.scene.wrap_options
  mesh_mode = wrap_options.source_mesh_mode

  # FIT shows the original vertices: nothing of its own to preserve
  if mesh_mode in {'ORIGINAL', 'SNAP', 'ATTRACT', 'STICK'}:
    vertex_store.store(wrap_options, mesh_mode.lower(), coords_world)


def clear_source_vertices_reserve(context):
  """
  Clear all source mesh vertices reserves
  """
  wrap_options = context.scene.wrap_options

  for stage in ('original', 'snap', 'attract', 'stick'):
    vertex_store.clear(wrap_options, stage)


############################################################
# Source mesh diagonal length

def get_source_mesh_diagonal_length(context):
  wrap_options = context.scene.wrap_options
  mesh_name = wrap_options.source_mesh_name

  obj = bpy.data.objects.get(mesh_name)

  # Get the 8 corners of the bounding box in world space
  world_bbox = np.array([obj.matrix_world @ Vector(v) for v in obj.bound_box])

  diag = np.ptp(world_bbox, 0)
  diag_len = np.linalg.norm(diag)

  return diag_len


############################################################
# Source mesh feedback

def set_source_mesh_vertices(context, data):
  """
  Set source mesh vertices from bytes (feedback)
  """
  wrap_options = context.scene.wrap_options
  mesh_name = wrap_options.source_mesh_name

  obj = bpy.data.objects.get(mesh_name)
  matrix = obj.matrix_world.inverted()

  mat = np.array(matrix)
  verts = np.frombuffer(data, np.float64).reshape((-1, 3))

  verts_loc = mat[:3].dot(np.vstack((verts.T, np.ones(len(verts))))).T

  vertices = [Vector(v) for v in verts_loc]
  set_mesh_vertices(obj, vertices, False)


############################################################
# Mesh vertex normal

def get_mesh_normals(obj, is_global=True):
  """
  Get mesh vertex normals
  """
  if not obj or obj.type != 'MESH':
    return []

  mesh = obj.data
  mat = obj.matrix_world.to_3x3()

  normals = []
  if is_global:
    normals = [mat @ v.normal for v in mesh.vertices]
  else:
    normals = [v.normal for v in mesh.vertices]

  return normals


def get_source_normals(context):
  """
  Get source mesh vertex normals
  """
  wrap_options = context.scene.wrap_options
  mesh_name = wrap_options.source_mesh_name

  obj = bpy.data.objects.get(mesh_name)
  return get_mesh_normals(obj)


def get_target_normals(context):
  """
  Get target mesh vertex normals
  """
  wrap_options = context.scene.wrap_options
  mesh_name = wrap_options.target_mesh_name

  obj = bpy.data.objects.get(mesh_name)
  return get_mesh_normals(obj)


############################################################
# Mesh faces

def get_mesh_faces(obj):
  """
  Get mesh faces
  """
  if not obj or obj.type != 'MESH':
    return []

  mesh = obj.data

  faces = [list(f.vertices) for f in mesh.polygons]

  return faces


def get_mesh_triangles(obj):
  """
  Get mesh triangles
  """
  if not obj or obj.type != 'MESH':
    return []

  # Ensure we are in Object Mode to access the mesh data directly
  if bpy.context.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')

  me = obj.data

  # Create a new bmesh from the mesh data
  bm = bmesh.new()
  bm.from_mesh(me)

  # Triangulate the faces
  bmesh.ops.triangulate(
      bm,
      faces=bm.faces[:],  # Operate on all faces
      quad_method='BEAUTY',
      ngon_method='BEAUTY'
  )

  bm.faces.ensure_lookup_table()

  tris = [[l.vert.index for l in f.loops] for f in bm.faces]

  return tris


def get_source_faces(context):
  wrap_options = context.scene.wrap_options
  mesh_name = wrap_options.source_mesh_name

  obj = bpy.data.objects.get(mesh_name)
  return get_mesh_faces(obj)


def get_source_triangles(context):
  wrap_options = context.scene.wrap_options
  mesh_name = wrap_options.source_mesh_name

  obj = bpy.data.objects.get(mesh_name)
  return get_mesh_triangles(obj)


def get_target_faces(context):
  wrap_options = context.scene.wrap_options
  mesh_name = wrap_options.target_mesh_name

  obj = bpy.data.objects.get(mesh_name)
  return get_mesh_faces(obj)


def get_target_triangles(context):
  wrap_options = context.scene.wrap_options
  mesh_name = wrap_options.target_mesh_name

  obj = bpy.data.objects.get(mesh_name)
  return get_mesh_triangles(obj)


############################################################
# Get vertices of face

def get_face_vertices(obj, face_index):
  """
  Get the vertices of a face on the BASE mesh.

  PATCHED: always returns a list (never None). The face index coming
  from obj.ray_cast() refers to the EVALUATED mesh (with modifiers
  applied), which can be out of range for the base mesh. In that case
  an empty list is returned and the caller must fall back to a global
  nearest-vertex search.
  """
  if not obj or obj.type != 'MESH':
    return []

  if face_index is None or face_index < 0:
    return []

  # Use bmesh for direct mesh data access
  bm = bmesh.new()
  bm.from_mesh(obj.data)

  # Ensure lookup tables are updated to access elements by index
  bm.faces.ensure_lookup_table()
  bm.verts.ensure_lookup_table()

  matrix = obj.matrix_world
  vertices = []

  if face_index < len(bm.faces):
    target_face = bm.faces[face_index]
    vertices = [
        (loop.vert.index, loop.vert.co.copy(), matrix @ loop.vert.co)
        for loop in target_face.loops]

  bm.free()
  return vertices


def find_nearest_vertex(location, vertices):
  """
  Find nearest vertice

  PATCHED: safely returns None when the vertex list is empty or None
  instead of raising "'NoneType' object is not iterable".
  """
  if not vertices:
    return None

  distances = [(i, l, w, (location - w).length_squared)
               for i, l, w in vertices]
  nearest = min(distances, key=itemgetter(3))
  return nearest


############################################################
# Global nearest vertex (KD-tree fallback)
# PATCHED: new. Used when the ray-cast face index does not exist on the
# base mesh (e.g. Mirror/Subsurf/other modifiers change the topology).

_kd_cache = {}


def _get_kd_tree(obj):
  """
  Build (and cache) a KD-tree of the object's BASE mesh vertices in
  world space. The cache is keyed by object name and invalidated when
  the vertex count or the world matrix changes.
  """
  mesh = obj.data
  key = obj.name
  stamp = (len(mesh.vertices), tuple(round(v, 6) for row in obj.matrix_world for v in row))

  cached = _kd_cache.get(key)
  if cached is not None and cached[0] == stamp:
    return cached[1]

  size = len(mesh.vertices)
  kd = kdtree.KDTree(size)
  matrix = obj.matrix_world
  for i, v in enumerate(mesh.vertices):
    kd.insert(matrix @ v.co, i)
  kd.balance()

  _kd_cache[key] = (stamp, kd)
  return kd


def find_nearest_vertex_global(obj, location):
  """
  Find the nearest BASE-mesh vertex to a world-space location using a
  KD-tree over all vertices. Returns (index, co_local, co_world) or
  None if the object has no vertices.
  """
  if not obj or obj.type != 'MESH' or len(obj.data.vertices) == 0:
    return None

  kd = _get_kd_tree(obj)
  co_world, index, _dist = kd.find(location)

  if index is None:
    return None

  co_local = obj.data.vertices[index].co.copy()
  return (index, co_local, co_world)


############################################################
# Script

if __name__ == '__main__':
  obj = bpy.data.objects.get('Cube')
  verts = get_face_vertices(obj, 0)
  print(f'{verts}')