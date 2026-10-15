# Mesh symmetry

import bpy
import numpy as np

from ..utils.mesh_helper import sync_basis_key


############################################################
# Topological symmetry

def get_mesh_polygons(obj):
  """
  Get all faces of mesh
  """
  if not obj or obj.type != 'MESH':
    return []

  # Ensure we are in Object Mode to access the mesh data directly
  # if bpy.context.mode != 'OBJECT':
  #   bpy.ops.object.mode_set(mode='OBJECT')

  # Get the list of all faces (polygons)
  faces = obj.data.polygons

  # Iterate through faces
  faces = [np.array(list(f.vertices), np.int32) for f in faces]

  return faces


def get_topo_symmetry(obj, first, second):
  """
  Get topological symmetry
  """
  faces = get_mesh_polygons(obj)
  if not faces:
    return [], []

  # Directional edge -> face
  edges_to_face = {(a, b): i
                   for i, f in enumerate(faces)
                   for a, b in zip(f, np.roll(f, -1))}

  if (first, second) not in edges_to_face or \
     (second, first) not in edges_to_face:
    return [], []

  # Face pairs
  f1_i = edges_to_face[(first, second)]
  f2_i = edges_to_face[(second, first)]

  v1_i = np.argmax(faces[f1_i] == first)
  v2_i = np.argmax(faces[f2_i] == first)

  face_syms = [(f1_i, v1_i, f2_i, v2_i)]

  face_map = {}
  vertex_map = {}

  # Process
  while face_syms:
    syms = []

    for f1, v1, f2, v2 in face_syms:
      if f1 in face_map:
        continue

      # Add into maps
      face_map |= {f1: f2, f2: f1}

      l1 = np.roll(faces[f1], -v1)
      l2 = np.roll(np.flip(faces[f2]), v2 + 1)

      vertex_map |= {a: b for a, b in zip(l1, l2)}
      vertex_map |= {a: b for a, b in zip(l2, l1)}

      # Iterate through neighbors
      s1 = [(edges_to_face[(b, a)], a)
            for a, b in zip(l1, np.roll(l1, -1))]

      s2 = [(edges_to_face[(a, b)], a)
            for a, b in zip(l2, np.roll(l2, -1))]

      syms.extend([
          ((c1, np.argmax(faces[c1] == d1),
            c2, np.argmax(faces[c2] == d2)))
          for (c1, d1), (c2, d2) in zip(s1, s2)
          if c1 not in face_map
      ])

    face_syms = syms

  # Check map
  if len(face_map) != len(faces):
    print('An error occurred in the index mapping of the symmetric face')
    return [], []

  if len(vertex_map) != len(obj.data.vertices):
    print("An error occurred in the index mapping of symmetric vertices")
    return [], []

  # Convert map to indexing
  face_idxs = np.array(
      [face_map[i] for i in range(len(faces))], np.int32)

  vertex_idxs = np.array(
      [vertex_map[i] for i in range(len(vertex_map))], np.int32)

  return face_idxs, vertex_idxs


############################################################
# Symmetrize coordinates

def symmetrize_vertex_coords(vertices, vertex_idxs):
  """
  Make the vertex coordinates symmetric
  """
  verts_mir = vertices[vertex_idxs]
  verts_mir[:, 0] = -verts_mir[:, 0]

  return (vertices + verts_mir) * 0.5


def symmetrize_mesh_coords(obj, vertex_idxs):
  """
  Make the vertex coordinates symmetric
  """
  mesh = obj.data
  vertices = np.empty(len(mesh.vertices) * 3, dtype=np.float32)

  mesh.vertices.foreach_get('co', vertices)
  vertices = vertices.reshape((-1, 3))

  verts_sym = symmetrize_vertex_coords(vertices, vertex_idxs)

  mesh.vertices.foreach_set('co', verts_sym.ravel())
  sync_basis_key(mesh)
