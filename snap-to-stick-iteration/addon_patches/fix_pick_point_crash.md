# Fix: crash when picking a point to symmetrize

## Error

```
TypeError: 'NoneType' object is not iterable
  utils/mesh_helper.py, line 408, in find_nearest_vertex
```

## Root cause

`get_hit_pos()` in `operators/op_pick_point.py` uses `obj.ray_cast()`, which casts
against the **evaluated** geometry (with modifiers applied). But
`get_face_vertices()` in `utils/mesh_helper.py` builds a bmesh from the **base**
mesh (`obj.data`).

If the source mesh has any topology-changing modifier (Mirror, Subdivision
Surface, Solidify, Triangulate, ...), the `face_index` returned by the ray cast
can be out of range for the base mesh. `get_face_vertices()` then returns
`None`, and `find_nearest_vertex()` crashes iterating over it.

## Fix (2 files)

### 1. `utils/mesh_helper.py`

Replace the existing `get_face_vertices` and `find_nearest_vertex` functions
(near the end of the file, around line 380-415) with the following, and add the
new `find_nearest_vertex_global` function:

```python
############################################################
# Get vertices of face

def get_face_vertices(obj, face_index):
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
  """
  if not vertices:
    return None

  distances = [(i, l, w, (location - w).length_squared)
               for i, l, w in vertices]
  nearest = min(distances, key=itemgetter(3))
  return nearest


def find_nearest_vertex_global(obj, location):
  """
  Find the nearest base-mesh vertex to a world-space location.

  Fallback for when the ray-cast face index does not exist on the base
  mesh (i.e. the object has topology-changing modifiers). Uses a KD-tree
  cached per object so repeated mouse-move lookups stay fast.
  """
  from mathutils import kdtree

  if not obj or obj.type != 'MESH':
    return None, None

  mesh = obj.data
  size = len(mesh.vertices)
  if size == 0:
    return None, None

  cache = find_nearest_vertex_global.__dict__.setdefault('_cache', {})
  matrix = obj.matrix_world
  key = (obj.name, size, tuple(round(v, 6) for col in matrix for v in col))

  kd = cache.get(key)
  if kd is None:
    cache.clear()  # keep at most one tree cached
    kd = kdtree.KDTree(size)
    for i, v in enumerate(mesh.vertices):
      kd.insert(matrix @ v.co, i)
    kd.balance()
    cache[key] = kd

  co, index, _dist = kd.find(location)
  if index is None:
    return None, None

  return index, Vector(co)
```

### 2. `operators/op_pick_point.py`

**a)** Update the import block at the top of the file to also import the new
fallback helper:

```python
from ..utils.mesh_helper import (
    get_mesh_vertices,
    get_face_vertices,
    find_nearest_vertex,
    find_nearest_vertex_global
)
```

**b)** Replace the `get_nearest_vertex` function (around line 150) with:

```python
def get_nearest_vertex(obj, hit_world, face_index):
  """
  Get nearest vertex of hit position
  """
  face_vertices = get_face_vertices(obj, face_index)
  nearest_vertice = find_nearest_vertex(hit_world, face_vertices)

  if nearest_vertice is None:
    # Ray cast hit the evaluated (modifier) geometry and the face index
    # does not exist on the base mesh - fall back to a global search.
    return find_nearest_vertex_global(obj, hit_world)

  vert_index = nearest_vertice[0]
  vert_world = nearest_vertice[2]

  return vert_index, vert_world
```

## What this changes

- No more crash: every code path now returns a value instead of `None` being
  iterated.
- Picking still works with topology-changing modifiers: when the face index
  does not match the base mesh, the nearest **base-mesh** vertex to the hit
  point is used instead. That keeps the picked `source_pick_index` valid for
  the wrap export (which uses base-mesh vertex indices).
- If nothing can be resolved, `(None, None)` is returned and the modal
  operator simply shows no highlight — clicking does nothing instead of
  raising an exception.

## Where to apply

```
C:\Users\Barya\AppData\Roaming\Blender Foundation\Blender\5.1\extensions\user_default\teca3d_wrap_for_blender\
  ├── operators\op_pick_point.py   (import block + get_nearest_vertex)
  └── utils\mesh_helper.py         (get_face_vertices, find_nearest_vertex, + new find_nearest_vertex_global)
```

Restart Blender (or disable/enable the extension) after editing so the
modules reload.

## Note

For best wrap results you may still want to apply topology-changing modifiers
on the source mesh before wrapping — the wrap export sends base-mesh vertices,
so the daemon never sees modifier-generated geometry either way.
