# Ring markers
#
# A ring marker is a pair of rings, one on the source mesh and one on the
# target mesh, each the cross-section of its mesh by a plane (center, normal)
# with a seam where the ring starts. The user adjusts both by hand; at Snap
# and Stick every ring pair becomes point pairs: each source vertex next to
# the source ring is matched to the point at the same fraction of the target
# ring, counted from the seams.
#
# The source ring is cut from the source's original shape and drawn on its
# current shape through the crossed edges, so it follows the wrap.

import json

import bpy
import numpy as np
from mathutils import Vector

from . import vertex_store
from .ring_slice import slice_ring


SIDES = ('source', 'target')

# Edit state: set by the ring edit operator, read by the drawing
state = {
    'editing': False,
    'hover': None,   # (ring uid, side, handle name)
}


############################################################
# Geometry

_geometry = {}


def _mesh_object(context, side):
  wrap_options = context.scene.wrap_options
  name = wrap_options.source_mesh_name if side == 'source' else wrap_options.target_mesh_name
  obj = bpy.data.objects.get(name)
  return obj if obj and obj.type == 'MESH' else None


def geometry(context, side):
  """
  World-space vertices and triangles the rings of `side` are cut from:
  the source's original shape, the target as shown. Cached.
  """
  obj = _mesh_object(context, side)
  if not obj:
    return None

  if side == 'source':
    wrap_options = context.scene.wrap_options
    stored = vertex_store.count(wrap_options, 'original')
    me = obj.data
    key = (obj.name, len(me.vertices), len(me.polygons), stored)
    cached = _geometry.get(side)
    if cached and cached[0] == key:
      return cached[1]
    if stored == len(me.vertices):
      V = vertex_store.get(wrap_options, 'original').astype(np.float64)
    else:
      V = np.empty(len(me.vertices) * 3)
      me.vertices.foreach_get('co', V)
      M = np.array(obj.matrix_world)
      V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    me.calc_loop_triangles()
    tris = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
    me.loop_triangles.foreach_get('vertices', tris)
  else:
    M = np.array(obj.matrix_world)
    key = (obj.name, tuple(M.ravel().round(6)))
    cached = _geometry.get(side)
    if cached and cached[0] == key:
      return cached[1]
    dg = context.evaluated_depsgraph_get()
    eo = obj.evaluated_get(dg)
    me = eo.to_mesh()
    V = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', V)
    V = V.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]
    me.calc_loop_triangles()
    tris = np.empty(len(me.loop_triangles) * 3, dtype=np.int32)
    me.loop_triangles.foreach_get('vertices', tris)
    eo.to_mesh_clear()

  data = (V, tris.reshape(-1, 3))
  _geometry[side] = (key, data)
  _rings.clear()
  return data


def clear_geometry():
  _geometry.clear()
  _rings.clear()


############################################################
# Rings

_rings = {}


def side_params(marker, side):
  return (Vector(getattr(marker, f'{side}_center')),
          Vector(getattr(marker, f'{side}_normal')),
          Vector(getattr(marker, f'{side}_seam')))


def set_side_params(marker, side, center=None, normal=None, seam=None):
  if center is not None:
    setattr(marker, f'{side}_center', center)
  if normal is not None:
    setattr(marker, f'{side}_normal', normal.normalized())
  if seam is not None:
    setattr(marker, f'{side}_seam', seam)


def ring(context, marker, side):
  """
  The marker's loop on `side` (a ring_slice.Loop with its seam set), or None
  """
  data = geometry(context, side)
  if data is None:
    return None
  center, normal, seam = side_params(marker, side)
  if normal.length < 1e-9:
    return None
  key = (marker.uid, side, tuple(center), tuple(normal), tuple(seam))
  if key in _rings:
    return _rings[key]
  V, tris = data
  loop = slice_ring(V, tris, np.array(center), np.array(normal))
  if loop is not None:
    loop.seam = float(loop.fraction(np.array(seam))[0])
  if len(_rings) > 256:
    _rings.clear()
  _rings[key] = loop
  return loop


def ring_display_points(context, marker, side, loop):
  """
  World-space points to draw: the target loop as is, the source loop moved
  with the source mesh's current shape (same crossed edges)
  """
  if side == 'target':
    return loop.pts
  obj = _mesh_object(context, 'source')
  if not obj or len(obj.data.vertices) <= int(max(loop.a.max(), loop.b.max())):
    return loop.pts
  if context.scene.wrap_options.source_mesh_mode == 'ORIGINAL':
    return loop.pts
  verts = obj.data.vertices
  M = obj.matrix_world
  ids = np.unique(np.concatenate([loop.a, loop.b]))
  co = {i: np.array(M @ verts[i].co) for i in ids.tolist()}
  A = np.array([co[i] for i in loop.a.tolist()])
  B = np.array([co[i] for i in loop.b.tolist()])
  return A + loop.t[:, None] * (B - A)


def handles(marker, side, loop):
  """
  Where the drag handles are: the center, the tip of the axis, the seam
  """
  center, normal, _ = side_params(marker, side)
  length = max(loop.radius() * 1.5, 1e-4)
  return {'center': center,
          'axis': center + normal.normalized() * length,
          'seam': Vector(loop.point_at(0.0)[0])}


def editable(context, side):
  """
  Source rings are cut from the original shape: adjust them only there
  """
  return side == 'target' or       context.scene.wrap_options.source_mesh_mode == 'ORIGINAL'


############################################################
# New ring at the center of a view

def place_at_view_center(context, marker, side, region, rv3d):
  """
  Center inside the mesh under the middle of the view, plane across the
  view's up direction, seam on the side facing the viewer
  """
  from bpy_extras import view3d_utils

  obj = _mesh_object(context, side)
  if not obj:
    return False
  coord = (region.width / 2.0, region.height / 2.0)
  direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
  origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)

  M = obj.matrix_world
  M_inv = M.inverted()
  o, dvec = M_inv @ origin, (M_inv.to_3x3() @ direction).normalized()
  hit, loc, _, _ = obj.ray_cast(o, dvec)
  if hit:
    # Midpoint between where the view ray enters and leaves the mesh
    hit2, loc2, _, _ = obj.ray_cast(loc + dvec * 1e-4, dvec)
    center = M @ ((loc + loc2) / 2.0 if hit2 else loc)
  else:
    center = rv3d.view_location.copy()

  up = rv3d.view_rotation @ Vector((0.0, 1.0, 0.0))
  set_side_params(marker, side, center=center, normal=up, seam=center)

  loop = ring(context, marker, side)
  if loop is None:
    return False

  # Seam: the ring point nearest to the viewer; center: the ring's middle
  toward_viewer = -(rv3d.view_rotation @ Vector((0.0, 0.0, -1.0)))
  i = int(np.argmax(loop.pts @ np.array(toward_viewer)))
  set_side_params(marker, side, center=Vector(loop.centroid()),
                  seam=Vector(loop.pts[i]))
  return True


def settle(context, marker, side):
  """
  After a drag: the center goes to the middle of its ring and the seam onto
  the ring, so the handles stay where the ring is
  """
  loop = ring(context, marker, side)
  if loop is None:
    return
  seam = Vector(loop.point_at(0.0)[0])
  set_side_params(marker, side, center=Vector(loop.centroid()), seam=seam)


############################################################
# Ring pairs -> point pairs

def expand(context):
  """
  Point pairs (source vertex index, target position) of every ring marker
  """
  wrap_options = context.scene.wrap_options
  src = geometry(context, 'source')
  tgt_obj = _mesh_object(context, 'target')
  if src is None or tgt_obj is None:
    return [], []
  V = src[0]
  M = tgt_obj.matrix_world
  M_inv = M.inverted()

  index, positions = [], []
  for marker in wrap_options.ring_markers:
    if not marker.show:
      continue   # hidden in the ring list: left out of the wrap too
    a = ring(context, marker, 'source')
    b = ring(context, marker, 'target')
    if a is None or b is None:
      continue
    center, normal, _ = side_params(marker, 'source')
    n_s = np.array(normal.normalized())
    n_t = np.array(side_params(marker, 'target')[1].normalized())

    # The source vertices next to the ring: the nearer end of each crossed edge
    near = np.where(a.t < 0.5, a.a, a.b)
    verts = np.unique(near)
    f = a.relative(a.fraction(V[verts]))
    if not a.closed or not b.closed:
      f = np.clip(f, 0.0, 1.0)
    off = (V[verts] - np.array(center)) @ n_s

    # Same fraction on the target ring, same offset along its axis, then
    # onto the target surface
    goal = b.point_at(f) + off[:, None] * n_t
    for v, g in zip(verts.tolist(), goal):
      ok, loc, _, _ = tgt_obj.closest_point_on_mesh(M_inv @ Vector(g))
      positions.append(list(M @ loc) if ok else list(g))
      index.append(v)
  return index, positions


def update_ring_pairs(context):
  """
  Store the expanded point pairs where Snap and Stick read them
  """
  index, positions = expand(context)
  context.scene.wrap_options.ring_pairs_data = json.dumps(
      {'source_index': index, 'target_pos': positions})
  return len(index)


def ring_pairs(wrap_options):
  """
  The stored ring point pairs: (source indices, target positions)
  """
  try:
    data = json.loads(wrap_options.ring_pairs_data or '{}')
  except ValueError:
    return [], []
  return data.get('source_index', []), data.get('target_pos', [])


############################################################
# Save / load

def markers_to_data(wrap_options):
  markers = list(wrap_options.ring_markers)
  index = {m.uid: i for i, m in enumerate(markers)}
  out = []
  for m in markers:
    d = {side: [list(v) for v in (getattr(m, f'{side}_center'),
                                  getattr(m, f'{side}_normal'),
                                  getattr(m, f'{side}_seam'))]
         for side in SIDES}
    d.update(name=m.name, show=m.show, mirror=index.get(m.mirror_uid, -1))
    out.append(d)
  return out


def data_to_markers(context, data):
  wrap_options = context.scene.wrap_options
  wrap_options.ring_markers.clear()
  made = []
  for d in data:
    m = new_marker(wrap_options, d.get('name'))
    m.show = d.get('show', True)
    for side in SIDES:
      c, n, s = (Vector(x) for x in d[side])
      set_side_params(m, side, center=c, normal=n, seam=s)
    made.append(m)
  for m, d in zip(made, data):
    j = d.get('mirror', -1)
    if 0 <= j < len(made):
      m.mirror_uid = made[j].uid
  update_ring_pairs(context)


def new_marker(wrap_options, name=None):
  m = wrap_options.ring_markers.add()
  wrap_options.ring_next_uid += 1
  m.uid = wrap_options.ring_next_uid
  m.name = name or f"Ring {m.uid}"
  wrap_options.ring_index = len(wrap_options.ring_markers) - 1
  return m


def find_marker(wrap_options, uid):
  """(index, marker) of the ring with this uid, or (None, None)"""
  if uid > 0:
    for i, m in enumerate(wrap_options.ring_markers):
      if m.uid == uid:
        return i, m
  return None, None


def ring_visible(wrap_options, marker):
  """Drawn (and used in the wrap): shown in the ring list"""
  return marker.show


def active_marker(wrap_options):
  """The ring picked in the ring list, or None"""
  i = wrap_options.ring_index
  return wrap_options.ring_markers[i] if 0 <= i < len(wrap_options.ring_markers) else None


def remove_marker(context, index):
  """Delete one ring; its mirror partner stays, unlinked"""
  wrap_options = context.scene.wrap_options
  markers = wrap_options.ring_markers
  if not 0 <= index < len(markers):
    return
  _, partner = find_marker(wrap_options, markers[index].mirror_uid)
  if partner is not None:
    partner.mirror_uid = 0
  markers.remove(index)
  wrap_options.ring_index = min(index, len(markers) - 1)
  update_ring_pairs(context)


############################################################
# Mirror: a ring and its partner on the other side of the body, mirror
# images across each mesh's middle (X) plane. On the source (symmetric) the
# partner is exact and stays in step with every adjustment; on the target
# (never quite symmetric) the partner starts mirrored, settled onto the limb
# it lands on, and is adjusted on its own from then on.

def mirror_plane(context, side):
  """X of the side's mesh middle plane (its bounding box centre)"""
  data = geometry(context, side)
  if data is None:
    return 0.0
  V = data[0]
  return 0.5 * float(V[:, 0].min() + V[:, 0].max())


def _mirrored(p, x0, vector=False):
  return Vector((-p.x, p.y, p.z)) if vector else Vector((2.0 * x0 - p.x, p.y, p.z))


def mirror_into(context, marker, partner, side, settle_after=True):
  """Set partner's ring on `side` to the mirror image of marker's"""
  x0 = mirror_plane(context, side)
  center, normal, seam = side_params(marker, side)
  set_side_params(partner, side, center=_mirrored(center, x0),
                  normal=_mirrored(normal, x0, vector=True), seam=_mirrored(seam, x0))
  if settle_after:
    settle(context, partner, side)


def sync_mirror(context, marker, sides=SIDES, settle_after=True):
  """After marker changed: its partner follows, mirrored"""
  _, partner = find_marker(context.scene.wrap_options, marker.mirror_uid)
  if partner is None:
    return
  for side in sides:
    mirror_into(context, marker, partner, side, settle_after)


def _side_suffix(context, marker):
  """'.L' when the source ring is on the character's left (+X: it faces -Y)"""
  return '.L' if side_params(marker, 'source')[0].x >= mirror_plane(context, 'source') else '.R'


def _base_name(name):
  for suffix in ('.L', '.R', '_L', '_R', ' L', ' R'):
    if name.endswith(suffix):
      return name[:-len(suffix)]
  return name


_SIDE_WORDS = {'left': 'right', 'right': 'left', 'Left': 'Right', 'Right': 'Left', 'LEFT': 'RIGHT', 'RIGHT': 'LEFT'}


def _swap_side_words(name):
  """'left wrist' -> 'right wrist'; None if the name says no side"""
  import re
  swapped = re.sub(r'\b(left|right|Left|Right|LEFT|RIGHT)\b', lambda m: _SIDE_WORDS[m.group(1)], name)
  return swapped if swapped != name else None


def add_mirror(context, marker):
  """Make marker's partner on the other side: a name with left / right in it
  gets the other word, any other name the suffixes .L and .R"""
  wrap_options = context.scene.wrap_options
  other_name = _swap_side_words(marker.name)
  if other_name is None:
    base = _base_name(marker.name)
    mine = _side_suffix(context, marker)
    other_name = base + ('.R' if mine == '.L' else '.L')
    marker.name = base + mine
  partner = new_marker(wrap_options, other_name)
  partner.show = marker.show
  marker.mirror_uid = partner.uid
  partner.mirror_uid = marker.uid
  sync_mirror(context, marker)
  for side in SIDES:
    if ring(context, partner, side) is None:   # the mirror plane misses that mesh
      remove_marker(context, len(wrap_options.ring_markers) - 1)
      marker.mirror_uid = 0
      return None
  update_ring_pairs(context)
  return partner


def clear_markers(context):
  wrap_options = context.scene.wrap_options
  wrap_options.ring_markers.clear()
  wrap_options.ring_pairs_data = ''
  _rings.clear()
