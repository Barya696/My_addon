# Ring slice
#
# A ring marker is the loop where a plane cuts a mesh around a center point.
# Pure numpy: the caller passes world-space vertices and triangles.
#
#   slice_ring()   the loop around the center, oriented counter-clockwise
#                  about the plane normal
#   Loop.point_at  a point at a fraction of the loop length from the seam
#   Loop.fraction  where a point lies along the loop, from the seam

import numpy as np


############################################################
# Plane basis

def plane_basis(normal):
  """
  Two unit vectors u, v with u x v = normal
  """
  n = normal / np.linalg.norm(normal)
  helper = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
  u = np.cross(helper, n)
  u /= np.linalg.norm(u)
  v = np.cross(n, u)
  return u, v


############################################################
# Loop

class Loop:
  """
  Points where the plane crosses mesh edges, in order around the ring.
  a, b, t: the crossed edge (vertex indices) and the crossing position on it
  """

  def __init__(self, a, b, t, pts, closed):
    self.a, self.b, self.t = a, b, t
    self.pts = pts
    self.closed = closed
    seg = np.diff(np.vstack([pts, pts[:1]]) if closed else pts, axis=0)
    self.lengths = np.linalg.norm(seg, axis=1)
    self.cum = np.concatenate([[0.0], np.cumsum(self.lengths)])
    self.total = max(float(self.cum[-1]), 1e-12)
    self.seam = 0.0  # fraction of the loop length where the ring starts

  def centroid(self):
    return self.pts.mean(axis=0)

  def radius(self):
    return float(np.linalg.norm(self.pts - self.centroid(), axis=1).mean())

  def _segments(self):
    p0 = self.pts
    p1 = np.roll(self.pts, -1, axis=0) if self.closed else self.pts[1:]
    return p0[:len(p1)], p1

  def fraction(self, points):
    """
    Absolute fraction (0..1 from the first point) of the nearest loop point
    to each of `points` (k, 3)
    """
    p0, p1 = self._segments()
    d = p1 - p0
    dd = np.maximum((d * d).sum(axis=1), 1e-30)
    q = np.atleast_2d(points)
    s = np.clip(((q[:, None, :] - p0[None]) * d[None]).sum(axis=2) / dd[None], 0.0, 1.0)
    near = p0[None] + s[:, :, None] * d[None]
    dist = np.linalg.norm(near - q[:, None, :], axis=2)
    i = np.argmin(dist, axis=1)
    return (self.cum[i] + s[np.arange(len(q)), i] * self.lengths[i]) / self.total

  def point_at(self, f):
    """
    Point at fraction f of the loop length, counted from the seam
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    x = (self.seam + f) % 1.0 if self.closed else np.clip(self.seam + f, 0.0, 1.0)
    s = x * self.total
    i = np.clip(np.searchsorted(self.cum, s, side='right') - 1, 0, len(self.lengths) - 1)
    w = (s - self.cum[i]) / np.maximum(self.lengths[i], 1e-30)
    p0, p1 = self._segments()
    return p0[i] + np.clip(w, 0.0, 1.0)[:, None] * (p1[i] - p0[i])

  def relative(self, absolute):
    """
    Absolute fractions -> fractions counted from the seam
    """
    if self.closed:
      return (absolute - self.seam) % 1.0
    return absolute - self.seam


############################################################
# Slice

def slice_loops(V, tris, center, normal):
  """
  Every loop where the plane through `center` with `normal` cuts the mesh
  """
  n = normal / np.linalg.norm(normal)
  d = (V - center) @ n
  side = d >= 0.0
  ts = side[tris]
  count = ts.sum(axis=1)
  cut = (count == 1) | (count == 2)
  T, S = tris[cut], ts[cut]
  if not len(T):
    return []

  # The two crossed edges of each cut triangle
  edges = []
  for i, j in ((0, 1), (1, 2), (2, 0)):
    edges.append((T[:, i], T[:, j], S[:, i] != S[:, j]))
  first = np.where(edges[0][2], 0, 1)
  second = np.where(edges[2][2], 2, 1)
  ea = np.stack([e[0] for e in edges], axis=1)
  eb = np.stack([e[1] for e in edges], axis=1)
  rows = np.arange(len(T))
  e1 = (ea[rows, first], eb[rows, first])
  e2 = (ea[rows, second], eb[rows, second])

  nv = len(V)

  def key(a, b):
    return np.minimum(a, b).astype(np.int64) * nv + np.maximum(a, b)

  k1, k2 = key(*e1), key(*e2)

  # Walk the segments into chains
  touching = {}
  for s, (x, y) in enumerate(zip(k1.tolist(), k2.tolist())):
    touching.setdefault(x, []).append(s)
    touching.setdefault(y, []).append(s)

  ends = (k1.tolist(), k2.tolist())
  used = np.zeros(len(T), dtype=bool)

  def walk(s, start_key):
    keys = []
    k = start_key
    while True:
      used[s] = True
      k = ends[1][s] if ends[0][s] == k else ends[0][s]
      keys.append(k)
      nxt = [x for x in touching[k] if not used[x]]
      if not nxt:
        return keys
      s = nxt[0]

  loops = []
  for s in range(len(T)):
    if used[s]:
      continue
    start = ends[0][s]
    forward = walk(s, start)
    closed = forward[-1] == start
    if closed:
      keys = [start] + forward[:-1]
    else:
      # Open chain (a hole in the mesh): walk back from the start as well
      back = [start]
      others = [x for x in touching[start] if not used[x]]
      if others:
        back += walk(others[0], start)
      keys = back[::-1] + forward
    if len(keys) < 3:
      continue
    keys = np.array(keys, dtype=np.int64)
    a, b = keys // nv, keys % nv
    t = d[a] / (d[a] - d[b])
    pts = V[a] + t[:, None] * (V[b] - V[a])
    loops.append(Loop(a, b, t, pts, closed))
  return loops


def _winding(pts2, c2):
  """
  Whether the 2D point c2 lies inside the polygon pts2
  """
  x, y = pts2[:, 0], pts2[:, 1]
  x1, y1 = np.roll(x, -1), np.roll(y, -1)
  crosses = ((y > c2[1]) != (y1 > c2[1]))
  xs = x + (c2[1] - y) * (x1 - x) / np.where(y1 - y == 0, 1e-30, y1 - y)
  return np.count_nonzero(crosses & (c2[0] < xs)) % 2 == 1


def slice_ring(V, tris, center, normal):
  """
  The loop around `center`: the smallest one that encloses it, else the
  nearest one. Oriented counter-clockwise about `normal`. None if the plane
  misses the mesh.
  """
  loops = slice_loops(V, tris, center, normal)
  if not loops:
    return None
  u, v = plane_basis(normal)

  def flat(p):
    q = p - center
    return np.stack([q @ u, q @ v], axis=-1)

  inside = [lp for lp in loops if _winding(flat(lp.pts), np.zeros(2))]
  if inside:
    loop = min(inside, key=lambda lp: lp.total)
  else:
    loop = min(loops, key=lambda lp: np.linalg.norm(lp.pts - center, axis=1).min())

  # Counter-clockwise about the normal (shoelace area)
  p2 = flat(loop.pts)
  area = (p2[:, 0] * np.roll(p2[:, 1], -1) - np.roll(p2[:, 0], -1) * p2[:, 1]).sum()
  if area < 0.0:
    loop = Loop(loop.a[::-1], loop.b[::-1], loop.t[::-1], loop.pts[::-1], loop.closed)
  return loop
