# Wrap solver
#
# Implements the wrapping math for the four requests the Blender add-on sends:
#   - wrap_init_request    : build the model + target acceleration
#   - wrap_snap_request     : point pairs pull, the shape follows (rough alignment)
#   - wrap_attract_request  : ICP-like surface attraction with dist/angle gates
#   - wrap_stick_request    : the surface sticks onto the target
#
# Snap and attract are Embedded Deformation on a tet cage built at init
# (ed_solver.py), as the original C++ wrapDaemon does. Stick is
# As-Rigid-As-Possible surface modelling on the source's own triangles
# (stick_solver.py): the point pairs and fixed vertices held, nearest-surface
# pulls within the stick distance and angle, rebuilt every step. The rules
# were measured on the original C++ wrapDaemon's recorded runs.

import hashlib
import json

import numpy as np

import ed_solver as ed
import stick_solver as ss
from scipy.spatial import cKDTree

try:
  import pypardiso   # Intel MKL PARDISO: ~6x faster than SuperLU on these systems
  pypardiso.PyPardisoSolver()   # loads MKL now: if it is missing, use SuperLU
except Exception:
  pypardiso = None
LINEAR_SOLVER = "PARDISO" if pypardiso is not None else "SuperLU"

EPS = 1.0e-12

# Stick (surface ARAP, stick_solver.py): the choices settled by the recorded
# C++ runs (eve, Stick alone from the C++ Attract result: cotangent weights
# give its 527 badly stretched edges, uniform 831; Init's rest shape drifts
# 4.5-5 mm from it, Stick's start 0.1 mm; nearest points on triangles land on
# eve as it does, nearest vertices stay 0.17 mm off; point-to-point onto
# vertices is 14-17 mm off on the probes)
STICK_WEIGHTING = "cotangent"    # edge weights: "uniform" or "cotangent"
STICK_REST = "start"             # rest shape: Init's ("init") or Stick's start ("start")
STICK_MATCH = "triangle"         # nearest target "vertex" or nearest point on a "triangle"
STICK_POINT_TO_PLANE = True      # surface rows: point-to-plane (True) or point-to-point
STICK_RAMP = None                # None: flat surface weight; 128: Attract-like ramp
STICK_INNER = 1                  # local/global alternations per streamed step
STICK_TOL = 1.0e-3               # stop once |E_prev - E| < STICK_TOL * E
STICK_MIN_STEPS = 1

# Bump on every solver change - printed at daemon startup and per request so
# it is immediately obvious which solver build a running daemon executes.
WRAP_SOLVER_VERSION = "33-mirrored-cage"


############################################################
# Geometry helpers

def triangulate(tris, quads):
  """Combine triangle list and quad list into a single (m, 3) int array."""
  faces = []
  for t in tris or []:
    if len(t) == 3:
      faces.append(t)
  for q in quads or []:
    if len(q) == 4:
      faces.append([q[0], q[1], q[2]])
      faces.append([q[0], q[2], q[3]])
    elif len(q) == 3:
      faces.append(q)
  if not faces:
    return np.zeros((0, 3), dtype=np.int64)
  return np.asarray(faces, dtype=np.int64)


def closed_triangulation(tris, quads):
  """One closed triangulation of a mesh sent as triangles and quads, for the
  cage's inside test (which counts surface crossings, so no face may be
  there twice). The add-on sends every face twice: the quads, and all their
  triangles; a plain client sends each face once, some as triangles and
  the rest as quads. So: the triangles alone when they close the surface,
  else both lists."""
  T = triangulate(tris, None)
  if len(T):
    E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    _, count = np.unique(E, axis=0, return_counts=True)
    if np.all(count == 2):
      return T
  return triangulate(tris, quads)


def stick_triangulation(tris, quads):
  """The triangles Stick's surface ARAP works on: the client's own
  triangulation, each face once. The add-on sends every face twice (the
  quads, and all faces as its BEAUTY triangles), so when the triangles cover
  every quad edge they are the whole surface and the quads are left out (on
  an open mesh the triangles do not close the surface, which is why
  closed_triangulation cannot decide this). A plain client sends each face
  once, some as triangles and the rest as quads: both lists. Repeated
  triangles are dropped, keeping the first."""
  T = triangulate(tris, None)
  if len(T):
    _, first = np.unique(np.sort(T, axis=1), axis=0, return_index=True)
    T = T[np.sort(first)]
  Q = [q for q in quads or [] if len(q) == 4]
  if not len(T):
    return triangulate(None, quads)
  if not Q:
    return T
  te = set(map(tuple, np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1).tolist()))
  Qa = np.asarray(Q, dtype=np.int64)
  qe = np.sort(np.concatenate([Qa[:, [k, (k + 1) % 4]] for k in range(4)]), axis=1)
  if all(tuple(e) in te for e in qe.tolist()):
    return T
  return np.concatenate([T, triangulate(None, quads)])


def vertex_normals(V, F):
  """Area-weighted vertex normals for vertices V (n,3) and triangles F (m,3)."""
  N = np.zeros_like(V)
  if len(F) == 0:
    N[:, 2] = 1.0
    return N

  e1 = V[F[:, 1]] - V[F[:, 0]]
  e2 = V[F[:, 2]] - V[F[:, 0]]
  fn = np.cross(e1, e2)  # area-weighted face normals

  np.add.at(N, F[:, 0], fn)
  np.add.at(N, F[:, 1], fn)
  np.add.at(N, F[:, 2], fn)

  norms = np.linalg.norm(N, axis=1)
  norms[norms < EPS] = 1.0
  return N / norms[:, None]


############################################################
# Closest point on target surface

class TargetSurface:
  """KD-tree accelerated closest-point-on-surface queries."""

  def __init__(self, V, F):
    self.V = V
    self.F = F
    if len(F) == 0:
      raise ValueError("Target mesh has no faces")

    self.tri_a = V[F[:, 0]]
    self.tri_b = V[F[:, 1]]
    self.tri_c = V[F[:, 2]]

    fn = np.cross(self.tri_b - self.tri_a, self.tri_c - self.tri_a)
    norms = np.linalg.norm(fn, axis=1)
    norms[norms < EPS] = 1.0
    self.face_normals = fn / norms[:, None]

    centroids = (self.tri_a + self.tri_b + self.tri_c) / 3.0
    self.tree = cKDTree(centroids)
    self.k = min(16, len(F))

    # Each triangle's reach from its centroid: a point at centroid distance d
    # is at least d - radius from the triangle (lets searches skip triangles)
    self.tri_radius = np.sqrt(np.max(np.stack(
        [((t - centroids) ** 2).sum(axis=1) for t in (self.tri_a, self.tri_b, self.tri_c)]), axis=0))
    self.max_radius = float(self.tri_radius.max())

  def closest_points(self, P):
    """For points P (n,3) return (closest (n,3), dist (n,), normal (n,3))."""
    _, idx = self.tree.query(P, k=self.k)
    if self.k == 1:
      idx = idx[:, None]

    n = len(P)
    best_pts = np.zeros((n, 3))
    best_d2 = np.full(n, np.inf)
    best_tri = np.zeros(n, dtype=np.int64)

    for j in range(idx.shape[1]):
      tri = idx[:, j]
      pts = _closest_point_on_triangles(
          P, self.tri_a[tri], self.tri_b[tri], self.tri_c[tri])
      d2 = np.einsum('ij,ij->i', P - pts, P - pts)
      better = d2 < best_d2
      best_pts[better] = pts[better]
      best_d2[better] = d2[better]
      best_tri[better] = tri[better]

    return best_pts, np.sqrt(best_d2), self.face_normals[best_tri]

def _closest_point_on_triangles(P, A, B, C):
  """Vectorized exact closest point on triangle (Ericson, Real-Time
  Collision Detection). All inputs are (n, 3)."""
  ab = B - A
  ac = C - A
  ap = P - A

  d1 = np.einsum('ij,ij->i', ab, ap)
  d2 = np.einsum('ij,ij->i', ac, ap)

  bp = P - B
  d3 = np.einsum('ij,ij->i', ab, bp)
  d4 = np.einsum('ij,ij->i', ac, bp)

  cp = P - C
  d5 = np.einsum('ij,ij->i', ab, cp)
  d6 = np.einsum('ij,ij->i', ac, cp)

  result = np.zeros_like(P)
  done = np.zeros(len(P), dtype=bool)

  # Vertex A region
  m = (d1 <= 0) & (d2 <= 0) & ~done
  result[m] = A[m]
  done |= m

  # Vertex B region
  m = (d3 >= 0) & (d4 <= d3) & ~done
  result[m] = B[m]
  done |= m

  # Edge AB region
  vc = d1 * d4 - d3 * d2
  m = (vc <= 0) & (d1 >= 0) & (d3 <= 0) & ~done
  denom = d1 - d3
  denom = np.where(np.abs(denom) < EPS, 1.0, denom)
  v = d1 / denom
  result[m] = A[m] + v[m, None] * ab[m]
  done |= m

  # Vertex C region
  m = (d6 >= 0) & (d5 <= d6) & ~done
  result[m] = C[m]
  done |= m

  # Edge AC region
  vb = d5 * d2 - d1 * d6
  m = (vb <= 0) & (d2 >= 0) & (d6 <= 0) & ~done
  denom = d2 - d6
  denom = np.where(np.abs(denom) < EPS, 1.0, denom)
  w = d2 / denom
  result[m] = A[m] + w[m, None] * ac[m]
  done |= m

  # Edge BC region
  va = d3 * d6 - d5 * d4
  m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0) & ~done
  denom = (d4 - d3) + (d5 - d6)
  denom = np.where(np.abs(denom) < EPS, 1.0, denom)
  w = (d4 - d3) / denom
  result[m] = B[m] + w[m, None] * (C[m] - B[m])
  done |= m

  # Interior
  m = ~done
  denom = va + vb + vc
  denom = np.where(np.abs(denom) < EPS, 1.0, denom)
  v = vb / denom
  w = vc / denom
  result[m] = A[m] + v[m, None] * ab[m] + w[m, None] * ac[m]

  return result


############################################################
# Session

class WrapSession:
  """Per-connection wrap state. Built by init, used by snap/attract/stick."""

  def __init__(self):
    self.initialized = False
    self.init_message = ''

  ############################################################
  # Init

  def init(self, req):
    V0 = np.asarray(req['source_coords'], dtype=np.float64)
    if V0.ndim != 2 or V0.shape[1] != 3 or len(V0) == 0:
      raise ValueError("source_coords must be a non-empty Nx3 list")

    F = triangulate(req.get('source_tris'), req.get('source_quads'))
    Vt = np.asarray(req['target_coords'], dtype=np.float64)
    Ft = triangulate(req.get('target_tris'), req.get('target_quads'))

    if len(Vt) == 0 or len(Ft) == 0:
      raise ValueError("target mesh is empty")

    n = len(V0)

    fixed = np.asarray(req.get('source_fixed') or [], dtype=np.int64)
    free = np.asarray(req.get('source_free') or [], dtype=np.int64)
    fixed = fixed[(fixed >= 0) & (fixed < n)]
    free = free[(free >= 0) & (free < n)]

    alpha = max(float(req.get('alpha', 1.0)), 1.0e-4)
    beta = max(float(req.get('beta', 0.1)), 1.0e-4)
    symmetric = bool(int(req.get('symmetric', 0)))
    cage_alpha, cage_beta = alpha, beta   # the cage takes them as sent

    # --- Parameter normalization -------------------------------------
    # The Blender add-on sends UI-scale values:
    #   alpha = flex            (UI range 50..200, default 100)
    #   beta  = flex * accuracy (UI range 50..1000, default 300)
    # Used raw, the damping term diag(beta) would drown every constraint
    # weight in the solver (300 vs ~70) and nothing would ever move.
    # Map them to O(1) internal stiffness/damping:
    #   more flex     -> softer mesh  (lower alpha)
    #   more accuracy -> more damping (higher beta, careful steps)
    if alpha > 10.0:  # values are UI-scale, not already normalized
      flex = alpha                       # 50 .. 200
      accuracy = max(beta / alpha, 0.1)  # 1 .. 5
      alpha = 100.0 / flex               # 2.0 (stiff) .. 0.5 (flexible)
      beta = 0.3 * accuracy              # 0.3 .. 1.5
    # ------------------------------------------------------------------

    # Content hash for the init-data cache
    h = hashlib.sha256()
    h.update(V0.tobytes())
    h.update(F.tobytes())
    h.update(Vt.tobytes())
    h.update(Ft.tobytes())
    h.update(np.float64([alpha, beta, float(symmetric)]).tobytes())
    h.update(ed.CAGE_VERSION.encode())
    content_hash = h.hexdigest()

    cached = req.get('data') or ''
    cache_valid = False
    cache_dict = {}
    if cached:
      try:
        cache_dict = json.loads(cached)
        cache_valid = cache_dict.get('hash') == content_hash
      except (json.JSONDecodeError, AttributeError):
        cache_valid = False

    if cache_valid and self.initialized and self.content_hash == content_hash:
      # Session already holds this exact model - nothing to rebuild
      return self.init_message

    self.V0 = V0
    self.F = F
    self.n = n
    self.fixed = fixed
    self.free_set = set(free.tolist())
    self.alpha = alpha
    self.beta = beta
    self.symmetric = symmetric
    self.content_hash = content_hash

    # Target acceleration structure
    self.target = TargetSurface(Vt, Ft)

    # Attract: the target's vertices in a kd-tree, and their unit normals as
    # the add-on sends them
    self.target_tree = cKDTree(Vt)
    Nt = np.asarray(req.get('target_normals') or [], dtype=np.float64)
    if Nt.shape != Vt.shape:
      Nt = vertex_normals(Vt, Ft)
    self.target_normals = Nt / np.maximum(np.linalg.norm(Nt, axis=1, keepdims=True), EPS)

    # Scene scale (bounding box diagonal of source)
    bb = V0.max(axis=0) - V0.min(axis=0)
    self.diag = max(float(np.linalg.norm(bb)), EPS)

    # Current positions (init always restarts from the original shape)
    self.x = V0.copy()

    # The latest snap point pairs and weights, which attract keeps
    self.pins = (np.zeros(0, dtype=np.int64), np.zeros((0, 3)))
    self.snap_force, self.snap_softness = 1.0, 1.0

    # Stick's surface ARAP: the client's triangulation of the source (each
    # face once) and what depends on its connectivity alone (edges, the
    # system's sparsity pattern and its PARDISO analysis), here once per
    # init; the solver for a rest shape and the target matcher are built on
    # first use
    self.surface_tris = stick_triangulation(req.get('source_tris'), req.get('source_quads'))
    self.stick_topology = ss.SurfaceTopology(self.surface_tris, len(V0))
    self.stick_arap = None
    self.stick_matcher = None

    # Snap's Embedded Deformation cage: from the add-on's cached init data
    # when it is for this exact model, else built here (nodes, edges and the
    # vertices' tet bindings; the rest positions come with every request)
    ed_data = None
    if cache_valid and cache_dict.get('ed'):
      try:
        ed_data = ed.deserialize_init(cache_dict['ed'])
      except Exception:
        ed_data = None
    if ed_data is None:
      cage_faces = closed_triangulation(req.get('source_tris'), req.get('source_quads'))
      ed_data = ed.init(V0, cage_faces, cage_alpha, cage_beta)
      del ed_data['rest']
    ed_data['rest'] = V0
    self.ed = ed.make_solver(ed_data)
    cached_ed = {k: v for k, v in ed_data.items() if k != 'rest'}
    print(f"[solver {WRAP_SOLVER_VERSION}] init: cage {self.ed.M} nodes, "
          f"{len(self.ed.edges) // 2} edges, {12 * self.ed.M} unknowns", flush=True)

    self.init_message = json.dumps({
        "hash": content_hash,
        "n": n,
        "version": int(req.get('version', 1)),
        "ed": ed.serialize_init(cached_ed),
    })
    self.initialized = True

    return self.init_message

  ############################################################
  # Snap
  #
  # Embedded Deformation (Sumner, Schmid, Pauly 2007) on the tet cage built
  # at init, see ed_solver.py: each cage node carries an affine and a
  # translation, all solved together by Gauss-Newton, with
  #   w_pair = w_fixed = 10^force / d^2,  w_reg = 10^(force - softness) / d^2
  # (d the source's bounding-box diagonal). As the original C++: a frame is
  # streamed after every step, a rejected one included (its positions,
  # before it is undone); it stops once an accepted step changes the energy
  # by less than 0.1% or after iter_num steps. Unlike it, a rejected step is
  # damped and retried rather than repeated unchanged. Pairs whose target is
  # exactly (0, 0, 0) or whose vertex is fixed are skipped. The solver keeps
  # its state between snaps: a new snap continues from the last one.

  def snap_iterations(self, req):
    """Generator: yields vertex positions after each iteration."""
    self._require_init()

    idx = np.asarray(req.get('pp_source_index') or [], dtype=np.int64)
    pos = np.asarray(req.get('pp_target_pos') or [], dtype=np.float64)
    force = float(req.get('force', 1.0))
    softness = float(req.get('softness', 1.0))
    iter_num = max(int(req.get('iter_num', 20)), 1)

    if len(idx) == 0 or len(idx) != len(pos):
      # Nothing to snap - still stream the current state once
      self.pins = (np.zeros(0, dtype=np.int64), np.zeros((0, 3)))
      yield self.x.copy()
      return

    valid = (idx >= 0) & (idx < self.n)
    idx, pos = idx[valid], pos[valid]
    self.pins = (idx.copy(), pos.copy())   # attract holds them too
    self.snap_force, self.snap_softness = force, softness

    w_con, w_reg = ed.weights(force, softness, ed.bbox_diag(self.ed.v0))
    terms = ed.constraint_terms(self.ed, idx, pos, self.fixed, w_con)
    gap = np.linalg.norm(pos - self.x[idx], axis=1)
    print(f"[solver {WRAP_SOLVER_VERSION}] snap: {len(idx)} pins, "
          f"pin gap mean {gap.mean():.4f} max {gap.max():.4f}, "
          f"force {force} softness {softness} iters {iter_num}, "
          f"w_pair {w_con:.4g} w_reg {w_reg:.4g}", flush=True)
    if not terms:
      # Every pair skipped and nothing fixed: nothing to solve for
      yield self.x.copy()
      return

    for it, v, E, accepted in self.ed.iterate(terms, w_reg, iter_num):
      print(f"[solver {WRAP_SOLVER_VERSION}] snap iteration {it + 1}: "
            f"E {E:.6g} {'accepted' if accepted else 'rejected'}", flush=True)
      yield v.copy()
    # A rejected last step is streamed but undone: carry on from the kept state
    self.x = self.ed.deform()

  ############################################################
  # Attract
  #
  # Snap's Embedded Deformation solver, continuing from where snap left it
  # (no reset), with snap's pairs and weights, plus the original C++'s
  # nearest-point constraints (ed.Attraction): each vertex (fixed and free
  # ones excepted) takes the nearest target vertex to where it is now, kept
  # only within attract_distance and when the vertex normals agree within
  # attract_angle, and is pulled along that target vertex's normal
  # (point-to-plane). That weight starts at w_reg / 128 and doubles after
  # every accepted step up to w_reg, and the matches are rebuilt after every
  # accepted step. Steps and stops as snap does.

  def attract_iterations(self, req):
    """Generator: yields vertex positions after each iteration."""
    self._require_init()

    reach = max(float(req.get('attract_distance', 0.05 * self.diag)), 1.0e-6 * self.diag)
    angle = float(req.get('attract_angle', 65.0))
    iter_num = max(int(req.get('iter_num', 20)), 1)

    w_con, w_reg = ed.weights(self.snap_force, self.snap_softness, ed.bbox_diag(self.ed.v0))
    pin_idx, pin_pos = self.pins
    base = ed.constraint_terms(self.ed, pin_idx, pin_pos, self.fixed, w_con)
    attraction = ed.Attraction(self.ed, base, self.target_tree, self.target.V,
                               self.target_normals, reach, angle, w_reg,
                               np.nonzero(self._movable_mask())[0],
                               lambda x: vertex_normals(x, self.F))
    print(f"[solver {WRAP_SOLVER_VERSION}] attract: reach {reach:.4f}, angle {angle}, "
          f"iters {iter_num}, force {self.snap_force} softness {self.snap_softness}, "
          f"{len(pin_idx)} pins", flush=True)

    for it, v, E, accepted in self.ed.iterate(None, w_reg, iter_num, retarget=attraction):
      print(f"[solver {WRAP_SOLVER_VERSION}] attract iteration {it + 1}: "
            f"E {E:.6g} {'accepted' if accepted else 'rejected'}, "
            f"{attraction.matched} matched, weight {attraction.w:.4g}", flush=True)
      yield v.copy()
    # A rejected last step is streamed but undone: carry on from the kept state
    self.x = self.ed.deform()

  ############################################################
  # Stick
  #
  # As-Rigid-As-Possible surface modelling (Sorkine & Alexa 2007) on the
  # source surface itself, as the original C++ does (stick_solver.py): a
  # rotation per vertex (local step) and the vertex positions (one sparse
  # SPD solve per step), with the request's point pairs and the fixed
  # vertices held at w_hard = 10^force / d^2 and nearest-surface pulls within
  # stick_distance and stick_angle at w_soft = 10^(force - softness) / d^2,
  # rebuilt every step (force and softness of the last snap). No step
  # rejection, no damping; it stops once |E_prev - E| < STICK_TOL * E or
  # after iter_num steps. The request's pairs are used for this stick only.

  def stick_iterations(self, req):
    """Generator: yields vertex positions after each iteration."""
    self._require_init()

    distance = max(float(req.get('stick_distance', 0.05 * self.diag)), 0.0)
    angle = float(req.get('stick_angle', 65.0))
    iter_num = max(int(req.get('iter_num', 10)), 1)
    pairs = req.get('pp_source_index') or []
    targets = req.get('pp_target_pos') or []

    if self.stick_arap is None or STICK_REST == "start":
      rest = self.V0 if STICK_REST == "init" else self.x
      self.stick_arap = ss.ARAPSolver(rest, self.stick_topology, STICK_WEIGHTING)
    if self.stick_matcher is None:
      self.stick_matcher = ss.SurfaceMatcher(self.target.V, self.target_normals,
                                             self.target, self.target_tree)
    print(f"[solver {WRAP_SOLVER_VERSION}] stick: reach {distance:.4f}, angle {angle}, "
          f"iters {iter_num}, force {self.snap_force} softness {self.snap_softness}, "
          f"{len(pairs)} pairs", flush=True)

    p = self.x
    for step, p, E, matched in ss.iterate(
        self.stick_arap, self.x, pairs, targets, self.fixed, self.stick_matcher,
        self.snap_force, self.snap_softness, distance, angle, iter_num,
        point_to_plane=STICK_POINT_TO_PLANE, match=STICK_MATCH, ramp=STICK_RAMP,
        inner=STICK_INNER, tol=STICK_TOL, min_steps=STICK_MIN_STEPS):
      print(f"[solver {WRAP_SOLVER_VERSION}] stick iteration {step + 1}: E {E:.6g}, "
            f"matched {matched}, pairs {len(pairs)}", flush=True)
      yield p.copy()
    self.x = p

  ############################################################
  # Helpers

  def _movable_mask(self):
    """Boolean mask of vertices that surface constraints may act on."""
    m = np.ones(self.n, dtype=bool)
    if len(self.fixed) > 0:
      m[self.fixed] = False
    if self.free_set:
      m[list(self.free_set)] = False
    return m

  def _require_init(self):
    if not self.initialized:
      raise RuntimeError(
          "Wrap has not been initialized - send wrap_init_request first")
