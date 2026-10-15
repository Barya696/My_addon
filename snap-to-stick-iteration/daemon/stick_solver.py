# As-Rigid-As-Possible surface modelling (Sorkine & Alexa, SGP 2007) for the
# Stick stage. numpy + scipy only; same conventions as ed_solver.py.
#
# The unknowns are the surface vertex positions p' themselves (not cage
# nodes), with a rotation R_i per vertex:
#
#   E = sum_i sum_j in N(i) w_ij || R_i (p_i - p_j) - (p'_i - p'_j) ||^2
#     + sum over constraints of w_c || r_c ||^2
#
# p the rest shape, directed mesh edges both ways. Constraints: the point
# pairs and the fixed vertices (weight w_hard), and nearest-surface pulls
# within the stick distance and angle (weight w_soft), rebuilt every step.
# Each streamed step is a local step (R_i from an SVD of the edge
# covariance), the constraints rebuilt, and a global step (one sparse SPD
# solve); no step rejection, no damping. It stops when the energy changes
# by less than tol * E or after iter_num steps.
#
# The switches of ARAPSolver / iterate (edge weighting, rest shape, how the
# surface is matched, a weight ramp, inner alternations, stop details) are
# set in wrap_solver.py from the original C++'s recorded runs.

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from ed_solver import SPDFactor


############################################################
# Mesh helpers

def mesh_edges(tris):
  """Undirected unique edges (E,2) of triangles (F,3); duplicate faces collapse."""
  e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
  return np.unique(np.sort(e, 1), axis=0)


def cotangent_weights(verts, topology):
  """w_ij = (cot a + cot b) / 2 over the triangles sharing edge ij, kept
  positive (an obtuse pair would make the Laplacian indefinite)."""
  tris = topology.tris
  acc = np.zeros(topology.n_edges)
  for k in range(3):
    a, b, c = tris[:, k], tris[:, (k + 1) % 3], tris[:, (k + 2) % 3]
    u, v = verts[b] - verts[a], verts[c] - verts[a]
    cot = (u * v).sum(1) / np.maximum(np.linalg.norm(np.cross(u, v), axis=1), 1e-20)
    np.add.at(acc, topology.opposite[k], 0.5 * cot)
  return np.maximum(acc, 1e-6)


def vertex_normals(verts, tris):
  """Area-weighted unit vertex normals."""
  fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
  n = np.zeros_like(verts)
  for k in range(3):
    np.add.at(n, tris[:, k], fn)
  return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)


def bbox_diag(verts):
  return float(np.linalg.norm(verts.max(0) - verts.min(0)))


############################################################
# Solver

class SurfaceTopology:
  """What Stick needs from the source surface's connectivity alone, built
  once per Init: the edges (directed, both ways), the edge opposite each
  triangle corner (where that corner's cotangent goes), and the sparsity
  pattern of the 3N x 3N point-to-plane system: the Laplacian's
  off-diagonal entries plus every vertex's full 3x3 diagonal block, kept
  even where a block is zero (an unmatched vertex). The pattern is then
  the same at every step of every Stick, so the one PARDISO factor here,
  shared by the Sticks of this Init, runs its analysis once."""

  def __init__(self, tris, N):
    self.N = N
    self.tris = np.asarray(tris, np.int64)
    und = mesh_edges(self.tris)
    self.n_edges = len(und)
    self.i = np.concatenate([und[:, 0], und[:, 1]])
    self.j = np.concatenate([und[:, 1], und[:, 0]])
    ek = und[:, 0] * N + und[:, 1]
    self.opposite = []
    for k in range(3):
      b, c = self.tris[:, (k + 1) % 3], self.tris[:, (k + 2) % 3]
      self.opposite.append(np.searchsorted(ek, np.minimum(b, c) * N + np.maximum(b, c)))

    # 3N pattern: (3i+c, 3j+c) for each directed edge, then each vertex's
    # 3x3 block; sorted by row and column, as CSR keeps it. pos_off / pos_blk
    # are where those entries land in the matrix's data array.
    n3, c3, v = 3 * N, np.arange(3), np.arange(N)
    r_off = (3 * self.i[:, None] + c3).ravel()
    c_off = (3 * self.j[:, None] + c3).ravel()
    r_blk = (3 * v[:, None, None] + c3[None, :, None]).repeat(3, 2).ravel()
    c_blk = (3 * v[:, None, None] + c3[None, None, :]).repeat(3, 1).ravel()
    keys, pos = np.unique(np.concatenate([r_off * n3 + c_off, r_blk * n3 + c_blk]), return_inverse=True)
    self.pos_off, self.pos_blk = pos[:len(r_off)], pos[len(r_off):]
    self.pos_diag = self.pos_blk.reshape(N, 3, 3)[:, c3, c3]
    self.indices = (keys % n3).astype(np.int32)
    self.indptr = np.concatenate([[0], np.cumsum(np.bincount(keys // n3, minlength=n3))]).astype(np.int32)
    self.factor = SPDFactor()

  def matrix(self, data):
    """The 3N x 3N matrix with this pattern and these values (zeros kept)."""
    return sp.csr_matrix((data, self.indices, self.indptr), shape=(3 * self.N,) * 2)


class ARAPSolver:
  """Unknowns p' (N,3); the rest shape p is fixed at construction."""

  def __init__(self, rest, topology, weighting="uniform"):
    self.top = topology
    self.p0 = np.asarray(rest, np.float64)
    self.N = len(self.p0)
    self.tris = topology.tris
    self.i, self.j = topology.i, topology.j   # directed, both ways
    w = np.ones(topology.n_edges) if weighting == "uniform" else cotangent_weights(self.p0, topology)
    self.w = np.concatenate([w, w])
    self.e0 = self.p0[self.i] - self.p0[self.j]
    self.deg = np.bincount(self.i, weights=self.w, minlength=self.N)   # weighted Laplacian's diagonal
    self.off3 = np.repeat(-self.w, 3)                                   # its off-diagonal, in the 3N system
    self._L = None
    self.R = np.tile(np.eye(3), (self.N, 1, 1))
    self.factor = topology.factor

  @property
  def L(self):
    """The N x N weighted Laplacian (point-to-point rows only)."""
    if self._L is None:
      W = sp.csr_matrix((self.w, (self.i, self.j)), shape=(self.N, self.N))
      self._L = (sp.diags(self.deg) - W).tocsr()
    return self._L

  # -- local step ------------------------------------------------------------
  def rotations(self, p):
    """R_i = V U^T from S_i = sum_j w_ij e0_ij d_ij^T = U S V^T (det fixed)."""
    d = p[self.i] - p[self.j]
    S = np.zeros((self.N, 3, 3))
    np.add.at(S, self.i, self.w[:, None, None] * self.e0[:, :, None] * d[:, None, :])
    U, _, Vt = np.linalg.svd(S)
    V = Vt.transpose(0, 2, 1)
    R = V @ U.transpose(0, 2, 1)
    neg = np.linalg.det(R) < 0
    if neg.any():
      V[neg, :, -1] *= -1.0
      R[neg] = V[neg] @ U[neg].transpose(0, 2, 1)
    self.R = R
    return R

  # -- energies --------------------------------------------------------------
  def arap_energy(self, p, R):
    r = np.einsum("eab,eb->ea", R[self.i], self.e0) - (p[self.i] - p[self.j])
    return float((self.w * (r * r).sum(1)).sum())

  def arap_rhs(self, R):
    t = 0.5 * self.w[:, None] * np.einsum("eab,eb->ea", R[self.i] + R[self.j], self.e0)
    b = np.zeros((self.N, 3))
    np.add.at(b, self.i, t)
    return b

  @staticmethod
  def con_energy(p, cons):
    E = 0.0
    for idx, q, n, w in cons:
      r = p[idx] - q
      E += float((w * ((r * n).sum(1) ** 2 if n is not None else (r * r).sum(1))).sum())
    return E

  # -- global step -----------------------------------------------------------
  def global_step(self, p, R, cons, point_to_plane):
    """cons: list of (idx, q, n, w); n is None for point-to-point rows.
    Solves H p_new = rhs as H (p_new - p) = rhs - H p, so that a step with
    nothing to change (rhs - H p zero to rounding) needs no factorisation,
    which matters when nothing constrains a piece and H is singular there.
    Returns the new p'."""
    b = self.arap_rhs(R)
    if not point_to_plane:
      # three N x N solves, one matrix
      diag = np.zeros(self.N)
      for idx, q, n, w in cons:
        np.add.at(diag, idx, w)
        np.add.at(b, idx, w[:, None] * q)
      H = (self.L + sp.diags(diag)).tocsr()
      res = b - H @ p
      if np.abs(res).max() <= 1e-12 * max(np.abs(b).max(), 1e-30):
        return p.copy()
      return p + np.column_stack([self.factor.solve(H, res[:, k]) for k in range(3)])
    # one 3N x 3N solve: point-to-plane rows couple the axes. Every vertex's
    # 3x3 block stays in the matrix, zero or not (the topology's pattern).
    blk = np.zeros((self.N, 3, 3))
    b3 = b.copy()
    for idx, q, n, w in cons:
      if n is None:
        np.add.at(blk, idx, w[:, None, None] * np.eye(3))
        np.add.at(b3, idx, w[:, None] * q)
      else:
        np.add.at(blk, idx, w[:, None, None] * n[:, :, None] * n[:, None, :])
        np.add.at(b3, idx, (w * (n * q).sum(1))[:, None] * n)
    data = np.empty(len(self.top.indices))
    data[self.top.pos_off] = self.off3
    data[self.top.pos_blk] = blk.ravel()
    data[self.top.pos_diag] += self.deg[:, None]
    H = self.top.matrix(data)
    res = b3.ravel() - H @ p.ravel()
    if np.abs(res).max() <= 1e-12 * max(np.abs(b3).max(), 1e-30):
      return p.copy()
    return p + self.factor.solve(H, res).reshape(self.N, 3)


############################################################
# Constraints

def validate_pairs(pairs, targets, fixed_idx, N):
  """As snap does: indices in range, targets exactly (0, 0, 0) and fixed
  vertices skipped."""
  pairs, targets = np.asarray(pairs, int), np.asarray(targets, float).reshape(-1, 3)
  if len(pairs) != len(targets):
    return np.zeros(0, int), np.zeros((0, 3))
  ok = (pairs >= 0) & (pairs < N) & ~np.all(targets == 0, axis=1) & ~np.isin(pairs, fixed_idx)
  return pairs[ok], targets[ok]


class SurfaceMatcher:
  """Nearest target vertex (kd-tree over the target vertices, their normals)
  or nearest point on a target triangle (target.closest_points -> point,
  distance, face normal), within reach and with the vertex normal within
  the angle of the target normal."""

  def __init__(self, target_verts, target_normals, target_surface=None, tree=None):
    self.Vt, self.Nt = target_verts, target_normals
    self.tree = tree if tree is not None else cKDTree(target_verts)
    self.surface = target_surface

  def match(self, p, src_normals, eligible, reach, cos_a, mode="vertex"):
    x = p[eligible]
    if mode == "vertex":
      d, nn = self.tree.query(x, distance_upper_bound=reach)
      ok = np.isfinite(d)
      idx, q, n = eligible[ok], self.Vt[nn[ok]], self.Nt[nn[ok]]
    else:
      q, d, n = self.surface.closest_points(x)
      ok = d <= reach
      idx, q, n = eligible[ok], q[ok], n[ok]
    face = (src_normals[idx] * n).sum(1) >= cos_a
    return idx[face], q[face], n[face]


############################################################
# Driver

def weights(force, softness, diag):
  d2 = diag * diag
  return 10.0 ** force / d2, 10.0 ** (force - softness) / d2   # w_hard, w_soft


def iterate(solver, p_start, pairs, targets, fixed_idx, matcher, force, softness,
            stick_distance, stick_angle, iter_num, point_to_plane=True, match="vertex",
            eligible=None, ramp=None, inner=1, tol=1e-3, min_steps=1):
  """Generator: yields (step, p', E, matched) after every step; the last one
  is the kept state."""
  N = solver.N
  fixed_idx = np.asarray(fixed_idx, int)
  pairs, targets = validate_pairs(pairs, targets, fixed_idx, N)
  w_hard, w_soft = weights(force, softness, bbox_diag(solver.p0))
  if eligible is None:
    eligible = np.setdiff1d(np.arange(N), np.concatenate([fixed_idx, pairs]))
  reach, cos_a = float(stick_distance), float(np.cos(np.radians(np.clip(stick_angle, 0.0, 180.0))))

  hard = []
  if len(pairs):
    hard.append((pairs, targets, None, np.full(len(pairs), w_hard)))
  if len(fixed_idx):
    hard.append((fixed_idx, solver.p0[fixed_idx], None, np.full(len(fixed_idx), w_hard)))

  p = np.asarray(p_start, np.float64).copy()
  w_cur = w_soft if ramp is None else w_soft / ramp
  E_prev = None
  for step in range(iter_num):
    for _ in range(inner):
      R = solver.rotations(p)
      idx, q, n = matcher.match(p, vertex_normals(p, solver.tris), eligible, reach, cos_a, match) \
          if reach > 0 else (np.zeros(0, int), np.zeros((0, 3)), np.zeros((0, 3)))
      soft = [(idx, q, n if point_to_plane else None, np.full(len(idx), w_cur))] if len(idx) else []
      cons = hard + soft
      p = solver.global_step(p, R, cons, point_to_plane)
    E = solver.arap_energy(p, R) + solver.con_energy(p, cons)
    yield step, p, E, len(idx)
    if E_prev is not None and step + 1 >= min_steps and abs(E_prev - E) < tol * E:
      break
    E_prev = E
    if ramp is not None:
      w_cur = min(w_cur * 2.0, w_soft)
