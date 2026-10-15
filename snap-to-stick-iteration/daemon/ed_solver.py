# Embedded Deformation (Sumner, Schmid, Pauly, SIGGRAPH 2007) on a tet cage.
# numpy + scipy only.
#
# Each cage node j carries an affine A_j and a translation t_j (start: I, 0).
# A source vertex v_i is bound to the 4 nodes of its enclosing tet with
# barycentric weights w_ij and deforms to
#
#   v~_i = sum_j w_ij [A_j (v_i - g_j) + g_j + t_j]
#
# Energy, minimised over all nodes at once by Gauss-Newton (one sparse
# Cholesky per iteration, Levenberg-Marquardt damping after a rejected step;
# accepting, stopping and streaming as the original C++ does, see iterate):
#
#   E = E_rot + w_reg E_reg + sum over constraint blocks of w_c E_con
#
#   E_rot = sum_j (a1.a2)^2 + (a1.a3)^2 + (a2.a3)^2
#                 + (1 - a1.a1)^2 + (1 - a2.a2)^2 + (1 - a3.a3)^2
#   E_reg = sum_j sum_k in N(j) || A_j (g_k - g_j) + g_j + t_j - (g_k + t_k) ||^2
#   E_con = sum_i || v~_i - q_i ||^2              (point pairs, fixed vertices)
#         + sum_i (n_i . (v~_i - q_i))^2          (attract: point-to-plane)
#
# with a_k the columns of A_j, cage edges in both directions, uniform weights.

import base64
import io

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy import ndimage
from scipy.spatial import cKDTree

try:
  import pypardiso   # Intel MKL PARDISO
  pypardiso.PyPardisoSolver()   # loads MKL now: if it is missing, use SuperLU
except Exception:
  pypardiso = None

_PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

# The cage: cells of (source bounding-box diagonal) / RESOLUTION, and the
# cells kept around the body grown by PAD cells
RESOLUTION = 64
PAD = 1

# Bumped whenever the cage a mesh gets changes, so a cage cached in the
# add-on's init data by an older build is rebuilt rather than reused
CAGE_VERSION = "mirrored-1"

# The step loop's stop tolerance, as in the original
STOP_TOL = 1.0e-3


############################################################
# Cage construction

def _sample_surface(verts, faces, spacing):
  """Points covering every triangle at <= `spacing` so no voxel is skipped."""
  tri = verts[faces]   # F,3,3
  e = np.linalg.norm(np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1],
                               tri[:, 0] - tri[:, 2]], 1), axis=2).max(1)
  n = np.maximum(1, np.ceil(e / spacing).astype(int))
  pts = [verts]
  for k in np.unique(n):
    sel = tri[n == k]
    i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
    m = (i + j) <= k
    u, v = i[m] / k, j[m] / k
    bary = np.stack([1 - u - v, u, v], 1)   # B,3
    pts.append(np.einsum("bk,fkd->fbd", bary, sel).reshape(-1, 3))
  return np.concatenate(pts)


def inside_cell_centres(verts, faces, origin, cell_size, dims):
  """Which cell centres are inside the closed triangle mesh: the parity of
  the surface crossings on a ray from each centre, cast along x, y and z,
  the majority of the three (a ray grazing an edge can miscount one axis).
  Unlike filling holes in the voxels, this keeps limbs that only come close
  (the legs, an arm and the torso) apart."""
  tri = verts[faces]   # F,3,3
  votes = np.zeros(dims, np.int8)
  for a in range(3):
    b, c = [k for k in range(3) if k != a]
    P = tri[:, :, [b, c]]   # each triangle seen along axis a
    # the centre lines (ib, ic) inside each triangle's bounding box
    lo = np.ceil((P.min(1) - origin[[b, c]]) / cell_size - 0.5).astype(int)
    hi = np.floor((P.max(1) - origin[[b, c]]) / cell_size - 0.5).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.array([dims[b], dims[c]]) - 1)
    nb, nc = np.maximum(hi[:, 0] - lo[:, 0] + 1, 0), np.maximum(hi[:, 1] - lo[:, 1] + 1, 0)
    count = nb * nc
    t = np.repeat(np.arange(len(tri)), count)
    k = np.arange(count.sum()) - np.repeat(np.cumsum(count) - count, count)
    ib = lo[t, 0] + k // np.maximum(nc[t], 1)
    ic = lo[t, 1] + k % np.maximum(nc[t], 1)
    q = origin[[b, c]] + (np.stack([ib, ic], 1) + 0.5) * cell_size
    # the line crosses the triangle where all three edge functions agree in sign
    p0, p1, p2 = P[t, 0], P[t, 1], P[t, 2]
    cross = lambda u, v, w: (v[:, 0] - u[:, 0]) * (w[:, 1] - u[:, 1]) - (v[:, 1] - u[:, 1]) * (w[:, 0] - u[:, 0])
    e0, e1, e2 = cross(p1, p2, q), cross(p2, p0, q), cross(p0, p1, q)
    area = e0 + e1 + e2
    hit = (((e0 >= 0) & (e1 >= 0) & (e2 >= 0)) | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))) & (area != 0)
    t, ib, ic = t[hit], ib[hit], ic[hit]
    w = np.stack([e0[hit], e1[hit], e2[hit]], 1) / area[hit, None]
    xa = (w * tri[t][:, :, a]).sum(1) - origin[a]   # where along the line
    # crossings below each centre on its line: odd means inside
    span = (dims[a] + 2) * cell_size
    line = ib * dims[c] + ic
    keys = np.sort(line * span + xa)
    ia, jb, jc = np.meshgrid(np.arange(dims[a]), np.arange(dims[b]), np.arange(dims[c]), indexing="ij")
    cl = jb * dims[c] + jc
    below = np.searchsorted(keys, cl * span + (ia + 0.5) * cell_size) - np.searchsorted(keys, cl * span)
    odd = (below % 2 == 1).astype(np.int8)   # indexed [a, b, c]
    order = np.argsort([a, b, c])
    votes += np.transpose(odd, order)
  return votes >= 2


def build_cage(verts, faces, cell_size, pad):
  """Cells of cell_size that hold surface samples or whose centre is inside
  the (closed) surface, grown by pad cells; every kept cube split into 6
  Freudenthal tets on a shared lattice.

  The cage is its own mirror image about the X centre plane of the mesh's
  bounding box: the lattice is centred on the box with an even number of
  cells along X (so that plane is a lattice plane), the kept cells are
  mirrored too, and the cubes left of the plane use the X-mirrored split.
  A plain Freudenthal split has one fixed body diagonal, which its mirror
  image does not share, so a symmetric body in it deforms differently on
  its left and right (eve's Snap: 18 mm between the shoulders, against the
  original's 0.85 mm). Across the plane the split stays conforming: the
  shared faces' diagonals agree."""
  lo, hi = verts.min(0), verts.max(0)
  dims = np.ceil((hi - lo) / cell_size).astype(int) + 2 * (pad + 1)
  dims[0] += dims[0] % 2
  origin = 0.5 * (lo + hi) - 0.5 * dims * cell_size

  pts = _sample_surface(verts, faces, cell_size * 0.5)
  idx = np.floor((pts - origin) / cell_size).astype(int)
  occ = np.zeros(dims, bool)
  occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
  occ |= inside_cell_centres(verts, faces, origin, cell_size, dims)
  if pad > 0:
    occ = ndimage.binary_dilation(occ, iterations=pad)
  occ |= occ[::-1]           # mirror the kept cells too

  cells = np.argwhere(occ)   # C,3
  vd = dims + 1              # lattice vertex grid
  left = cells[:, 0] < dims[0] // 2
  step_x = np.where(left, -1, 1)

  def vid(c):
    return (c[:, 0] * vd[1] + c[:, 1]) * vd[2] + c[:, 2]

  tets = []
  for p in _PERMS:   # Freudenthal / Kuhn split, X-mirrored left of the centre
    c = cells.copy()
    c[left, 0] += 1          # mirrored cubes start at their +X corner
    corners = [c]
    for ax in p:
      c = c.copy()
      c[:, ax] += step_x if ax == 0 else 1
      corners.append(c)
    tets.append(np.stack([vid(k) for k in corners], 1))
  tets = np.stack(tets, 1)   # C,6,4 lattice ids

  used, inv = np.unique(tets, return_inverse=True)
  tets = inv.reshape(tets.shape)   # compact node ids
  nodes = origin + np.stack(np.unravel_index(used, vd), 1) * cell_size

  cell_row = -np.ones(dims, int)
  cell_row[cells[:, 0], cells[:, 1], cells[:, 2]] = np.arange(len(cells))

  T = tets.reshape(-1, 4)
  pairs = np.concatenate([T[:, [a, b]] for a, b in
                          [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]])
  pairs = np.unique(np.sort(pairs, 1), axis=0)
  edges = np.concatenate([pairs, pairs[:, ::-1]])   # directed, both ways

  return dict(nodes=nodes, tets=tets, cells=cells, cell_row=cell_row,
              origin=origin, cell_size=cell_size, edges=edges)


def bind_vertices(cage, verts):
  """Return (bind_idx N,4) node ids and (bind_w N,4) barycentric weights."""
  nodes, tets = cage["nodes"], cage["tets"]
  cidx = np.floor((verts - cage["origin"]) / cage["cell_size"]).astype(int)
  cidx = np.clip(cidx, 0, np.array(cage["cell_row"].shape) - 1)
  row = cage["cell_row"][cidx[:, 0], cidx[:, 1], cidx[:, 2]]
  bad = row < 0
  if bad.any():   # vertex outside cage
    centers = cage["origin"] + (cage["cells"] + 0.5) * cage["cell_size"]
    row[bad] = cKDTree(centers).query(verts[bad])[1]
  cand = tets[row]   # N,6,4
  P = nodes[cand]    # N,6,4,3
  T = np.stack([P[..., 1, :] - P[..., 0, :], P[..., 2, :] - P[..., 0, :],
                P[..., 3, :] - P[..., 0, :]], -1)   # N,6,3,3
  lam = np.linalg.solve(T, (verts[:, None, :] - P[..., 0, :])[..., None])[..., 0]
  w = np.concatenate([1 - lam.sum(-1, keepdims=True), lam], -1)   # N,6,4
  best = w.min(-1).argmax(1)   # tet containing the point
  ar = np.arange(len(verts))
  return cand[ar, best], w[ar, best]


############################################################
# Solver

class SPDFactor:
  """Sparse Cholesky for the Gauss-Newton matrix H (symmetric positive
  definite). Its pattern is the same from one iteration to the next, so
  PARDISO's analysis (the fill-reducing ordering) runs only when the
  pattern changes and each iteration does the numeric factorisation alone.
  Without pypardiso: SuperLU in symmetric mode with a minimum-degree
  ordering (with row pivoting that ordering is lost and a factorisation
  takes minutes)."""

  def __init__(self):
    self.pardiso = pypardiso.PyPardisoSolver(mtype=2) if pypardiso is not None else None
    self.pattern = None

  def solve(self, H, b):
    H = H.tocsr()
    H.sum_duplicates()
    if self.pardiso is None:
      lu = spla.splu(H.tocsc(), permc_spec="MMD_AT_PLUS_A",
                     diag_pivot_thresh=0.0, options=dict(SymmetricMode=True))
      return lu.solve(b)
    U = sp.triu(H, format="csr")   # PARDISO's symmetric types take the upper triangle
    self.pardiso._check_A(U)
    b = np.ascontiguousarray(b, dtype=np.float64)
    if (self.pattern is None or not np.array_equal(self.pattern[0], U.indptr)
        or not np.array_equal(self.pattern[1], U.indices)):
      self.pardiso.set_phase(11)   # analysis
      self.pardiso._call_pardiso(U, b)
      self.pattern = (U.indptr.copy(), U.indices.copy())
    self.pardiso.set_phase(22)     # numeric factorisation
    self.pardiso._call_pardiso(U, b)
    self.pardiso.set_phase(33)     # solve
    return self.pardiso._call_pardiso(U, b)


class EDSolver:
  """X layout per node j: A_j row-major at 12j..12j+8 (A[r,c] -> 12j+3r+c),
  t_j at 12j+9..12j+11."""

  def __init__(self, cage, rest_verts, bind_idx, bind_w):
    self.g = cage["nodes"]
    self.M = len(self.g)
    self.edges = cage["edges"]
    self.v0 = rest_verts
    self.bi, self.bw = bind_idx, bind_w
    self.J_reg, self.c_reg = self._reg_jacobian()
    self.factor = SPDFactor()
    self.reset()

  def reset(self):
    self.X = np.tile(np.r_[np.eye(3).ravel(), 0.0, 0.0, 0.0], self.M)

  # -- skinning --------------------------------------------------------------
  def deform(self, X=None):
    X = self.X if X is None else X
    Xn = X.reshape(self.M, 12)
    A, t = Xn[:, :9].reshape(self.M, 3, 3), Xn[:, 9:]
    j = self.bi
    g = self.g[j]   # N,4,3
    out = np.einsum("nsrc,nsc->nsr", A[j], self.v0[:, None, :] - g) + g + t[j]
    return (self.bw[..., None] * out).sum(1)

  # -- constant Jacobians ----------------------------------------------------
  def con_jacobian(self, vi):
    """Residual for vertices vi is J @ X + const - q (3 rows per vertex)."""
    n = len(vi)
    j, w = self.bi[vi], self.bw[vi]
    d = self.v0[vi][:, None, :] - self.g[j]   # n,4,3
    c = np.arange(3)
    rb = 3 * np.arange(n)[:, None]
    rows, cols, vals = [], [], []
    for s in range(4):
      js, ws, ds = j[:, s], w[:, s], d[:, s]
      R = (rb + c[None, :])[:, :, None].repeat(3, 2)
      C = 12 * js[:, None, None] + 3 * c[None, :, None] + c[None, None, :]
      V = (ws[:, None, None] * ds[:, None, :]).repeat(3, 1)
      rows += [R.ravel(), (rb + c).ravel()]
      cols += [C.ravel(), (12 * js[:, None] + 9 + c).ravel()]
      vals += [V.ravel(), np.repeat(ws, 3)]
    J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3 * n, 12 * self.M))
    const = (w[..., None] * self.g[j]).sum(1).ravel()
    return J, const

  def plane_jacobian(self, vi, normals):
    """Point-to-plane residuals for vertices vi along unit normals (one row
    per vertex): n_i . v~_i - n_i . q_i is J @ X + const - (n_i . q_i)."""
    k = len(vi)
    j, w = self.bi[vi], self.bw[vi]
    d = self.v0[vi][:, None, :] - self.g[j]   # k,4,3
    n = normals
    c = np.arange(3)
    # d/dA_s[r,c] = w_s n_r d_s[c] at 12 j_s + 3 r + c;  d/dt_s[r] = w_s n_r
    colA = 12 * j[:, :, None, None] + 3 * c[None, None, :, None] + c[None, None, None, :]
    valA = w[:, :, None, None] * n[:, None, :, None] * d[:, :, None, :]
    colT = 12 * j[:, :, None] + 9 + c[None, None, :]
    valT = w[:, :, None] * n[:, None, :]
    cols = np.concatenate([colA.reshape(k, 36), colT.reshape(k, 12)], 1).ravel()
    vals = np.concatenate([valA.reshape(k, 36), valT.reshape(k, 12)], 1).ravel()
    J = sp.csr_matrix((vals, (np.repeat(np.arange(k), 48), cols)), shape=(k, 12 * self.M))
    const = (n * (w[..., None] * self.g[j]).sum(1)).sum(1)
    return J, const

  def _reg_jacobian(self):
    E = self.edges
    j, k = E[:, 0], E[:, 1]
    ev = self.g[k] - self.g[j]
    n = len(E)
    c = np.arange(3)
    rb = 3 * np.arange(n)[:, None]
    R = (rb + c[None, :])[:, :, None].repeat(3, 2)
    C = 12 * j[:, None, None] + 3 * c[None, :, None] + c[None, None, :]
    V = ev[:, None, :].repeat(3, 1)
    rows = [R.ravel(), (rb + c).ravel(), (rb + c).ravel()]
    cols = [C.ravel(), (12 * j[:, None] + 9 + c).ravel(), (12 * k[:, None] + 9 + c).ravel()]
    vals = [V.ravel(), np.ones(3 * n), -np.ones(3 * n)]
    J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(3 * n, 12 * self.M))
    return J, (self.g[j] - self.g[k]).ravel()

  # -- nonlinear rotation term -----------------------------------------------
  def _rot(self, X):
    A = X.reshape(self.M, 12)[:, :9].reshape(self.M, 3, 3)
    a = [A[:, :, 0], A[:, :, 1], A[:, :, 2]]   # columns
    dot = lambda p, q: (p * q).sum(1)
    r = np.stack([dot(a[0], a[1]), dot(a[0], a[2]), dot(a[1], a[2]),
                  dot(a[0], a[0]) - 1, dot(a[1], a[1]) - 1, dot(a[2], a[2]) - 1], 1)
    spec = [(0, 0, a[1]), (0, 1, a[0]), (1, 0, a[2]), (1, 2, a[0]), (2, 1, a[2]),
            (2, 2, a[1]), (3, 0, 2 * a[0]), (4, 1, 2 * a[1]), (5, 2, 2 * a[2])]
    jj, rr = np.arange(self.M), np.arange(3)
    rows, cols, vals = [], [], []
    for q, col, val in spec:
      rows.append(np.repeat(6 * jj + q, 3))
      cols.append((12 * jj[:, None] + 3 * rr[None, :] + col).ravel())
      vals.append(val.ravel())
    J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(6 * self.M, 12 * self.M))
    return J, r.ravel()

  # -- Gauss-Newton driver ---------------------------------------------------
  def iterate(self, terms, w_reg, iters, w_rot=1.0, tol=STOP_TOL, retarget=None):
    """Generator: yields (iteration, deformed vertices, E, accepted) after
    every iteration, rejected ones included:
      - plain Gauss-Newton; after a rejected step each diagonal entry of H is
        multiplied by (1 + lambda), lambda growing tenfold per rejection from
        1e-6 and shrinking tenfold per accepted step (the C++ keeps lambda
        at 0, so a rejected step comes back unchanged every time)
      - as the original C++: a step is accepted when the energy does not grow
        (E_new <= E); the streamed vertices are the step's, even when it is
        then undone; it stops after an accepted step with
        |E - E_new| < tol * E_new, or after iters steps.
    terms: list of (J, const, q_flat, weight) linear constraint blocks.
    retarget: optional function of the kept vertex positions, called at the
    start and after every accepted step, that returns the terms for the
    steps that follow (attract rebuilds its matches and weight)."""
    X = self.X.copy()
    if retarget is not None:
      terms = retarget(self.deform(X))

    def system(X):
      Jr, rr = self._rot(X)
      Js = [np.sqrt(w_rot) * Jr, np.sqrt(w_reg) * self.J_reg]
      rs = [np.sqrt(w_rot) * rr, np.sqrt(w_reg) * (self.J_reg @ X + self.c_reg)]
      for J, c, q, w in terms:
        Js.append(np.sqrt(w) * J)
        rs.append(np.sqrt(w) * (J @ X + c - q))
      return sp.vstack(Js).tocsr(), np.concatenate(rs)

    J, r = system(X)
    E, lam = r @ r, 0.0
    for it in range(iters):
      g = J.T @ r
      if not g.any():
        # Already at a stationary point (no constraints, or every one met):
        # the Gauss-Newton step is zero. H may be singular here (a rigid
        # motion of the whole cage costs nothing), so do not factorise it
        delta = np.zeros_like(X)
      else:
        H = J.T @ J
        if lam:
          H = H + lam * sp.diags(H.diagonal())
        delta = self.factor.solve(H, -g)
      Xn = X + delta
      Jn, rn = system(Xn)
      En = rn @ rn
      accepted = En <= E
      converged = accepted and abs(E - En) < tol * En
      if accepted:
        X, J, r, E = Xn, Jn, rn, En
        lam = 0.0 if lam < 1e-9 else lam / 10
        if retarget is not None and not converged:
          terms = retarget(self.deform(X))
          J, r = system(X)
          E = r @ r
      else:
        lam = max(lam * 10, 1e-6)   # undo the step; damp and retry
      self.X = X
      yield it, self.deform(Xn), En, accepted
      if converged:
        break

  def solve(self, terms, w_reg, iters, on_frame=None, w_rot=1.0, tol=STOP_TOL):
    for it, v, E, accepted in self.iterate(terms, w_reg, iters, w_rot, tol):
      if on_frame:
        on_frame(it, v, E, accepted)
    return self.deform()


############################################################
# Protocol-level entry points

def bbox_diag(verts):
  return float(np.linalg.norm(verts.max(0) - verts.min(0)))


def cage_params_from_alpha_beta(alpha, beta, diag):
  # TODO: tune this mapping against recorded exe output (Task 5).
  return diag / RESOLUTION, PAD


def init(verts, faces, alpha, beta):
  """faces: one closed triangulation of the source (the inside test counts
  surface crossings, so no face may be there twice)."""
  diag = bbox_diag(verts)
  cell, pad = cage_params_from_alpha_beta(alpha, beta, diag)
  cage = build_cage(verts, faces, cell, pad)
  bi, bw = bind_vertices(cage, verts)
  return dict(nodes=cage["nodes"], edges=cage["edges"], bind_idx=bi, bind_w=bw,
              rest=verts.astype(np.float64), diag=diag)


def make_solver(init_data):
  cage = dict(nodes=init_data["nodes"], edges=init_data["edges"])
  return EDSolver(cage, init_data["rest"], init_data["bind_idx"], init_data["bind_w"])


def weights(force, softness, diag):
  d2 = diag * diag
  return 10.0 ** force / d2, 10.0 ** (force - softness) / d2   # w_con, w_reg


def constraint_terms(solver, pairs, targets, fixed_idx, w_con):
  """Pairs whose target is exactly (0,0,0) or whose vertex is fixed are skipped."""
  terms = []
  pairs, targets = np.asarray(pairs, int), np.asarray(targets, float).reshape(-1, 3)
  fixed_idx = np.asarray(fixed_idx, int)
  valid = ~np.all(targets == 0, axis=1) & ~np.isin(pairs, fixed_idx)
  if valid.any():
    J, c = solver.con_jacobian(pairs[valid])
    terms.append((J, c, targets[valid].ravel(), w_con))
  if len(fixed_idx):
    J, c = solver.con_jacobian(fixed_idx)
    terms.append((J, c, solver.v0[fixed_idx].ravel(), w_con))
  return terms


def snap(solver, pairs, targets, fixed_idx, force, softness, iter_num, on_frame=None):
  w_con, w_reg = weights(force, softness, bbox_diag(solver.v0))
  terms = constraint_terms(solver, pairs, targets, fixed_idx, w_con)
  return solver.solve(terms, w_reg, iter_num, on_frame)


class Attraction:
  """Attract's constraints, as the original C++ builds them: each attracted
  vertex takes the nearest target vertex to where it is now, kept only if it
  is within reach and the two vertex normals agree within the angle; the
  residual is point-to-plane along that target vertex's normal. The weight
  starts at w_reg / 128 and doubles after every accepted step, up to w_reg;
  the matches are rebuilt after every accepted step. The point pairs and
  fixed vertices stay held (base terms).

  normals_of: function giving the source's vertex normals at positions x."""

  def __init__(self, solver, base, tree, target_verts, target_normals, reach, angle_deg,
               w_reg, attracted, normals_of):
    self.solver, self.base, self.tree = solver, base, tree
    self.tv, self.tn = target_verts, target_normals
    self.reach, self.cos = reach, np.cos(np.radians(np.clip(angle_deg, 0.0, 180.0)))
    self.w_reg, self.attracted, self.normals_of = w_reg, attracted, normals_of
    self.w = None
    self.matched = 0

  def __call__(self, x):
    self.w = self.w_reg / 128.0 if self.w is None else min(2.0 * self.w, self.w_reg)
    vi = self.attracted
    d, nn = self.tree.query(x[vi], distance_upper_bound=self.reach)
    ok = np.isfinite(d)
    vi, nn = vi[ok], nn[ok]
    ok = (self.normals_of(x)[vi] * self.tn[nn]).sum(1) >= self.cos
    vi, nn = vi[ok], nn[ok]
    self.matched = len(vi)
    if not len(vi):
      return self.base
    n = self.tn[nn]
    J, c = self.solver.plane_jacobian(vi, n)
    return self.base + [(J, c, (n * self.tv[nn]).sum(1), self.w)]


############################################################
# wrap_init_data (de)serialisation

def serialize_init(init_data):
  buf = io.BytesIO()
  np.savez_compressed(buf, **init_data)
  return base64.b64encode(buf.getvalue()).decode()


def deserialize_init(s):
  with np.load(io.BytesIO(base64.b64decode(s))) as z:
    d = {k: z[k] for k in z.files}
  d["diag"] = float(d["diag"])
  return d
