# Wrap Standalone - single-file Blender add-on
#
# The COMPLETE wrap pipeline (Snap / Attract / Stick) running entirely
# inside Blender. No daemon, no sockets, no external dependencies:
# only numpy (bundled with Blender) and Blender's own BVH tree.
#
# Install: Edit > Preferences > Add-ons > Install... > pick this file,
# enable "Wrap Standalone". Panel appears in the 3D View sidebar (N key)
# under the "Wrap" tab.
#
# Workflow (same as the daemon-based add-on):
#   1. Select the SOURCE object, click "Set Source".
#   2. Select the TARGET object, click "Set Target".
#   3. "Pick Point Pair": click a point on the SOURCE, then the matching
#      point on the TARGET. Repeat for as many pairs as you want. ESC or
#      right-click to stop picking.
#   4. Snap -> Attract -> Stick. "Original" restores the source shape.
#
# Solver: cotangent-Laplacian elastic model (numpy CG solver), global
# similarity alignment + multiscale RBF proportional warp for snap,
# annealed non-rigid ICP for attract, progressive projection for stick.

bl_info = {
    "name": "Wrap Standalone",
    "author": "v0",
    "version": (1, 0, 0),
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar > Wrap",
    "description": "Standalone mesh wrapping (snap/attract/stick), no server needed",
    "category": "Mesh",
}

import bpy
import gpu
import numpy as np
from bpy.props import (
    BoolProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from gpu_extras.batch import batch_for_shader
from mathutils import Vector
from mathutils.bvhtree import BVHTree

EPS = 1.0e-12
SOLVER_VERSION = "standalone-4-threaded"

############################################################
# numpy sparse helpers (no scipy in Blender)

class CSRLite:
  """Minimal CSR-style sparse matrix supporting matvec via bincount."""

  def __init__(self, rows, cols, vals, n):
    order = np.argsort(rows, kind='stable')
    self.rows = rows[order]
    self.cols = cols[order]
    self.vals = vals[order]
    self.n = n

  def matvec(self, x):
    """y = A @ x for x of shape (n,) or (n, 3)."""
    if x.ndim == 1:
      return np.bincount(self.rows, weights=self.vals * x[self.cols],
                         minlength=self.n)
    out = np.empty((self.n, x.shape[1]))
    for d in range(x.shape[1]):
      out[:, d] = np.bincount(self.rows, weights=self.vals * x[self.cols, d],
                              minlength=self.n)
    return out

  def diagonal(self):
    d = np.zeros(self.n)
    on_diag = self.rows == self.cols
    np.add.at(d, self.rows[on_diag], self.vals[on_diag])
    return d


def triangulate_mesh(obj):
  """World-space vertices (n,3) float64 + triangle indices (m,3) int64."""
  dg = bpy.context.evaluated_depsgraph_get()
  eo = obj.evaluated_get(dg)
  mesh = eo.to_mesh()
  mesh.calc_loop_triangles()

  n = len(mesh.vertices)
  V = np.empty(n * 3, dtype=np.float64)
  mesh.vertices.foreach_get('co', V)
  V = V.reshape(-1, 3)

  m = len(mesh.loop_triangles)
  F = np.empty(m * 3, dtype=np.int64)
  mesh.loop_triangles.foreach_get('vertices', F)
  F = F.reshape(-1, 3)

  MW = np.array(obj.matrix_world, dtype=np.float64)
  V = V @ MW[:3, :3].T + MW[:3, 3]

  eo.to_mesh_clear()
  return V, F


def cotangent_laplacian(V, F):
  """Row-normalized cotangent Laplacian as CSRLite."""
  n = len(V)
  i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]

  def cot(a, b, c):
    u = V[b] - V[a]
    v = V[c] - V[a]
    cross = np.linalg.norm(np.cross(u, v), axis=1)
    cross[cross < EPS] = EPS
    return np.clip(np.einsum('ij,ij->i', u, v) / cross, -10.0, 10.0)

  c0 = cot(i0, i1, i2)
  c1 = cot(i1, i2, i0)
  c2 = cot(i2, i0, i1)

  rows = np.concatenate([i1, i2, i2, i0, i0, i1])
  cols = np.concatenate([i2, i1, i0, i2, i1, i0])
  vals = np.concatenate([c0, c0, c1, c1, c2, c2]) * 0.5

  # weights matrix W; L = D - W, row-normalized
  d = np.zeros(n)
  np.add.at(d, rows, vals)
  d_safe = np.abs(d.copy())
  d_safe[d_safe < EPS] = 1.0

  all_rows = np.concatenate([rows, np.arange(n)])
  all_cols = np.concatenate([cols, np.arange(n)])
  all_vals = np.concatenate([-vals / d_safe[rows], d / d_safe])
  return CSRLite(all_rows, all_cols, all_vals, n)


def vertex_normals(V, F):
  N = np.zeros_like(V)
  fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
  for k in range(3):
    np.add.at(N, F[:, k], fn)
  norms = np.linalg.norm(N, axis=1)
  norms[norms < EPS] = 1.0
  return N / norms[:, None]


############################################################
# Wrap solver state (module-level session)

class WrapState:
  source_name = ""
  target_name = ""
  V0 = None            # rest shape, world space
  x = None             # current positions, world space
  F = None
  L = None             # CSRLite laplacian
  L_diag = None
  bvh = None           # target BVH (world space)
  diag = 1.0
  mirror = None
  sim_R = None
  sim_s = 1.0
  sim_t = None
  adj_rows = None      # adjacency for dilation/relaxation
  adj_cols = None
  adj_deg = None
  pairs = []           # [(src_world_co, dst_world_co, src_vert_index)]
  initialized = False


S = WrapState()


def build_state(context):
  props = context.scene.wrap_sa
  src = bpy.data.objects.get(props.source_object)
  tgt = bpy.data.objects.get(props.target_object)
  if src is None or tgt is None:
    return "Set both Source and Target objects first"

  V0, F = triangulate_mesh(src)
  Vt, Ft = triangulate_mesh(tgt)
  if len(F) == 0 or len(Ft) == 0:
    return "Source and target must have faces"

  # Remember the very first rest shape per source so "Original" can always
  # restore it, even after re-initialization on a deformed mesh
  if getattr(S, 'first_source', None) != src.name:
    S.first_source = src.name
    S.first_V0 = V0.copy()

  S.source_name = src.name
  S.target_name = tgt.name
  S.V0 = V0
  S.x = V0.copy()
  S.F = F
  S.L = cotangent_laplacian(V0, F)
  S.L_diag = S.L.diagonal()
  # tolist() is parsed in C by FromPolygons - orders of magnitude faster
  # than a Python Vector()/tuple() comprehension on large meshes
  S.bvh = BVHTree.FromPolygons(Vt.tolist(), Ft.tolist())
  bb = V0.max(axis=0) - V0.min(axis=0)
  S.diag = max(float(np.linalg.norm(bb)), EPS)

  # adjacency (uniform) for gate dilation / tangential relaxation
  rows = np.concatenate([F[:, 0], F[:, 1], F[:, 1], F[:, 2], F[:, 2], F[:, 0]])
  cols = np.concatenate([F[:, 1], F[:, 0], F[:, 2], F[:, 1], F[:, 0], F[:, 2]])
  key = rows.astype(np.int64) * len(V0) + cols
  _, uniq = np.unique(key, return_index=True)
  S.adj_rows = rows[uniq]
  S.adj_cols = cols[uniq]
  deg = np.bincount(S.adj_rows, minlength=len(V0)).astype(np.float64)
  deg[deg < 1.0] = 1.0
  S.adj_deg = deg

  # symmetry map in the ORIGINAL local frame (mirror across source local X=0)
  S.mirror = None
  S.sim_R = np.eye(3)
  S.sim_s = 1.0
  S.sim_t = np.zeros(3)
  if props.symmetric:
    # Vectorized mirror matching (a Python KDTree loop here froze Blender
    # for many seconds on 50k+ vert meshes). Quantize local coordinates
    # and match x-mirrored keys with a hash join - pure numpy, milliseconds.
    MW = np.array(src.matrix_world, dtype=np.float64)
    inv = np.linalg.inv(MW)
    Vl = V0 @ inv[:3, :3].T + inv[:3, 3]
    tol = 1.0e-4 * S.diag
    q = np.round(Vl / tol).astype(np.int64)
    qm = q * np.array([-1, 1, 1])
    key = (q[:, 0] << 42) ^ (q[:, 1] << 21) ^ q[:, 2]
    key_m = (qm[:, 0] << 42) ^ (qm[:, 1] << 21) ^ qm[:, 2]
    order = np.argsort(key)
    pos_in_sorted = np.searchsorted(key[order], key_m)
    pos_in_sorted = np.clip(pos_in_sorted, 0, len(key) - 1)
    cand = order[pos_in_sorted]
    ok_mask = np.abs(Vl[cand] - Vl * np.array([-1.0, 1.0, 1.0])).max(axis=1) < tol
    if ok_mask.all():
      S.mirror = cand
      # store the local->world frame so mirroring happens in local space
      S.sim_R = MW[:3, :3] / np.cbrt(max(np.linalg.det(MW[:3, :3]), EPS))
      S.sim_s = np.cbrt(max(np.linalg.det(MW[:3, :3]), EPS))
      S.sim_t = MW[:3, 3]
    else:
      print(f"[wrap {SOLVER_VERSION}] mesh not symmetric "
            f"({(~ok_mask).sum()} unmatched verts) - symmetry disabled")

  S.initialized = True
  return None


# BVH queries are Python-loop bound: on a 50k+ vert mesh a full pass takes
# seconds. Never do a full pass inside a single modal timer tick - chunk it
# and yield between chunks so the Blender UI stays responsive (no freezing).
CHUNK = 4096


def closest_points_iter(P):
  """Chunked generator version of closest-point queries.

  Usage inside an operation generator:
      C, D, N = yield from closest_points_iter(P)
  Yields None after each chunk so the modal timer can breathe.
  """
  n = len(P)
  C = np.empty((n, 3))
  D = np.empty(n)
  N = np.empty((n, 3))
  find = S.bvh.find_nearest
  for i0 in range(0, n, CHUNK):
    hi = min(i0 + CHUNK, n)
    for i in range(i0, hi):
      co, normal, _, dist = find(Vector(P[i]))
      if co is None:
        C[i] = P[i]
        D[i] = np.inf
        N[i] = (0.0, 0.0, 1.0)
      else:
        C[i] = co
        D[i] = dist
        N[i] = normal
    if hi < n:
      yield None
  return C, D, N


def symmetrize(x):
  if S.mirror is None:
    return x
  xl = ((x - S.sim_t) @ S.sim_R) / S.sim_s
  xm = xl[S.mirror] * np.array([-1.0, 1.0, 1.0])
  xl = 0.5 * (xl + xm)
  return S.sim_s * (xl @ S.sim_R.T) + S.sim_t


def neighbor_average(x):
  out = np.zeros_like(x)
  for d in range(3):
    out[:, d] = np.bincount(S.adj_rows, weights=x[S.adj_cols, d],
                            minlength=len(x))
  return out / S.adj_deg[:, None]


def elastic_solve_iter(cons_idx, cons_pos, cons_w, alpha, beta, iters=80):
  """Solve (alpha*L + diag(w+beta)) x = alpha*L@x0 + beta*x + W c via CG.

  Chunked generator: yields every few CG iterations so a large solve never
  blocks the UI for more than a fraction of a second. Warm-started from the
  current positions, so convergence is fast in practice.
  Usage: S.x = yield from elastic_solve_iter(...)
  """
  n = len(S.V0)
  w = np.zeros(n)
  rhs = alpha * S.L.matvec(S.V0) + beta * S.x
  if len(cons_idx) > 0:
    np.add.at(w, cons_idx, cons_w)
    np.add.at(rhs, cons_idx, cons_w[:, None] * cons_pos)

  diag_A = alpha * S.L_diag + w + beta
  diag_A[diag_A < EPS] = EPS

  def A(v):
    return alpha * S.L.matvec(v) + (w + beta)[:, None] * v

  x = S.x.copy()
  r = rhs - A(x)
  z = r / diag_A[:, None]
  p = z.copy()
  rz = np.einsum('ij,ij->', r, z)
  for k in range(iters):
    Ap = A(p)
    pAp = np.einsum('ij,ij->', p, Ap)
    if abs(pAp) < EPS:
      break
    a = rz / pAp
    x += a * p
    r -= a * Ap
    if np.linalg.norm(r) < 1.0e-8 * n:
      break
    z = r / diag_A[:, None]
    rz_new = np.einsum('ij,ij->', r, z)
    p = z + (rz_new / max(rz, EPS)) * p
    rz = rz_new
    if (k + 1) % 15 == 0:
      yield None
  return symmetrize(x)


def similarity_transform(src, dst):
  if len(src) == 0:
    return None, None, None
  mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
  if len(src) == 1:
    return np.eye(3), 1.0, mu_d - mu_s
  if len(src) == 2:
    ds, dd = src[1] - src[0], dst[1] - dst[0]
    ls, ld = np.linalg.norm(ds), np.linalg.norm(dd)
    if ls < EPS or ld < EPS:
      return np.eye(3), 1.0, mu_d - mu_s
    scale = float(np.clip(ld / ls, 0.2, 5.0))
    u, v = ds / ls, dd / ld
    axis = np.cross(u, v)
    sin_a, cos_a = np.linalg.norm(axis), float(np.dot(u, v))
    if sin_a < EPS:
      R = np.eye(3) if cos_a > 0 else -np.eye(3) + 2.0 * np.outer(u, u)
    else:
      axis = axis / sin_a
      K = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
      R = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)
    return R, scale, mu_d - scale * (R @ mu_s)
  src_c, dst_c = src - mu_s, dst - mu_d
  cov = dst_c.T @ src_c / len(src)
  U, D, Vt = np.linalg.svd(cov)
  Ssign = np.eye(3)
  if np.linalg.det(U) * np.linalg.det(Vt) < 0:
    Ssign[2, 2] = -1.0
  R = U @ Ssign @ Vt
  var_s = np.einsum('ij,ij->', src_c, src_c) / len(src)
  scale = float(np.clip(np.trace(np.diag(D) @ Ssign) / max(var_s, EPS),
                        0.2, 5.0))
  return R, scale, mu_d - scale * (R @ mu_s)


def rbf_warp_field(src, dst, softness, diag):
  """Multiscale Gaussian RBF: coarse regularized layer for wide regional
  motion + fine exact-interpolation layer. Robust to dense pin clusters."""
  k = len(src)
  if k == 0:
    return None
  disp = dst - src
  if np.max(np.linalg.norm(disp, axis=1)) < 1.0e-9 * diag:
    return None
  soft = 0.5 + 0.5 * max(softness, 0.2)

  def kernel(A, B, sig_b):
    d = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    return np.exp(-(d / sig_b[None, :]) ** 2)

  if k == 1:
    sigma_f = np.array([0.35 * diag]) * soft
    W_f = np.linalg.solve(kernel(src, src, sigma_f) + 1e-8 * np.eye(k), disp)
    W_c = np.zeros((1, 3))
    sigma_c = sigma_f
  else:
    d2 = np.linalg.norm(src[:, None, :] - src[None, :, :], axis=2)
    np.fill_diagonal(d2, np.inf)
    nn = d2.min(axis=1)
    finite = d2[np.isfinite(d2)]
    med = float(np.median(finite)) if len(finite) else 0.3 * diag
    sigma_c = np.full(k, max(med, 0.15 * diag) * soft)
    Kc = kernel(src, src, sigma_c)
    W_c = np.linalg.solve(Kc + 0.1 * np.eye(k), disp)
    residual = disp - Kc @ W_c
    sigma_f = np.clip(nn, 0.06 * diag, 0.5 * diag) * soft
    W_f = np.linalg.solve(kernel(src, src, sigma_f) + 1e-8 * np.eye(k),
                          residual)

  def field(P):
    out = np.empty_like(P)
    step = 20000
    for i0 in range(0, len(P), step):
      chunk = P[i0:i0 + step]
      vals = kernel(chunk, src, sigma_f) @ W_f
      if k > 1:
        vals += kernel(chunk, src, sigma_c) @ W_c
      out[i0:i0 + step] = vals
    return out

  return field


def surface_constraints_iter(distance, cos_limit, base_weight, slide=False):
  closest, dist, tgt_n = yield from closest_points_iter(S.x)
  src_n = vertex_normals(S.x, S.F)
  cos_angle = np.einsum('ij,ij->i', src_n, tgt_n)
  accept = (dist < distance) & (cos_angle >= cos_limit)

  near = dist < distance * 2.0
  for _ in range(3):
    acc_f = accept.astype(np.float64)
    frac = np.bincount(S.adj_rows, weights=acc_f[S.adj_cols],
                       minlength=len(S.x)) / S.adj_deg
    grow = near & ~accept & (frac >= 0.5)
    if not grow.any():
      break
    accept |= grow

  idx = np.nonzero(accept)[0]
  if len(idx) == 0:
    return idx, np.zeros((0, 3)), np.zeros(0), closest, dist, tgt_n
  if slide:
    delta = closest[idx] - S.x[idx]
    along = np.einsum('ij,ij->i', delta, tgt_n[idx])
    pos = S.x[idx] + along[:, None] * tgt_n[idx]
  else:
    pos = closest[idx]
  d = dist[idx]
  falloff = 1.0 - np.clip(d / max(distance, EPS), 0.0, 1.0)
  w = base_weight * (0.3 + 0.7 * falloff)
  return idx, pos, w, closest, dist, tgt_n


############################################################
# Wrap operations (generators yielding after each iteration)

def snap_iterations(props):
  pairs = S.pairs
  if not pairs:
    yield
    return
  idx = np.array([p[2] for p in pairs], dtype=np.int64)
  pos = np.array([p[1] for p in pairs], dtype=np.float64)
  force = max(props.snap_force, 1e-3)
  softness = max(props.snap_softness, 1e-3)
  iter_num = max(props.iterations, 6)

  alpha = 100.0 / max(props.flex, 1.0)
  beta = 0.3 * max(props.accuracy, 0.1)

  print(f"[wrap {SOLVER_VERSION}] snap: {len(idx)} pins, "
        f"gap mean {np.linalg.norm(pos - S.x[idx], axis=1).mean():.4f}")

  # Phase 1: global similarity alignment
  n_align = max(iter_num // 4, 1)
  R, sc, tr = similarity_transform(S.x[idx], pos)
  if R is not None:
    x_start = S.x.copy()
    x_aligned = sc * (x_start @ R.T) + tr
    for it in range(n_align):
      t = (it + 1) / n_align
      S.x = x_start + t * (x_aligned - x_start)
      yield True
    S.V0 = sc * (S.V0 @ R.T) + tr
    S.sim_s = sc * S.sim_s
    S.sim_R = R @ S.sim_R
    S.sim_t = sc * (S.sim_t @ R.T) + tr

  # Phase 2: multiscale RBF proportional warp
  n_warp = max(iter_num // 3, 1)
  warp = rbf_warp_field(S.x[idx], pos, softness, S.diag)
  if warp is not None:
    x_start = S.x.copy()
    x_warped = x_start + warp(x_start)
    wd = np.linalg.norm(x_warped - x_start, axis=1)
    print(f"[wrap {SOLVER_VERSION}] rbf warp: "
          f"{(wd > 0.01 * S.diag).mean() * 100:.0f}% verts move >1% diag")
    for it in range(n_warp):
      t = (it + 1) / n_warp
      tt = t * t * (3.0 - 2.0 * t)
      S.x = symmetrize(x_start + tt * (x_warped - x_start))
      yield True
    S.V0 = S.V0 + warp(S.V0)

  # Phase 3: elastic refinement with exact pin projection
  used = (n_align if R is not None else 0) + (n_warp if warp else 0)
  n_refine = max(iter_num - used, 1)
  alpha_eff = alpha * max(softness, 0.25)
  for it in range(n_refine):
    t = (it + 1) / n_refine
    w = np.full(len(idx), force * 200.0 * t)
    S.x = yield from elastic_solve_iter(idx, pos, w, alpha_eff, beta)
    S.x[idx] += (t * t) * (pos - S.x[idx])
    yield True
    # SINGLE-PASS MODE: only the first (gentlest) refinement iteration is
    # applied - repeated passes progressively degrade smoothness.
    break


def attract_iterations(props):
  distance = max(props.attract_distance * 0.01 * S.diag, 1e-6 * S.diag)
  cos_user = np.cos(np.radians(np.clip(props.attract_angle, 0.0, 180.0)))
  iter_num = max(props.iterations, 6)
  alpha = 100.0 / max(props.flex, 1.0)
  beta = 0.3 * max(props.accuracy, 0.1)

  _, dist0, _ = yield from closest_points_iter(S.x)
  r0 = max(distance, float(np.percentile(dist0, 95)) * 1.25)

  for it in range(iter_num):
    t = (it + 1) / iter_num
    s = (1.0 - t) ** 2
    r_it = distance + (r0 - distance) * s
    cos_it = cos_user * (1.0 - s) + (-1.0) * s
    w_it = 10.0 + 190.0 * t
    alpha_it = alpha * (1.0 * s + 0.15 * (1.0 - s))
    idx, pos, w, closest, dist, tgt_n = yield from surface_constraints_iter(
        r_it, cos_it, w_it, slide=(t > 0.5))
    S.x = yield from elastic_solve_iter(idx, pos, w, alpha_it, beta)

    # tangential relaxation, preserving distance along the local normal.
    # Pre-solve distances are reused as the offset estimate to save a
    # full BVH pass per iteration (small approximation, huge speedup).
    avg = neighbor_average(S.x)
    moved = S.x + 0.35 * (avg - S.x)
    closest2, dist2, n2 = yield from closest_points_iter(moved)
    keep = dist < r_it
    off = np.minimum(dist[keep], dist2[keep])
    sign = np.sign(np.einsum('ij,ij->i', moved[keep] - closest2[keep],
                             n2[keep]))[:, None]
    moved[keep] = closest2[keep] + off[:, None] * n2[keep] * sign
    S.x = symmetrize(moved)
    yield True
    # SINGLE-PASS MODE: stop after the first (gentlest) iteration.
    # Repeated relaxation/projection passes degrade smoothness.
    break


def stick_iterations(props):
  distance = max(props.stick_distance * 0.01 * S.diag, 1e-6 * S.diag)
  cos_user = np.cos(np.radians(np.clip(props.stick_angle, 0.0, 180.0)))
  iter_num = max(props.iterations // 2, 4)
  alpha = 100.0 / max(props.flex, 1.0)
  beta = 0.3 * max(props.accuracy, 0.1)

  pairs = S.pairs
  pp_idx = np.array([p[2] for p in pairs], dtype=np.int64) if pairs else \
      np.zeros(0, dtype=np.int64)
  pp_pos = np.array([p[1] for p in pairs], dtype=np.float64) if pairs else \
      np.zeros((0, 3))

  _, dist0, _ = yield from closest_points_iter(S.x)
  r0 = max(distance, float(np.percentile(dist0, 95)) * 1.25)

  for it in range(iter_num):
    t = (it + 1) / iter_num
    s = (1.0 - t) ** 2
    r_it = distance + (r0 - distance) * s
    cos_it = cos_user * (1.0 - s) + (-1.0) * s
    w_it = 100.0 + 1900.0 * t
    alpha_it = alpha * (0.7 * s + 0.08 * (1.0 - s))
    idx, pos, w, _, _, _ = yield from surface_constraints_iter(
        r_it, cos_it, w_it)
    if len(pp_idx) > 0:
      idx = np.concatenate([idx, pp_idx])
      pos = np.concatenate([pos, pp_pos]) if len(pos) else pp_pos
      w = np.concatenate([w, np.full(len(pp_idx), 1000.0)])
    S.x = yield from elastic_solve_iter(idx, pos, w, alpha_it, beta)

    blend = t * t
    closest, dist, tgt_n = yield from closest_points_iter(S.x)
    src_n = vertex_normals(S.x, S.F)
    cos_angle = np.einsum('ij,ij->i', src_n, tgt_n)
    accept = (dist < distance) & (cos_angle >= cos_user)
    near = dist < distance * 2.0
    for _ in range(3):
      acc_f = accept.astype(np.float64)
      frac = np.bincount(S.adj_rows, weights=acc_f[S.adj_cols],
                         minlength=len(S.x)) / S.adj_deg
      grow = near & ~accept & (frac >= 0.5)
      if not grow.any():
        break
      accept |= grow
    S.x[accept] += blend * (closest[accept] - S.x[accept])

    # single relaxation pass (was 2): halves the BVH cost per iteration
    relax = 0.5 * blend
    avg = neighbor_average(S.x)
    moved = S.x + relax * (avg - S.x)
    closest2, _, _ = yield from closest_points_iter(moved)
    moved[accept] = closest2[accept]
    S.x = moved
    if len(pp_idx) > 0:
      S.x[pp_idx] += blend * (pp_pos - S.x[pp_idx])
    S.x = symmetrize(S.x)
    yield True
    # SINGLE-PASS MODE: stop after the first (gentlest) iteration.
    # Repeated hard projections + relaxation passes degrade smoothness.
    break


############################################################
# Write positions back to the source object

def apply_positions(context, positions=None):
  """Write world-space positions into the source mesh. Pass an explicit
  snapshot when the solver thread may still be mutating S.x."""
  src = bpy.data.objects.get(S.source_name)
  if src is None:
    return
  x = S.x if positions is None else positions
  inv = np.linalg.inv(np.array(src.matrix_world, dtype=np.float64))
  local = x @ inv[:3, :3].T + inv[:3, 3]
  flat = local.astype(np.float32).ravel()
  src.data.vertices.foreach_set('co', flat)
  src.data.update()


############################################################
# Point-pair picking (modal) + overlay drawing

_draw_handle = None


def _draw_pairs():
  shader = gpu.shader.from_builtin('UNIFORM_COLOR')
  gpu.state.point_size_set(9.0)

  if S.pairs:
    coords = []
    for src_co, dst_co, _ in S.pairs:
      coords.append(tuple(src_co))
      coords.append(tuple(dst_co))
    batch = batch_for_shader(shader, 'POINTS', {"pos": coords})
    shader.uniform_float("color", (1.0, 0.1, 0.1, 1.0))
    batch.draw(shader)
    lines = batch_for_shader(shader, 'LINES', {"pos": coords})
    shader.uniform_float("color", (1.0, 0.6, 0.1, 0.8))
    lines.draw(shader)

  # Pending picks: blue = source point, green = target point
  gpu.state.point_size_set(12.0)
  if getattr(S, 'pending_src', None) is not None:
    batch = batch_for_shader(
        shader, 'POINTS', {"pos": [tuple(S.pending_src[0])]})
    shader.uniform_float("color", (0.2, 0.5, 1.0, 1.0))
    batch.draw(shader)
  if getattr(S, 'pending_dst', None) is not None:
    batch = batch_for_shader(
        shader, 'POINTS', {"pos": [tuple(S.pending_dst)]})
    shader.uniform_float("color", (0.2, 1.0, 0.4, 1.0))
    batch.draw(shader)

  gpu.state.point_size_set(1.0)


def ensure_draw_handler():
  global _draw_handle
  if _draw_handle is None:
    _draw_handle = bpy.types.SpaceView3D.draw_handler_add(
        _draw_pairs, (), 'WINDOW', 'POST_VIEW')


def remove_draw_handler():
  global _draw_handle
  if _draw_handle is not None:
    bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
    _draw_handle = None


def _ray_cast_screen(context, event):
  """Ray-cast the scene under the mouse. Returns (obj_name, world_co)."""
  from bpy_extras import view3d_utils
  region = context.region
  rv3d = context.region_data
  coord = (event.mouse_region_x, event.mouse_region_y)
  direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, coord)
  origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, coord)
  dg = context.evaluated_depsgraph_get()
  hit, loc, _, _, obj, _ = context.scene.ray_cast(dg, origin, direction)
  if not hit:
    return None
  return obj.name, tuple(loc)


class _SinglePickMixin:
  """One click on the required object; every other event passes through,
  so you can freely orbit / pan / zoom while picking. ESC cancels."""

  pick_kind = 'SOURCE'  # or 'TARGET'

  def modal(self, context, event):
    context.area.tag_redraw()
    if event.type in {'RIGHTMOUSE', 'ESC'} and event.value == 'PRESS':
      context.workspace.status_text_set(None)
      return {'CANCELLED'}

    if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
      hit = _ray_cast_screen(context, event)
      if hit is None:
        return {'RUNNING_MODAL'}
      obj_name, world_co = hit
      props = context.scene.wrap_sa
      wanted = (props.source_object if self.pick_kind == 'SOURCE'
                else props.target_object)
      if obj_name != wanted:
        self.report({'WARNING'}, f"Click on the {self.pick_kind} mesh")
        return {'RUNNING_MODAL'}

      if self.pick_kind == 'SOURCE':
        d = np.linalg.norm(S.x - np.array(world_co), axis=1)
        vi = int(np.argmin(d))
        S.pending_src = (S.x[vi].copy(), vi)
      else:
        S.pending_dst = np.array(world_co)

      context.workspace.status_text_set(None)
      # Auto-commit when both halves are picked
      if (getattr(S, 'pending_src', None) is not None and
          getattr(S, 'pending_dst', None) is not None):
        context.workspace.status_text_set(
            "Both points set - press Pair to commit (or re-pick either)")
      return {'FINISHED'}

    return {'PASS_THROUGH'}

  def invoke(self, context, event):
    props = context.scene.wrap_sa
    if not S.initialized or S.source_name != props.source_object:
      err = build_state(context)
      if err:
        self.report({'ERROR'}, err)
        return {'CANCELLED'}
    ensure_draw_handler()
    context.workspace.status_text_set(
        f"Click a point on the {self.pick_kind} - navigate freely, "
        "ESC to cancel")
    context.window_manager.modal_handler_add(self)
    return {'RUNNING_MODAL'}


class WRAPSA_OT_pick_source(bpy.types.Operator, _SinglePickMixin):
  bl_idname = "wrapsa.pick_source"
  bl_label = "Source Point"
  bl_description = ("Click one point on the SOURCE mesh (navigate freely "
                    "first; ESC cancels)")
  pick_kind = 'SOURCE'


class WRAPSA_OT_pick_target(bpy.types.Operator, _SinglePickMixin):
  bl_idname = "wrapsa.pick_target"
  bl_label = "Target Point"
  bl_description = ("Click one point on the TARGET mesh (navigate freely "
                    "first; ESC cancels)")
  pick_kind = 'TARGET'


class WRAPSA_OT_add_pair(bpy.types.Operator):
  bl_idname = "wrapsa.add_pair"
  bl_label = "Pair"
  bl_description = "Commit the picked source + target points as a pair"

  @classmethod
  def poll(cls, context):
    return (getattr(S, 'pending_src', None) is not None and
            getattr(S, 'pending_dst', None) is not None)

  def execute(self, context):
    src_co, vi = S.pending_src
    S.pairs.append((src_co, S.pending_dst.copy(), vi))
    S.pending_src = None
    S.pending_dst = None
    context.workspace.status_text_set(None)
    for area in context.screen.areas:
      area.tag_redraw()
    self.report({'INFO'}, f"Pair added ({len(S.pairs)} total)")
    return {'FINISHED'}


############################################################
# Wrap operators (solver on a BACKGROUND THREAD, modal timer applies
# results). All heavy math uses only numpy + the prebuilt BVH - no bpy
# access - so it is safe off the main thread. Blender data is written
# exclusively from modal() on the main thread. The UI never freezes:
# the worker can take as long as it needs while the viewport stays live.

class _WrapRunnerMixin:
  _timer = None
  _thread = None
  _lock = None
  _latest = None      # newest positions produced by the worker
  _dirty = False      # new positions waiting to be applied
  _done = False
  _error = None
  _cancel = False

  def _worker(self, gen):
    try:
      for val in gen:
        if self._cancel:
          break
        if val:  # one full iteration finished
          with self._lock:
            self._latest = S.x.copy()
            self._dirty = True
    except Exception as exc:  # surface worker errors in the UI
      import traceback
      traceback.print_exc()
      self._error = str(exc)
    finally:
      self._done = True

  def modal(self, context, event):
    if event.type == 'ESC':
      self._cancel = True
    if event.type == 'TIMER':
      apply_now = None
      with self._lock:
        if self._dirty:
          apply_now = self._latest
          self._dirty = False
      if apply_now is not None:
        apply_positions(context, apply_now)
        context.area.tag_redraw()
      if self._done and not self._dirty:
        if self._error:
          self.report({'ERROR'}, f"Wrap failed: {self._error}")
        return self._finish(context)
    return {'RUNNING_MODAL'}

  def _finish(self, context):
    wm = context.window_manager
    if self._timer:
      wm.event_timer_remove(self._timer)
    if self._thread is not None:
      self._thread.join(timeout=2.0)
    context.workspace.status_text_set(None)
    context.area.tag_redraw()
    return {'FINISHED'}

  def _start(self, context, gen):
    import threading
    props = context.scene.wrap_sa
    if not S.initialized or S.source_name != props.source_object:
      err = build_state(context)
      if err:
        self.report({'ERROR'}, err)
        return {'CANCELLED'}
    self._lock = threading.Lock()
    self._latest = None
    self._dirty = False
    self._done = False
    self._error = None
    self._cancel = False
    self._thread = threading.Thread(target=self._worker, args=(gen,),
                                    daemon=True)
    self._thread.start()
    wm = context.window_manager
    self._timer = wm.event_timer_add(0.05, window=context.window)
    wm.modal_handler_add(self)
    context.workspace.status_text_set(
        f"{self.bl_label} running... (ESC to cancel)")
    return {'RUNNING_MODAL'}


class WRAPSA_OT_init(bpy.types.Operator):
  bl_idname = "wrapsa.init"
  bl_label = "Init"
  bl_description = ("Build the solver state: elastic model, target BVH, "
                    "symmetry map. Runs automatically before Snap if skipped")

  def execute(self, context):
    err = build_state(context)
    if err:
      self.report({'ERROR'}, err)
      return {'CANCELLED'}
    self.report(
        {'INFO'},
        f"Init OK: {S.n} source verts, symmetric="
        f"{'yes' if S.mirror is not None else 'no'}")
    return {'FINISHED'}


class WRAPSA_OT_snap(bpy.types.Operator, _WrapRunnerMixin):
  bl_idname = "wrapsa.snap"
  bl_label = "Snap"
  bl_description = "Global alignment + proportional RBF warp + elastic pin refinement"

  def execute(self, context):
    if not S.pairs:
      self.report({'ERROR'}, "Pick at least one point pair first")
      return {'CANCELLED'}
    return self._start(context, snap_iterations(context.scene.wrap_sa))


class WRAPSA_OT_attract(bpy.types.Operator, _WrapRunnerMixin):
  bl_idname = "wrapsa.attract"
  bl_label = "Attract"
  bl_description = "Annealed non-rigid ICP surface attraction"

  def execute(self, context):
    return self._start(context, attract_iterations(context.scene.wrap_sa))


class WRAPSA_OT_stick(bpy.types.Operator, _WrapRunnerMixin):
  bl_idname = "wrapsa.stick"
  bl_label = "Stick"
  bl_description = "Strong projection onto the target surface"

  def execute(self, context):
    return self._start(context, stick_iterations(context.scene.wrap_sa))


class WRAPSA_OT_set_source(bpy.types.Operator):
  bl_idname = "wrapsa.set_source"
  bl_label = "Set Source"

  def execute(self, context):
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
      self.report({'ERROR'}, "Select a mesh object")
      return {'CANCELLED'}
    context.scene.wrap_sa.source_object = obj.name
    S.initialized = False
    S.pairs.clear()
    return {'FINISHED'}


class WRAPSA_OT_set_target(bpy.types.Operator):
  bl_idname = "wrapsa.set_target"
  bl_label = "Set Target"

  def execute(self, context):
    obj = context.active_object
    if obj is None or obj.type != 'MESH':
      self.report({'ERROR'}, "Select a mesh object")
      return {'CANCELLED'}
    context.scene.wrap_sa.target_object = obj.name
    S.initialized = False
    return {'FINISHED'}


class WRAPSA_OT_original(bpy.types.Operator):
  bl_idname = "wrapsa.original"
  bl_label = "Original"
  bl_description = "Restore the source mesh to its original shape"

  def execute(self, context):
    props = context.scene.wrap_sa
    err = build_state(context) if (not S.initialized or
                                   S.source_name != props.source_object) \
        else None
    if err:
      self.report({'ERROR'}, err)
      return {'CANCELLED'}
    # Restore the FIRST-seen rest shape (not the current re-init snapshot,
    # which may already be deformed)
    S.x = S.first_V0.copy() if getattr(S, 'first_V0', None) is not None \
        else S.V0.copy()
    S.V0 = S.x.copy()
    S.sim_R = np.eye(3)
    S.sim_s = 1.0
    S.sim_t = np.zeros(3)
    apply_positions(context)
    context.area.tag_redraw()
    return {'FINISHED'}


class WRAPSA_OT_clear_pairs(bpy.types.Operator):
  bl_idname = "wrapsa.clear_pairs"
  bl_label = "Clear Pairs"

  def execute(self, context):
    S.pairs.clear()
    S.pending_src = None
    S.pending_dst = None
    for area in context.screen.areas:
      area.tag_redraw()
    return {'FINISHED'}


############################################################
# Properties + panel

class WrapSAProps(bpy.types.PropertyGroup):
  source_object: StringProperty(name="Source")
  target_object: StringProperty(name="Target")
  symmetric: BoolProperty(name="Symmetric", default=False,
                          description="Mirror deformation across the source's local X=0")
  flex: FloatProperty(name="Flex", default=100.0, min=50.0, max=200.0)
  accuracy: FloatProperty(name="Accuracy", default=3.0, min=0.1, max=10.0)
  snap_force: FloatProperty(name="Snap Force", default=7.0, min=1.0, max=9.0)
  snap_softness: FloatProperty(name="Snap Softness", default=2.0, min=0.0,
                               max=3.0)
  attract_distance: FloatProperty(name="Attract Dist %", default=3.0,
                                  min=0.1, max=50.0,
                                  description="Percent of mesh diagonal")
  attract_angle: FloatProperty(name="Attract Angle", default=30.0, min=1.0,
                               max=180.0)
  stick_distance: FloatProperty(name="Stick Dist %", default=1.7, min=0.1,
                                max=50.0,
                                description="Percent of mesh diagonal")
  stick_angle: FloatProperty(name="Stick Angle", default=50.0, min=1.0,
                             max=180.0)
  iterations: IntProperty(name="Iterations", default=20, min=6, max=60)


class WRAPSA_PT_panel(bpy.types.Panel):
  bl_label = "Wrap Standalone"
  bl_space_type = 'VIEW_3D'
  bl_region_type = 'UI'
  bl_category = "Wrap"

  def draw(self, context):
    layout = self.layout
    props = context.scene.wrap_sa

    col = layout.column(align=True)
    row = col.row(align=True)
    row.operator("wrapsa.set_source")
    row.label(text=props.source_object or "-")
    row = col.row(align=True)
    row.operator("wrapsa.set_target")
    row.label(text=props.target_object or "-")
    col.prop(props, "symmetric")

    layout.separator()
    col = layout.column(align=True)
    row = col.row(align=True)
    src_set = getattr(S, 'pending_src', None) is not None
    dst_set = getattr(S, 'pending_dst', None) is not None
    row.operator("wrapsa.pick_source",
                 icon='RADIOBUT_ON' if src_set else 'RADIOBUT_OFF')
    row.operator("wrapsa.pick_target",
                 icon='RADIOBUT_ON' if dst_set else 'RADIOBUT_OFF')
    col.operator("wrapsa.add_pair", icon='LINKED')
    row = col.row(align=True)
    row.operator("wrapsa.clear_pairs")
    row.label(text=f"{len(S.pairs)} pairs")

    layout.separator()
    col = layout.column(align=True)
    col.prop(props, "flex")
    col.prop(props, "accuracy")
    col.prop(props, "snap_force")
    col.prop(props, "snap_softness")
    col.prop(props, "attract_distance")
    col.prop(props, "attract_angle")
    col.prop(props, "stick_distance")
    col.prop(props, "stick_angle")
    col.prop(props, "iterations")

    layout.separator()
    col = layout.column(align=True)
    col.scale_y = 1.4
    col.operator(
        "wrapsa.init",
        icon='CHECKMARK' if S.initialized else 'FILE_REFRESH')
    col.operator("wrapsa.snap", icon='SNAP_ON')
    col.operator("wrapsa.attract", icon='FORCE_MAGNETIC')
    col.operator("wrapsa.stick", icon='UV_SYNC_SELECT')
    col.operator("wrapsa.original", icon='LOOP_BACK')


CLASSES = (
    WrapSAProps,
    WRAPSA_OT_set_source,
    WRAPSA_OT_set_target,
    WRAPSA_OT_pick_source,
    WRAPSA_OT_pick_target,
    WRAPSA_OT_add_pair,
    WRAPSA_OT_clear_pairs,
    WRAPSA_OT_init,
    WRAPSA_OT_snap,
    WRAPSA_OT_attract,
    WRAPSA_OT_stick,
    WRAPSA_OT_original,
    WRAPSA_PT_panel,
)


def register():
  for cls in CLASSES:
    bpy.utils.register_class(cls)
  bpy.types.Scene.wrap_sa = PointerProperty(type=WrapSAProps)


def unregister():
  remove_draw_handler()
  del bpy.types.Scene.wrap_sa
  for cls in CLASSES:
    bpy.utils.unregister_class(cls)


if __name__ == "__main__":
  register()
