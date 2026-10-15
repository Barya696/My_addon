# Rig fit
#
# Poses a skinned mesh's deform bones so the mesh matches a target shape with
# the same vertex order, and bakes what posing cannot reach into the base
# shape (from fit_rig_to_shape.py, tested standalone):
#
#   Fitter.fit_pose()  inverse skinning with the weights locked; connected
#                      bones rotate about their joint; refined against
#                      Blender's own deformation (B-bones included); bones
#                      whose skin folds get a stiffer bend and are refit
#   Fitter.place()     put the (swapped) deform bones at the solved transforms
#   Fitter.bake()      base shape that the posed rig deforms onto the target

import bpy
import numpy as np
from mathutils import Matrix

from .rig_swap import write_matrices

FIT_ROUNDS = 40     # Blender-in-the-loop rounds per pose fit
FOLD_ATTEMPTS = 8   # refits with a stiffer bend on bones whose skin folds
BAKE_STEPS = 20     # damped Newton steps of the base-shape bake
BASE_PRIOR = 0.001  # initial bend stiffness of connected bones (relative)


def log(*args):
  print('[Wrap Fit]', *args, flush=True)


############################################################
# Inverse skinning
#
# Model: plain linear blend skinning with the mesh's weights. Each deform
# bone j has a world-space rigid transform (R_j, t_j) from rest to pose.

def rms(a, b):
  return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def procrustes(H):
  """
  Rotation R maximizing trace(R H) (H = sum source target^T)
  """
  U, _, Vt = np.linalg.svd(H)
  d = -1.0 if np.linalg.det(Vt.T @ U.T) < 0 else 1.0
  return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def skin(V0, W, R, t):
  out = V0 * (1.0 - W.sum(axis=1))[:, None]
  for j in range(W.shape[1]):
    idx = np.nonzero(W[:, j] > 1e-6)[0]
    out[idx] += W[idx, j, None] * (V0[idx] @ R[j].T + t[j])
  return out


def deform_weights(obj, rig):
  """
  Normalized weights (n, b) of the deform bones that have a vertex group
  """
  deform = {b.name for b in rig.data.bones if b.use_deform}
  groups = [g for g in obj.vertex_groups if g.name in deform]
  column = {g.index: i for i, g in enumerate(groups)}
  W = np.zeros((len(obj.data.vertices), len(groups)))
  for v in obj.data.vertices:
    for g in v.groups:
      i = column.get(g.group)
      if i is not None:
        W[v.index, i] = g.weight
  total = W.sum(axis=1, keepdims=True)
  W = np.divide(W, total, out=np.zeros_like(W), where=total > 0.0)
  return [g.name for g in groups], W


def bone_info(rig, names, W):
  """
  Hierarchy of the deform bones. A connected bone can only rotate about its
  parent's tail; its subtree is everything connected below it.
  """
  A = rig.matrix_world
  index = {n: i for i, n in enumerate(names)}
  bones = [rig.data.bones[n] for n in names]
  parent = [index[b.parent.name]
            if b.use_connect and b.parent and b.parent.name in index else -1
            for b in bones]
  children = [[c for c in range(len(names)) if parent[c] == i]
              for i in range(len(names))]

  def chain_depth(i):
    d = 0
    while parent[i] >= 0:
      i, d = parent[i], d + 1
    return d

  def below(i):
    out = [i]
    for c in children[i]:
      out += below(c)
    return out

  columns = [np.nonzero(W[:, j] > 1e-6)[0] for j in range(len(names))]
  subtree = [below(i) for i in range(len(names))]
  return {
      'parent': parent,
      'head0': np.array([list(A @ b.head_local) for b in bones]),
      'length': np.array([(A @ b.tail_local - A @ b.head_local).length
                          for b in bones]),
      'order': sorted(range(len(names)), key=chain_depth),
      'subtree': subtree,
      'sub_idx': [np.unique(np.concatenate([columns[k] for k in s]))
                  for s in subtree],
  }


def solve_subtree(V0, V1, W, info, alpha, R=None, t=None, sweeps=300,
                  tol=1e-6):
  """
  Coordinate descent over joints, root first. Each update moves a bone
  together with everything below it: a rotation about its head for a
  connected bone, a rigid transform for a free one. A connected bone's
  rotation is also pulled toward its parent's (its rest bend) by
      mu_j ||G - R_parent R_j^T||^2,  mu_j = alpha_j sum(s^2) L_j^2
  Every update is the exact least-squares optimum for its block.
  """
  parent, head0, length, order, subtree, sub_idx = (
      info[k] for k in ('parent', 'head0', 'length', 'order', 'subtree',
                        'sub_idx'))
  b = W.shape[1]
  R = np.tile(np.eye(3), (b, 1, 1)) if R is None else R.copy()
  t = np.zeros((b, 3)) if t is None else t.copy()
  pred = skin(V0, W, R, t)

  for _ in range(sweeps):
    before = rms(pred, V1)
    for j in order:
      S, idx = subtree[j], sub_idx[j]
      p = V0[idx]
      c = np.zeros((len(idx), 3))
      s = np.zeros(len(idx))
      for k in S:
        wk = W[idx, k]
        c += wk[:, None] * (p @ R[k].T + t[k])
        s += wk
      others = pred[idx] - c
      target = V1[idx] - others

      if parent[j] >= 0:
        h = R[j] @ head0[j] + t[j]
        H = (c - s[:, None] * h).T @ (target - s[:, None] * h)
        mu = alpha[j] * float(s @ s) * length[j] ** 2
        if mu > 0.0:
          H = H + mu * (R[parent[j]] @ R[j].T).T
        G = procrustes(H)
        g = h - G @ h
      else:
        s2 = max(float(s @ s), 1e-12)
        c_hat = (s @ c) / s2
        q_hat = (s @ target) / s2
        G = procrustes((c - s[:, None] * c_hat).T
                       @ (target - s[:, None] * q_hat))
        g = q_hat - G @ c_hat

      for k in S:
        R[k] = G @ R[k]
        t[k] = G @ t[k] + g
      pred[idx] = others + c @ G.T + s[:, None] * g

    if before - rms(pred, V1) <= tol * before:
      break
  return R, t


############################################################
# Fitter: Blender-side evaluation

class Fitter:
  def __init__(self, obj, rig, V0, target):
    self.obj = obj
    self.rig = rig
    self.V0 = V0
    self.target = target
    self.M = np.array(obj.matrix_world)
    self.M_inv = np.linalg.inv(self.M)
    self.A = rig.matrix_world.copy()
    self.A_inv = self.A.inverted()
    self.names, self.W = deform_weights(obj, rig)
    self.info = bone_info(rig, self.names, self.W)
    diag = float(np.linalg.norm(V0.max(axis=0) - V0.min(axis=0)))
    self.eps = 5e-4 * diag  # finite-difference offset

  # Mesh

  def set_base_world(self, B):
    local = B @ self.M_inv[:3, :3].T + self.M_inv[:3, 3]
    co = local.astype(np.float32).ravel()
    self.obj.data.vertices.foreach_set('co', co)
    # With shape keys, Blender evaluates the reference key
    keys = self.obj.data.shape_keys
    if keys and keys.reference_key:
      keys.reference_key.data.foreach_set('co', co)
    self.obj.data.update()

  def mesh_world(self):
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    eo = self.obj.evaluated_get(dg)
    me = eo.to_mesh()
    co = np.empty(len(me.vertices) * 3)
    me.vertices.foreach_get('co', co)
    eo.to_mesh_clear()
    return co.reshape(-1, 3) @ self.M[:3, :3].T + self.M[:3, 3]

  def deformed(self, B):
    self.set_base_world(B)
    return self.mesh_world()

  def jacobians(self, B, D0):
    """
    Per-vertex 3x3 deformation at base B: the armature deforms every vertex
    on its own, so three offset evaluations give all of them
    """
    J = np.empty((len(B), 3, 3))
    for k in range(3):
      offset = np.zeros(3)
      offset[k] = self.eps
      J[:, :, k] = (self.deformed(B + offset) - D0) / self.eps
    self.set_base_world(B)
    return J

  # Pose

  def place(self, R, t):
    """
    Put the swapped deform bones at the solved transforms, return the mesh
    """
    items = []
    for j, name in enumerate(self.names):
      T = np.eye(4)
      T[:3, :3], T[:3, 3] = R[j], t[j]
      pose_bone = self.rig.pose.bones[name]
      items.append((pose_bone, self.A_inv @ Matrix(T.tolist()) @ self.A
                    @ pose_bone.bone.matrix_local))
    write_matrices(items)
    return self.mesh_world()

  def refine(self, R, t, alpha):
    """
    Blender evaluates the real mesh; its difference to the skinning model
    (B-bones and the like) is taken off the target and the model re-solved.
    Keeps the best round; the incoming pose is not a candidate, so a raised
    bend stiffness always takes effect.
    """
    best = (float('inf'), R, t)
    stale = 0
    for k in range(FIT_ROUNDS):
      actual = self.place(R, t)
      if k > 0:
        err = rms(actual, self.target)
        stale = 0 if err < best[0] * 0.995 else stale + 1
        if err < best[0]:
          best = (err, R.copy(), t.copy())
        if stale >= 4:
          break
      correction = actual - skin(self.V0, self.W, R, t)
      R, t = solve_subtree(self.V0, self.target - correction, self.W,
                           self.info, alpha, R, t, sweeps=25, tol=1e-5)
    if best[0] == float('inf'):
      return R, t
    return best[1], best[2]

  def fit_pose(self):
    """
    Pose fit with fold prevention. Returns the solved transforms.
    """
    alpha = np.full(len(self.names), BASE_PRIOR)
    R, t = solve_subtree(self.V0, self.target, self.W, self.info, alpha)
    parent = self.info['parent']

    for attempt in range(FOLD_ATTEMPTS):
      R, t = self.refine(R, t, alpha)
      D = self.place(R, t)
      folded = np.nonzero(np.linalg.det(self.jacobians(self.V0, D)) <= 0)[0]
      log(f"pose fit {attempt + 1}: {rms(D, self.target) * 1000:.2f} mm rms, "
          f"{len(folded)} vertices folded")
      if not len(folded):
        break
      # Stiffer bend on the bones whose skin folds (and a little on their
      # parents), then refit
      for j in set(np.argmax(self.W[folded], axis=1).tolist()):
        alpha[j] = max(alpha[j] * 10.0, 0.01)
        if parent[j] >= 0:
          alpha[parent[j]] = max(alpha[parent[j]], alpha[j] * 0.3)
    else:
      log(f"  {len(folded)} vertices still fold, the bake leaves them where "
          f"the rig can reach")
    return R, t

  # Bake

  def bake(self):
    """
    Base shape that the posed rig deforms onto the target: per-vertex damped
    Newton (Levenberg-Marquardt); a step is kept only for the vertices it
    brings closer.
    """
    B = self.V0.copy()
    D = self.deformed(B)
    err = np.linalg.norm(D - self.target, axis=1)
    lam = np.full(len(B), 1e-6)
    I3 = np.eye(3)
    for _ in range(BAKE_STEPS):
      J = self.jacobians(B, D)
      Jt = np.transpose(J, (0, 2, 1))
      delta = np.linalg.solve(Jt @ J + lam[:, None, None] * I3,
                              (Jt @ (self.target - D)[:, :, None]))[:, :, 0]
      D_try = self.deformed(B + delta)
      err_try = np.linalg.norm(D_try - self.target, axis=1)
      better = err_try < err
      B[better] += delta[better]
      D[better] = D_try[better]
      err[better] = err_try[better]
      lam = np.where(better, lam * 0.3, lam * 10.0)
    self.set_base_world(B)
    return err
