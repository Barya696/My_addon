# Fit Rig To Shape
#
# Standalone script (Blender Text Editor, Blender 4.x / 5.x). Poses the
# armature of a skinned mesh so the mesh matches a target shape with the same
# vertex order (e.g. the Wrap add-on's Stick result), then bakes what posing
# cannot reach into the mesh's base shape, so the posed mesh matches the
# target exactly. Not tied to Rigify or to any bone names.
#
#   FIT    1. Pose: inverse skinning with the weights locked. Connected deform
#             bones only rotate about their joint; each bone moves together
#             with everything below it. Refined against Blender's own
#             deformation (B-bones included). Bones whose skin folds get a
#             stiffer bend and are refit until nothing folds.
#          2. The pose is handed to the bones that drive the deform bones:
#             their constraints are swapped temporarily (bind), the followed
#             pose is baked into those bones, and the constraints are put back
#             exactly as they were.
#          3. The remaining difference is baked into the base shape.
#   UNDO   Puts the pose and the base shape back as they were before FIT.
#
# Usage: select the skinned mesh (or set TARGET = 'WRAP'), set MODE, Run Script.
# FIT takes about a minute on a 50k vertex character; progress is printed to
# the system console.
#
# Note: the Wrap add-on's Original / Snap / Attract / Stick buttons rewrite the
# mesh from their own snapshots, which replaces the baked base shape. Run UNDO
# before using them again.

import json
import time

import bpy
import numpy as np
from mathutils import Matrix

MODE = 'FIT'     # 'FIT' | 'UNDO'
TARGET = 'WRAP'  # 'WRAP': latest Wrap add-on stage, or the name of a mesh
#                  object with the same vertices as the selected mesh

FIT_ROUNDS = 40     # Blender-in-the-loop rounds per pose fit
FOLD_ATTEMPTS = 8   # refits with a stiffer bend on bones whose skin folds
BAKE_STEPS = 20     # damped Newton steps of the base-shape bake
BASE_PRIOR = 0.001  # initial bend stiffness of connected bones (relative)

STATE_KEY = 'fit_rig_to_shape'
BASE_ATTRIBUTE = 'fit_rig_base'


def log(*args):
  print('[Fit Rig]', *args, flush=True)


############################################################
# Bind (from swap_def_bind.py)
#
# SWAP mutes the deform bones' constraints and binds the bones that drive them
# to the deform bones with temporary constraints, so the deform bones can be
# posed directly. RESTORE bakes the followed pose into those bones, removes
# the temporary constraints and unmutes the originals exactly as they were.

BIND_MAX_DEPTH = 8
BIND_MAX_PASSES = 6
BIND_EPS = 1e-5
BIND_STATE_KEY = 'swap_def_bind'
BIND_PREFIX = 'SwapDEF '


def is_driven(rig, pose_bone, con):
  if not rig.animation_data:
    return False
  path = f'pose.bones["{pose_bone.name}"].constraints["{con.name}"]'
  return any(d.data_path.startswith(path) for d in rig.animation_data.drivers)


def depth(pose_bone):
  d = 0
  while pose_bone.parent:
    pose_bone = pose_bone.parent
    d += 1
  return d


def matrix_delta(a, b):
  return max(abs(a[i][j] - b[i][j]) for i in range(4) for j in range(4))


def write_matrices(items):
  """
  items: [(pose_bone, pose_space_matrix)], written parents first so each
  PoseBone.matrix setter sees a fresh parent matrix. Repeated until nothing
  moves: a bone's parent can also depend on a sibling through a constraint.
  """
  by_depth = {}
  for pose_bone, matrix in items:
    by_depth.setdefault(depth(pose_bone), []).append((pose_bone, matrix))
  for _ in range(BIND_MAX_PASSES):
    changed = 0.0
    for level in sorted(by_depth):
      for pose_bone, matrix in by_depth[level]:
        delta = matrix_delta(pose_bone.matrix, matrix)
        if delta > BIND_EPS:
          pose_bone.matrix = matrix
          changed = max(changed, delta)
      bpy.context.view_layer.update()
    if changed <= BIND_EPS:
      break


class Binder:
  """
  Walks from a deform bone to the free bones that place it and records one
  binding per (bone, kind): ('FULL' | 'HEAD' | 'TAIL' | 'SCALE', deform name)
  """

  def __init__(self, rig, deform_names):
    self.rig = rig
    self.deform_names = deform_names
    self.binds = {}
    self.limits = set()

  def limit(self, bone, why):
    self.limits.add(f"{bone.name}: {why}")

  def bind(self, bone, kind, source):
    self.binds.setdefault(bone.name, {})[kind] = source

  def live_constraints(self, bone, muted=None):
    cons = [c for c in bone.constraints
            if (c.influence > 0.0 or is_driven(self.rig, bone, c))]
    if muted is not None:
      return [c for c in cons if muted.get(c.name)]
    return [c for c in cons if c.enabled]

  def resolve(self, bone, source, kind, level=0, muted=None):
    cons = self.live_constraints(bone, muted)

    if not cons:
      if level == 0:
        return
      if bone.name in self.deform_names:
        self.limit(bone, 'chain leads back to a deform bone')
        return
      if kind != 'SCALE' and bone.bone.use_connect and bone.parent:
        self.limit(bone, 'connected control, its head stays on the parent tail')
      self.bind(bone, kind, source)
      return

    if level >= BIND_MAX_DEPTH:
      self.limit(bone, 'constraint chain too deep')
      return

    placed = False
    oriented = False
    for con in cons:
      if self.invert(bone, con, source, kind, level):
        placed = True
        if con.type == 'COPY_TRANSFORMS':
          oriented = True

    if placed and kind == 'FULL' and not oriented:
      # Head and tail are placed but nothing sets the roll. Stretch To keeps
      # the owner's roll, so bind the bone itself ahead of its constraints.
      if bone.name in self.deform_names:
        self.limit(bone, 'roll comes from its parent')
      else:
        self.bind(bone, 'FULL', source)

    if placed or kind != 'FULL':
      return

    # Nothing places this bone, it sits where its parent puts it
    parent = bone.parent
    if not parent or parent.name in self.deform_names:
      self.limit(bone, 'position comes from an unplaceable parent')
      return
    offset = (bone.matrix_basis.inverted() @ bone.bone.matrix_local.inverted()
              @ parent.bone.matrix_local)
    if matrix_delta(offset, Matrix.Identity(4)) > BIND_EPS:
      self.limit(bone, 'parent is offset from this bone, offset ignored')
    self.resolve(parent, source, 'FULL', level + 1)

  def invert(self, bone, con, source, kind, level):
    """
    Push one constraint onwards. Returns True if it sets the bone's location.
    """
    rig = self.rig
    ctype = con.type

    if ctype not in {'COPY_TRANSFORMS', 'COPY_LOCATION', 'COPY_SCALE',
                     'STRETCH_TO'}:
      self.limit(bone, f'{ctype} still applies')
      return False
    if con.target != rig or con.subtarget not in rig.pose.bones:
      self.limit(bone, f'{ctype} target is not a bone of this armature')
      return False
    if getattr(con, 'head_tail', 0.0) != 0.0:
      self.limit(bone, f'{ctype} head/tail is not 0')
      return False

    target = rig.pose.bones[con.subtarget]
    driven = is_driven(rig, bone, con)
    if con.influence != 1.0 and not driven:
      self.limit(bone, f'{ctype} has partial influence {con.influence:.2f}')
      return False
    if driven:
      self.limit(bone, f'{ctype} influence is driven, all targets bound')

    if ctype == 'COPY_TRANSFORMS':
      if con.mix_mode != 'REPLACE':
        self.limit(bone, f'Copy Transforms mix mode {con.mix_mode}')
        return False
      if con.target_space != con.owner_space:
        self.limit(bone, 'Copy Transforms spaces differ')
        return False
      self.resolve(target, source, kind, level + 1)
      return True

    if ctype == 'COPY_LOCATION':
      if kind == 'SCALE':
        return False
      if not (con.use_x and con.use_y and con.use_z) or con.use_offset \
          or con.invert_x or con.invert_y or con.invert_z:
        self.limit(bone, 'Copy Location is partial, inverted or offset')
        return False
      if con.target_space != con.owner_space:
        self.limit(bone, 'Copy Location spaces differ')
        return False
      self.resolve(target, source, 'HEAD' if kind != 'TAIL' else 'TAIL',
                   level + 1)
      return True

    if ctype == 'COPY_SCALE':
      if kind in {'HEAD', 'TAIL'}:
        return False
      if not (con.use_x and con.use_y and con.use_z) or con.use_offset \
          or con.power != 1.0 or con.use_make_uniform:
        self.limit(bone, 'Copy Scale uses offset, power or uniform')
        return False
      self.resolve(target, source, 'SCALE', level + 1)
      return False

    # STRETCH_TO: the target's head sits on this bone's tail
    if kind != 'FULL':
      return False
    self.resolve(target, source, 'TAIL', level + 1)
    return False


def bind_resolve_all(rig, state):
  deform_names = {name for name, _, _, _ in state['muted']}
  muted_by_bone = {}
  for bone_name, con_name, enabled, _ in state['muted']:
    muted_by_bone.setdefault(bone_name, {})[con_name] = enabled

  binder = Binder(rig, deform_names)
  for bone_name, muted in muted_by_bone.items():
    pose_bone = rig.pose.bones.get(bone_name)
    if pose_bone:
      binder.resolve(pose_bone, bone_name, 'FULL', 0, muted)
  return binder


def add_binding(rig, bone, kind, source):
  if kind == 'FULL':
    con = bone.constraints.new('COPY_TRANSFORMS')
  elif kind == 'SCALE':
    con = bone.constraints.new('COPY_SCALE')
  else:
    con = bone.constraints.new('COPY_LOCATION')
    con.head_tail = 1.0 if kind == 'TAIL' else 0.0
  con.name = f'{BIND_PREFIX}{kind}'
  con.target = rig
  con.subtarget = source
  con.target_space = 'WORLD'
  con.owner_space = 'WORLD'
  con.show_expanded = False
  # Ahead of the bone's own constraints so they still run after it
  bone.constraints.move(len(bone.constraints) - 1, 0)
  return con


def bind_swap(rig):
  if rig.get(BIND_STATE_KEY):
    raise RuntimeError("The rig is already swapped")

  # Deform bones drop to their basis pose once their constraints are muted.
  # Remember where they are so the swap does not move anything.
  held = [(pose_bone, pose_bone.matrix.copy())
          for pose_bone in rig.pose.bones if pose_bone.bone.use_deform]

  muted = []
  for pose_bone in rig.pose.bones:
    if pose_bone.bone.use_deform:
      for con in pose_bone.constraints:
        muted.append([pose_bone.name, con.name, con.enabled,
                      getattr(con, 'subtarget', None)])
  state = {'muted': muted, 'bound': []}

  # Work out the bindings while the constraints still point at their targets
  binder = bind_resolve_all(rig, state)

  # Mute and clear the targets: Blender keeps a muted constraint's
  # dependency, and with the controls bound the other way that is a cycle
  for bone_name, con_name, _, subtarget in muted:
    con = rig.pose.bones[bone_name].constraints[con_name]
    con.enabled = False
    if subtarget is not None and con.target == rig:
      con.subtarget = ''

  bpy.context.view_layer.update()
  write_matrices(held)

  order = {'FULL': 0, 'SCALE': 1, 'HEAD': 2, 'TAIL': 2}
  for bone_name, kinds in binder.binds.items():
    bone = rig.pose.bones[bone_name]
    # Added in reverse so that after each move-to-front the order is
    # FULL, SCALE, HEAD/TAIL: a head placement only overrides location
    for kind in sorted(kinds, key=lambda k: order[k], reverse=True):
      con = add_binding(rig, bone, kind, kinds[kind])
      state['bound'].append([bone_name, con.name])

  rig[BIND_STATE_KEY] = json.dumps(state)
  bpy.context.view_layer.update()

  log(f"Swapped {len(muted)} constraints on {len({m[0] for m in muted})} "
      f"deform bones, {len(binder.binds)} bones follow them")
  for line in sorted(binder.limits):
    log(f"  limited {line}")


def bind_restore(rig):
  state = rig.get(BIND_STATE_KEY)
  if not state:
    return
  state = json.loads(state)

  bpy.context.view_layer.update()

  # Bake: the bound bones' followed matrices become their own transforms
  bound_bones = {name for name, _ in state['bound'] if name in rig.pose.bones}
  baked = [(rig.pose.bones[name], rig.pose.bones[name].matrix.copy())
           for name in bound_bones]

  for bone_name, con_name in state['bound']:
    pose_bone = rig.pose.bones.get(bone_name)
    con = pose_bone and pose_bone.constraints.get(con_name)
    if con:
      pose_bone.constraints.remove(con)
  bpy.context.view_layer.update()
  write_matrices(baked)

  # The bound bones now hold the pose. Clear what was typed into deform
  # bones that Copy Transforms fully determines, so nothing is applied twice.
  for bone_name, con_name, enabled, _ in state['muted']:
    pose_bone = rig.pose.bones.get(bone_name)
    con = pose_bone and pose_bone.constraints.get(con_name)
    if con and enabled and con.type == 'COPY_TRANSFORMS' \
        and con.mix_mode == 'REPLACE' and con.influence == 1.0:
      pose_bone.matrix_basis = Matrix.Identity(4)

  for bone_name, con_name, enabled, subtarget in state['muted']:
    pose_bone = rig.pose.bones.get(bone_name)
    con = pose_bone and pose_bone.constraints.get(con_name)
    if con:
      if subtarget is not None:
        con.subtarget = subtarget
      con.enabled = enabled

  del rig[BIND_STATE_KEY]
  bpy.context.view_layer.update()


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
      'length': np.array([b.length for b in bones]),
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
    self.obj.data.vertices.foreach_set('co', local.astype(np.float32).ravel())
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


############################################################
# Inputs

def base_world(obj):
  co = np.empty(len(obj.data.vertices) * 3)
  obj.data.vertices.foreach_get('co', co)
  M = np.array(obj.matrix_world)
  return co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]


def armature_of(obj):
  for mod in obj.modifiers:
    if mod.type == 'ARMATURE' and mod.object and mod.show_viewport:
      return mod.object
  return None


def wrap_snapshot(collection):
  coords = np.empty(len(collection) * 3)
  collection.foreach_get('coord', coords)
  return coords.reshape(-1, 3)


def get_inputs(context):
  """
  (mesh, rig, rest shape, target shape), all world space
  """
  if TARGET == 'WRAP':
    ws = getattr(context.scene, 'wrap_options', None)
    if ws is None:
      raise RuntimeError("TARGET = 'WRAP' needs the Wrap add-on")
    obj = bpy.data.objects.get(ws.source_mesh_name)
    if not obj:
      raise RuntimeError("The Wrap add-on has no source mesh")

    stage = ws.wrap_process_stage
    if stage >= 4:
      name, result = 'Stick', ws.source_stick_vertices
    elif stage == 3:
      name, result = 'Attract', ws.source_attract_vertices
    elif stage == 2:
      name, result = 'Snap', ws.source_snap_vertices
    else:
      raise RuntimeError("Run at least Wrap Snap first")

    # The fit starts from the original shape
    if ws.source_mesh_mode != 'ORIGINAL':
      ws.source_mesh_mode = 'ORIGINAL'
    log(f"target: Wrap {name} result of '{obj.name}'")
    rest, target = wrap_snapshot(ws.source_original_vertices), wrap_snapshot(result)
  else:
    obj = context.object
    target_obj = bpy.data.objects.get(TARGET)
    if not obj or obj.type != 'MESH':
      raise RuntimeError("Select the skinned mesh")
    if not target_obj or target_obj.type != 'MESH':
      raise RuntimeError(f"No mesh object named '{TARGET}'")
    log(f"target: mesh object '{target_obj.name}'")
    rest, target = base_world(obj), base_world(target_obj)

  rig = armature_of(obj)
  if not rig:
    raise RuntimeError(f"'{obj.name}' has no armature modifier with an armature")
  n = len(obj.data.vertices)
  if len(rest) != n or len(target) != n:
    raise RuntimeError("The target does not have the same vertices as the mesh")
  return obj, rig, rest, target


############################################################
# State (for UNDO)

def save_state(obj, rig):
  pose = {pb.name: [v for row in pb.matrix_basis for v in row]
          for pb in rig.pose.bones}
  rig[STATE_KEY] = json.dumps({'mesh': obj.name, 'pose': pose})

  attrs = obj.data.attributes
  if attrs.get(BASE_ATTRIBUTE):
    attrs.remove(attrs[BASE_ATTRIBUTE])
  attr = attrs.new(BASE_ATTRIBUTE, 'FLOAT_VECTOR', 'POINT')
  co = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
  obj.data.vertices.foreach_get('co', co)
  attr.data.foreach_set('vector', co)


def undo(context):
  obj = context.object
  rig = obj if obj and obj.type == 'ARMATURE' else (armature_of(obj) if obj else None)
  if not rig and TARGET == 'WRAP' and getattr(context.scene, 'wrap_options', None):
    mesh = bpy.data.objects.get(context.scene.wrap_options.source_mesh_name)
    rig = mesh and armature_of(mesh)
  if not rig or not rig.get(STATE_KEY):
    log("Nothing to undo")
    return

  bind_restore(rig)  # in case FIT stopped while swapped

  state = json.loads(rig[STATE_KEY])
  for name, m in state['pose'].items():
    pose_bone = rig.pose.bones.get(name)
    if pose_bone:
      pose_bone.matrix_basis = Matrix([m[0:4], m[4:8], m[8:12], m[12:16]])

  mesh = bpy.data.objects.get(state['mesh'])
  attr = mesh and mesh.data.attributes.get(BASE_ATTRIBUTE)
  if attr:
    co = np.empty(len(mesh.data.vertices) * 3, dtype=np.float32)
    attr.data.foreach_get('vector', co)
    mesh.data.vertices.foreach_set('co', co)
    mesh.data.attributes.remove(attr)
    mesh.data.update()

  del rig[STATE_KEY]
  bpy.context.view_layer.update()
  log(f"Undone: pose of '{rig.name}' and base shape of '{state['mesh']}' restored")


############################################################
# Fit

def fit(context):
  obj, rig, rest, target = get_inputs(context)
  if rig.get(STATE_KEY):
    log("Already fitted - run UNDO first")
    return

  start = time.time()
  save_state(obj, rig)

  # Start from the rest pose and the rest shape
  for pose_bone in rig.pose.bones:
    pose_bone.matrix_basis = Matrix.Identity(4)
  fitter = Fitter(obj, rig, rest, target)
  fitter.set_base_world(rest)
  log(f"'{rig.name}': {len(fitter.names)} deform bones, "
      f"{len(rest)} vertices")

  # 1-2. Pose, handed to the bones that drive the deform bones
  bind_swap(rig)
  try:
    R, t = fitter.fit_pose()
    fitter.place(R, t)
  finally:
    bind_restore(rig)
  pose_err = np.linalg.norm(fitter.mesh_world() - target, axis=1) * 1000
  log(f"pose only: {pose_err.mean():.2f} mm mean, {pose_err.max():.1f} mm max")

  # 3. Base shape
  err = fitter.bake() * 1000
  log(f"pose + base shape: {err.mean():.4f} mm mean, {err.max():.4f} mm max, "
      f"{np.count_nonzero(err > 1.0)} vertices off by more than 1 mm")
  log(f"done in {time.time() - start:.0f} s (MODE = 'UNDO' puts everything back)")


############################################################
# Script

def main():
  if MODE == 'FIT':
    fit(bpy.context)
  elif MODE == 'UNDO':
    undo(bpy.context)
  else:
    log(f"Unknown MODE {MODE!r}")


if __name__ == "__main__":
  main()
