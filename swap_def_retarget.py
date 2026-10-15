# Swap DEF constraints (retarget version)
#
# Standalone script (Blender Text Editor, Blender 4.x / 5.x). Lets the deform
# bones be posed directly, then hands the pose to the rig's main controls.
#
#   SWAP     mute every constraint on every deform bone, keep the pose, and
#            bind the bones they follow with temporary constraints, so the
#            rig follows the deform bones live while you pose them. Main
#            controls that move rigidly with a deform bone (IK hand and foot
#            targets, torso, ...) ride along with it live too.
#   RESTORE  retarget: solve the main controls (IK targets and poles, torso,
#            fingers, ...) so the rig reproduces the deform pose through its
#            own constraints, on the active side of IK/FK style switches (snap
#            the other side with the rig's own IK/FK snapping, or set
#            RT_OTHER_SIDE = True); put only the leftover on secondary bones
#            (tweaks); unmute the original constraints exactly as they were.
#            The deform bones end up where you left them. Works the same
#            whatever the rig's units, position or scale.
#
# Nothing is specific to Rigify: main controls are posable bones with a shape
# that no constraint drives, switches are custom properties that drive Copy
# Transforms influences, and the solve uses Blender's own evaluation.
#
# Usage: select the armature, set MODE below, then Run Script.
#   1. MODE = 'SWAP', Run Script. Show the deform bones' bone collection if it
#      is hidden, then pose the deform bones. The rig follows them live, IK
#      targets included; that live view is a preview, RESTORE does the solve.
#   2. MODE = 'RESTORE', Run Script. Takes about a minute on a full body
#      rig (Blender is busy meanwhile); progress and results are printed to the
#      system console (Window > Toggle System Console).

import json
import math
import re

import bpy
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

MODE = 'SWAP'  # 'SWAP' | 'RESTORE'

MAX_DEPTH = 8    # constraint chain length followed from a deform bone
MAX_PASSES = 6   # settling passes when writing bone matrices
EPS = 1e-5

STATE_KEY = 'swap_def_bind'
BIND_PREFIX = 'SwapDEF '
LIVE_MAIN = True  # main controls (IK targets, ...) ride along live while swapped


############################################################
# Helpers

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


def matrix_delta(a, b, size=0.1):
  """
  Largest difference between two matrices: the rotation part as is, the
  movement in tenths of `size` (a rig size, see rig_size), so rigs in any
  unit compare the same. The default compares raw armature units.
  """
  turn = max(abs(a[i][j] - b[i][j]) for i in range(3) for j in range(3))
  move = max(abs(a[i][3] - b[i][3]) for i in range(3)) / (10.0 * size)
  return max(turn, move)


def write_matrices(items):
  """
  items: [(pose_bone, pose_space_matrix)], written parents first so each
  PoseBone.matrix setter sees a fresh parent matrix. Repeated until nothing
  moves: a bone's parent can also depend on a sibling through a constraint
  (Rigify's MCH tweak parents), which one ordered pass does not cover.
  """
  size = rig_size(items[0][0].id_data) if items else 0.1
  by_depth = {}
  for pose_bone, matrix in items:
    by_depth.setdefault(depth(pose_bone), []).append((pose_bone, matrix))
  for _ in range(MAX_PASSES):
    changed = 0.0
    for level in sorted(by_depth):
      for pose_bone, matrix in by_depth[level]:
        delta = matrix_delta(pose_bone.matrix, matrix, size)
        if delta > EPS:
          pose_bone.matrix = matrix
          changed = max(changed, delta)
      bpy.context.view_layer.update()
    if changed <= EPS:
      break


############################################################
# Binder
#
# Walks from a deform bone to the free bones that place it and records one
# binding per (bone, kind): ('FULL' | 'HEAD' | 'TAIL' | 'SCALE', deform name)

class Binder:
  def __init__(self, rig, deform_names):
    self.rig = rig
    self.deform_names = deform_names
    self.binds = {}   # bone name -> {kind: deform bone name}
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
    """
    bone should follow deform bone `source` (fully, head only, head on the
    source's tail, or scale only)
    """
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

    if level >= MAX_DEPTH:
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
      # Copy Location sets the head, Stretch To aims at the tail, nothing
      # sets the roll. Stretch To keeps the owner's roll, so bind the bone
      # itself ahead of its own constraints: it carries the roll.
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
    if matrix_delta(offset, Matrix.Identity(4), rig_size(self.rig)) > EPS:
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


def resolve_all(rig, state):
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


############################################################
# Temporary constraints

def add_binding(rig, bone, kind, source):
  if kind == 'FULL':
    # Location and rotation only: a deform bone's stretch scale comes from a
    # Stretch To and is rebuilt from the TAIL bindings. Copying it here would
    # scale the follower and stretch its children a second time.
    con = bone.constraints.new('COPY_ROTATION')
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
  # Ahead of the bone's own constraints so they still run after it: a
  # mechanism bone bound for its roll keeps its Copy Location / Stretch To
  bone.constraints.move(len(bone.constraints) - 1, 0)
  return con


def snap(rig, state):
  """
  Nothing to do: the bindings are live constraints. Kept so the same
  harness drives both scripts.
  """
  bpy.context.view_layer.update()


############################################################
# Swap

def swap(rig, quiet=False):
  if rig.get(STATE_KEY):
    print("Already swapped - run RESTORE first")
    return

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
  binder = resolve_all(rig, state)

  # Main controls that ride along with a deform bone while swapped (e.g. an
  # IK hand target with the hand), worked out while the rig is still intact
  follow = live_follow_pairs(rig, muted, set(binder.binds)) if LIVE_MAIN else {}

  # Remember where the bound bones are, RESTORE puts them back before retargeting
  state['basis'] = {name: [v for row in rig.pose.bones[name].matrix_basis for v in row]
                    for name in binder.binds}

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
      if kind == 'FULL':
        loc = add_binding(rig, bone, 'HEAD', kinds[kind])
        loc.name = f'{BIND_PREFIX}FULL location'
        state['bound'].append([bone_name, loc.name])

  rig[STATE_KEY] = json.dumps(state)
  bpy.context.view_layer.update()

  # The bindings can move bones that deform bones are parented under (e.g. a
  # spine tweak that carries the shoulders). Put the deform bones back where
  # they were, now that the bindings are in place.
  write_matrices(held)

  # Live following of the main controls: a Child Of with its offset frozen,
  # so nothing jumps. Skipped where it would make a dependency cycle.
  live = []
  for name, def_name in follow.items():
    pose_bone = rig.pose.bones[name]
    target = rig.pose.bones[def_name]
    if name in dependencies(rig, target):
      continue
    con = pose_bone.constraints.new('CHILD_OF')
    con.name = f'{BIND_PREFIX}live'
    con.target = rig
    con.subtarget = def_name
    # Blender applies a bone's Child Of inverse with the armature object's
    # transform already taken out: the target's own pose-space inverse
    con.inverse_matrix = target.matrix.inverted()
    con.show_expanded = False
    pose_bone.constraints.move(len(pose_bone.constraints) - 1, 0)
    state['bound'].append([name, con.name])
    live.append(f"{name} -> {def_name}")
  rig[STATE_KEY] = json.dumps(state)
  bpy.context.view_layer.update()
  write_matrices(held)

  if quiet:
    return
  print(f"Muted {len(muted)} constraints on {len({m[0] for m in muted})} "
        f"deform bones, {len(binder.binds)} bones follow them through "
        f"{len(state['bound'])} temporary constraints")
  print(f"Main controls following live ({len(live)}): {', '.join(live)}")
  for line in sorted(binder.limits):
    print(f"Limited {line}")


############################################################
# Live following of the main controls

def live_follow_pairs(rig, muted, already_bound):
  """
  For each main control, the deform bone that moves rigidly with it: nudge
  the control and compare its own motion with every deform bone's motion
  """
  deform = [pb for pb in rig.pose.bones if pb.bone.use_deform]
  direct = {subtarget for _, _, _, subtarget in muted if subtarget}
  controls = [pb for pb in main_controls(rig, direct)
              if pb.name not in already_bound]
  nudge = Matrix.Rotation(0.05, 4, 'X') @ Matrix.Rotation(0.03, 4, 'Z')
  size = rig_size(rig)

  bpy.context.view_layer.update()
  before = {pb.name: pb.matrix.copy() for pb in deform}
  pairs = {}
  for pb in controls:
    basis = pb.matrix_basis.copy()
    c0 = pb.matrix.copy()
    pb.matrix_basis = basis @ nudge
    bpy.context.view_layer.update()
    moved_c = pb.matrix @ c0.inverted()

    best = None
    for d in deform:
      moved_d = d.matrix @ before[d.name].inverted()
      if matrix_delta(moved_d, Matrix.Identity(4), size) < 1e-6:
        continue
      key = (round(matrix_delta(moved_d, moved_c, size), 4),
             (d.bone.head_local - pb.bone.head_local).length)
      if best is None or key < best[0]:
        best = (key, d.name)

    pb.matrix_basis = basis
    if best and best[0][0] < 0.02:  # moves (almost) rigidly with the control
      pairs[pb.name] = best[1]
  bpy.context.view_layer.update()
  return pairs


def dependencies(rig, pose_bone):
  """
  Every bone this bone's pose depends on: parents and constraint targets
  """
  out, stack = set(), [pose_bone]
  while stack:
    pb = stack.pop()
    nxt = [pb.parent] if pb.parent else []
    for c in pb.constraints:
      if not c.enabled or (c.influence == 0.0 and not is_driven(rig, pb, c)):
        continue
      for owner, sub in ((getattr(c, 'target', None), getattr(c, 'subtarget', '')),
                         (getattr(c, 'pole_target', None), getattr(c, 'pole_subtarget', ''))):
        if owner == rig and sub in rig.pose.bones:
          nxt.append(rig.pose.bones[sub])
      for t in getattr(c, 'targets', []):
        if t.target == rig and t.subtarget in rig.pose.bones:
          nxt.append(rig.pose.bones[t.subtarget])
    for b in nxt:
      if b.name not in out:
        out.add(b.name)
        stack.append(b)
  return out


############################################################
# Bind helpers

def remove_bindings(rig, state):
  for bone_name, con_name in state['bound']:
    pose_bone = rig.pose.bones.get(bone_name)
    con = pose_bone and pose_bone.constraints.get(con_name)
    if con:
      pose_bone.constraints.remove(con)


def unmute(rig, state):
  # Clear what was typed into deform bones that Copy Transforms fully
  # determines, so nothing is applied twice
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

  del rig[STATE_KEY]
  bpy.context.view_layer.update()


############################################################
# Retarget: the deform pose goes onto the main controls
#
# Main controls: posable bones with a shape that no constraint drives. A bone
# a deform bone follows directly, riding on driven bones (a tweak), is
# secondary: it only gets what the main controls cannot reach.
#
# The main controls are solved against Blender's own evaluation of the rig
# (constraints, IK solver, B-bones), so no constraint has to be inverted by
# hand. Switches that pick between Copy Transforms sources (IK/FK) are
# flipped and the other side solved too, so the pose holds in every mode.

RT_ITERATIONS = 120
# Distances are measured in rig sizes (the mean deform bone length), so a rig
# built in meters, centimeters or anything else is posed the same way
RT_STEP = 1e-3          # finite-difference step for turning: radians
RT_MOVE_STEP = 0.01     # finite-difference step for moving: rig sizes
RT_REGULARIZE = 0.04    # preference for the smallest change of the controls
RT_MOVE_WEIGHT = 0.87   # moving 1.15 rig sizes weighs like turning 1 radian
RT_TOLERANCE = 1e-4     # stop when a step improves the total by less than this
RT_INFLUENCE_POWER = 1.0  # a channel's change costs more the more deform bones it moves
RT_GLOBAL_SHARE = 0.9     # a control moving (nearly) all of the deform bones...
RT_GLOBAL_GAIN = 0.05     # ...is kept only if it improves the fit by at least this much
RT_STRETCH = 0.3          # moving a control 1 unit stretches a deform bone by more: only turn it
RT_OTHER_SIDE = False     # also solve the inactive side of IK/FK switches (else snap it in the rig)


def live_constraints(pose_bone):
  return any(c.enabled and c.influence > 0.0 for c in pose_bone.constraints)


def main_controls(rig, direct_targets):
  out = []
  for pb in rig.pose.bones:
    if pb.bone.use_deform or pb.parent is None or pb.bone.hide \
        or not pb.custom_shape or live_constraints(pb):
      continue
    if pb.name in direct_targets \
        and any(live_constraints(a) for a in pb.parent_recursive):
      continue  # secondary: rides on driven bones (e.g. a tweak)
    out.append(pb)
  return out


def deform_features(rig, names):
  """
  Head, tail and a point along the X axis of every deform bone: position,
  direction, length and roll
  """
  bpy.context.view_layer.update()
  out = np.empty((len(names), 3, 3))
  for i, name in enumerate(names):
    pb = rig.pose.bones[name]
    m = pb.matrix
    head = m.translation
    out[i, 0] = head
    out[i, 1] = pb.tail
    out[i, 2] = head + m.col[0].xyz.normalized() * pb.bone.length
  return out.reshape(-1)


class Channels:
  """
  The free transform channels of a set of bones (locks respected), as a
  parameter vector of changes from where they started
  """

  def __init__(self, bones, respect_locks=True, turn_only=frozenset()):
    self.bones = bones
    self.base = []
    self.index = []
    for i, pb in enumerate(bones):
      if pb.rotation_mode == 'AXIS_ANGLE':
        self.base.append(None)
        continue
      rot = (pb.rotation_quaternion.copy() if pb.rotation_mode == 'QUATERNION'
             else pb.rotation_euler.copy())
      self.base.append((pb.location.copy(), rot))
      if not pb.bone.use_connect and pb.name not in turn_only:
        for a in range(3):
          if not (respect_locks and pb.lock_location[a]):
            self.index.append((i, 'L', a))
      for a in range(3):
        if not (respect_locks and pb.lock_rotation[a]):
          self.index.append((i, 'R', a))
    self.p = np.zeros(len(self.index))
    self.columns = {}
    for col, (i, _, _) in enumerate(self.index):
      self.columns.setdefault(i, []).append(col)

  def key(self, col):
    i, kind, a = self.index[col]
    return (self.bones[i].name, kind, a)

  def apply_bone(self, i, p):
    pb = self.bones[i]
    if self.base[i] is None:
      return
    loc, rot = Vector(), Vector()
    for col in self.columns.get(i, []):
      _, kind, a = self.index[col]
      (loc if kind == 'L' else rot)[a] = p[col]
    base_loc, base_rot = self.base[i]
    pb.location = base_loc + loc
    if pb.rotation_mode == 'QUATERNION':
      angle = rot.length
      delta = Quaternion(rot / angle, angle) if angle > 1e-12 else Quaternion()
      pb.rotation_quaternion = base_rot @ delta
    else:
      pb.rotation_euler = Euler((base_rot[0] + rot[0], base_rot[1] + rot[1],
                                 base_rot[2] + rot[2]), pb.rotation_mode)

  def apply(self, p):
    for i in range(len(self.bones)):
      self.apply_bone(i, p)


def feature_weights(rig, names):
  """
  A deform bone without constraints is posed directly afterwards: only where
  its parent puts it (its head) is the controls' job, not its rotation
  """
  fw = np.ones((len(names), 3, 3))
  for i, name in enumerate(names):
    if not live_constraints(rig.pose.bones[name]):
      fw[i, 1:] = 0.0
  return fw.reshape(-1)


def fit_rms(rig, names, target):
  r = (deform_features(rig, names) - target) * feature_weights(rig, names)
  return float(np.sqrt((r @ r) / (len(target) // 3)))


def mm(rig):
  """
  Armature units to world millimeters, for the log
  """
  return 1000.0 * sum(rig.matrix_world.to_scale()) / 3.0


def rig_size(rig):
  """
  Mean rest length of the deform bones, in armature units: the unit the
  solver measures distances in
  """
  lengths = [b.length for b in rig.data.bones if b.use_deform]
  return float(np.mean(lengths)) if lengths else 1.0


def moved_bones(rig, controls, names):
  """
  For each control, the deform bones whose matrix changes when the control
  is turned a little
  """
  bpy.context.view_layer.update()
  before = {n: rig.pose.bones[n].matrix.copy() for n in names}
  nudge = Matrix.Rotation(0.05, 4, 'X') @ Matrix.Rotation(0.05, 4, 'Z')
  size = rig_size(rig)
  out = {}
  for pb in controls:
    basis = pb.matrix_basis.copy()
    pb.matrix_basis = basis @ nudge
    bpy.context.view_layer.update()
    out[pb.name] = {n for n in names
                    if matrix_delta(rig.pose.bones[n].matrix, before[n], size) > 1e-6}
    pb.matrix_basis = basis
  bpy.context.view_layer.update()
  return out


def pivot_helpers(controls, moved):
  """
  A normal child control moves a subset of the deform bones its nearest
  ancestor control moves (a shoulder under the spine, a phalanx under the
  previous one). A child that moves anything its ancestor does not pivots
  from somewhere else (e.g. heel / ball pivots under an IK foot): an
  alternative way to do the ancestor's job. Helpers stay where they are.
  """
  names = {pb.name for pb in controls}
  out = set()
  # Parents first, each compared with its nearest ancestor control that is
  # not itself a helper (helpers can hang under helpers)
  for pb in sorted(controls, key=lambda b: len(b.parent_recursive)):
    if not moved[pb.name]:
      continue
    ancestor = next((a for a in pb.parent_recursive
                     if a.name in names and a.name not in out), None)
    if ancestor and moved[ancestor.name] \
        and not moved[pb.name] <= moved[ancestor.name]:
      out.add(pb.name)
  return out


def stretch_handles(rig, controls, names, carriers):
  """
  Controls whose movement stretches a deform bone (a toe or a finger joint
  pulled off its parent) rather than carrying the bones along (IK targets, a
  torso), under a carrier control that can move them instead (the IK foot,
  the previous phalanx). They are only turned, the way an animator poses
  them. Without a carrier above (a chest under a held whole-body torso),
  moving is the only way to place them.
  """
  bpy.context.view_layer.update()

  def lengths():
    return np.array([(rig.pose.bones[n].tail - rig.pose.bones[n].head).length
                     for n in names])

  rest = lengths()
  step = RT_MOVE_STEP * rig_size(rig)
  out = set()
  for pb in controls:
    if pb.bone.use_connect \
        or not any(a.name in carriers for a in pb.parent_recursive):
      continue
    loc = pb.location.copy()
    for a in range(3):
      if pb.lock_location[a]:
        continue
      d = Vector()
      d[a] = step
      pb.location = loc + d
      bpy.context.view_layer.update()
      stretch = np.abs(lengths() - rest).max() / step
      pb.location = loc
      if stretch > RT_STRETCH:
        out.add(pb.name)
        break
  bpy.context.view_layer.update()
  return out


def solve_controls(rig, channels, names, target, label, skip=frozenset()):
  """
  Damped Gauss-Newton on the channels against Blender's evaluation.
  Finite-difference Jacobian: one full pass to learn which deform bones each
  channel moves, then channels that move different bones share one
  evaluation (refreshed with a full pass every few iterations). Channels that
  move no deform bone are dropped.
  """
  # Residuals and movements in rig sizes
  size = rig_size(rig)
  fw = feature_weights(rig, names) / size

  def residual():
    return (deform_features(rig, names) - target) * fw

  step = np.array([RT_MOVE_STEP * size if kind == 'L' else RT_STEP
                   for _, kind, _ in channels.index])

  def nudge(cols):
    p = channels.p.copy()
    for col in cols:
      p[col] += step[col]
    for i in {channels.index[col][0] for col in cols}:
      channels.apply_bone(i, p)
    r_nudged = residual()
    for i in {channels.index[col][0] for col in cols}:
      channels.apply_bone(i, channels.p)
    return r_nudged

  # Smallest change: mu * sum (w_c p_c)^2, rotations in radians, movements
  # in rig sizes weighted by RT_MOVE_WEIGHT
  w2 = np.array([(RT_MOVE_WEIGHT / size if kind == 'L' else 1.0) ** 2
                 for _, kind, _ in channels.index])

  def total(r_vec, p):
    return r_vec @ r_vec + RT_REGULARIZE * float((w2 * p * p).sum())

  n_points = len(target) // 3
  r = residual()
  start_rms = np.sqrt((r @ r) / n_points) * size
  cost = total(r, channels.p)
  keep, support, groups = None, {}, []
  weighted = False
  lam = 1e-3
  iterations = 0
  evaluations = 0
  full_next = True

  for it in range(RT_ITERATIONS):
    iterations += 1
    was_full = full_next or keep is None
    full_next = False
    if was_full:
      cols = range(len(channels.index)) if keep is None else keep
      J = np.zeros((len(r), len(channels.index)))
      for col in cols:
        J[:, col] = (nudge([col]) - r) / step[col]
        evaluations += 1
      if keep is None:
        keep = [c for c in range(len(channels.index))
                if np.abs(J[:, c]).max() > 1e-7 and channels.key(c) not in skip]
      # Which deform bones each channel moves, then greedy grouping of
      # channels that move disjoint sets of bones
      support = {c: np.nonzero(np.abs(J[:, c]) > 1e-9)[0] for c in keep}
      if not weighted:
        # A control that moves much of the body (e.g. a torso) has to be
        # justified by much of the body: its change costs in proportion to
        # the number of deform bones it moves
        for c in keep:
          moved = len(np.unique(support[c] // 9))
          w2[c] *= max(moved, 1) ** RT_INFLUENCE_POWER
        weighted = True
        cost = total(r, channels.p)
      groups, used = [], []
      for c in sorted(keep, key=lambda c: -len(support[c])):
        bones_c = set((support[c] // 9).tolist())
        for g, u in zip(groups, used):
          if not (u & bones_c) and all(channels.index[c][0] != channels.index[x][0] for x in g):
            g.append(c)
            u |= bones_c
            break
        else:
          groups.append([c])
          used.append(set(bones_c))
    else:
      J = np.zeros((len(r), len(channels.index)))
      for g in groups:
        r_nudged = nudge(g)
        evaluations += 1
        for c in g:
          rows = support[c]
          J[rows, c] = (r_nudged[rows] - r[rows]) / step[c]

    Jk = J[:, keep]
    A = Jk.T @ Jk + RT_REGULARIZE * np.diag(w2[keep])
    g_vec = Jk.T @ r + RT_REGULARIZE * w2[keep] * channels.p[keep]

    improved = False
    for _ in range(10):
      step = np.zeros(len(channels.p))
      step[keep] = np.linalg.solve(A + lam * np.diag(np.diag(A)), -g_vec)
      channels.apply(channels.p + step)
      r_new = residual()
      evaluations += 1
      cost_new = total(r_new, channels.p + step)
      if cost_new < cost:
        channels.p += step
        r, previous, cost = r_new, cost, cost_new
        lam = max(lam / 3.0, 1e-9)
        improved = True
        break
      lam *= 5.0
    channels.apply(channels.p)
    if np.sqrt((r @ r) / n_points) < 1e-6:
      break
    if not improved or previous - cost < RT_TOLERANCE * previous:
      # A grouped Jacobian can be off after large moves: retry with the full
      # one before giving up
      if was_full:
        break
      full_next = True

  r = residual()
  end_rms = np.sqrt((r @ r) / n_points) * size
  print(f"  {label}: {len(keep)} channels on {len({channels.index[c][0] for c in keep})} "
        f"bones, deform error {start_rms * mm(rig):.2f} -> {end_rms * mm(rig):.3f} mm rms "
        f"({iterations} iterations, {evaluations} evaluations, {len(groups)} groups)")
  return {channels.key(c) for c in keep}


class MeshesOff:
  """
  Meshes deformed by the rig skip their armature modifier while solving
  (faster evaluations), switched back on afterwards in any case
  """

  def __init__(self, rig):
    self.mods = [m for o in bpy.data.objects if o.type == 'MESH'
                 for m in o.modifiers
                 if m.type == 'ARMATURE' and m.object == rig and m.show_viewport]

  def __enter__(self):
    for m in self.mods:
      m.show_viewport = False

  def __exit__(self, *exc):
    for m in self.mods:
      m.show_viewport = True
    bpy.context.view_layer.update()


def deform_error(rig, names, wanted):
  bpy.context.view_layer.update()
  errors = [((rig.pose.bones[n].matrix.translation - wanted[n].translation).length, n)
            for n in names]
  e = np.array([x for x, _ in errors])
  worst = max(errors)
  return (f"{np.sqrt(np.mean(e ** 2)) * mm(rig):.3f} mm rms, "
          f"worst {worst[0] * mm(rig):.2f} mm ({worst[1]})")


def copy_switches(rig):
  """
  Custom properties that drive Copy Transforms influences (e.g. IK/FK)
  """
  out = set()
  if not rig.animation_data:
    return []
  for d in rig.animation_data.drivers:
    if '.constraints[' not in d.data_path or not d.data_path.endswith('.influence'):
      continue
    try:
      con = rig.path_resolve(d.data_path.rsplit('.', 1)[0])
    except ValueError:
      continue
    if con.type != 'COPY_TRANSFORMS':
      continue
    for var in d.driver.variables:
      for tgt in var.targets:
        m = re.fullmatch(r'pose\.bones\["(.+?)"\]\["(.+?)"\]', tgt.data_path or '')
        if tgt.id == rig and m and m.group(1) in rig.pose.bones:
          out.add((m.group(1), m.group(2)))
  return sorted(out)


def switch_sources(rig, switches):
  """
  (owner, source) for every Copy Transforms whose influence one of the
  switches drives, e.g. (ORG-upper_arm.L, upper_arm_fk.L)
  """
  props = {f'pose.bones["{b}"]["{p}"]' for b, p in switches}
  out = []
  for d in rig.animation_data.drivers:
    if not d.data_path.endswith('.influence'):
      continue
    if not any(t.data_path in props for v in d.driver.variables for t in v.targets):
      continue
    try:
      con = rig.path_resolve(d.data_path.rsplit('.', 1)[0])
    except ValueError:
      continue
    if con.type == 'COPY_TRANSFORMS':
      owner = rig.pose.bones[d.data_path.split('"')[1]]
      # Every Copy Transforms source on this bone: the driven one and the
      # one it overrides (e.g. FK under IK)
      for c in owner.constraints:
        if c.type == 'COPY_TRANSFORMS' and c.subtarget in rig.pose.bones:
          pair = (owner, rig.pose.bones[c.subtarget])
          if pair not in out:
            out.append(pair)
  return out


def set_switch(rig, switch, value):
  rig.pose.bones[switch[0]][switch[1]] = value
  rig.update_tag()


def retarget(rig, direct_targets, names, target):
  controls = main_controls(rig, direct_targets)
  moved = moved_bones(rig, controls, names)

  # Alternative pivots of another control (heel / ball under an IK foot)
  # stay where they are
  helpers = pivot_helpers(controls, moved)
  if helpers:
    print(f"  left at rest, alternative pivots of another control: {', '.join(sorted(helpers))}")
  controls = [pb for pb in controls if pb.name not in helpers]

  # Whole-body controls (e.g. a torso) stay where they are unless the whole
  # body needs them: solve without them, then keep their contribution only
  # if it improves the fit by at least RT_GLOBAL_GAIN
  whole = [pb for pb in controls if len(moved[pb.name]) > RT_GLOBAL_SHARE * len(names)]
  local = [pb for pb in controls if pb not in whole]

  # Controls that would stretch the deform bones if moved, while a control
  # above them can carry them instead, are only turned
  turn = stretch_handles(rig, controls, names, {pb.name for pb in local})
  if turn:
    print(f"  only turned, moving them stretches the deform bones: {', '.join(sorted(turn))}")

  active =solve_controls(rig, Channels(local, turn_only=turn), names, target,
                          'main controls')
  if whole:
    # Converge without them first, so the comparison is fair (a second solve
    # always gains a little just by iterating further)
    active = solve_controls(rig, Channels(local, turn_only=turn), names, target,
                            'main controls, converged')
    start = {pb.name: pb.matrix_basis.copy() for pb in controls}
    before = fit_rms(rig, names, target)
    active_all = solve_controls(rig, Channels(controls, turn_only=turn), names, target,
                                f'with {len(whole)} whole-body controls')
    gain = 1.0 - fit_rms(rig, names, target) / max(before, 1e-12)
    listed = ', '.join(pb.name for pb in whole)
    if gain < RT_GLOBAL_GAIN:
      for pb in controls:
        pb.matrix_basis = start[pb.name]
      bpy.context.view_layer.update()
      print(f"  {listed}: improve the fit by only {gain:.1%}, left where they were")
    else:
      active = active_all
      print(f"  {listed}: improve the fit by {gain:.1%}, kept")

  # The other side of each switch (e.g. FK while IK is active); off by
  # default: snap it with the rig's own IK/FK snapping instead
  if not RT_OTHER_SIDE:
    return
  switches = [s for s in copy_switches(rig)
              if isinstance(rig.pose.bones[s[0]].get(s[1]), (int, float))
              and rig.pose.bones[s[0]][s[1]] in (0, 1, 0.0, 1.0)]
  if switches:
    values = {s: rig.pose.bones[s[0]][s[1]] for s in switches}
    # Snap: each free Copy Transforms source of the other side goes where the
    # bone that copies it is now (exact for FK chains), then flip and polish
    sources = [(o, t) for o, t in switch_sources(rig, switches)
               if t in controls]
    write_matrices([(t, o.matrix.copy()) for o, t in sources])
    for s in switches:
      set_switch(rig, s, 1 - values[s])
    # Only channels that do nothing in the active mode (e.g. the FK controls
    # while IK is active), so the active mode keeps its solution
    solve_controls(rig, Channels(controls, turn_only=turn), names, target,
                   f"other side of {len(switches)} switches", skip=active)
    for s in switches:
      set_switch(rig, s, values[s])
    bpy.context.view_layer.update()


############################################################
# Restore

def restore(rig):
  state = rig.get(STATE_KEY)
  if not state:
    print("Nothing to restore")
    return
  state = json.loads(state)

  bpy.context.view_layer.update()
  names = [pb.name for pb in rig.pose.bones if pb.bone.use_deform]
  wanted = {name: rig.pose.bones[name].matrix.copy() for name in names}
  target = deform_features(rig, names)
  direct = {subtarget for _, _, _, subtarget in state['muted'] if subtarget}

  # 1. Back to the normal rig: drop the live bindings, put the bound bones
  #    back where they were before SWAP, unmute the deform constraints
  remove_bindings(rig, state)
  for name, m in state.get('basis', {}).items():
    if name in rig.pose.bones:
      rig.pose.bones[name].matrix_basis = Matrix([m[0:4], m[4:8], m[8:12], m[12:16]])
  unmute(rig, state)

  with MeshesOff(rig):
    # 2. Main controls, and the other side of IK/FK style switches
    retarget(rig, direct, names, target)
    print(f"  deform heads after the main controls: {deform_error(rig, names, wanted)}")

    # 3. The leftover goes onto the bones the deform bones follow (tweaks,
    #    ...), solved the same way, smallest change first
    before = {pb.name: pb.matrix_basis.copy() for pb in rig.pose.bones}
    now = {'muted': [[pb.name, c.name, c.enabled, getattr(c, 'subtarget', None)]
                     for pb in rig.pose.bones if pb.bone.use_deform
                     for c in pb.constraints]}
    followers = [rig.pose.bones[n] for n in resolve_all(rig, now).binds]
    solve_controls(rig, Channels(followers), names, target,
                   'leftover on the bones the deform bones follow')

    # Deform bones without constraints are posed directly
    write_matrices([(rig.pose.bones[n], wanted[n]) for n in names
                    if not live_constraints(rig.pose.bones[n])])

  print(f"  deform heads after RESTORE: {deform_error(rig, names, wanted)}")
  moved = []
  for pb in rig.pose.bones:
    if pb.bone.use_deform:
      continue
    d = before[pb.name].inverted() @ pb.matrix_basis
    loc = d.translation.length
    ang = math.degrees(d.to_quaternion().angle)
    if loc > 1e-5 or ang > 1e-3:
      moved.append((loc, min(ang, 360 - ang), pb.name))
  print(f"Restored. Leftover on {len(moved)} bones"
        + (", largest: " + ", ".join(f"{n} {l * mm(rig):.1f} mm / {a:.1f} deg"
                                     for l, a, n in sorted(moved, reverse=True)[:5])
           if moved else ""))


############################################################
# Script

def main(mode):
  rig = bpy.context.object
  if not rig or rig.type != 'ARMATURE':
    print("Select the armature first")
    return
  if mode == 'SWAP':
    swap(rig)
  elif mode == 'RESTORE':
    restore(rig)
  else:
    print(f"Unknown MODE {mode!r}")


if __name__ == "__main__":
  main(MODE)
