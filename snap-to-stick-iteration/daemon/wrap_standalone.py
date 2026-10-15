# Standalone wrap tester - NO server/client, NO Blender.
#
# Drives wrap_solver.py directly with an interactive matplotlib UI that
# mirrors the add-on workflow:
#
#   Init -> pick point pairs -> Snap -> Attract -> Stick -> Original (reset)
#
# Usage:
#   python wrap_standalone.py                      # demo spheres
#   python wrap_standalone.py source.obj target.obj  # your own meshes (triangulated OBJ)
#   python wrap_standalone.py --selftest           # headless smoke test
#
# Requirements (same as the daemon): numpy, scipy, matplotlib
#   pip install numpy scipy matplotlib
#
# Point pairs: click "Pick Pair", then click a vertex on the SOURCE (blue),
# then the matching vertex on the TARGET (gray). Repeat for more pairs.
# Keep wrap_solver.py in the same folder.

import sys

import numpy as np

from wrap_solver import WrapSession, WRAP_SOLVER_VERSION

# Panel presets (Medium, matching the add-on)
PRESET = {
    'init_flex': 100.0, 'init_accuracy': 3.0,
    'snap_force': 7.0, 'snap_softness': 2.0,
    'attract_distance_pct': 1.0, 'attract_angle': 30.0,
    'stick_distance_pct': 0.5, 'stick_angle': 50.0,
    'snap_iters': 20, 'attract_iters': 30, 'stick_iters': 10,
}


############################################################
# Mesh helpers

def make_uv_sphere(r, nu=32, nv=24, scale=(1, 1, 1), offset=(0, 0, 0)):
  verts = [np.array([0, 0, r * scale[2]]) + np.array(offset)]
  for i in range(1, nv):
    theta = np.pi * i / nv
    for j in range(nu):
      phi = 2 * np.pi * j / nu
      verts.append(np.array([
          r * np.sin(theta) * np.cos(phi) * scale[0],
          r * np.sin(theta) * np.sin(phi) * scale[1],
          r * np.cos(theta) * scale[2]]) + np.array(offset))
  verts.append(np.array([0, 0, -r * scale[2]]) + np.array(offset))
  verts = np.array(verts)

  faces = []
  def ring(i, j):
    return 1 + (i - 1) * nu + (j % nu)
  for j in range(nu):
    faces.append([0, ring(1, j), ring(1, j + 1)])
  for i in range(1, nv - 1):
    for j in range(nu):
      a, b = ring(i, j), ring(i, j + 1)
      c, d = ring(i + 1, j + 1), ring(i + 1, j)
      faces.append([a, b, c])
      faces.append([a, c, d])
  bot = len(verts) - 1
  for j in range(nu):
    faces.append([bot, ring(nv - 1, j + 1), ring(nv - 1, j)])
  return verts, faces


def load_obj(path):
  """Minimal OBJ loader (v / f, triangulates polygons as fans)."""
  verts, faces = [], []
  with open(path) as f:
    for line in f:
      if line.startswith('v '):
        verts.append([float(x) for x in line.split()[1:4]])
      elif line.startswith('f '):
        ids = [int(p.split('/')[0]) - 1 for p in line.split()[1:]]
        for k in range(1, len(ids) - 1):
          faces.append([ids[0], ids[k], ids[k + 1]])
  return np.asarray(verts, dtype=np.float64), faces


def demo_meshes():
  """Bumpy offset target so the full pipeline is exercised."""
  sv, sf = make_uv_sphere(1.0, scale=(0.5, 0.4, 1.0))
  tv, tf = make_uv_sphere(1.0, scale=(0.62, 0.5, 1.05), offset=(1.8, 0.3, 0.1))
  c = tv - np.array([1.8, 0.3, 0.1])
  r = np.linalg.norm(c, axis=1, keepdims=True)
  bump = 0.05 * np.sin(4 * c[:, 0:1]) * np.cos(5 * c[:, 1:2])
  tv = np.array([1.8, 0.3, 0.1]) + c / r * (r + bump)
  return sv, sf, tv, tf


############################################################
# Wrap driver (same call sequence the add-on makes, minus sockets)

class WrapDriver:

  def __init__(self, sv, sf, tv, tf, symmetric=False):
    self.sv0 = sv.copy()
    self.sf, self.tv, self.tf = sf, tv, tf
    self.session = WrapSession()
    self.symmetric = symmetric
    self.pairs = []          # list of (source_vertex_index, target_position)
    self.init()

  def init(self):
    flex = PRESET['init_flex']
    self.session.init({
        'source_coords': self.sv0.tolist(),
        'source_tris': self.sf, 'source_quads': [],
        'target_coords': self.tv.tolist(),
        'target_tris': self.tf, 'target_quads': [],
        'alpha': flex, 'beta': flex * PRESET['init_accuracy'],
        'symmetric': 1 if self.symmetric else 0,
    })
    print(f"[standalone] init ok - solver {WRAP_SOLVER_VERSION}, "
          f"{len(self.sv0)} src verts, {len(self.tf)} tgt tris")

  def reset(self):
    self.pairs.clear()
    self.session = WrapSession()
    self.init()

  def _pp(self):
    idx = [p[0] for p in self.pairs]
    pos = [list(p[1]) for p in self.pairs]
    return idx, pos

  def snap(self):
    idx, pos = self._pp()
    if not idx:
      print("[standalone] snap: pick at least one point pair first")
      return iter(())
    return self.session.snap_iterations({
        'pp_source_index': idx, 'pp_target_pos': pos,
        'force': PRESET['snap_force'],
        'softness': PRESET['snap_softness'],
        'iter_num': PRESET['snap_iters'],
    })

  def attract(self):
    d = PRESET['attract_distance_pct'] * 0.01 * self.session.diag
    return self.session.attract_iterations({
        'attract_distance': d,
        'attract_angle': PRESET['attract_angle'],
        'iter_num': PRESET['attract_iters'],
    })

  def stick(self):
    idx, pos = self._pp()
    d = PRESET['stick_distance_pct'] * 0.01 * self.session.diag
    return self.session.stick_iterations({
        'pp_source_index': idx, 'pp_target_pos': pos,
        'stick_distance': d,
        'stick_angle': PRESET['stick_angle'],
        'iter_num': PRESET['stick_iters'],
    })

  def mean_dist(self):
    _, dist, _ = self.session.target.closest_points(self.session.x)
    return dist.mean()


############################################################
# Interactive UI

def run_ui(driver):
  import matplotlib
  import matplotlib.pyplot as plt
  from matplotlib.widgets import Button

  fig = plt.figure(f"Wrap standalone tester - solver {WRAP_SOLVER_VERSION}",
                   figsize=(11, 8))
  ax = fig.add_subplot(111, projection='3d')
  fig.subplots_adjust(bottom=0.16)

  state = {'picking': None}  # None | 'source' | 'target'

  F = np.asarray(driver.sf)
  step_f = max(len(F) // 1500, 1)          # decimate wireframe for speed
  tgt_step = max(len(driver.tv) // 4000, 1)

  def redraw(status=""):
    ax.cla()
    x = driver.session.x
    # target point cloud (gray)
    ax.scatter(*driver.tv[::tgt_step].T, s=2, c='0.75', alpha=0.5,
               label='target', picker=state['picking'] == 'target',
               pickradius=3)
    # source wireframe (blue)
    seg = x[F[::step_f]]
    for tri in seg:
      t = np.vstack([tri, tri[:1]])
      ax.plot(t[:, 0], t[:, 1], t[:, 2], c='tab:blue', lw=0.3, alpha=0.6)
    # source vertices for picking
    ax.scatter(*x.T, s=1, c='tab:blue', alpha=0.15,
               picker=state['picking'] == 'source', pickradius=3)
    # point pairs (red)
    for i, p in driver.pairs:
      ax.plot([x[i, 0], p[0]], [x[i, 1], p[1]], [x[i, 2], p[2]],
              c='red', lw=1.0, alpha=0.8)
      ax.scatter(*x[i], s=25, c='red')
      ax.scatter(*p, s=25, c='darkred', marker='s')

    all_pts = np.vstack([x, driver.tv])
    lo, hi = all_pts.min(0), all_pts.max(0)
    ctr, rad = (lo + hi) / 2, (hi - lo).max() / 2
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(ctr[2] - rad, ctr[2] + rad)
    ax.set_title(f"{status}   pairs: {len(driver.pairs)}   "
                 f"mean dist: {driver.mean_dist():.4f}")
    fig.canvas.draw_idle()

  def animate(gen, label):
    n = 0
    for _ in gen:
      n += 1
      if n % 2 == 0 or True:
        redraw(f"{label} iter {n}")
        plt.pause(0.01)
    redraw(f"{label} done")
    print(f"[standalone] {label}: {n} iterations, "
          f"mean dist {driver.mean_dist():.5f}")

  # --- pick handling ------------------------------------------------------
  pending = {}

  def on_pick(event):
    if state['picking'] is None or not len(event.ind):
      return
    i = int(event.ind[0])
    if state['picking'] == 'source':
      pending['src'] = i * 1  # scatter over full x -> direct index
      state['picking'] = 'target'
      redraw("now click the matching TARGET vertex (gray)")
    elif state['picking'] == 'target':
      ti = i * tgt_step       # target scatter is decimated
      driver.pairs.append((pending.pop('src'), driver.tv[ti].copy()))
      state['picking'] = None
      redraw("pair added")

  fig.canvas.mpl_connect('pick_event', on_pick)

  # --- buttons (mirror the add-on panel) ----------------------------------
  def add_button(x, label, cb):
    bax = fig.add_axes([x, 0.04, 0.11, 0.06])
    b = Button(bax, label)
    b.on_clicked(cb)
    return b

  buttons = []  # keep references alive
  buttons.append(add_button(0.05, 'Pick Pair', lambda e: (
      state.update(picking='source'),
      redraw("click a SOURCE vertex (blue)"))))
  buttons.append(add_button(0.18, 'Auto Pairs', lambda e: (
      auto_pairs(), redraw("auto pairs added"))))
  buttons.append(add_button(0.31, 'Snap', lambda e: animate(
      driver.snap(), 'snap')))
  buttons.append(add_button(0.44, 'Attract', lambda e: animate(
      driver.attract(), 'attract')))
  buttons.append(add_button(0.57, 'Stick', lambda e: animate(
      driver.stick(), 'stick')))
  buttons.append(add_button(0.70, 'Original', lambda e: (
      driver.reset(), redraw("reset to original"))))
  buttons.append(add_button(0.83, 'Clear Pairs', lambda e: (
      driver.pairs.clear(), redraw("pairs cleared"))))

  def auto_pairs():
    """Scatter a handful of pairs automatically (nearest target vertex to
    widely-spread source vertices) - quick way to test without clicking."""
    n = len(driver.sv0)
    for i in [0, n // 5, 2 * n // 5, 3 * n // 5, 4 * n // 5, n - 1]:
      d = np.linalg.norm(driver.tv - driver.session.x[i], axis=1)
      driver.pairs.append((i, driver.tv[int(d.argmin())].copy()))

  redraw("ready - pick pairs, then Snap / Attract / Stick")
  plt.show()


############################################################
# Headless self test

def selftest():
  sv, sf, tv, tf = demo_meshes()
  d = WrapDriver(sv, sf, tv, tf)
  n = len(sv)
  for i in [0, n // 4, n // 2, 3 * n // 4, n - 1]:
    dd = np.linalg.norm(tv - d.session.x[i], axis=1)
    d.pairs.append((i, tv[int(dd.argmin())].copy()))
  print(f"[selftest] initial mean dist {d.mean_dist():.4f}")
  for _ in d.snap():
    pass
  print(f"[selftest] after snap    {d.mean_dist():.4f}")
  for _ in d.attract():
    pass
  print(f"[selftest] after attract {d.mean_dist():.4f}")
  for _ in d.stick():
    pass
  print(f"[selftest] after stick   {d.mean_dist():.4f}")
  print("[selftest] PASS" if d.mean_dist() < 0.02 else "[selftest] FAIL")


if __name__ == '__main__':
  args = [a for a in sys.argv[1:] if not a.startswith('--')]
  if '--selftest' in sys.argv:
    selftest()
  else:
    if len(args) >= 2:
      sv, sf = load_obj(args[0])
      tv, tf = load_obj(args[1])
    else:
      sv, sf, tv, tf = demo_meshes()
    run_ui(WrapDriver(sv, sf, tv, tf))
