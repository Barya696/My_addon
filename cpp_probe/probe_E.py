# Probe E: how Stick treats the vertices around a held vertex.
#
# Five flat source patches (20 x 20 quads, 10 mm, 0.2 m square) at z = 0,
# 0.6 m apart along X (the same layout as probes B and C, so the same bbox
# diagonal, about 2.608 m, and with the MEDIUM preset the same reaches:
# Attract 26.1 mm / 30 deg, Stick 13.0 mm / 50 deg). Under each patch its own
# flat target (0.3 m square, same facing). In every patch three pairs hold
# the centre vertex (10, 10) and its neighbours (11, 10) and (10, 11) where
# they are (z = 0): three points that are not in a line, so Snap's system has
# a unique solution (one held vertex would leave the patch free to turn about
# it). After Attract the patch lies on its target except for a tent around
# that held corner; Stick decides how the tent ends up. The cases differ in
# one thing at a time:
#   E1  target 5 mm below, target grid 25 mm (as probes B/C)
#   E2  target 5 mm below, target grid 10 mm, vertices under the source's
#   E3  target 5 mm below, target grid 10 mm, shifted half a cell (5 mm, 5 mm)
#   E4  target 2 mm below, target grid 25 mm
#   E5  target 10 mm below, target grid 25 mm
# E1-E3 separate a grid / edge effect; E1, E4, E5 show whether the tent
# scales with the height (a weight law) or not (a gate or threshold).
#
#   blender -b --python probe_E.py -- <out dir> <cpp|py|dry>
# cpp: the C++ daemon already listening on the add-on's port; the add-on adds
#      the login from its preferences (never read, printed or saved here).
# py:  the Python daemon on PY_PORT. dry: build the scene, save it as
#      probe_E_scene.blend and the add-on's triangles, contact no daemon.
# Saves <out dir>/probe_E_<mode>.npz: meshes, the add-on's triangles (its
# BEAUTY triangulation, as sent at Init), pairs, groups, every streamed step
# of Snap, Attract and Stick in float64 with arrival times, and the add-on's
# distances and angles. Adapted from probe.py (probes A-D).
import bpy, os, sys, time, types, threading, json, numpy as np
OUT, MODE = sys.argv[sys.argv.index('--') + 1:][:2]
os.makedirs(OUT, exist_ok=True)
PY_PORT = 31012
if not any(k.endswith('teca3d_wrap_for_blender') for k in sys.modules):
    import addon_utils
    addon_utils.enable('bl_ext.user_default.teca3d_wrap_for_blender', default_set=False)
pkg = next(k for k in sys.modules if k.endswith('teca3d_wrap_for_blender'))
M = lambda n: sys.modules[f'{pkg}.{n}']
ow, wc, sm, mh, vh = (M('operators.op_wrap_operators'), M('client.wrap_client'), M('client.socket_manager'),
                      M('utils.mesh_helper'), M('utils.view_helper'))
if MODE == 'py':
    fake = types.SimpleNamespace(server_port=PY_PORT, user_email='', user_password='')
    sm.get_wrap_prefs = wc.get_wrap_prefs = lambda context: fake

def say(*a):
    print("PROBE", *a, flush=True)

def grid(nx, ny, sx, sy, z, cx=0.0, cy=0.0):
    xs = np.linspace(-sx / 2, sx / 2, nx + 1) + cx; ys = np.linspace(-sy / 2, sy / 2, ny + 1) + cy
    V = np.array([[x, y, z] for y in ys for x in xs])
    Fq = [[j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i]
          for j in range(ny) for i in range(nx)]
    return V, Fq

def make_object(name, V, Fq):
    me = bpy.data.meshes.new(name); me.from_pydata(V.tolist(), [], Fq); me.update()
    ob = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(ob)
    return ob

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

CASES = [("E1 target 5 mm below, 25 mm grid", -0.005, 12, 0.0),
         ("E2 target 5 mm below, 10 mm grid aligned", -0.005, 30, 0.0),
         ("E3 target 5 mm below, 10 mm grid shifted half a cell", -0.005, 30, 0.005),
         ("E4 target 2 mm below, 25 mm grid", -0.002, 12, 0.0),
         ("E5 target 10 mm below, 25 mm grid", -0.010, 12, 0.0)]
CY = 0.05   # the scene sits 50 mm off the X axis: no held vertex at (0, 0, 0), which the add-on and daemons read as "no target"
Vs, Fs, Vt, Ft, groups, pairs = [], [], [], [], {}, []
for k, (label, dz, nt, shift) in enumerate(CASES):
    cx = (k - 2) * 0.6
    v, f = grid(20, 20, 0.2, 0.2, 0.0, cx, CY); off = sum(len(a) for a in Vs)
    Vs.append(v); Fs += [[i + off for i in q] for q in f]; groups[label] = list(range(off, off + len(v)))
    pairs += [(off + c, v[c].copy()) for c in (220, 221, 241)]   # (10, 10), (11, 10), (10, 11), held where they are
    v, f = grid(nt, nt, 0.3, 0.3, dz, cx + shift, CY + shift); off = sum(len(a) for a in Vt)
    Vt.append(v); Ft += [[i + off for i in q] for q in f]
Vs, Vt = np.vstack(Vs), np.vstack(Vt)
src = make_object("probe_source", Vs, Fs); tgt = make_object("probe_target", Vt, Ft)

ctx = bpy.context; ws = ctx.scene.wrap_options
ws.source_mesh_mode = 'ORIGINAL'; ws.source_is_symmetrical = False
vh.reset_source_mesh_options(ctx)
ws.source_mesh_name = src.name; ws.source_diagonal_length = mh.get_source_mesh_diagonal_length(ctx)
mh.preserve_source_vertices(ctx)
ws.target_mesh_name = tgt.name
ws.point_pairs.clear()
for i, p in pairs:
    item = ws.point_pairs.add(); item.source_vertex_index = i; item.target_pos = p
ws.wrap_preset = 'MEDIUM'
settings = {'diagonal_m': float(ws.source_diagonal_length), 'attract_pct': float(ws.wrap_attract_distance), 'attract_angle': float(ws.wrap_attract_angle),
            'stick_pct': float(ws.wrap_stick_distance), 'stick_angle': float(ws.wrap_stick_angle),
            'flex': float(ws.wrap_init_flex), 'accuracy': float(ws.wrap_init_accuracy)}
src_tris = np.array(mh.get_source_triangles(ctx), dtype=np.int64)   # what the add-on sends at Init
tgt_tris = np.array(mh.get_target_triangles(ctx), dtype=np.int64)
say(f"E/{MODE}: source {len(Vs)} vertices, {len(Fs)} quads, {len(src_tris)} triangles; target {len(Vt)} vertices, {len(Ft)} quads, "
    f"{len(tgt_tris)} triangles; {len(pairs)} pairs; settings {json.dumps(settings)}")

blocks = []
if MODE == 'dry':
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "probe_E_scene.blend"), copy=True)
else:
    stage = {'name': None, 't0': 0.0}
    _recv = ow.receive_data_block
    def recording_receive(buffer):
        buffer, block = _recv(buffer)
        if block:
            blocks.append((stage['name'], time.time() - stage['t0'], np.frombuffer(block, np.float64).copy().reshape(-1, 3)))
        return buffer, block
    ow.receive_data_block = recording_receive

    class Runner:
        def __init__(self):
            self.mutex = threading.Lock(); self.mesh_data = None; self.start_time = 0; self.messages = []
        def report(self, kind, msg):
            self.messages.append(msg)

    t0 = time.time(); resp = wc.wrap_init(ctx)
    say(f"init: {time.time() - t0:.1f} s, reply code {resp.get('code') if resp else None}, {resp.get('status') if resp else 'no reply'}")
    if resp and resp.get('code') == 0:
        ws.wrap_process_stage = 1
        for name, fn in (('snap', ow.do_wrap_snap), ('attract', ow.do_wrap_attract), ('stick', ow.do_wrap_stick)):
            stage['name'] = name; stage['t0'] = time.time(); r = Runner(); fn(r, ctx)
            if r.mesh_data:
                mh.set_source_mesh_vertices(ctx, r.mesh_data); mh.preserve_source_vertices(ctx)
            n = sum(1 for b in blocks if b[0] == name)
            say(f"{name}: {time.time() - stage['t0']:.1f} s, {n} blocks | {' / '.join(r.messages)}")
np.savez_compressed(os.path.join(OUT, f"probe_E_{MODE}.npz"), Vs=Vs, Vt=Vt, Fs=np.array(Fs), Ft=np.array(Ft),
                    source_tris=src_tris, target_tris=tgt_tris,
                    pairs_idx=np.array([i for i, _ in pairs], dtype=int), pairs_pos=np.array([p for _, p in pairs]).reshape(-1, 3),
                    **{f"{st}_blocks": np.array([b[2] for b in blocks if b[0] == st]) for st in ('snap', 'attract', 'stick')},
                    **{f"{st}_times": np.array([b[1] for b in blocks if b[0] == st]) for st in ('snap', 'attract', 'stick')},
                    groups=json.dumps(groups), settings=json.dumps(settings))
say("saved", os.path.join(OUT, f"probe_E_{MODE}.npz"))
sm.global_socket = None
