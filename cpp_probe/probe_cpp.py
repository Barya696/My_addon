# Black-box probe of the original C++ wrapDaemon (TECA 3D 2026 v1.0.0).
#
#   1. start the C++ daemon on the add-on's port (31008):  wrapDaemon.exe 31008
#   2. blender -b <copy of the 88-pair scene> --python probe_cpp.py -- <out dir> <req88.pkl>
#
# Runs with the user's Blender preferences (no --factory-startup): Init goes through the
# installed add-on's own wrap_init(), which adds the login from its preferences. This script
# never reads, prints or saves the email or password. Init contacts the vendor's server.
#
# Every test saves <out>/<name>.npz (request JSON without credentials, reply, stream times,
# final vertices) as soon as it finishes, and appends one line to <out>/log.jsonl.
import bpy, os, sys, time, json, pickle, re
import numpy as np

OUT, REQ88 = sys.argv[sys.argv.index('--') + 1:][:2]
os.makedirs(OUT, exist_ok=True)

if not any(k.endswith('teca3d_wrap_for_blender') for k in sys.modules):
    import addon_utils
    addon_utils.enable('bl_ext.user_default.teca3d_wrap_for_blender', default_set=False)
pkg = next(k for k in sys.modules if k.endswith('teca3d_wrap_for_blender'))
M = lambda n: sys.modules[f'{pkg}.{n}']
wc, sm = M('client.wrap_client'), M('client.socket_manager')
ctx = bpy.context
ws = ctx.scene.wrap_options


def say(*a):
    print('PROBE', *a, flush=True)


def log(rec):
    with open(os.path.join(OUT, 'log.jsonl'), 'a') as f:
        f.write(json.dumps(rec) + '\n')
    say(json.dumps({k: v for k, v in rec.items() if k not in ('request',)})[:400])


def redact(s):
    # the Init handle starts with the account email
    return re.sub(r'^[^_]*@[^_]*_', '<email>_', s) if isinstance(s, str) else s


# ------------------------------------------------------------ data
req88 = pickle.load(open(REQ88, 'rb'))
rest = np.array([list(v.coord) for v in ws.source_original_vertices], float)
assert np.allclose(rest, np.array(req88['init']['source_coords'])), 'scene rest != req88 rest'
tris = np.array(req88['init']['source_tris'], int)
P_IDX = np.array(req88['snap']['pp_source_index'], int)
P_POS = np.array(req88['snap']['pp_target_pos'], float)
N = len(rest)
diag = float(np.linalg.norm(rest.max(0) - rest.min(0)))

fn = np.cross(rest[tris[:, 1]] - rest[tris[:, 0]], rest[tris[:, 2]] - rest[tris[:, 0]])
vn = np.zeros_like(rest)
for k in range(3):
    np.add.at(vn, tris[:, k], fn)
vn /= np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-30)

np.savez_compressed(os.path.join(OUT, 'mesh.npz'), rest=rest, tris=tris, normals=vn,
                    pairs_idx=P_IDX, pairs_pos=P_POS, diag=diag)
say(f'{N} vertices, {len(tris)} tris, {len(P_IDX)} pairs, bbox diagonal {diag:.4f} m')


# ------------------------------------------------------------ protocol
def set_timeout():
    if sm.global_socket:
        sm.global_socket.settimeout(600)


def init(name):
    t0 = time.perf_counter()
    resp = wc.wrap_init(ctx)                  # the add-on adds the login itself
    t = time.perf_counter() - t0
    set_timeout()
    ok = bool(resp) and resp.get('code') == 0
    if ok and resp.get('message'):
        wc.set_wrap_init_data(ctx, resp['message'])   # as the Init operator does
    r = {k: (redact(v) if k == 'message' else v) for k, v in (resp or {}).items()}
    log({'test': name, 'kind': 'init', 'time': t, 'reply': r, 'reply_keys': sorted((resp or {}).keys()),
         'flex': ws.wrap_init_flex, 'accuracy': ws.wrap_init_accuracy})
    return ok


def run(req):
    """Send one Snap/Attract/Stick request; return reply, stream times, blocks."""
    t0 = time.perf_counter()
    if not sm.send_on_socket(ctx, (json.dumps(req) + '\n').encode('utf-8')):
        return None, np.zeros(0), []
    set_timeout()
    line = sm.receive_on_socket(ctx)
    t_ack = time.perf_counter() - t0
    resp = json.loads(line) if line else None
    times, blocks = [t_ack], []
    if resp and resp.get('code') == 0:
        buf = b''
        while True:
            buf, blk = sm.receive_data_block(buf)
            if not blk:
                break
            times.append(time.perf_counter() - t0)
            blocks.append(np.frombuffer(blk, np.float64).reshape(-1, 3).copy())
    times.append(time.perf_counter() - t0)
    return resp, np.array(times), blocks


def snap_req(idx, pos, force=7.0, softness=2.0, iters=20, extra=None):
    r = {'request': 'wrap_snap_request', 'pp_source_index': [int(i) for i in idx],
         'pp_target_pos': [[float(c) for c in p] for p in pos],
         'force': float(force), 'softness': float(softness), 'iter_num': int(iters)}
    if extra:
        r.update(extra)
    return r


def test(name, req, keep_all=False):
    resp, times, blocks = run(req)
    final = blocks[-1] if blocks else np.zeros((0, 3))
    steps = [float(np.abs(blocks[0] - rest).max())] if blocks else []
    steps += [float(np.abs(blocks[i] - blocks[i - 1]).max()) for i in range(1, len(blocks))]
    rec = {'test': name, 'kind': req['request'], 'reply': resp, 'n_blocks': len(blocks),
           'times': [round(float(t), 4) for t in times], 'max_step_m': steps,
           'request': {k: v for k, v in req.items() if k not in ('pp_source_index', 'pp_target_pos')}}
    if len(final):
        rec['max_dev_from_rest_m'] = float(np.linalg.norm(final - rest, axis=1).max())
        rec['finite'] = bool(np.isfinite(final).all())
    log(rec)
    np.savez_compressed(os.path.join(OUT, f'{name}.npz'), request=json.dumps(req), reply=json.dumps(resp),
                        times=times, final=final,
                        **({'blocks': np.array(blocks)} if keep_all and blocks else {}))
    return final, len(blocks)


def reset(tag):
    # pairs at their rest positions: the zero-energy state (identity) is the minimum
    return test(f'reset_{tag}', snap_req(P_IDX, rest[P_IDX]))


# ------------------------------------------------------------ test geometry (generic)
c = rest.mean(0)
m = int(np.argmin(np.linalg.norm(rest - c, axis=1)))          # surface vertex nearest the centroid
anchors = [m]
d = np.linalg.norm(rest - rest[m], axis=1)
for _ in range(4):                                             # farthest-point sampling: extremities
    a = int(np.argmax(d)); anchors.append(a)
    d = np.minimum(d, np.linalg.norm(rest - rest[a], axis=1))
anchors = anchors[1:]
tang = np.cross(vn[m], [0.0, 0.0, 1.0]); tang /= np.linalg.norm(tang)
b = int(np.argmin(np.linalg.norm(rest - (rest[m] + 0.03 * tang), axis=1)))
geo = {'moved': m, 'anchors': anchors, 'second': b, 'ab_dist_m': float(np.linalg.norm(rest[b] - rest[m])),
       'n_moved': vn[m].tolist(), 'n_second': vn[b].tolist()}
json.dump(geo, open(os.path.join(OUT, 'geometry.json'), 'w'))
say('geometry', json.dumps(geo))

# ------------------------------------------------------------ run
if not init('init1'):
    say('Init failed: stopping (is the login set in the add-on preferences?)')
    sm.global_socket = None
    sys.exit(0)
init('init2')                                                  # same Init again, now with the cached handle

orig = test('snap_orig', snap_req(P_IDX, P_POS), keep_all=True)
test('snap_orig_again', snap_req(P_IDX, P_POS), keep_all=True)  # warm start?

init('init3')
rev = np.arange(len(P_IDX))[::-1]
test('snap_reversed', snap_req(P_IDX[rev], P_POS[rev]))
init('init4')
shuf = np.random.default_rng(0).permutation(len(P_IDX))
test('snap_shuffled', snap_req(P_IDX[shuf], P_POS[shuf]))

reset('a')
for n in (1, 2, 5, 10, 20, 40, 200):                          # time vs iter_num
    test(f'snap_iter_{n}', snap_req(P_IDX, P_POS, iters=n), keep_all=(n == 40))
    reset(f'iter_{n}')

test('snap_translate_x100mm', snap_req(P_IDX, rest[P_IDX] + [0.1, 0.0, 0.0]))   # every pair +100 mm X
reset('translate')

one_idx = anchors + [m]
for f, s in ((7, 2), (7, 0), (7, 1), (7, 3), (5, 2), (9, 2)):
    pos = np.vstack([rest[anchors], rest[m] + 0.05 * vn[m]])
    test(f'snap_single_f{f}_s{s}', snap_req(one_idx, pos, force=f, softness=s))
    reset(f'single_f{f}_s{s}')

two_idx = anchors + [m, b]
pos = np.vstack([rest[anchors], rest[m] + 0.025 * vn[m], rest[b] - 0.025 * vn[b]])
test('snap_two_opposite', snap_req(two_idx, pos))
reset('two')

test('snap_with_proportional_keys', snap_req(P_IDX, P_POS, extra={'proportional_size': 1.0,
                                                                  'proportional_falloff': 'SMOOTH'}))
reset('prop')
for f, s in ((9, 0), (5, 3)):
    test(f'snap_orig_f{f}_s{s}', snap_req(P_IDX, P_POS, force=f, softness=s))
    reset(f'orig_f{f}_s{s}')

# full chain from a fresh Init: ground truth for Attract and Stick
init('init5')
test('chain_snap', snap_req(P_IDX, P_POS), keep_all=True)
test('chain_attract', dict(req88['attract']), keep_all=True)
test('chain_stick', dict(req88['stick']), keep_all=True)

# Init parameters: each change of flex/accuracy is a new build on the server
for flex, acc in ((50.0, 3.0), (200.0, 3.0), (100.0, 1.0), (100.0, 5.0)):
    ws.wrap_init_flex, ws.wrap_init_accuracy = flex, acc
    if init(f'init_flex{flex:g}_acc{acc:g}'):
        test(f'snap_flex{flex:g}_acc{acc:g}', snap_req(P_IDX, P_POS))

say('done')
sm.global_socket = None
