# Analyse the results of probe_cpp.py:  python analyze_cpp.py <results dir>
# Prints the numbers per task and writes ground_truth/*.npz next to the results dir.
import os, sys, json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

RES = sys.argv[1]
GT = os.path.join(os.path.dirname(RES.rstrip('/\\')), 'ground_truth')
os.makedirs(GT, exist_ok=True)

mesh = np.load(os.path.join(RES, 'mesh.npz'))
rest, tris, vn = mesh['rest'], mesh['tris'], mesh['normals']
P_IDX, P_POS, diag = mesh['pairs_idx'], mesh['pairs_pos'], float(mesh['diag'])
geo = json.load(open(os.path.join(RES, 'geometry.json')))
N = len(rest)
E = np.unique(np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1), axis=0)
L0 = np.linalg.norm(rest[E[:, 0]] - rest[E[:, 1]], axis=1)
logs = {}
for line in open(os.path.join(RES, 'log.jsonl')):
    r = json.loads(line); logs[r['test']] = r


def final(name):
    p = os.path.join(RES, name + '.npz')
    return np.load(p)['final'] if os.path.exists(p) else None


def bad_edges(X):
    r = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1) / np.maximum(L0, 1e-12)
    return int(((r > 1.5) | (r < 1 / 1.5)).sum())


def gaps(X, idx, pos):
    return np.linalg.norm(X[idx] - pos, axis=1)


def mm(x):
    return f'{1000 * x:.4g} mm'


def h(t):
    print('\n' + '=' * 8, t)


print(f'N = {N} vertices, E = {len(E)} edges, F = {len(tris)} triangles, bbox diagonal {diag:.4f} m')

# ------------------------------------------------------------ Task 2.1 Init
h('Task 2.1 Init')
for k, r in logs.items():
    if r['kind'] == 'init':
        msg = r['reply'].get('message', '')
        print(f"{k:24s} {r['time']:7.2f} s  code {r['reply'].get('code')}  status {r['reply'].get('status')!r}  "
              f"keys {r['reply_keys']}  message {msg!r} ({len(msg)} chars)  flex {r['flex']} acc {r['accuracy']}")

# ------------------------------------------------------------ Task 2.2 iter_num
h('Task 2.2 Snap time vs iter_num')
print('iter_num  blocks  total s  per-block s (after ack)       reply')
for n in (1, 2, 5, 10, 20, 40, 200):
    r = logs.get(f'snap_iter_{n}')
    if r:
        t = np.array(r['times']); per = np.diff(t[:-1]) if len(t) > 2 else []
        print(f"{n:8d}  {r['n_blocks']:6d}  {t[-1]:7.2f}  {np.round(per, 2).tolist()}  {r['reply']}")
for a, b_ in ((20, 200), (20, 40)):
    A, B = final(f'snap_iter_{a}'), final(f'snap_iter_{b_}')
    if A is not None and B is not None and len(A) and len(B):
        print(f'iter {a} vs {b_}: max deviation {mm(np.linalg.norm(A - B, axis=1).max())}')
r = logs.get('snap_orig')
if r:
    print('snap_orig steps (max vertex move per streamed block):', [mm(s) for s in r['max_step_m']])

# ------------------------------------------------------------ Task 2.3 order + warm start
h('Task 2.3 order independence')
O = final('snap_orig')
for k in ('snap_reversed', 'snap_shuffled'):
    X = final(k)
    if O is not None and X is not None and len(X):
        print(f'{k}: max deviation from original order {np.linalg.norm(X - O, axis=1).max():.3e} m, '
              f'blocks {logs[k]["n_blocks"]} vs {logs["snap_orig"]["n_blocks"]}')
X = final('snap_orig_again')
if O is not None and X is not None and len(X):
    print(f'snap_orig_again (same request, no Init between): blocks {logs["snap_orig_again"]["n_blocks"]}, '
          f'max deviation {mm(np.linalg.norm(X - O, axis=1).max())}, first-block move {mm(logs["snap_orig_again"]["max_step_m"][0] if logs["snap_orig_again"]["max_step_m"] else 0)} from rest')
resets = {k: r.get('max_dev_from_rest_m') for k, r in logs.items() if k.startswith('reset_')}
if resets:
    print('reset quality (max distance from rest after pairs-at-rest Snap):',
          {k: f'{v:.2e} m' for k, v in resets.items() if v is not None})

# ------------------------------------------------------------ Task 2.4/2.5 single pair
h('Task 2.4/2.5 single moved pair (+50 mm along its normal; 4 far anchors held at rest)')
m, anchors = geo['moved'], geo['anchors']
W = coo_matrix((np.r_[L0, L0], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(N, N)).tocsr()
geod = dijkstra(W, indices=m)
eucl = np.linalg.norm(rest - rest[m], axis=1)
bins = np.array([0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.5, 2.5])
for f, s in ((7, 2), (7, 0), (7, 1), (7, 3), (5, 2), (9, 2)):
    X = final(f'snap_single_f{f}_s{s}')
    if X is None or not len(X):
        continue
    if not np.isfinite(X).all():
        print(f'force {f} softness {s}: NOT FINITE'); continue
    d = np.linalg.norm(X - rest, axis=1)
    target = rest[m] + 0.05 * vn[m]
    frac = float(np.dot(X[m] - rest[m], vn[m]) / 0.05)
    peak = d[m]
    prof = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (geod >= lo) & (geod < hi)
        prof.append(f'{lo * 100:g}-{hi * 100:g}cm: {1000 * np.median(d[sel]):.3g}' if sel.any() else '')
    def radius(level, dist):
        far = dist[d >= level * peak]
        return float(far.max()) if len(far) else 0.0
    print(f'force {f} softness {s}: pair moved {frac:.4f} of 50 mm (gap {mm(np.linalg.norm(X[m] - target))}), '
          f'blocks {logs[f"snap_single_f{f}_s{s}"]["n_blocks"]}; geodesic radius to 50% / 10% / 1% of peak: '
          f'{radius(0.5, geod):.3f} / {radius(0.1, geod):.3f} / {radius(0.01, geod):.3f} m; '
          f'min displacement {d.min():.2e} m, vertices < 1 um: {(d < 1e-6).sum()}, anchors max {mm(d[anchors].max())}')
    print('   median displacement (mm) by geodesic distance:', ' | '.join(p for p in prof if p))

# ------------------------------------------------------------ Task 2.6 two opposite pairs
h('Task 2.6 two pairs ~30 mm apart pulled +25 / -25 mm along their normals')
X = final('snap_two_opposite')
if X is not None and len(X):
    a, b = geo['moved'], geo['second']
    u = rest[b] - rest[a]; L = np.linalg.norm(u); u /= L
    n = vn[a]
    rel = rest - rest[a]
    s_ = rel @ u
    off = rel - np.outer(s_, u)
    side = np.linalg.norm(off - np.outer(off @ n, n), axis=1)
    sel = (side < 0.004) & (s_ > -0.06) & (s_ < L + 0.06) & (np.abs(rel @ n) < 0.02)
    order = np.argsort(s_[sel])
    dn = ((X - rest) @ n)[sel][order]
    ss = s_[sel][order]
    print(f'a-b distance {mm(L)}; {sel.sum()} vertices in the 4 mm slab; displacement along n_a (mm) vs position (mm):')
    print('   ' + ' '.join(f'{1000 * p:.0f}:{1000 * q:+.1f}' for p, q in zip(ss, dn)))
    print(f'   pair a moved {mm(np.dot(X[a] - rest[a], vn[a]))} (target +25), pair b {mm(np.dot(X[b] - rest[b], vn[b]))} (target -25); bad edges {bad_edges(X)}')

# ------------------------------------------------------------ Task 2.7 translation
h('Task 2.7 all pairs +100 mm X')
X = final('snap_translate_x100mm')
if X is not None and len(X):
    dv = X - rest
    dist = np.linalg.norm(dv, axis=1)
    print(f'moved > 99 mm: {(dist > 0.099).mean() * 100:.2f}% of vertices; deviation from a pure +100 mm X shift: '
          f'max {mm(np.linalg.norm(dv - [0.1, 0, 0], axis=1).max())}, mean {mm(np.linalg.norm(dv - [0.1, 0, 0], axis=1).mean())}')

# ------------------------------------------------------------ Task 2.8
h('Task 2.8 request with proportional_size / proportional_falloff keys')
r = logs.get('snap_with_proportional_keys')
if r:
    X = final('snap_with_proportional_keys'); R0 = final('snap_iter_20')
    dev = np.linalg.norm(X - R0, axis=1).max() if (X is not None and len(X) and R0 is not None and len(R0)) else None
    print(f"reply {r['reply']}, blocks {r['n_blocks']}, max deviation from the same Snap without them: {dev}")

# ------------------------------------------------------------ orig 88 quality + Task 3.1
h('88-pair Snap quality and Task 3.1 flex / accuracy')
rows = [('snap_orig', 'default 100/3, f7 s2'), ('snap_orig_f9_s0', 'force 9 softness 0'), ('snap_orig_f5_s3', 'force 5 softness 3')]
rows += [(f'snap_flex{fl:g}_acc{ac:g}', f'flex {fl:g} accuracy {ac:g}') for fl, ac in ((50, 3), (200, 3), (100, 1), (100, 5))]
for k, lab in rows:
    X = final(k)
    if X is None or not len(X):
        continue
    g = gaps(X, P_IDX, P_POS)
    t = np.array(logs[k]['times'])
    dev = np.linalg.norm(X - O, axis=1) if O is not None else np.zeros(1)
    print(f'{lab:24s} blocks {logs[k]["n_blocks"]:2d}, {t[-1]:5.2f} s ({np.mean(np.diff(t[:-1])) if len(t) > 2 else 0:.2f} s/step); '
          f'pair gap mean {mm(g.mean())} max {mm(g.max())}; bad edges {bad_edges(X)}; vs default: mean {mm(dev.mean())} max {mm(dev.max())}')

# ------------------------------------------------------------ Task 4 chain
h('Task 4 Attract / Stick (from a fresh Init)')
for k in ('chain_snap', 'chain_attract', 'chain_stick'):
    r = logs.get(k)
    if r:
        t = np.array(r['times'])
        print(f"{k}: reply {r['reply']}, blocks {r['n_blocks']}, total {t[-1]:.2f} s, per step {np.round(np.diff(t[:-1]), 2).tolist()}")
        X = final(k)
        if X is not None and len(X):
            g = gaps(X, P_IDX, P_POS)
            print(f'   pair gap mean {mm(g.mean())} max {mm(g.max())}; bad edges {bad_edges(X)}')
X = final('chain_snap')
if X is not None and O is not None and len(X):
    print(f'chain_snap vs snap_orig (both from a fresh Init): max deviation {np.linalg.norm(X - O, axis=1).max():.3e} m')

# ------------------------------------------------------------ Task 6 ground truth
h('Task 6 ground truth')
gt = ['snap_orig', 'snap_translate_x100mm', 'snap_single_f7_s2', 'snap_two_opposite', 'snap_orig_f9_s0',
      'snap_orig_f5_s3', 'chain_attract', 'chain_stick']
for k in gt:
    p = os.path.join(RES, k + '.npz')
    if not os.path.exists(p):
        continue
    z = np.load(p, allow_pickle=False)
    X = z['final']
    req = json.loads(str(z['request']))
    idx = np.array(req.get('pp_source_index', P_IDX), int)
    pos = np.array(req.get('pp_target_pos', P_POS), float)
    np.savez_compressed(os.path.join(GT, k + '.npz'), request=str(z['request']), rest=rest, tris=tris, result=X,
                        start=('snap_orig' if k.startswith('chain_') else 'rest (fresh Init, or after a pairs-at-rest Snap)'),
                        pair_gap=gaps(X, idx, pos), bad_edges=bad_edges(X), stream_times=z['times'])
    print(f'{k}: saved, pair gap max {mm(gaps(X, idx, pos).max())}, bad edges {bad_edges(X)}')
