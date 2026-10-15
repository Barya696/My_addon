# Package the C++ daemon recordings of 2026-10-11 as a regression set (Task 6).
#   python package_ground_truth.py <session bf918229 scratchpad> <req88.pkl> <out dir>
# One .npz per configuration: meshes, the exact requests (no email/password), and the C++
# result of every streamed step of Snap, Attract and Stick with their arrival times.
import os, sys, json, pickle
import numpy as np

SRC, REQ88, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
MEDIUM = {'flex': 100.0, 'accuracy': 3.0, 'force': 7.0, 'softness': 2.0,
          'attract_pct': 1.0, 'attract_angle': 30.0, 'stick_pct': 0.5, 'stick_angle': 50.0}


def edges_of(faces):
    e = []
    for f in faces:
        f = [int(i) for i in f if i >= 0]
        e += [(f[k], f[(k + 1) % len(f)]) for k in range(len(f))]
    return np.unique(np.sort(np.array(e), axis=1), axis=0)


def quality(rest, X, E, idx, pos):
    L0 = np.linalg.norm(rest[E[:, 0]] - rest[E[:, 1]], axis=1)
    r = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1) / np.maximum(L0, 1e-12)
    g = np.linalg.norm(X[idx] - pos, axis=1)
    return {'pair_gap_mean_mm': float(1000 * g.mean()), 'pair_gap_max_mm': float(1000 * g.max()),
            'edges_ratio_outside_1.5': int(((r > 1.5) | (r < 1 / 1.5)).sum())}


summary = {}
# ------------------------------------------------------------ eve, 88 pairs
req = pickle.load(open(REQ88, 'rb'))
init = {k: v for k, v in req['init'].items() if k not in ('email', 'password')}
rest = np.array(init['source_coords'], float)
idx = np.array(req['snap']['pp_source_index'], int); pos = np.array(req['snap']['pp_target_pos'], float)
E = edges_of(init['source_quads'] or init['source_tris'])
rec = {}
for st in ('snap', 'attract', 'stick'):
    rec[f'{st}_blocks'] = np.load(os.path.join(SRC, f'cpp_{st}_blocks.npy'))
    rec[f'{st}_times'] = np.load(os.path.join(SRC, f'cpp_{st}_times.npy'))
q = {st: quality(rest, rec[f'{st}_blocks'][-1].astype(float), E, idx, pos) for st in ('snap', 'attract', 'stick')}
np.savez_compressed(os.path.join(OUT, 'eve88_medium.npz'),
                    init_request=json.dumps({k: v for k, v in init.items()
                                             if k not in ('source_coords', 'source_tris', 'source_quads', 'target_coords',
                                                          'target_normals', 'target_tris', 'target_quads')}),
                    snap_request=json.dumps(req['snap']), attract_request=json.dumps(req['attract']),
                    stick_request=json.dumps(req['stick']),
                    source_coords=rest, source_tris=np.array(init['source_tris'], int),
                    source_quads=np.array(init['source_quads'], int),
                    target_coords=np.array(init['target_coords'], float), target_normals=np.array(init['target_normals'], float),
                    target_tris=np.array(init['target_tris'], int), quality=json.dumps(q), **rec)
summary['eve88_medium'] = {'source_vertices': len(rest), 'pairs': len(idx), 'steps': {st: len(rec[f'{st}_blocks']) for st in ('snap', 'attract', 'stick')}, 'quality': q}

# ------------------------------------------------------------ small probes A-D (built by probe.py, MEDIUM preset)
names = {'A': 'bar_both_ends_turned_90deg', 'B': 'attract_facing', 'C': 'attract_reach', 'D': 'attract_stick_angle'}
for p, label in names.items():
    z = np.load(os.path.join(SRC, f'probe_{p}_cpp.npz'), allow_pickle=True)
    Vs, Fs = z['Vs'], z['Fs']
    diag = float(np.linalg.norm(Vs.max(0) - Vs.min(0)))
    pidx, ppos = z['pairs_idx'].astype(int), z['pairs_pos'].astype(float)
    snap = {'request': 'wrap_snap_request', 'pp_source_index': pidx.tolist(), 'pp_target_pos': ppos.tolist(),
            'force': MEDIUM['force'], 'softness': MEDIUM['softness'], 'iter_num': 20}
    attract = {'request': 'wrap_attract_request', 'attract_distance': MEDIUM['attract_pct'] * diag * 0.01,
               'attract_angle': MEDIUM['attract_angle'], 'iter_num': 20}
    stick = {'request': 'wrap_stick_request', 'pp_source_index': pidx.tolist(), 'pp_target_pos': ppos.tolist(),
             'stick_distance': MEDIUM['stick_pct'] * diag * 0.01, 'stick_angle': MEDIUM['stick_angle'], 'iter_num': 10}
    Ep = edges_of(Fs)
    q = {}
    for st in ('snap', 'attract', 'stick'):
        b = z[f'{st}_blocks']
        if b.size:
            X = b[-1].astype(float)
            q[st] = quality(Vs, X, Ep, pidx, ppos) if np.isfinite(X).all() else 'not finite'
    np.savez_compressed(os.path.join(OUT, f'probe{p}_{label}.npz'),
                        settings=json.dumps(dict(MEDIUM, source_bbox_diagonal=diag, note='requests rebuilt from the probe scene; '
                                                 'distances = percent x bbox diagonal x 0.01 as the add-on does')),
                        snap_request=json.dumps(snap), attract_request=json.dumps(attract), stick_request=json.dumps(stick),
                        source_coords=Vs, source_quads=Fs, target_coords=z['Vt'], target_quads=z['Ft'],
                        groups=z['groups'], quality=json.dumps(q),
                        **{f'{st}_{k}': z[f'{st}_{k}'] for st in ('snap', 'attract', 'stick') for k in ('blocks', 'times')})
    summary[f'probe{p}_{label}'] = {'source_vertices': len(Vs), 'pairs': len(pidx),
                                    'steps': {st: int(len(z[f'{st}_blocks'])) for st in ('snap', 'attract', 'stick')}, 'quality': q}

json.dump(summary, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
print(json.dumps(summary, indent=1))
