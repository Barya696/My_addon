# C++ wrapDaemon: how Snap, Attract and Stick work (2026-10-13)

**Binary.** The original is `wrapDaemon (3).exe`:
- UPX-packed, 20.5 MB, sha256 `2fb149e5…`.
- Unpacked: `My addon\wrapDaemon_unpacked\wrapDaemon_unpacked.exe`, 89 MB.
- PE time stamp 2026-06-19, MSVC x64, built with vcpkg (`E:\works\c\vcpkg`).
- The Ghidra decompile is `Desktop\Fable 5.1 work\wrapDaemon_unpacked.exe.c`.

**Not the C++ daemon.** `add-on-wrap-for-blender\bin\wrapDaemon.exe` and `backups\wrapDaemon_cpp_before_py21_2026-10-13.exe` (sha256 `1000fcec…`) are PyInstaller / Python 3.14 builds.

## Answers

| Question | Answer |
|---|---|
| Is Snap order-independent? | **Yes, by construction.** Pairs are written into per-vertex arrays, and every energy and Jacobian sums over vertex index. The only effect of order is that two pairs on the same vertex resolve as "last one wins". A black-box deviation was **not measured**, because Init currently fails (see "Blocked"). |
| Does single-pair displacement have finite support? | **No.** Snap is one global sparse solve over a tetrahedral deformation graph; the solver loop has no falloff kernel. With 1 or 2 pairs (or collinear pairs) the system is singular: MKL prints "Zero pivot detected" every step. At least 3 non-collinear pairs are needed. |
| Which linear algebra / geometry libraries? | **Intel oneMKL**, statically linked: PARDISO/DSS for the sparse solve, LAPACK `dgesvd` for 3×3 SVD. **FLANN** (kd-tree), **CGAL** `Surface_mesh<Epick>` with **GMP**, and Intel OpenMP (`libiomp5md`). **No Eigen, SuiteSparse/CHOLMOD, libigl, Ceres, nanoflann, OpenMesh or geogram.** Networking: gRPC 1.65.5, protobuf, OpenSSL, libsodium, LZ4, zlib, asio. |
| What does `wrap_init_data` contain? | **Only a 36-character handle**, `<account email>_wrap-init_261011-162524`: the name of a file on the vendor's server, so its size has no relation to N, E or F. The real Init data is a **tetrahedral mesh** that the server builds and the daemon downloads (LZ4-compressed and libsodium-encrypted): `int32 n`, then `n × 3` 4-byte coordinates, then `int32 m`, then `m × 4` int32 tet indices, i.e. 8 + 12n + 16m bytes. |
| Time per Snap iteration, linear in iter_num? | On eve (54,412 vertices) one Gauss–Newton step takes **≈0.72 s**, and time is linear in the number of steps. **But `iter_num` is only an upper bound**: Snap stops once `|ΔE| < 0.001·E`. On eve it stopped after **4 steps** (max vertex move per step: 231.8 → 16.4 → 2.2 → 0.16 mm). A scene with all pairs at rest (E ≈ 0) runs all 20. |
| Falloff drags vs global solve? | **Global solve, high confidence** (read from the code, matching the recorded C++ runs). Snap is **Embedded Deformation** (Sumner, Schmid & Pauly 2007) on a server-built tet mesh, solved by Gauss–Newton with MKL PARDISO, with **soft** pairs. It is not biharmonic, and Snap is not ARAP. Attract is the same solver plus nearest-point constraints. Stick is a separate **surface ARAP** solver (Sorkine & Alexa 2007). |

## Task 1: strings and imports

**What I ran:** Python string scan (ASCII and UTF-16, length ≥ 6) of the unpacked exe, 102,585 strings. `pefile` imports and sections. RTTI class names (`.?AV…`).

**Imports**
- abseil, c-ares, gmp (5 `__gmpn_*` functions), libcrypto/libssl, libiomp5md (43 OpenMP entry points), libprotobuf.
- libsodium: `crypto_pwhash`, `crypto_secretstream_xchacha20poly1305_*`.
- lz4, re2, zlib, MSVCP140.
- CRT math: `sqrt pow cos exp log`.

**Statically linked (evidence from strings and RTTI)**
- **MKL**: about 4,700 `mkl_`/pardiso strings, "Zero pivot detected", "\*\*\* Error in PARDISO". The FEAST "Extended Eigensolvers" messages are MKL's own; they are not Eigen.
- **FLANN**: `flann::KDTreeSingleIndex<L2<double>>`, KMeans, "Building kdtree tree...".
- **CGAL**: `Surface_mesh<Point_3<Epick>>` property maps, "Undecidable conversion of CGAL::Uncertain".
- **boost::container**, **asio** (TCP server), **gRPC + protobuf**: `hub.proto` has `HubService/UploadFile`, `DownloadFile`, `FileData`, `MetaData{name, app, func}`.
- A glaze-style JSON reader: one parser function per key.

**The application's own classes:** only `CPnt3` (a 3-D point) and the `hub::` gRPC stubs.

**No algorithm words** (`arap`, `laplacian`, `energy`, `rigid`, …) appear as strings.
- The daemon's own console text is obfuscated: "Wrap Daemon, TECA 3D…", "Building, n", "Build failed", "Success, x, n" and the request names are not in the string table.
- So there is **no ordered list of application format strings**. The only format strings belong to FLANN, gRPC and MKL.
- The only source paths are vcpkg and gRPC headers.

**Plain JSON keys**
- Request keys: `password source_name source_coords source_tris source_quads source_free source_fixed target_name target_coords target_normals target_tris target_quads symmetric pp_source_index pp_target_pos softness iter_num attract_distance attract_angle stick_distance stick_angle`.
- Short keys (`email data alpha beta force version`) are compared as integer constants inside the code instead.
- Reply keys: `response code status message`.
- The **report uploaded to the server** at Init contains `email`, both meshes, `source_free`, `source_fixed`, `data`, `symmetric`, `alpha`, `beta`, `version`, `init_name`, `elapsed`, `hdd_serial_number`, `cpu_id` and `stamp`.

**Conclusion:** MKL PARDISO (a sparse direct solve) + FLANN + CGAL. Task 5 confirms the global solve.

## Task 5: the algorithm, from the decompiled code

### Call chain

| Request | Handler | Thread | Solver |
|---|---|---|---|
| Snap | `FUN_144a62ea0` | `FUN_144a53660` | `FUN_144aa4720(iter_num, reach² = 0, angle = 0)` |
| Attract | `FUN_144a61a60` | `FUN_144a53590` | `FUN_144aa4720(iter_num, reach², angle)`, **the same solver as Snap** |
| Stick | `FUN_144a63110` | `FUN_144a53730` | `FUN_144a8a540` (ARAP) |
| Init | `FUN_144a61c00` | | `FUN_144aa30d0` → `FUN_144aa4230` sets up the solver |

### The deformation model (Snap and Attract)

- **Nodes** `g_k` are the tet-mesh vertices. Each node carries an affine transform: a 3×3 matrix `A_k` plus a translation `t_k`, 12 unknowns per node. Init sets `A = I`, `t = 0`.
- **Binding** (`FUN_144aa6a60`, `FUN_144aa6690`): each source vertex is bound to one tet.
  - A FLANN kd-tree over the tet centroids finds the 20 nearest tets.
  - Barycentric coordinates come from signed volumes.
  - The vertex keeps the tet that contains it, or the one it is least outside of. Negative coordinates are clamped to 0 and the rest renormalised.
- **Deformed vertex** (`FUN_144aa3190`): `x_i = Σ_{k∈tet(i)} b_ik · (A_k (v_i − g_k) + g_k + t_k)`.
- **Node neighbours** (`FUN_144aa1810`) are the edges of the tets.

### The energy

`E = E_rot + w_reg·E_reg + E_con`

| Term | Function | Definition |
|---|---|---|
| `E_rot` | `FUN_144aa21b0` | Over the columns `c` of each `A_k`: `Σ_k (c1·c2)² + (c1·c3)² + (c2·c3)² + (‖c1‖²−1)² + (‖c2‖²−1)² + (‖c3‖²−1)²`. Weight 1. |
| `E_reg` | `FUN_144aa0a80` | `Σ_k Σ_{j∈N(k)} ‖A_k(g_j − g_k) + g_k + t_k − (g_j + t_j)‖²` |
| `E_con` | `FUN_144aa1a80` | `Σ_i w_i ‖x_i − c_i‖²` (point-to-point), or `w_i (n_i·(x_i − c_i))²` where a normal is set (point-to-plane) |

### The weights

Set by `FUN_144aa5d30`, then scaled by `FUN_144aa0930`. `D` is the bounding-box diagonal of the source rest vertices; `FUN_144abfae0` returns its **square**.

| Weight | Value | With defaults (force 7, softness 2) |
|---|---|---|
| `w_reg` | `10^(force − softness) / D²` | `1e5 / D²` |
| `w_pair` = `w_fixed` | `10^force / D²` | `1e7 / D²` |
| Attract weight | starts at `w_reg / 128`, doubles after every accepted step, capped at `w_reg` | |

### The constraints (`FUN_144aa3400`)

- **Pairs**: point-to-point onto the target, weight `w_pair`. Pairs are **soft**: on eve they end 0.45 mm mean / 1.13 mm max off target.
- **`source_fixed` vertices**: held at their **rest** position, also with `w_pair`.
- **Attract** (only when reach > 0, so never in Snap):
  - Each free vertex takes the FLANN nearest **target vertex** of its deformed position.
  - The match is kept only if distance² ≤ reach² and `n_src·n_tgt ≥ cos(angle)`.
  - The residual is point-to-plane with that target vertex's normal.
  - The weight is the per-target weight × the attract weight.
  - Matches are rebuilt after every accepted step, as in ICP.

### The solver

`FUN_144aa4720` runs the loop; `FUN_144aa5db0` computes each step.

- **Normal matrix**: `H = w_reg·JᵀJ_reg + JᵀJ_con + JᵀJ_rot` (12×12 node blocks), with each diagonal entry multiplied by `(1 + λ)`.
- **Right-hand side**: `b = g_rot + w_reg·g_reg + g_con`.
- **Each step**: MKL PARDISO factorises and solves; then `x += dx`.
- **Accept or undo**:
  - If `E_new ≤ E_prev`, the step is kept and `λ ×= 0.001`.
  - Otherwise the step is undone and `λ ÷= 0.001`.
  - **λ starts at 0**, so this is plain, undamped Gauss–Newton. That is why fewer than 3 pairs gives a zero pivot.
- **Stop**: when `|E_prev − E_new| < 0.001·E_new`; otherwise after `iter_num` steps.
- **Constants** (from the constructor): λ₀ = 0, factor 0.001, tolerance 0.001.
- **Streaming**: the vertex positions are sent after every step (`FUN_144aa4d50`), including a step that is then undone.
- **Not reset per Snap**: I found no reset of the node transforms on the Snap path (Init does reset them), so a second Snap probably starts from the previous result. Not confirmed black-box.

### Stick (`FUN_144a8a540`)

- ARAP on the surface mesh itself: `Σ_i Σ_j w_ij ‖R_i(p_i − p_j) − (p'_i − p'_j)‖²` (`FUN_144a88080`), plus a constraint energy.
- Each `R_i` comes from LAPACK `dgesvd` on a 3×3 matrix, with `R = U·Vᵀ` (`FUN_144a88bc0`).
- The global step is a sparse PARDISO solve. Constraints are rebuilt every iteration.
- Weights `10^(force − softness)` and `10^force` (`FUN_144a8c190`).
- Stops when `|ΔE| < tol·E`. There is no step rejection.

### Init

- The daemon uploads the report above (both meshes, `alpha = flex`, `beta = flex × accuracy`, disk serial and CPU id) to the hub (216.24.190.16) over gRPC.
- The server builds the data ("Building, 1…30").
- The daemon downloads the result, decrypts it (key from `crypto_pwhash`) and decompresses it. That result is the tet mesh.
- `alpha` and `beta` go only into that report, so they must set the tet-mesh resolution on the server side.

## Task 2: runtime behaviour

**New runs are blocked** (see "Blocked"). These numbers come from the C++ recordings of 2026-10-11 (88 pairs, MEDIUM preset).

| Item | Result |
|---|---|
| 2.1 Init | 28.7 s, a server build. The reply has `response, code, status, message`; `message` is the 36-character handle. A second, identical Init could not be timed. |
| 2.2 Snap | 4 streamed steps arriving at 0.88 / 1.61 / 2.34 / 3.05 s. Max move per step 231.75 / 16.38 / 2.18 / 0.16 mm, the fast convergence of Gauss–Newton. Pair gap after Snap: 0.449 mm mean, 1.134 mm max. |
| 2.2 small probes | Probes B/C/D (all pairs at rest, E ≈ 0) always run all 20 steps, because the stop test cannot fire at E = 0. Probe A (bar, 674 vertices) takes about 0.67 s per step, as eve does, so the cost follows the tet-mesh size, not the surface vertex count. |
| 2.3 order | Order-independent by code; not measured. |
| 2.4–2.5 single pair | A single pair is singular in the C++. A test needs anchors; `probe_cpp.py` holds 4 far-away vertices at rest. |
| 2.6 two opposite pairs | Prepared in `probe_cpp.py`, not run. |
| 2.7 +100 mm translation | `E_rot` and `E_reg` do not change under a common translation, so the prediction is that every vertex moves exactly 100 mm. Not run. |
| 2.8 Snap request schema | `request, pp_source_index, pp_target_pos, force, softness, iter_num`. The project copy of `wrap_client.py` also sends `proportional_size` and `proportional_falloff`, which are not in the C++ key table. Whether the C++ ignores or rejects them is untested. |

Attract: 12 steps of about 1.0 s each. Stick: 10 steps of about 2.7 s each.

## Task 3: Init parameters

- **3.1 flex and accuracy:** these only reach the server, so they control how the tet mesh is built. Not measured.
- **3.2 changing the mesh between Init and Snap:** Snap requests carry no coordinates. The daemon solves from the rest positions it received at Init, so editing the Blender mesh between Init and Snap cannot change the C++ result; the streamed result simply overwrites the mesh. `wrap_init_data` is a server file name tied to the account and a time stamp, and every Init uploads the meshes again.

## Task 4: "Fit"

There is no `wrap_fit_request`, neither in the C++ key table nor in the add-on. The requests are Init, Snap, Attract and Stick.

- **Attract** is Snap's solver plus nearest-point (point-to-plane) constraints with a weight ramp. In other words, it is "Snap with ICP correspondence updates".
- **Stick** is a different algorithm: surface ARAP.
- Snap with `iter_num → ∞` equals Snap stopped at about 4–5 steps, because of the 0.1 % energy stop.

## Task 6: ground truth

`ground_truth\` holds five configurations taken from the C++ recordings of 2026-10-11:

| File | Configuration |
|---|---|
| `eve88_medium.npz` | Eve with the 88 pairs. Every Snap (4), Attract (12) and Stick (10) step, the arrival times, the exact requests (without credentials) and all meshes. |
| `probeA_bar_both_ends_turned_90deg.npz` | A bar whose two ends are pinned to the bar turned 90°. |
| `probeB_attract_facing.npz` | Attract against surfaces facing the same and the opposite way. |
| `probeC_attract_reach.npz` | Attract against surfaces at 30–105 % of the reach. |
| `probeD_attract_stick_angle.npz` | Attract and Stick against surfaces tilted 0–80°. |

Each file also has quality numbers: pair gap, and the count of edges stretched or shrunk by more than ×1.5. `summary.json` lists them. The probe requests were rebuilt from each probe scene with the MEDIUM preset.

## Blocked: new black-box runs

- **What happened:** `probe_cpp.py` was run on 2026-10-13 against the original packed exe. Init contacted the vendor server, printed "Building, 1 … 30" and then "Build failed", and returned code 4 "Failed to create init data" after 82 s.
- **Likely cause:** this is exactly how the daemon behaved on 2026-10-11 without a login, so the Blender 5.2 add-on preferences most likely have no working email/password right now.
- **To rerun:** once the login is set in the add-on preferences, rerun the probe and the analysis with the commands below. They cover Tasks 2.1–2.8, 3.1, 4 and 6 on the real 88-pair scene; Init contacts the vendor server and uploads the meshes.

```
wrapDaemon.exe 31008                    (the packed C++ exe, run from a folder holding its 12 DLLs)
blender -b <copy of the 88-pair scene> --python cpp_probe\probe_cpp.py -- cpp_probe\results <req88.pkl>
python cpp_probe\analyze_cpp.py cpp_probe\results
```

## What this means for a Python replacement

- **Everything except the tet mesh can be copied exactly:** the model, energies, weights, constraint rules, solver, stop test and Stick's ARAP.
- **The tet mesh is the one unknown.** The server builds it from `alpha` and `beta`, and it is not in the client.
  - Probe A (a 1 m bar) costs as much per step as eve (2.06 m diagonal), which fits a node spacing proportional to the diagonal (for example `D/alpha`). That is a guess, not a measurement.
- **Matching the C++ within 1 mm will depend on how closely a local tet mesh** (TetGen, CGAL Mesh_3 or a regular lattice) **matches the server's.**
- **C++ pairs are soft** (0.45 mm mean after Snap), while solver 22's pairs are exact.
