# wrapDaemon

A local wrap server for the **add-on-wrap-for-blender** Blender add-on. It implements
the exact TCP protocol the add-on's client code expects, so the add-on UI works
unchanged — Wrap Init, Snap, Attract, and Stick all run against this daemon.

## What it implements

| Request | Behavior |
|---|---|
| `wrap_init_request` | Builds the elastic model (cotangent Laplacian) and target KD-tree; returns a cacheable init-data string. Credentials are accepted unconditionally (local use). |
| `wrap_snap_request` | Soft point-pair constraints, weight ramped over `iter_num` (20) iterations. `force` scales pull strength, `softness` scales deformation spread. |
| `wrap_attract_request` | ICP-style surface attraction: per iteration, each vertex is pulled to its closest point on the target if within `attract_distance` and `attract_angle`. |
| `wrap_stick_request` | Strong surface projection plus point pairs, landing vertices on the target surface. |

Snap/attract/stick stream per-iteration vertex positions as binary feedback blocks
(`4-byte LE size` + `float64 Nx3` payload) with a client ack after each block and a
zero-size terminator — exactly matching `client/socket_manager.py` in the add-on.
`symmetric`, `source_fixed` (pinned) and `source_free` (excluded from attraction)
are honored.

## Run locally (any OS)

```bash
cd daemon
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Start the server (port must match the add-on preference, default 31008)
.venv/bin/python wrap_daemon.py 31008
```

Verify everything with the end-to-end test (in a second terminal):

```bash
.venv/bin/python test_client.py 31008
```

The test replicates the add-on's exact socket behavior and drives a full
init → snap → attract → stick pipeline on a synthetic sphere/ellipsoid pair.

## Use with Blender (Windows)

### Option A: build wrapDaemon.exe (recommended)

On a Windows machine with Python 3.10+ installed:

```bat
cd daemon
build_windows.bat
```

Then copy `dist\wrapDaemon.exe` into the add-on's `bin\` folder:

```
add-on-wrap-for-blender\bin\wrapDaemon.exe
```

Now the add-on's **Wrap Start** button launches the daemon automatically with the
port configured in the add-on preferences, and detects it via
`Get-Process -Name wrapDaemon` — exactly as designed.

### Option B: run with Python manually

Start the daemon yourself before pressing Wrap Start:

```bash
python wrap_daemon.py 31008
```

Note: the add-on checks for a process literally named `wrapDaemon`, so with a manual
Python run it may report "Failed to start the Wrap server" — you can ignore that;
the socket client connects regardless, and Init/Snap/Attract/Stick will work.

## Flags

- `--verbose` — extra logging (request details, tracebacks).

## Files

- `wrap_daemon.py` — entry point: threaded TCP server, per-connection sessions
- `wrap_protocol.py` — JSON line framing + feedback block streaming/acks
- `wrap_solver.py` — the wrapping math (elastic solve, closest-point queries)
- `test_client.py` — end-to-end test replicating the add-on's protocol
- `build_windows.bat` — PyInstaller one-file build for Windows
