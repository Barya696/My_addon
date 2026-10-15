#!/usr/bin/env python3
# End-to-end test client for wrapDaemon.
#
# Replicates the exact socket behavior of the Blender add-on
# (client/socket_manager.py + client/wrap_client.py):
#   - newline-delimited JSON requests/responses
#   - binary feedback blocks with ack lines
#
# Scenario: a UV sphere (source) is wrapped onto a scaled/offset ellipsoid
# (target) through the full pipeline: init -> snap -> attract -> stick.
#
# Usage: python test_client.py [port]

import json
import socket
import sys

import numpy as np


############################################################
# Client (mirrors socket_manager.py)

class WrapClient:

  def __init__(self, host='127.0.0.1', port=31008):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.connect((host, port))
    self.reader = self.sock.makefile('rb')

  def close(self):
    self.reader.close()
    self.sock.close()

  def send_request(self, data):
    """Exactly like wrap_client.send_data: JSON line out, JSON line back."""
    message = json.dumps(data) + '\n'
    self.sock.sendall(message.encode('utf-8'))
    line = self.reader.readline().decode('utf-8').rstrip()
    return json.loads(line) if line else None

  def _read_exact(self, n):
    """Read exactly n bytes via the SAME buffered reader used for JSON
    lines. Mixing reader.readline() with raw sock.recv() desyncs the
    stream: readline() over-reads into its private buffer, and recv()
    starts mid-stream with a garbage size header."""
    data = b''
    while len(data) < n:
      chunk = self.reader.read(n - len(data))
      if not chunk:
        raise ConnectionError("server closed connection")
      data += chunk
    return data

  def receive_data_block(self, buffer):
    """Like the PATCHED socket_manager.receive_data_block (buffered reads)."""
    while len(buffer) < 4:
      buffer += self._read_exact(4 - len(buffer))

    size = int.from_bytes(buffer[:4], byteorder='little')
    buffer = buffer[4:]

    while len(buffer) < size:
      buffer += self._read_exact(size - len(buffer))

    ack = json.dumps({
        "status": "success",
        "code": "feedback received",
        "length": size + 4,
    }) + '\n'
    self.sock.sendall(ack.encode('utf-8'))

    return buffer[size:], buffer[:size]

  def recv_feedback(self, expected_n):
    """Like op_wrap_operators.recv_feedback: read blocks until zero block."""
    buffer = b''
    frames = []
    while True:
      buffer, block = self.receive_data_block(buffer)
      if not block:
        break
      verts = np.frombuffer(block, np.float64).reshape((-1, 3))
      assert len(verts) == expected_n, \
          f"frame has {len(verts)} verts, expected {expected_n}"
      frames.append(verts)
    return frames


############################################################
# Test meshes

def make_uv_sphere(radius=1.0, rings=16, segments=24):
  """UV sphere with quads on the body and triangles at the poles."""
  verts = [[0.0, 0.0, radius]]
  for r in range(1, rings):
    phi = np.pi * r / rings
    for s in range(segments):
      theta = 2.0 * np.pi * s / segments
      verts.append([
          radius * np.sin(phi) * np.cos(theta),
          radius * np.sin(phi) * np.sin(theta),
          radius * np.cos(phi),
      ])
  verts.append([0.0, 0.0, -radius])

  tris, quads = [], []
  # Top cap
  for s in range(segments):
    tris.append([0, 1 + s, 1 + (s + 1) % segments])
  # Body quads
  for r in range(rings - 2):
    ra = 1 + r * segments
    rb = 1 + (r + 1) * segments
    for s in range(segments):
      s2 = (s + 1) % segments
      quads.append([ra + s, rb + s, rb + s2, ra + s2])
  # Bottom cap
  last = len(verts) - 1
  rb = 1 + (rings - 2) * segments
  for s in range(segments):
    tris.append([last, rb + (s + 1) % segments, rb + s])

  return np.asarray(verts), tris, quads


def surface_distance(V, Vt, Ft):
  """Mean distance from V to the target surface (dense check, test only)."""
  sys.path.insert(0, '.')
  from wrap_solver import TargetSurface
  surf = TargetSurface(Vt, np.asarray(Ft))
  _, dist, _ = surf.closest_points(V)
  return float(dist.mean())


############################################################
# Test run

def main():
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 31008

  # Source: unit sphere. Target: ellipsoid, offset so snap has work to do.
  src_v, src_tris, src_quads = make_uv_sphere(radius=1.0)
  tgt_v, tgt_tris, tgt_quads = make_uv_sphere(radius=1.0, rings=20, segments=28)
  tgt_v = tgt_v * np.array([1.4, 1.1, 0.8]) + np.array([0.3, 0.15, 0.1])

  # Triangulate target for normals / distance checks
  tgt_f = list(tgt_tris)
  for q in tgt_quads:
    tgt_f.append([q[0], q[1], q[2]])
    tgt_f.append([q[0], q[2], q[3]])

  from wrap_solver import vertex_normals
  tgt_normals = vertex_normals(tgt_v, np.asarray(tgt_f))

  n = len(src_v)
  client = WrapClient(port=port)
  print(f"Connected. Source {n} verts, target {len(tgt_v)} verts.")

  ############################################################
  # 1. Init
  init_req = {
      "request": "wrap_init_request",
      "email": "test@example.com",
      "password": "anything",
      "source_name": "TestSphere",
      "source_coords": src_v.tolist(),
      "source_tris": src_tris,
      "source_quads": src_quads,
      "source_free": [],
      "source_fixed": [],
      "target_name": "TestEllipsoid",
      "target_coords": tgt_v.tolist(),
      "target_normals": tgt_normals.tolist(),
      "target_tris": tgt_tris,
      "target_quads": tgt_quads,
      "data": '',
      "symmetric": 0,
      "alpha": 1.0,       # flex
      "beta": 0.5,        # flex * accuracy
      "version": 1,
  }
  resp = client.send_request(init_req)
  assert resp and resp['code'] == 0, f"init failed: {resp}"
  assert resp['message'], "init returned no cacheable message"
  print(f"[PASS] init: code=0, message cached ({len(resp['message'])} chars)")

  # Re-init with cached data (should also succeed)
  init_req['data'] = resp['message']
  resp2 = client.send_request(init_req)
  assert resp2 and resp2['code'] == 0, f"cached init failed: {resp2}"
  assert resp2['message'] == resp['message'], "cache message mismatch"
  print("[PASS] init with cached data: code=0, same message")

  d0 = surface_distance(src_v, tgt_v, tgt_f)
  print(f"       mean distance to target before wrap: {d0:.4f}")

  ############################################################
  # 2. Snap - use 6 point pairs (poles + equator) like user-picked pairs
  pp_index = [0, n - 1]
  pp_target = [tgt_v[0].tolist(), tgt_v[len(tgt_v) - 1].tolist()]
  equator_src = 1 + 7 * 24  # ring 8 of 16, first segment
  for k in range(4):
    i = equator_src + k * 6
    p = src_v[i] * np.array([1.4, 1.1, 0.8]) + np.array([0.3, 0.15, 0.1])
    pp_index.append(int(i))
    pp_target.append(p.tolist())

  resp = client.send_request({
      "request": "wrap_snap_request",
      "pp_source_index": pp_index,
      "pp_target_pos": pp_target,
      "force": 1.0,
      "softness": 1.0,
      "iter_num": 20,
  })
  assert resp and resp['code'] == 0, f"snap failed: {resp}"
  frames = client.recv_feedback(n)
  assert len(frames) == 20, f"expected 20 snap frames, got {len(frames)}"
  d_snap = surface_distance(frames[-1], tgt_v, tgt_f)
  print(f"[PASS] snap: 20 frames streamed, mean dist {d0:.4f} -> {d_snap:.4f}")
  assert d_snap < d0, "snap did not move source toward target"

  ############################################################
  # 3. Attract
  diag = float(np.linalg.norm(src_v.max(0) - src_v.min(0)))
  resp = client.send_request({
      "request": "wrap_attract_request",
      "attract_distance": 30.0 * diag * 0.01,
      "attract_angle": 65.0,
      "iter_num": 20,
  })
  assert resp and resp['code'] == 0, f"attract failed: {resp}"
  frames = client.recv_feedback(n)
  assert len(frames) == 20, f"expected 20 attract frames, got {len(frames)}"
  d_attract = surface_distance(frames[-1], tgt_v, tgt_f)
  print(f"[PASS] attract: 20 frames, mean dist {d_snap:.4f} -> {d_attract:.4f}")
  assert d_attract < d_snap, "attract did not improve fit"

  ############################################################
  # 4. Stick
  resp = client.send_request({
      "request": "wrap_stick_request",
      "pp_source_index": pp_index,
      "pp_target_pos": pp_target,
      "stick_distance": 50.0 * diag * 0.01,
      "stick_angle": 80.0,
      "iter_num": 10,
  })
  assert resp and resp['code'] == 0, f"stick failed: {resp}"
  frames = client.recv_feedback(n)
  assert len(frames) == 10, f"expected 10 stick frames, got {len(frames)}"
  d_stick = surface_distance(frames[-1], tgt_v, tgt_f)
  print(f"[PASS] stick: 10 frames, mean dist {d_attract:.4f} -> {d_stick:.4f}")
  assert d_stick < d_attract, "stick did not improve fit"
  assert d_stick < 0.02 * diag, \
      f"final fit too loose: {d_stick:.4f} (diag {diag:.3f})"

  ############################################################
  # 5. Error handling: unknown request
  resp = client.send_request({"request": "bogus_request"})
  assert resp and resp['code'] != 0, "unknown request should return an error"
  print(f"[PASS] unknown request rejected: code={resp['code']}")

  client.close()

  # 6. Fresh connection without init -> snap must fail cleanly
  client2 = WrapClient(port=port)
  resp = client2.send_request({
      "request": "wrap_snap_request",
      "pp_source_index": [0], "pp_target_pos": [[0, 0, 0]],
      "force": 1.0, "softness": 1.0, "iter_num": 5,
  })
  assert resp and resp['code'] != 0, "snap before init should fail"
  print(f"[PASS] snap before init rejected: code={resp['code']}")
  client2.close()

  print("\nAll tests passed.")
  print(f"Fit summary: start {d0:.4f} -> snap {d_snap:.4f} "
        f"-> attract {d_attract:.4f} -> stick {d_stick:.4f}")


if __name__ == '__main__':
  main()
