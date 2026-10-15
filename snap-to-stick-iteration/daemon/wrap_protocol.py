# Wrap daemon protocol helpers
#
# Wire protocol (matches the Blender add-on's client/socket_manager.py):
#   - Requests/responses: newline-delimited UTF-8 JSON
#   - Feedback stream (snap/attract/stick): repeated binary blocks of
#       [4-byte little-endian uint32 size][payload]
#     where payload is float64 world-space vertex positions, shape (N, 3).
#     After EVERY block (including the zero-size terminator) the client sends
#     an ack JSON line: {"status": "success", "code": "feedback received",
#     "length": size + 4}. The server must read this ack before sending the
#     next block (flow control).

import json
import struct

import numpy as np


class ProtocolError(Exception):
  pass


class ClientConnection:
  """Wraps a connected socket with line-based JSON and block streaming."""

  def __init__(self, sock):
    self.sock = sock
    self.reader = sock.makefile('rb')

  def close(self):
    try:
      self.reader.close()
    except Exception:
      pass
    try:
      self.sock.close()
    except Exception:
      pass

  ############################################################
  # JSON lines

  def read_json_line(self):
    """Read one newline-terminated JSON message. Returns None on EOF."""
    line = self.reader.readline()
    if not line:
      return None

    text = line.decode('utf-8').strip()
    if not text:
      return {}

    try:
      return json.loads(text)
    except json.JSONDecodeError as e:
      raise ProtocolError(f"Invalid JSON from client: {e}")

  def send_json_line(self, obj):
    message = json.dumps(obj) + '\n'
    self.sock.sendall(message.encode('utf-8'))

  ############################################################
  # Feedback blocks

  def send_feedback_block(self, vertices: np.ndarray):
    """Send one feedback block (float64 Nx3) and wait for the client ack."""
    payload = np.ascontiguousarray(vertices, dtype=np.float64).tobytes()
    header = struct.pack('<I', len(payload))
    self.sock.sendall(header + payload)
    self._read_ack()

  def send_feedback_end(self):
    """Send the zero-size terminator block and wait for the final ack."""
    self.sock.sendall(struct.pack('<I', 0))
    self._read_ack()

  def _read_ack(self):
    ack = self.read_json_line()
    if ack is None:
      raise ProtocolError("Connection closed while waiting for feedback ack")


############################################################
# Response helpers

def success_response(message=''):
  return {"code": 0, "status": "success", "message": message}


def error_response(code, status, message=''):
  return {"code": code, "status": status, "message": message}
