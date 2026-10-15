# Socket manager
#
# PATCHED: receive_data_block now reads through the SAME buffered reader
# (global_reader) that receive_on_socket uses for JSON lines.
#
# Why: global_reader = global_socket.makefile('rb') is an io.BufferedReader.
# Its readline() does not stop reading at the newline on the OS level - it
# pulls a full internal buffer's worth of bytes (up to 8 KB) from the kernel,
# finds the '\n', and keeps the remainder in its private buffer. The daemon
# sends the JSON ack immediately followed by binary vertex blocks on the same
# TCP connection, so readline() frequently steals the first bytes of the
# binary stream. The old receive_data_block() then read from the RAW socket
# (global_socket.recv), which cannot see those stolen bytes - the 4-byte size
# header it parsed was garbage and every block after was misaligned, which
# scrambled the decoded float64 vertex coordinates.
#
# The fix: ALL reads go through global_reader. Writes (sendall) are unbuffered
# and remain on the raw socket, which is safe.

import json
import socket

from ..preferences.preferences import get_wrap_prefs

global_socket = None
global_reader = None


############################################################
# Init

def init_socket(context):
  global global_socket, global_reader

  if global_socket:
    return

  wrap_prefs = get_wrap_prefs(context)
  server_port = wrap_prefs.server_port

  global_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

  try:
    server_host = '127.0.0.1'
    global_socket.connect((server_host, server_port))
    global_reader = global_socket.makefile('rb')
  except ConnectionRefusedError:
    print(
        f"Connection failed. Is the Wrap server running at {server_host}:{server_port}?")
    global_socket = None
  except Exception as e:
    print(f"An error occurred: {e}")
    global_socket = None


############################################################
# Close helper

def close_socket():
  """Close and reset the socket and its buffered reader."""
  global global_socket, global_reader

  if global_reader:
    try:
      global_reader.close()
    except Exception:
      pass
    global_reader = None

  if global_socket:
    try:
      global_socket.close()
    except Exception:
      pass
    global_socket = None


############################################################
# Send

def send_on_socket(context, data):
  global global_socket, global_reader

  if not global_socket:
    init_socket(context)

  if not global_socket:
    return False

  try:
    global_socket.sendall(data)
    return True
  except Exception as e:
    print(f"An error occurred: {e}")
    close_socket()

  return False


############################################################
# Receive

def receive_on_socket(context) -> str:
  global global_socket, global_reader

  if not global_socket:
    init_socket(context)

  if not global_socket:
    return ''

  try:
    line = global_reader.readline()
    data = line.decode('utf-8').rstrip()
    return data
  except Exception as e:
    print(f"An error occurred: {e}")
    close_socket()

  return ''


############################################################
# Data block

def _read_exact(n: int) -> bytes:
  """Read exactly n bytes through the buffered reader.

  MUST use global_reader (not global_socket.recv) so that any bytes the
  buffered reader pulled in during readline() are consumed in order.
  """
  global global_reader

  data = b''
  while len(data) < n:
    chunk = global_reader.read(n - len(data))
    if not chunk:
      raise ConnectionError("Connection closed while reading data block")
    data += chunk
  return data


def receive_data_block(buffer: bytes) -> tuple[bytes, bytes]:
  global global_socket, global_reader

  if not global_socket:
    return buffer, b''

  try:
    # Get size (4-byte little-endian header) - via the buffered reader
    while len(buffer) < 4:
      buffer += _read_exact(4 - len(buffer))

    size = int.from_bytes(buffer[:4], byteorder='little')
    buffer = buffer[4:]

    while len(buffer) < size:
      buffer += _read_exact(size - len(buffer))

    # Response (writes stay on the raw socket - safe)
    message_string = json.dumps({
        "status": "success",
        "code": "feedback received",
        "length": size + 4,
    }) + '\n'
    message_bytes = message_string.encode('utf-8')

    global_socket.sendall(message_bytes)

    return buffer[size:], buffer[:size]
  except Exception as e:
    print(f"An error occurred: {e}")
    close_socket()

  return buffer, b''
