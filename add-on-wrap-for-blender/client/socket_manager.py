# Socket manager

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
    global_reader.close()
    global_reader = None

    global_socket.close()
    global_socket = None

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
    global_reader.close()
    global_reader = None

    global_socket.close()
    global_socket = None

  return ''


############################################################
# Data block

def receive_data_block(buffer: bytes) -> tuple[bytes, bytes]:
  global global_socket, global_reader

  if not global_socket:
    return buffer, b''

  try:
    # IMPORTANT: read through global_reader (the buffered file object),
    # NOT global_socket.recv() directly. Both wrap the SAME TCP connection
    # that JSON acks are read from via global_reader.readline(). readline()
    # can pull more bytes from the kernel than it returns and keep the rest
    # in its own internal buffer; a raw socket.recv() call right after has
    # no visibility into that buffered data and starts reading from the
    # wrong offset in the stream. That desync is what was corrupting the
    # vertex blocks - the 4-byte size header and/or payload would be read
    # from the wrong position, producing garbage/misplaced vertex data
    # after np.frombuffer() reshaped it in mesh_helper.py.

    # Get size
    while len(buffer) < 4:
      chunk = global_reader.read(4 - len(buffer))
      if not chunk:
        raise ConnectionError("Socket closed while reading block size")
      buffer += chunk

    size = int.from_bytes(buffer[:4], byteorder='little')
    buffer = buffer[4:]

    while len(buffer) < size:
      chunk = global_reader.read(size - len(buffer))
      if not chunk:
        raise ConnectionError("Socket closed while reading block payload")
      buffer += chunk

    # Response
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
    global_reader.close()
    global_reader = None

    global_socket.close()
    global_socket = None

  return buffer, b''
