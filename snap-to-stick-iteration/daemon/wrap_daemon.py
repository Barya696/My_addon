#!/usr/bin/env python3
# wrapDaemon - local wrap server for the "add-on-wrap-for-blender" add-on
#
# Usage:
#   python wrap_daemon.py [port]        (default port: 31008)
#
# The Blender add-on launches this as bin/wrapDaemon.exe <port> and talks to
# it over TCP at 127.0.0.1:<port>. See wrap_protocol.py for the wire format.

import socket
import socketserver
import sys
import threading
import traceback

from wrap_protocol import (
    ClientConnection,
    ProtocolError,
    success_response,
    error_response,
)
from wrap_solver import WrapSession, WRAP_SOLVER_VERSION, LINEAR_SOLVER

VERBOSE = '--verbose' in sys.argv


def log(*args):
  print('[wrapDaemon]', *args, flush=True)


def vlog(*args):
  if VERBOSE:
    log(*args)


############################################################
# Request handlers

def handle_init(session: WrapSession, conn: ClientConnection, req: dict):
  # Local daemon: credentials are accepted unconditionally.
  email = req.get('email', '')
  vlog(f"init request from '{email}' "
       f"(source: {req.get('source_name')}, target: {req.get('target_name')})")

  message = session.init(req)
  conn.send_json_line(success_response(message))
  log(f"init ok: {session.n} source verts, "
      f"{len(session.target.F)} target tris, "
      f"alpha={session.alpha}, beta={session.beta}, "
      f"symmetric={session.symmetric}")


def handle_iterative(session: WrapSession, conn: ClientConnection,
                     req: dict, name: str):
  """Shared flow for snap / attract / stick: respond, then stream feedback."""
  if not session.initialized:
    conn.send_json_line(error_response(2, "not initialized"))
    return

  if name == 'snap':
    iterator = session.snap_iterations(req)
  elif name == 'attract':
    iterator = session.attract_iterations(req)
  else:
    iterator = session.stick_iterations(req)

  conn.send_json_line(success_response())
  log(f"{name} started (iter_num={req.get('iter_num')})")

  count = 0
  try:
    for vertices in iterator:
      conn.send_feedback_block(vertices)
      count += 1
  finally:
    conn.send_feedback_end()

  log(f"{name} complete ({count} iterations streamed)")


HANDLERS = {
    'wrap_init_request': handle_init,
    'wrap_snap_request': lambda s, c, r: handle_iterative(s, c, r, 'snap'),
    'wrap_attract_request': lambda s, c, r: handle_iterative(s, c, r, 'attract'),
    'wrap_stick_request': lambda s, c, r: handle_iterative(s, c, r, 'stick'),
}


############################################################
# Server

class WrapRequestHandler(socketserver.BaseRequestHandler):

  def handle(self):
    self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn = ClientConnection(self.request)
    session = WrapSession()
    peer = self.client_address
    log(f"client connected: {peer[0]}:{peer[1]}")

    try:
      while True:
        try:
          req = conn.read_json_line()
        except ProtocolError as e:
          log(f"protocol error: {e}")
          conn.send_json_line(error_response(3, "bad request", str(e)))
          continue

        if req is None:
          break  # client disconnected

        request_name = req.get('request', '')
        log(f"request received: '{request_name}'")
        handler = HANDLERS.get(request_name)

        if handler is None:
          log(f"unknown request: '{request_name}'")
          conn.send_json_line(
              error_response(4, f"unknown request '{request_name}'"))
          continue

        try:
          handler(session, conn, req)
        except (ValueError, RuntimeError) as e:
          log(f"{request_name} failed: {e}")
          conn.send_json_line(error_response(1, str(e)))
        except (BrokenPipeError, ConnectionResetError):
          raise
        except Exception as e:
          log(f"{request_name} crashed: {e}")
          if VERBOSE:
            traceback.print_exc()
          conn.send_json_line(error_response(5, f"internal error: {e}"))

    except (BrokenPipeError, ConnectionResetError):
      pass
    finally:
      conn.close()
      log(f"client disconnected: {peer[0]}:{peer[1]}")


class ThreadedTCPServer(socketserver.ThreadingTCPServer):
  allow_reuse_address = True
  daemon_threads = True


############################################################
# Main

def main():
  port = 31008
  for arg in sys.argv[1:]:
    if arg.isdigit():
      port = int(arg)
      break

  server = ThreadedTCPServer(('127.0.0.1', port), WrapRequestHandler)
  log(f"solver version: {WRAP_SOLVER_VERSION} (linear solver: {LINEAR_SOLVER})")
  log(f"listening on 127.0.0.1:{port}")

  try:
    server.serve_forever()
  except KeyboardInterrupt:
    log("shutting down")
  finally:
    server.shutdown()
    server.server_close()


if __name__ == '__main__':
  main()
