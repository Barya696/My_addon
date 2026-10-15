#!/usr/bin/env python3
"""
Diagnostic logger: captures every request from the Blender add-on and logs
the exact parameters sent. Use this to reverse-engineer what the working
addon sends, so we can replicate the algorithm.

Usage:
  python wrap_daemon_logger.py 31008
  
Then in Blender, do: Init → Snap
Output goes to stdout and daemon_log.txt
"""

import json
import socket
import sys
import threading
import struct

LOG_FILE = "daemon_log.txt"

def log(msg):
    """Print and write to both stdout and file."""
    print(f"[logger] {msg}", flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(f"[logger] {msg}\n")

def handle_client(conn, addr):
    """Handle one client connection, log all requests."""
    log(f"client connected: {addr}")
    
    source_verts = None  # Will be set by init request
    
    try:
        # Read JSON lines until client disconnects
        buffer = b''
        while True:
            chunk = conn.recv(4096)
            if not chunk:
                break
            
            buffer += chunk
            
            # Process complete JSON lines
            while b'\n' in buffer:
                line, buffer = buffer.split(b'\n', 1)
                if not line:
                    continue
                
                try:
                    req = json.loads(line.decode('utf-8'))
                    req_type = req.get('request', 'unknown')
                    log(f"\n=== REQUEST: {req_type} ===")
                    
                    # Log all parameters
                    for key in sorted(req.keys()):
                        value = req[key]
                        if isinstance(value, list):
                            log(f"  {key}: list of {len(value)} items")
                            if key in ['pp_source_index', 'pp_target_pos']:
                                # Show first few items for point pairs
                                if value:
                                    log(f"    first items: {value[:min(3, len(value))]}")
                        elif isinstance(value, dict):
                            log(f"  {key}: dict with keys {list(value.keys())}")
                        elif isinstance(value, (int, float, bool)):
                            log(f"  {key}: {value}")
                        elif isinstance(value, str):
                            log(f"  {key}: '{value}'")
                        else:
                            log(f"  {key}: {type(value).__name__}")
                    
                    # Send appropriate response
                    if req_type == 'wrap_init_request':
                        # Store source vertex count for snap responses
                        source_verts = len(req.get('source_coords', []))
                        ack = json.dumps({"status": "init_ok"}).encode() + b'\n'
                        conn.send(ack)
                        log(f"  sent init_ok ack")
                    
                    elif req_type == 'wrap_snap_request':
                        # Log point pairs
                        pp_idx = req.get('pp_source_index', [])
                        pp_pos = req.get('pp_target_pos', [])
                        log(f"  point pairs: {len(pp_idx)} pins")
                        if pp_idx:
                            log(f"    indices: {pp_idx[:5]}")
                        
                        # Send ack
                        ack = json.dumps({"status": "snap_ok"}).encode() + b'\n'
                        conn.send(ack)
                        log(f"  sent snap_ok ack")
                        
                        # Stream back dummy vertex data so Blender doesn't timeout
                        # This is just to keep the connection alive, not real results
                        if source_verts:
                            for i in range(min(5, source_verts)):
                                # Dummy: vertex i remains at its original position
                                vert_data = struct.pack('<fff', 0.0, 0.0, 0.0)
                                size = len(vert_data)
                                conn.send(struct.pack('<I', size))
                                conn.send(vert_data)
                        
                        # Send feedback line to mark end of iteration
                        feedback = json.dumps({"iteration": 1, "dist": 0.5}).encode() + b'\n'
                        conn.send(feedback)
                    
                    else:
                        # Generic ack for unknown requests
                        ack = json.dumps({"status": "ok"}).encode() + b'\n'
                        conn.send(ack)
                    
                except json.JSONDecodeError as e:
                    log(f"JSON decode error: {e}, line: {line[:100]}")
    
    except Exception as e:
        log(f"error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        conn.close()
        log("client disconnected")

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 31008
    
    # Clear log file
    open(LOG_FILE, 'w').close()
    
    log(f"starting diagnostic logger on 127.0.0.1:{port}")
    log(f"output going to {LOG_FILE}")
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', port))
    server.listen(1)
    
    try:
        while True:
            conn, addr = server.accept()
            thread = threading.Thread(target=handle_client, args=(conn, addr))
            thread.daemon = True
            thread.start()
    except KeyboardInterrupt:
        log("shutting down...")
    finally:
        server.close()

if __name__ == '__main__':
    main()
