# Client for wrap

import json
import numpy as np

from .socket_manager import send_on_socket, receive_on_socket
from ..core import vertex_store
from ..preferences.preferences import get_wrap_prefs
from ..utils.point_pair_helper import get_point_pair_index_pos
from ..utils.mesh_helper import (
    get_target_vertices,
    get_target_normals,
    get_source_faces,
    get_source_triangles,
    get_target_faces,
    get_target_triangles
)


############################################################
# Global variables

global_version_number = 1


############################################################
# Send

def send_data(context, data):
  """
  Send wrap init request to server
  """
  response = None

  message_string = json.dumps(data) + '\n'
  message_bytes = message_string.encode('utf-8')

  if send_on_socket(context, message_bytes):
    received = receive_on_socket(context)

    if received:
      try:
        response = json.loads(received)
      except json.JSONDecodeError:
        print("Failed to decode JSON from server response")

  return response


############################################################
# Wrap init

def set_wrap_init_data(context, data):
  """
  Set wrap init data
  """
  wrap_options = context.scene.wrap_options

  init_data = ''
  try:
    init_data = json.dumps({
        'flex': wrap_options.wrap_init_flex,
        'accuracy': wrap_options.wrap_init_accuracy,
        'data': data
    })
  except json.JSONDecodeError:
    print("Unable to dump the initialization data as a JSON string")

  wrap_options.wrap_init_data = init_data


def get_wrap_init_data(context):
  """
  Get send data for wrap init
  """
  global global_version_number

  wrap_options = context.scene.wrap_options

  init_data = wrap_options.wrap_init_data
  flex = wrap_options.wrap_init_flex
  accuracy = wrap_options.wrap_init_accuracy

  data = ''
  if init_data:
    try:
      data_dict = json.loads(init_data)
      if 'flex' in data_dict:
        if flex == data_dict['flex'] and accuracy == data_dict['accuracy']:
          data = data_dict['data']
    except json.JSONDecodeError:
      print("Failed to decode JSON from wrap_init_data property")
    except Exception:
      print("An exception occurred while parsing the wrap_init_data property")

  wrap_prefs = get_wrap_prefs(context)

  user_email = wrap_prefs.user_email
  user_password = wrap_prefs.user_password

  source_name = wrap_options.source_mesh_name
  source_coords = vertex_store.get(wrap_options, 'original').tolist()
  source_faces = get_source_faces(context)
  if len(source_faces[0]) != 3:
    source_tris = get_source_triangles(context)

  source_free = [int(x) for x in wrap_options.free_index.split()]
  source_fixed = [int(x) for x in wrap_options.fixed_index.split()]

  target_name = wrap_options.target_mesh_name
  target_coords = get_target_vertices(context)
  target_normals = get_target_normals(context)
  target_faces = get_target_faces(context)
  if len(target_faces[0]) != 3:
    target_tris = get_target_triangles(context)

  return {
      "request": "wrap_init_request",
      "email": user_email,
      "password": user_password,
      "source_name": source_name,
      "source_coords": source_coords,
      "source_tris": source_faces if len(source_faces[0]) == 3 else source_tris,
      "source_quads": source_faces if len(source_faces[0]) == 4 else [],
      "source_free": source_free,
      "source_fixed": source_fixed,
      "target_name": target_name,
      "target_coords": [list(v) for v in target_coords],
      "target_normals": [list(v) for v in target_normals],
      "target_tris": target_faces if len(target_faces[0]) == 3 else target_tris,
      "target_quads": target_faces if len(target_faces[0]) == 4 else [],
      "data": data,
      # The source is symmetric, the target is not (a pose is only symmetric
      # at rest): the solver must not make the result symmetric
      "symmetric": 0,
      "alpha": flex,
      "beta": flex * accuracy,
      "version": global_version_number
  }


def wrap_init(context):
  """
  Send wrap init request to server
  """
  data_to_send = get_wrap_init_data(context)
  return send_data(context, data_to_send)


############################################################
# Wrap snap

def get_wrap_snap_data(context):
  """
  Get send data for wrap snap
  """
  wrap_options = context.scene.wrap_options

  source_index, target_pos = get_point_pair_index_pos(context)

  # Snap drags each pair like G with proportional editing: send the scene's
  # own proportional size and falloff (Connected Only is always off)
  tool_settings = context.scene.tool_settings

  return {
      "request": "wrap_snap_request",
      "pp_source_index": source_index,
      "pp_target_pos": target_pos,
      "force": wrap_options.wrap_snap_force,
      "softness": wrap_options.wrap_snap_softness,
      "proportional_size": tool_settings.proportional_size,
      "proportional_falloff": tool_settings.proportional_edit_falloff,
      "iter_num": 20
  }


def wrap_snap(context):
  """
  Send wrap init request to server
  """
  data_to_send = get_wrap_snap_data(context)
  return send_data(context, data_to_send)


############################################################
# Wrap attract

def get_wrap_attract_data(context):
  """
  Get send data for wrap attract
  """
  wrap_options = context.scene.wrap_options

  diag_len = wrap_options.source_diagonal_length

  distance = wrap_options.wrap_attract_distance
  angle = wrap_options.wrap_attract_angle

  return {
      "request": "wrap_attract_request",
      "attract_distance": distance * diag_len * 0.01,
      "attract_angle": angle,
      "iter_num": 20
  }


def wrap_attract(context):
  """
  Send wrap init request to server
  """
  data_to_send = get_wrap_attract_data(context)
  return send_data(context, data_to_send)


############################################################
# Wrap stick

def get_wrap_stick_data(context):
  """
  Get send data for wrap stick
  """
  wrap_options = context.scene.wrap_options

  source_index, target_pos = get_point_pair_index_pos(context)

  diag_len = wrap_options.source_diagonal_length

  distance = wrap_options.wrap_stick_distance
  angle = wrap_options.wrap_stick_angle

  return {
      "request": "wrap_stick_request",
      "pp_source_index": source_index,
      "pp_target_pos": target_pos,
      "stick_distance": distance * diag_len * 0.01,
      "stick_angle": angle,
      "iter_num": 10
  }


def wrap_stick(context):
  """
  Send wrap init request to server
  """
  data_to_send = get_wrap_stick_data(context)
  return send_data(context, data_to_send)
