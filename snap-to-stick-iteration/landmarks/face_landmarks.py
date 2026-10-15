# Face landmarks for the Wrap add-on
#
# Runs Deep-MVLM (DTU-3D set, 73 landmarks, untextured "geometry+depth"
# model) on one head mesh, on the CPU, and writes the landmarks as JSON.
#
#   python face_landmarks.py head.obj landmarks.json [--views 32]
#
# The mesh must be in millimetres, centred near the origin, face looking along
# +Z with Y up (what the add-on exports). Fewer views is faster, more views
# averages out more noise (Deep-MVLM's default is 96).

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MVLM = os.path.join(HERE, 'Deep-MVLM')
CONFIG = os.path.join('configs', 'DTU3D-geometry+depth.json')


def main():
  ap = argparse.ArgumentParser(description='Face landmarks for the Wrap add-on')
  ap.add_argument('mesh', help='head mesh (.obj), millimetres, face along +Z')
  ap.add_argument('out', help='output .json')
  ap.add_argument('--views', type=int, default=32, help='number of rendered views')
  args = ap.parse_args()
  mesh = os.path.abspath(args.mesh)
  out = os.path.abspath(args.out)

  # Deep-MVLM finds its model and writes its logs relative to its own folder
  os.chdir(MVLM)
  sys.path.insert(0, MVLM)
  from parse_config import ConfigParser
  import deepmvlm

  config_args = argparse.ArgumentParser()
  config_args.add_argument('-c', '--config')
  config_args.add_argument('-d', '--device')
  config_args.add_argument('-n', '--name')
  sys.argv = [sys.argv[0], '-c', CONFIG]
  config = ConfigParser(config_args, timestamp=False)
  config.config['data_loader']['args']['n_views'] = args.views

  start = time.time()
  landmarks = deepmvlm.DeepMVLM(config).predict_one_file(mesh)
  with open(out, 'w') as f:
    json.dump({'landmark_set': 'DTU-3D',
               'landmarks': [[float(x) for x in p] for p in landmarks],
               'views': args.views,
               'seconds': round(time.time() - start, 1)}, f)
  print(f'{len(landmarks)} landmarks in {time.time() - start:.0f} s -> {out}')


if __name__ == '__main__':
  main()
