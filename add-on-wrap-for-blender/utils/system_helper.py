# System helper

import subprocess
from pathlib import Path


############################################################
# Global variables

global_wrap_server_name = 'wrapDaemon'


############################################################
# Wrap server

def start_process(exePath: str):
  try:
    # subprocess.Popen(f'start {exePath}', shell=True)
    subprocess.Popen(["powershell", "-Command", f"Start-Process {exePath}"])
  except Exception as e:
    print(f"Error: {e}")


#  NPM(K)    PM(M)      WS(M)     CPU(s)      Id  SI ProcessName
#  ------    -----      -----     ------      --  -- -----------
#      12     3.59      10.42       0.03    8496   1 wrapDaemon

def get_process_pid(name: str) -> int:
  pid = -1

  try:
    command = f'Get-Process -Name {name} -ErrorAction SilentlyContinue'
    output = subprocess.run(
        ["powershell", "-Command", command], capture_output=True, text=True)

    if output.returncode == 0:
      lines = [l for l in output.stdout.splitlines() if l.strip()]
      if len(lines) == 3:
        items = lines[2].strip().split()
        if items[-1] == name:
          pid = int(items[-3])
  except Exception as e:
    print(f"Error: {e}")

  return pid


def kill_process(pid: int) -> bool:
  try:
    output = subprocess.run(['taskkill', '/F', '/PID', str(pid)], check=True)
    return output.returncode == 0
  except Exception as e:
    print(f"Error: {e}")

  return False


############################################################
# Path

def get_addon_dir():
  return Path(__file__).resolve().parent.parent


############################################################
# Script

if __name__ == '__main__':

  addon_dir = get_addon_dir()

  if False:
    processName = global_wrap_server_name
    start_process(f'{addon_dir}\\bin\\{processName}.exe')

    pid = get_process_pid(processName)
    kill_process(pid)

  print('Pause')
