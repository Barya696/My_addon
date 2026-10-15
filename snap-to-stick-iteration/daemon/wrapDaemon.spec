# -*- mode: python ; coding: utf-8 -*-
import os
import sys

# Intel MKL for PARDISO (pypardiso). pypardiso looks for mkl_rt under
# <prefix>/Library/bin, so the files go to the same place inside the build
# (_internal\Library\bin): the runtime, the core, the threading layer and
# every CPU's kernels.
MKL_DIR = os.path.join(sys.prefix, 'Library', 'bin')
MKL_FILES = ['mkl_rt.3.dll', 'mkl_core.3.dll', 'mkl_intel_thread.3.dll', 'libiomp5md.dll',
             'mkl_def.3.dll', 'mkl_mc3.3.dll', 'mkl_avx2.3.dll', 'mkl_avx512.3.dll', 'mkl_avx10.3.dll']
mkl_binaries = [(os.path.join(MKL_DIR, f), os.path.join('Library', 'bin')) for f in MKL_FILES]

a = Analysis(
    ['wrap_daemon.py'],
    pathex=[],
    binaries=mkl_binaries,
    datas=[],
    hiddenimports=['pypardiso'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# One-folder build: dist\wrapDaemon\ holds wrapDaemon.exe and its _internal
# folder. Unlike a one-file exe it runs as a single process (the add-on's
# Start check and its stop on quit look for exactly one wrapDaemon process)
# and has nothing to unpack at startup.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='wrapDaemon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='wrapDaemon',
)
