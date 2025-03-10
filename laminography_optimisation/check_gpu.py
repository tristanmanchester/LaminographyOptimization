#!/usr/bin/env python
"""Check GPU info for laminography optimisation."""

import sys
print(f"Python executable: {sys.executable}")

try:
    import astra
    print(f"ASTRA is available")
    print(f"CUDA available: {astra.astra.use_cuda()}")
    if astra.astra.use_cuda():
        print("Testing CUDA...")
        astra.test_CUDA()
except ImportError:
    print("ASTRA not available")

try:
    import torch
    print(f"PyTorch available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
except ImportError:
    print("PyTorch not available")

try:
    from cil.framework import ImageGeometry, AcquisitionGeometry
    from cil.recon import FBP
    print("CIL is available")
except ImportError:
    print("CIL not available")

# Try to get memory usage info 
try:
    import os
    import psutil
    process = psutil.Process(os.getpid())
    print(f"Memory used by Python: {process.memory_info().rss / (1024 * 1024):.2f} MB")
    print(f"Available system memory: {psutil.virtual_memory().available / (1024 * 1024 * 1024):.2f} GB")
    print(f"Total system memory: {psutil.virtual_memory().total / (1024 * 1024 * 1024):.2f} GB")
except ImportError:
    print("psutil not available")