#!/usr/bin/env python3
"""
GPU utility functions for detecting and using CUDA/ROCm GPUs
"""
import os
import logging
import torch


def is_gpu_available():
    """
    Check if GPU is available (CUDA or ROCm).
    Returns True if GPU is available, False otherwise.
    """
    return torch.cuda.is_available()


def get_gpu_count():
    """Get the number of available GPUs."""
    if is_gpu_available():
        return torch.cuda.device_count()
    return 0


def get_gpu_name(device_id=0):
    """Get the name of the GPU at device_id."""
    if is_gpu_available() and device_id < get_gpu_count():
        return torch.cuda.get_device_name(device_id)
    return None


def setup_gpu(gpu_id=-1):
    """
    Setup GPU device.
    
    Args:
        gpu_id: GPU ID to use (-1 for CPU)
    
    Returns:
        device: torch.device object
        gpu_id: Actual GPU ID used (-1 if CPU)
    """
    if gpu_id >= 0:
        if is_gpu_available():
            if gpu_id >= get_gpu_count():
                logging.warning(
                    f'GPU {gpu_id} requested but only {get_gpu_count()} GPUs available. '
                    f'Using GPU 0 instead.'
                )
                gpu_id = 0
            torch.cuda.set_device(gpu_id)
            device = torch.device('cuda')
            logging.info(f'Using GPU {gpu_id}: {get_gpu_name(gpu_id)}')
            return device, gpu_id
        else:
            logging.warning(
                f'GPU {gpu_id} requested but GPU is not available. '
                f'Falling back to CPU.'
            )
            return torch.device('cpu'), -1
    else:
        logging.info('Using CPU')
        return torch.device('cpu'), -1


def diagnose_gpu():
    """
    Diagnose GPU availability and print diagnostic information.
    """
    print("=" * 60)
    print("GPU Diagnostic Information")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    
    # Check HIP/ROCm
    if hasattr(torch.version, 'hip'):
        hip_version = torch.version.hip
        print(f"HIP version: {hip_version}")
        if hip_version:
            print("  -> ROCm/HIP build detected")
    
    # Check CUDA
    if hasattr(torch.version, 'cuda'):
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")
    
    print(f"\nGPU Available: {is_gpu_available()}")
    print(f"GPU Count: {get_gpu_count()}")
    
    if is_gpu_available():
        for i in range(get_gpu_count()):
            print(f"  GPU {i}: {get_gpu_name(i)}")
    else:
        print("\nGPU is not available. Possible reasons:")
        print("  1. ROCm driver not installed or not properly configured")
        print("  2. GPU not recognized by the system")
        print("  3. Environment variables not set (ROCM_PATH, HIP_PLATFORM)")
        print("\nChecking environment variables:")
        print(f"  ROCM_PATH: {os.environ.get('ROCM_PATH', 'Not set')}")
        print(f"  HIP_PLATFORM: {os.environ.get('HIP_PLATFORM', 'Not set')}")
        print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # Try to create a tensor on GPU
    if is_gpu_available():
        try:
            x = torch.randn(2, 3).cuda()
            print(f"\n✓ Successfully created tensor on GPU: {x.device}")
            del x
        except Exception as e:
            print(f"\n✗ Error creating tensor on GPU: {e}")
    
    print("=" * 60)


if __name__ == '__main__':
    diagnose_gpu()

