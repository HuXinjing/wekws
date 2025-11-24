#!/bin/bash
# ROCm 环境检查和配置脚本

echo "=========================================="
echo "ROCm Environment Check and Setup"
echo "=========================================="

# 检查是否在 WSL2
if grep -qEi "(Microsoft|WSL)" /proc/version 2>/dev/null; then
    echo "✓ Running in WSL2"
    IS_WSL2=true
else
    echo "✗ Not running in WSL2"
    IS_WSL2=false
fi

# 检查 ROCm 安装
echo ""
echo "1. Checking ROCm installation..."
ROCM_PATHS=("/opt/rocm" "/opt/rocm-5.7" "/usr/local/rocm")
ROCM_FOUND=false

for path in "${ROCM_PATHS[@]}"; do
    if [ -d "$path" ]; then
        echo "  ✓ Found ROCm at: $path"
        ROCM_FOUND=true
        export ROCM_PATH="$path"
        export PATH="$ROCM_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$ROCM_PATH/lib:$LD_LIBRARY_PATH"
        break
    fi
done

if [ "$ROCM_FOUND" = false ]; then
    echo "  ✗ ROCm not found"
    echo ""
    echo "  To install ROCm for WSL2:"
    echo "    sudo apt update"
    echo "    sudo apt install rocm-dkms rocm-libs rocm-dev"
fi

# 检查 rocm-smi
echo ""
echo "2. Checking rocm-smi..."
if command -v rocm-smi &> /dev/null; then
    echo "  ✓ rocm-smi found"
    rocm-smi
else
    echo "  ✗ rocm-smi not found"
fi

# 检查 PyTorch
echo ""
echo "3. Checking PyTorch ROCm support..."
cd "$(dirname "$0")"
source .venv/bin/activate 2>/dev/null || true
python3 << 'EOF'
import torch
print(f"  PyTorch version: {torch.__version__}")
if hasattr(torch.version, 'hip'):
    hip_ver = torch.version.hip
    print(f"  HIP version: {hip_ver}")
    if hip_ver:
        print("  ✓ PyTorch has ROCm support")
    else:
        print("  ✗ PyTorch ROCm support not available")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("  ✗ GPU not available")
EOF

# 检查环境变量
echo ""
echo "4. Environment variables:"
echo "  ROCM_PATH: ${ROCM_PATH:-Not set}"
echo "  HIP_PLATFORM: ${HIP_PLATFORM:-Not set}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"

# 建议
echo ""
echo "=========================================="
echo "Recommendations:"
echo "=========================================="

if [ "$ROCM_FOUND" = false ]; then
    echo "1. Install ROCm runtime (if you have AMD GPU)"
    echo "2. Or use CPU training by setting gpus='-1' in run_frame_stream.sh"
else
    if ! command -v rocm-smi &> /dev/null; then
        echo "1. ROCm installed but rocm-smi not in PATH"
        echo "   Add to ~/.bashrc:"
        echo "   export PATH=\$ROCM_PATH/bin:\$PATH"
    fi
fi

echo ""
echo "To test GPU availability:"
echo "  python wekws/utils/gpu_utils.py"

