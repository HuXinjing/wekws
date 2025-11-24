#!/bin/bash
# 修复 ROCm 仓库配置脚本

echo "=========================================="
echo "ROCm Repository Configuration Fix"
echo "=========================================="

# 检测系统版本
if [ -f /etc/os-release ]; then
    . /etc/os-release
    UBUNTU_CODENAME=$(lsb_release -cs 2>/dev/null || echo "noble")
    echo "Detected Ubuntu codename: $UBUNTU_CODENAME"
else
    UBUNTU_CODENAME="noble"
    echo "Using default codename: $UBUNTU_CODENAME"
fi

echo ""
echo "Current ROCm repository configuration:"
cat /etc/apt/sources.list.d/rocm.list 2>/dev/null || echo "  Not configured"

echo ""
echo "Options:"
echo "1. Update to noble (Ubuntu 24.04)"
echo "2. Update to ROCm 6.1 with noble"
echo "3. Keep jammy (Ubuntu 22.04) - may have compatibility issues"
echo "4. Cancel"

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Updating to noble..."
        echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ noble main" | sudo tee /etc/apt/sources.list.d/rocm.list
        echo "✓ Updated to noble"
        ;;
    2)
        echo ""
        echo "Updating to ROCm 6.1 with noble..."
        echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.1/ noble main" | sudo tee /etc/apt/sources.list.d/rocm.list
        echo "✓ Updated to ROCm 6.1"
        ;;
    3)
        echo ""
        echo "Keeping jammy..."
        echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list
        echo "✓ Kept jammy"
        ;;
    4)
        echo "Cancelled"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "Updated configuration:"
cat /etc/apt/sources.list.d/rocm.list

echo ""
echo "Next steps:"
echo "1. Run: sudo apt update"
echo "2. Try: sudo apt install rocm-dkms rocm-libs rocm-dev"
echo "3. Or use: sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms"

