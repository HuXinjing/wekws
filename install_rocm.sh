#!/bin/bash
# ROCm 安装脚本 for WSL2

set -e

echo "=========================================="
echo "ROCm Installation Script for WSL2"
echo "=========================================="

# 检查系统版本
echo ""
echo "1. Checking system version..."
if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "  OS: $ID"
    echo "  Version: $VERSION_ID"
    
    # 确定 Ubuntu 版本对应的代号
    case "$VERSION_ID" in
        "22.04")
            UBUNTU_CODENAME="jammy"
            ;;
        "20.04")
            UBUNTU_CODENAME="focal"
            ;;
        "18.04")
            UBUNTU_CODENAME="bionic"
            ;;
        *)
            echo "  ⚠ Unknown Ubuntu version, using jammy as default"
            UBUNTU_CODENAME="jammy"
            ;;
    esac
    echo "  Codename: $UBUNTU_CODENAME"
else
    echo "  ⚠ Cannot determine OS version, using jammy as default"
    UBUNTU_CODENAME="jammy"
fi

# 检查是否已添加仓库
echo ""
echo "2. Checking ROCm repository..."
if [ -f /etc/apt/sources.list.d/rocm.list ]; then
    echo "  ✓ ROCm repository already configured"
    cat /etc/apt/sources.list.d/rocm.list
else
    echo "  ✗ ROCm repository not configured"
    echo ""
    echo "3. Adding ROCm repository..."
    
    # 添加 GPG key
    echo "  Adding GPG key..."
    wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - 2>/dev/null || \
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add - 2>/dev/null || \
    {
        echo "  ⚠ Failed to add GPG key, trying alternative method..."
        sudo apt-key adv --fetch-keys https://repo.radeon.com/rocm/rocm.gpg.key
    }
    
    # 添加仓库
    echo "  Adding repository..."
    ROCM_VERSION="5.7"
    echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_VERSION}/ ${UBUNTU_CODENAME} main" | sudo tee /etc/apt/sources.list.d/rocm.list
    
    echo "  ✓ Repository added"
fi

# 更新包列表
echo ""
echo "4. Updating package list..."
sudo apt update

# 检查可用的 ROCm 包
echo ""
echo "5. Checking available ROCm packages..."
apt-cache search rocm | grep -E "^rocm-" | head -10 || echo "  ⚠ No ROCm packages found"

# 尝试安装
echo ""
echo "6. Attempting to install ROCm..."
echo "  Note: This may take a while and requires sudo privileges"

# 尝试不同的包名组合
PACKAGES=(
    "rocm-dkms rocm-libs rocm-dev"
    "rocm-5.7 rocm-dev-5.7"
    "rocm-core rocm-dev"
)

INSTALLED=false
for pkg_list in "${PACKAGES[@]}"; do
    echo ""
    echo "  Trying: $pkg_list"
    if sudo apt install -y $pkg_list 2>&1 | grep -q "Unable to locate package"; then
        echo "    ✗ Packages not found"
        continue
    else
        echo "    ✓ Installation attempted"
        INSTALLED=true
        break
    fi
done

if [ "$INSTALLED" = false ]; then
    echo ""
    echo "=========================================="
    echo "Installation Failed"
    echo "=========================================="
    echo ""
    echo "Possible reasons:"
    echo "1. ROCm may not be available for your Ubuntu version"
    echo "2. WSL2 GPU support may require Windows-side configuration"
    echo "3. Your AMD GPU may not be supported in WSL2"
    echo ""
    echo "Alternative: Use CPU training"
    echo "  Set gpus='-1' in run_frame_stream.sh"
    echo ""
    echo "For WSL2 GPU support, you may need:"
    echo "1. Windows 11 22H2 or later"
    echo "2. AMD driver with WSL2 support"
    echo "3. GPU passthrough enabled"
else
    echo ""
    echo "=========================================="
    echo "Installation Complete"
    echo "=========================================="
    echo ""
    echo "Setting up environment variables..."
    echo 'export ROCM_PATH=/opt/rocm' >> ~/.bashrc
    echo 'export PATH=$ROCM_PATH/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo ""
    echo "Please run: source ~/.bashrc"
    echo "Then verify: rocm-smi"
fi

