# ROCm 安装指南 - Ubuntu 24.04 (Noble)

## 问题

Ubuntu 24.04 使用 `noble` 代号，但 ROCm 5.7 仓库配置的是 `jammy` (22.04)。

## 解决方案

### 方案 1: 使用更新的 ROCm 版本（推荐）

Ubuntu 24.04 可能需要更新的 ROCm 版本（6.x）：

```bash
# 1. 更新仓库配置为 noble
sudo sed -i 's/jammy/noble/g' /etc/apt/sources.list.d/rocm.list

# 或者使用更新的 ROCm 版本
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.1/ noble main" | sudo tee /etc/apt/sources.list.d/rocm.list

# 2. 更新并安装
sudo apt update
sudo apt install rocm-dkms rocm-libs rocm-dev
```

### 方案 2: 使用 amdgpu-install 工具（WSL2 推荐）

这是 AMD 官方推荐的 WSL2 安装方式：

```bash
# 1. 添加 AMD 仓库
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/rocm.gpg

# 2. 添加仓库
echo "deb [arch=amd64] https://repo.radeon.com/amdgpu/latest/ubuntu/ noble main" | sudo tee /etc/apt/sources.list.d/amdgpu.list
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/latest/ noble main" | sudo tee /etc/apt/sources.list.d/rocm.list

# 3. 更新
sudo apt update

# 4. 安装 amdgpu-install
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/noble/amdgpu-install_*.deb
sudo apt install ./amdgpu-install_*.deb

# 5. 安装 ROCm（WSL2 模式）
sudo amdgpu-install -y --usecase=wsl,rocm --no-dkms
```

### 方案 3: 继续使用 jammy 仓库（可能不兼容）

如果上述方法都不行，可以尝试强制使用 jammy 仓库：

```bash
# 恢复为 jammy
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

# 更新（可能会有警告）
sudo apt update

# 尝试安装（可能会有依赖问题）
sudo apt install -y rocm-dkms rocm-libs rocm-dev
```

### 方案 4: 使用 CPU 训练（最简单）

如果 ROCm 安装遇到问题，可以先用 CPU 训练：

```bash
# 在 run_frame_stream.sh 中设置
gpus="-1"
```

## 验证安装

安装完成后：

```bash
# 设置环境变量
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# 验证
rocm-smi
rocminfo

# 测试 PyTorch
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

## 注意事项

1. **WSL2 GPU 支持要求**：
   - Windows 11 22H2 或更高版本
   - AMD 驱动支持 WSL2 GPU 直通
   - 在 Windows 中启用 GPU 直通

2. **GPU 型号支持**：
   - 不是所有 AMD GPU 都支持 ROCm
   - 查看 AMD 官方支持列表

3. **如果安装失败**：
   - 检查 Windows 端是否有 AMD GPU
   - 检查 WSL2 GPU 直通是否启用
   - 考虑使用 CPU 训练

