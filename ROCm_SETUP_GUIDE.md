# ROCm GPU 配置指南

## 当前状态

- ✅ PyTorch 已安装 ROCm 版本 (2.3.1+rocm5.7)
- ❌ ROCm 运行时未安装
- ❌ GPU 未被识别

## 在 WSL2 中使用 AMD GPU

如果你在 WSL2 环境中，AMD GPU 支持需要特殊配置：

### 方案 1: 安装 ROCm for WSL2（推荐）

1. **检查 WSL2 版本**
   ```bash
   wsl --version
   ```

2. **安装 ROCm for WSL2**
   ```bash
   # 添加 ROCm 仓库
   wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
   echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list
   
   # 安装 ROCm
   sudo apt update
   sudo apt install rocm-dkms rocm-libs rocm-dev
   ```

3. **设置环境变量**
   ```bash
   export ROCM_PATH=/opt/rocm
   export PATH=$ROCM_PATH/bin:$PATH
   export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
   ```

4. **验证安装**
   ```bash
   rocm-smi
   ```

### 方案 2: 使用 CPU 训练（临时方案）

如果暂时无法配置 ROCm，可以先用 CPU 训练：

```bash
# 在 run_frame_stream.sh 中设置
gpus="-1"
```

### 方案 3: 检查 GPU 是否在 Windows 端可用

在 Windows PowerShell 中运行：
```powershell
# 检查 AMD GPU
Get-WmiObject Win32_VideoController | Where-Object {$_.Name -like "*AMD*"}
```

## 快速诊断命令

运行以下命令进行诊断：

```bash
cd /home/jason/myWorkspace/wekws
source .venv/bin/activate
python wekws/utils/gpu_utils.py
```

## 常见问题

### Q: 为什么 PyTorch 显示 ROCm 版本但 GPU 不可用？
A: PyTorch 的 ROCm 版本只是编译时支持，还需要安装 ROCm 运行时库。

### Q: WSL2 中可以使用 AMD GPU 吗？
A: 可以，但需要：
- Windows 11 22H2 或更高版本
- 支持 GPU 直通的 AMD 驱动
- 在 WSL2 中安装 ROCm

### Q: 如何检查我的 GPU 型号？
A: 在 Windows 中：
- 打开设备管理器 → 显示适配器
- 或在 PowerShell 中运行上面的命令

## 下一步

1. 如果确认有 AMD GPU，按照方案 1 安装 ROCm
2. 如果暂时无法配置，使用方案 2 进行 CPU 训练
3. 运行诊断脚本确认配置

