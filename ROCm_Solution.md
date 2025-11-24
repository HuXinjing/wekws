# ROCm 安装问题解决方案

## 问题

ROCm 5.7 **不支持** Ubuntu 24.04 (noble)，仓库返回 404 错误。

## 解决方案

### 方案 1: 使用 ROCm 6.x（如果支持 noble）

```bash
# 更新为 ROCm 6.1
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.1/ noble main" | sudo tee /etc/apt/sources.list.d/rocm.list

# 更新
sudo apt update

# 尝试安装
sudo apt install rocm-dkms rocm-libs rocm-dev
```

### 方案 2: 使用 jammy 仓库（可能有兼容性问题）

强制使用 Ubuntu 22.04 的仓库，但可能会有依赖冲突：

```bash
# 改回 jammy
echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

# 更新（忽略架构警告）
sudo apt update

# 强制安装（可能会有问题）
sudo apt install -y --allow-unauthenticated rocm-dkms rocm-libs rocm-dev
```

### 方案 3: 使用 CPU 训练（推荐，最简单）

由于 ROCm 在 Ubuntu 24.04 上的支持有限，**建议先使用 CPU 训练**：

```bash
# 修改 run_frame_stream.sh
cd examples/hi_xiaowen/s0
# 编辑 run_frame_stream.sh，设置：
gpus="-1"  # 使用 CPU

# 然后运行训练
bash run_frame_stream.sh 2 2
```

CPU 训练虽然慢一些，但可以正常工作，适合：
- 测试和开发
- 小规模数据集
- 暂时无法配置 GPU 的情况

### 方案 4: 降级到 Ubuntu 22.04（如果必须使用 GPU）

如果必须使用 GPU 训练，可以考虑：
1. 创建新的 WSL2 实例使用 Ubuntu 22.04
2. 或者在 Docker 容器中使用 Ubuntu 22.04

## 推荐方案

**建议使用方案 3（CPU 训练）**，因为：
1. ✅ 最简单，无需额外配置
2. ✅ 可以立即开始训练
3. ✅ 适合开发和测试
4. ✅ 后续可以再配置 GPU

## 快速开始 CPU 训练

```bash
cd /home/jason/myWorkspace/wekws/examples/hi_xiaowen/s0

# 修改配置
sed -i 's/gpus="0"/gpus="-1"/' run_frame_stream.sh

# 运行训练
bash run_frame_stream.sh 2 2
```

