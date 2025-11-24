# hi_xiaowen 数据集帧流式训练指南

本指南说明如何使用帧流式训练脚本来训练 hi_xiaowen 数据集。

## 快速开始

### 1. 数据准备（如果还没有准备）

```bash
# 进入示例目录
cd examples/hi_xiaowen/s0

# 运行数据准备阶段（stage -3 到 0）
bash run_frame_stream.sh -3 0
```

这会完成：
- 下载数据集（如果需要）
- 准备 Kaldi 格式文件（wav.scp, text）
- 计算 CMVN
- 生成 data.list 文件

### 2. 帧流式训练

#### 方式一：标准训练（推荐）

标准训练模式：模型支持流式推理，但训练时使用批量处理，速度较快。

```bash
# 单GPU训练
bash run_frame_stream.sh 2 2

# 或者直接使用 Python 命令
python wekws/bin/train_frame_stream.py \
    --config conf/frame_stream_tcn_ctc.yaml \
    --train_data data/train/data.list \
    --cv_data data/dev/data.list \
    --model_dir exp/frame_stream_tcn_ctc \
    --gpus 0 \
    --num_keywords 2599 \
    --min_duration 50 \
    --cmvn_file data/train/global_cmvn \
    --norm_var \
    --num_workers 8
```

#### 方式二：帧流式训练模式（逐帧处理）

帧流式训练模式：训练时逐帧处理，完全模拟真实流式推理场景，但训练速度较慢。

```bash
# 修改 run_frame_stream.sh 中的 frame_stream_mode=true
# 或者直接使用 Python 命令
python wekws/bin/train_frame_stream.py \
    --config conf/frame_stream_tcn_ctc.yaml \
    --train_data data/train/data.list \
    --cv_data data/dev/data.list \
    --model_dir exp/frame_stream_tcn_ctc \
    --gpus 0 \
    --num_keywords 2599 \
    --min_duration 50 \
    --cmvn_file data/train/global_cmvn \
    --norm_var \
    --num_workers 8 \
    --frame_stream_mode  # 启用帧流式训练模式
```

### 3. 模型评估

```bash
# 运行评估（stage 3）
bash run_frame_stream.sh 3 3
```

## 配置文件说明

### `conf/frame_stream_tcn_ctc.yaml`

这是基于原始 `ds_tcn_ctc.yaml` 修改的帧流式训练配置，主要变化：

1. **特征提取配置**：
   ```yaml
   feature_extraction_conf:
       feature_type: 'fbank'
       num_mel_bins: 40
       frame_shift: 10
       frame_length: 25
   ```

2. **上下文扩展**（重要）：
   ```yaml
   context_expansion: true
   context_expansion_conf:
       left: 2
       right: 2
   ```
   - 这会扩展特征维度：`input_dim = 40 * (2 + 1 + 2) = 200`

3. **模型配置**：
   ```yaml
   model:
       input_dim: 200  # 必须与 context_expansion 配置匹配
       hidden_dim: 256
       backbone:
           type: tcn
           ds: true
   ```

## 关键参数说明

### 训练参数

- `--config`: 配置文件路径
- `--train_data`: 训练数据列表文件（data.list）
- `--cv_data`: 验证数据列表文件（data.list）
- `--model_dir`: 模型保存目录
- `--num_keywords`: 关键词数量（hi_xiaowen 为 2599）
- `--min_duration`: 关键词最小帧数
- `--cmvn_file`: CMVN 文件路径
- `--norm_var`: 是否归一化方差
- `--frame_stream_mode`: 是否启用帧流式训练模式（可选）

### 配置参数

- `context_expansion`: 是否使用上下文扩展
  - `true`: 使用上下文扩展，`input_dim = num_mel_bins * (left + 1 + right)`
  - `false`: 不使用，`input_dim = num_mel_bins`
- `frame_skip`: 帧下采样率
  - `1`: 不跳过
  - `3`: 每3帧取1帧
- `batch_size`: 批次大小（根据GPU内存调整）

## 训练模式对比

| 模式 | 训练方式 | 速度 | 流式推理支持 | 适用场景 |
|------|---------|------|------------|---------|
| 标准训练 | 批量处理 | 快 | ✅ 支持 | 推荐，大多数场景 |
| 帧流式训练 | 逐帧处理 | 慢 | ✅ 完全模拟 | 需要极端流式场景验证 |

**注意**：即使使用标准训练模式，模型仍然支持流式推理（通过 cache 机制）。

## 使用其他 Backbone

如果你想使用 LSTM 或 Transformer backbone，可以修改配置文件：

### LSTM 配置示例

```yaml
model:
    input_dim: 200
    hidden_dim: 256
    preprocessing:
        type: linear
    backbone:
        type: lstm
        num_layers: 2
        dropout: 0.1
        bidirectional: false
    classifier:
        type: identity
    activation:
        type: identity
```

### Transformer 配置示例

```yaml
model:
    input_dim: 200
    hidden_dim: 256
    preprocessing:
        type: none  # Transformer 通常不需要预处理
    backbone:
        type: transformer
        num_layers: 4
        nhead: 8
        dim_feedforward: 2048
        dropout: 0.1
        activation: relu
        max_cache_len: 1000
    classifier:
        type: identity
    activation:
        type: identity
```

## 常见问题

### 1. input_dim 不匹配

**错误**：`RuntimeError: size mismatch`

**解决**：确保 `model.input_dim` 与 `context_expansion` 配置匹配：
- 有 `context_expansion`: `input_dim = num_mel_bins * (left + 1 + right)`
- 无 `context_expansion`: `input_dim = num_mel_bins`

### 2. 训练速度慢

**解决**：
- 不使用 `--frame_stream_mode`（标准训练模式）
- 增加 `batch_size`（如果GPU内存允许）
- 使用 `frame_skip: 3` 减少帧数

### 3. GPU 内存不足

**解决**：
- 减小 `batch_size`
- 减小 `hidden_dim`
- 减少 `num_layers`

## 完整训练流程示例

```bash
# 1. 数据准备
bash run_frame_stream.sh -3 0

# 2. 训练（标准模式）
bash run_frame_stream.sh 2 2

# 3. 评估
bash run_frame_stream.sh 3 3

# 4. 导出模型（可选）
bash run_frame_stream.sh 4 4
```

## 参考

- 原始训练脚本：`run_ctc.sh`
- 帧流式训练脚本：`run_frame_stream.sh`
- 配置文件：`conf/frame_stream_tcn_ctc.yaml`
- 帧流式训练文档：`examples/frame_stream_kws/README.md`

