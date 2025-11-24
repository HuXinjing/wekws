# 帧流式关键词检测配置示例

本目录包含用于帧流式关键词检测训练的配置文件示例。

## 配置文件说明

### 1. `frame_stream_simple.yaml`
- **推荐用于快速开始**
- 使用FSMN backbone
- 简化的配置，包含基本必需参数
- 适合大多数场景

### 2. `frame_stream_fsmn_ctc.yaml`
- **详细配置，包含完整注释**
- 使用FSMN backbone + CTC loss
- 包含所有参数的详细说明
- 适合需要自定义配置的场景

### 3. `frame_stream_mdtc.yaml`
- 使用MDTC (Multi-scale Depthwise Temporal Convolution) backbone
- 另一种支持流式推理的架构选择
- 适合想要尝试不同架构的场景

## 关键配置说明

### 特征提取配置
```yaml
feature_extraction_conf:
    num_mel_bins: 80      # Mel滤波器数量
    frame_shift: 10       # 帧移（毫秒），10ms = 160样本
    frame_length: 25       # 帧长（毫秒），25ms = 400样本
```

### 上下文扩展（重要）
```yaml
context_expansion: true
context_expansion_conf:
    left: 2               # 左侧上下文帧数
    right: 2              # 右侧上下文帧数
```
**注意**：这些值必须与推理时的配置完全一致！

### 输入维度计算
- **有context_expansion**: `input_dim = num_mel_bins * (left + 1 + right)`
  - 例如：`80 * (2 + 1 + 2) = 400`
- **无context_expansion**: `input_dim = num_mel_bins`
  - 例如：`80`

### 帧下采样
```yaml
frame_skip: 3  # 每3帧取1帧
```
**注意**：frame_skip会影响推理时的时序，需要保持一致。

## 使用示例

### 训练命令
```bash
# 使用简化配置
python wekws/bin/train_frame_stream.py \
    --config examples/frame_stream_kws/conf/frame_stream_simple.yaml \
    --train_data data/train/data.list \
    --cv_data data/valid/data.list \
    --model_dir exp/frame_stream \
    --gpus 0 \
    --num_keywords 1 \
    --dict ./dict

# 启用帧流式训练模式（可选）
python wekws/bin/train_frame_stream.py \
    --config examples/frame_stream_kws/conf/frame_stream_simple.yaml \
    --train_data data/train/data.list \
    --cv_data data/valid/data.list \
    --model_dir exp/frame_stream \
    --gpus 0 \
    --num_keywords 1 \
    --dict ./dict \
    --frame_stream_mode  # 启用帧流式训练
```

### 推理命令（使用训练好的模型）
```bash
python wekws/bin/frame_stream_kws_ctc.py \
    --config exp/frame_stream/config.yaml \
    --checkpoint exp/frame_stream/final.pt \
    --token_file dict/tokens.txt \
    --lexicon_file dict/lexicon.txt \
    --keywords "你好小文" \
    --wav_path test.wav \
    --threshold 0.5
```

## 配置参数调优建议

### 批次大小
- GPU内存充足：`batch_size: 256`
- GPU内存有限：`batch_size: 128` 或更小

### 学习率
- 初始学习率：`lr: 0.001`
- 如果训练不稳定，可以降低到 `0.0005` 或 `0.0001`

### 模型大小
- **小模型**（快速推理）：
  - `hidden_dim: 64`
  - `num_layers: 3`
- **中等模型**（平衡）：
  - `hidden_dim: 128`
  - `num_layers: 4`
- **大模型**（高精度）：
  - `hidden_dim: 256`
  - `num_layers: 6`

### 数据增强
- 训练初期：使用较强的数据增强（`spec_aug`, `dither`）
- 训练后期：可以适当减弱数据增强

## 注意事项

1. **配置一致性**：训练和推理的配置必须一致，特别是：
   - `frame_shift` 和 `frame_length`
   - `context_expansion` 的 `left` 和 `right`
   - `frame_skip`
   - `num_mel_bins`

2. **采样率**：必须使用16kHz采样率（`resample_rate: 16000`）

3. **流式推理**：模型本身支持流式推理（通过cache机制），即使使用标准训练也能进行流式推理

4. **帧流式训练**：`--frame_stream_mode` 选项会显著降低训练速度，通常只在需要确保极端流式场景表现时使用

