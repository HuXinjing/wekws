#!/bin/bash
# Copyright 2021  Binbin Zhang(binbzha@qq.com)
#           2023  Jing Du(thuduj12@163.com)
#
# 帧流式训练脚本 - 用于 hi_xiaowen 数据集
# Frame Streaming Training Script for hi_xiaowen dataset

. ./path.sh

stage=$1
stop_stage=$2
num_keywords=2599

# 使用帧流式训练配置
config=conf/frame_stream_tcn_ctc.yaml
norm_mean=true
norm_var=true
gpus="0"

checkpoint=
dir=exp/frame_stream_tcn_ctc
average_model=true
num_average=30
if $average_model ;then
  score_checkpoint=$dir/avg_${num_average}.pt
else
  score_checkpoint=$dir/final.pt
fi

download_dir=./data/local # your data dir

. tools/parse_options.sh || exit 1;
window_shift=50

# 是否启用帧流式训练模式（逐帧处理，模拟真实流式推理）
# 注意：这会显著降低训练速度，通常只在需要确保极端流式场景表现时使用
frame_stream_mode=false

# 数据准备阶段（与原始脚本相同）
if [ ${stage} -le -3 ] && [ ${stop_stage} -ge -3 ]; then
  echo "Download and extracte all datasets"
  local/mobvoi_data_download.sh --dl_dir $download_dir
fi

if [ ${stage} -le -2 ] && [ ${stop_stage} -ge -2 ]; then
  echo "Preparing datasets..."
  mkdir -p dict
  echo "<FILLER> -1" > dict/dict.txt
  echo "<HI_XIAOWEN> 0" >> dict/dict.txt
  echo "<NIHAO_WENWEN> 1" >> dict/dict.txt
  awk '{print $1}' dict/dict.txt > dict/words.txt

  for folder in train dev test; do
    mkdir -p data/$folder
    for prefix in p n; do
      mkdir -p data/${prefix}_$folder
      json_path=$download_dir/mobvoi_hotword_dataset_resources/${prefix}_$folder.json
      local/prepare_data.py $download_dir/mobvoi_hotword_dataset $json_path \
        dict/dict.txt data/${prefix}_$folder
    done
    cat data/p_$folder/wav.scp data/n_$folder/wav.scp > data/$folder/wav.scp
    cat data/p_$folder/text data/n_$folder/text > data/$folder/text
    rm -rf data/p_$folder data/n_$folder
  done
fi

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
  # 使用 Paraformer Large 转录负样本
  git clone https://www.modelscope.cn/datasets/thuduj12/mobvoi_kws_transcription.git
  for folder in train dev test; do
    if [ -f data/$folder/text ];then
      mv data/$folder/text data/$folder/text.label
    fi
    cp mobvoi_kws_transcription/$folder.text data/$folder/text
  done

  awk '{print $1, $2-1}' mobvoi_kws_transcription/tokens.txt > dict/dict.txt
  sed -i 's/& 1/<filler> 1/' dict/dict.txt
  echo '<SILENCE>' > dict/words.txt
  echo '<EPS>' >> dict/words.txt
  echo '<BLK>' >> dict/words.txt
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo "Compute CMVN and Format datasets"
  tools/compute_cmvn_stats.py --num_workers 16 --train_config $config \
    --in_scp data/train/wav.scp \
    --out_cmvn data/train/global_cmvn

  for x in train dev test; do
    tools/wav_to_duration.sh --nj 8 data/$x/wav.scp data/$x/wav.dur

    # 生成 data.list 文件
    tools/make_list.py data/$x/wav.scp data/$x/text \
      data/$x/wav.dur data/$x/data.list
  done
fi

# 帧流式训练阶段
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Start frame streaming training ..."
  mkdir -p $dir
  cmvn_opts=
  $norm_mean && cmvn_opts="--cmvn_file data/train/global_cmvn"
  $norm_var && cmvn_opts="$cmvn_opts --norm_var"
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')

  # 可选：使用预训练模型作为 checkpoint
  # checkpoint=mobvoi_kws_transcription/23.pt

  # 构建训练命令
  train_cmd="wekws/bin/train_frame_stream.py"
  train_args="--config $config \
    --train_data data/train/data.list \
    --cv_data data/dev/data.list \
    --model_dir $dir \
    --num_workers 8 \
    --num_keywords $num_keywords \
    --min_duration 50 \
    --seed 666 \
    $cmvn_opts"

  # 如果启用帧流式训练模式
  if $frame_stream_mode; then
    train_args="$train_args --frame_stream_mode"
    echo "Frame streaming mode enabled (逐帧处理模式)"
  else
    echo "Standard training mode (标准训练模式，模型仍支持流式推理)"
  fi

  # 如果有 checkpoint，添加它
  if [ ! -z "$checkpoint" ]; then
    train_args="$train_args --checkpoint $checkpoint"
  fi

  # 单GPU训练
  if [ $num_gpus -eq 1 ]; then
    python $train_cmd --gpus $gpus $train_args
  else
    # 多GPU训练
    torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
      $train_cmd --gpus $gpus $train_args
  fi
fi

# 模型平均和评估
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Do model average, Compute FRR/FAR ..."
  if $average_model; then
    python wekws/bin/average_model.py \
      --dst_model $score_checkpoint \
      --src_path $dir  \
      --num ${num_average} \
      --val_best
  fi
  result_dir=$dir/test_$(basename $score_checkpoint)
  mkdir -p $result_dir
  
  # 使用帧流式推理脚本
  python wekws/bin/frame_stream_kws_ctc.py \
    --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --token_file data/tokens.txt \
    --lexicon_file data/lexicon.txt \
    --keywords "你好小文,你好问问" \
    --wav_path test.wav \
    --threshold 0.5

  # 或者使用批量评估
  python wekws/bin/stream_score_ctc.py \
    --config $dir/config.yaml \
    --test_data data/test/data.list \
    --gpu 0  \
    --batch_size 256 \
    --checkpoint $score_checkpoint \
    --score_file $result_dir/score.txt  \
    --num_workers 8  \
    --keywords "你好小文,你好问问" \
    --token_file data/tokens.txt \
    --lexicon_file data/lexicon.txt

  python wekws/bin/compute_det_ctc.py \
      --keywords "你好小文,你好问问" \
      --test_data data/test/data.list \
      --window_shift $window_shift \
      --step 0.001  \
      --score_file $result_dir/score.txt \
      --token_file data/tokens.txt \
      --lexicon_file data/lexicon.txt
fi

# 模型导出
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  jit_model=$(basename $score_checkpoint | sed -e 's:.pt$:.zip:g')
  onnx_model=$(basename $score_checkpoint | sed -e 's:.pt$:.onnx:g')
  python wekws/bin/export_jit.py \
    --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --jit_model $dir/$jit_model
  python wekws/bin/export_onnx.py \
    --config $dir/config.yaml \
    --checkpoint $score_checkpoint \
    --onnx_model $dir/$onnx_model
fi

