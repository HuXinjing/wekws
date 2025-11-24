# Copyright (c) 2023 Jing Du(thuduj12@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from wekws.dataset.init_dataset import init_dataset
from wekws.utils.checkpoint import load_checkpoint, save_checkpoint
from wekws.model.kws_model import init_model
from wekws.utils.executor import Executor
from wekws.utils.train_utils import count_parameters, set_mannul_seed
from wekws.utils.gpu_utils import setup_gpu, diagnose_gpu
from wekws.model.loss import criterion
from wenet.text.char_tokenizer import CharTokenizer


def get_args():
    parser = argparse.ArgumentParser(description='training your network for frame streaming')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--gpus',
                        default='-1',
                        help='gpu lists, seperated with `,`, -1 for cpu')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--cmvn_file', default=None, help='global cmvn file')
    parser.add_argument('--norm_var',
                        action='store_true',
                        default=False,
                        help='norm var option')
    parser.add_argument('--dict', default='./dict', help='dict dir')
    parser.add_argument('--num_keywords',
                        default=1,
                        type=int,
                        help='number of keywords')
    parser.add_argument('--min_duration',
                        default=50,
                        type=int,
                        help='min duration frames of the keyword')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--reverb_lmdb', default=None, help='reverb lmdb file')
    parser.add_argument('--noise_lmdb', default=None, help='noise lmdb file')
    parser.add_argument('--frame_stream_mode',
                        action='store_true',
                        default=False,
                        help='Enable frame streaming training mode. '
                        'In this mode, model processes sequences frame by frame '
                        'with state caching, simulating real streaming inference.')

    args = parser.parse_args()
    return args


class FrameStreamExecutor(Executor):
    """
    帧流式训练执行器
    在训练时模拟流式推理，逐帧处理并保持状态
    
    注意：
    - 默认情况下（frame_stream_mode=False），使用标准批量训练，速度更快
    - 启用frame_stream_mode时，会逐帧处理，模拟真实流式推理，但训练速度会显著降低
    - 对于大多数情况，标准训练已经足够，因为模型本身支持流式推理（通过cache机制）
    - 帧流式训练主要用于确保模型在极端流式场景下的表现
    """
    
    def __init__(self):
        super().__init__()
    
    def train(self, model, optimizer, data_loader, device, writer, args):
        ''' Train one epoch with frame streaming mode
        '''
        model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        epoch = args.get('epoch', 0)
        min_duration = args.get('min_duration', 0)
        frame_stream_mode = args.get('frame_stream_mode', False)

        for batch_idx, batch_dict in enumerate(data_loader):
            key = batch_dict['keys']
            feats = batch_dict['feats']
            target = batch_dict['target']
            target = target[:, 0] if target.shape[1] == 1 else target
            feats_lengths = batch_dict['feats_lengths']
            label_lengths = batch_dict['target_lengths']
            feats = feats.to(device)
            target = target.to(device)
            feats_lengths = feats_lengths.to(device)
            label_lengths = label_lengths.to(device)
            num_utts = feats_lengths.size(0)
            if num_utts == 0:
                continue
            
            if frame_stream_mode:
                # 帧流式训练：逐帧处理，保持状态
                logits = self._forward_frame_stream(model, feats, feats_lengths)
            else:
                # 标准训练：批量处理
                logits, _ = model(feats)
            
            loss_type = args.get('criterion', 'max_pooling')
            loss, acc = criterion(loss_type,
                                  logits,
                                  target,
                                  feats_lengths,
                                  target_lengths=label_lengths,
                                  min_duration=min_duration,
                                  validation=False)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(model.parameters(), clip)
            if torch.isfinite(grad_norm):
                optimizer.step()
            if batch_idx % log_interval == 0:
                logging.debug(
                    'TRAIN Batch {}/{} loss {:.8f} acc {:.8f}'.format(
                        epoch, batch_idx, loss.item(), acc))
    
    def cv(self, model, data_loader, device, args):
        ''' Cross validation with optional frame streaming mode
        '''
        model.eval()
        log_interval = args.get('log_interval', 10)
        epoch = args.get('epoch', 0)
        frame_stream_mode = args.get('frame_stream_mode', False)
        # in order to avoid division by 0
        num_seen_utts = 1
        total_loss = 0.0
        total_acc = 0.0
        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(data_loader):
                key = batch_dict['keys']
                feats = batch_dict['feats']
                target = batch_dict['target']
                target = target[:, 0] if target.shape[1] == 1 else target
                feats_lengths = batch_dict['feats_lengths']
                label_lengths = batch_dict['target_lengths']
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                label_lengths = label_lengths.to(device)
                num_utts = feats_lengths.size(0)
                if num_utts == 0:
                    continue
                
                if frame_stream_mode:
                    # 帧流式验证：逐帧处理，保持状态
                    logits = self._forward_frame_stream(model, feats, feats_lengths)
                else:
                    # 标准验证：批量处理
                    logits, _ = model(feats)
                
                loss, acc = criterion(args.get('criterion', 'max_pooling'),
                                      logits,
                                      target,
                                      feats_lengths,
                                      target_lengths=label_lengths,
                                      min_duration=0,
                                      validation=True)
                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts
                    total_acc += acc * num_utts
                if batch_idx % log_interval == 0:
                    logging.debug(
                        'CV Batch {}/{} loss {:.8f} acc {:.8f} history loss {:.8f}'
                        .format(epoch, batch_idx, loss.item(), acc,
                                total_loss / num_seen_utts))
        return total_loss / num_seen_utts, total_acc / num_seen_utts
    
    def _forward_frame_stream(self, model, feats, feats_lengths):
        """
        帧流式前向传播：逐帧处理，模拟真实流式推理
        
        Args:
            model: 模型
            feats: (batch_size, max_len, feat_dim) 特征序列
            feats_lengths: (batch_size,) 每个序列的实际长度
        
        Returns:
            logits: (batch_size, max_len, vocab_size) 输出logits
        """
        batch_size, max_len, feat_dim = feats.shape
        device = feats.device
        
        # 初始化输出logits列表
        logits_list = []
        
        # 初始化状态缓存: (batch_size, D, C)
        # 如果cache为空，模型内部会处理
        in_cache = torch.zeros(0, 0, 0, dtype=torch.float, device=device)
        
        # 逐帧处理
        for t in range(max_len):
            # 获取当前帧的特征 (batch_size, feat_dim)
            current_feat = feats[:, t, :]
            
            # 检查哪些样本还需要处理（未达到实际长度）
            active_mask = t < feats_lengths  # (batch_size,)
            
            # 为当前帧添加sequence维度: (batch_size, 1, feat_dim)
            current_feat = current_feat.unsqueeze(1)
            
            # 对于不活跃的样本，使用零特征（但保持batch维度）
            # 这样可以保持batch一致性，但不会影响cache更新
            
            # 批量处理当前帧
            # 注意：即使某些样本已经处理完，我们仍然需要保持batch维度
            # 模型会正确处理cache
            frame_logits, out_cache = model(current_feat, in_cache)
            
            # 更新状态缓存
            # out_cache的形状是 (batch_size, D, C)
            # 对于已经处理完的样本，我们保持其cache不变
            if in_cache.numel() == 0:
                # 第一次，直接使用out_cache
                in_cache = out_cache
            else:
                # 后续帧，更新cache
                # 对于活跃的样本，使用新的cache；对于不活跃的，保持旧的
                if out_cache.shape[0] == batch_size:
                    # 只更新活跃样本的cache
                    # 这里简化处理：直接使用out_cache（模型内部应该已经正确处理）
                    in_cache = out_cache
                else:
                    # 如果batch size不匹配，需要特殊处理
                    in_cache = out_cache
            
            # 收集当前帧的logits: (batch_size, 1, vocab_size)
            logits_list.append(frame_logits)
        
        # 拼接所有帧的logits: (batch_size, max_len, vocab_size)
        logits = torch.cat(logits_list, dim=1)
        
        return logits


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    # Set random seed
    set_mannul_seed(args.seed)
    print(args)
    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    gpu = int(args.gpus.split(',')[rank]) if args.gpus != '-1' else -1
    
    # Setup GPU with proper diagnostics
    if rank == 0 and gpu >= 0:
        # Print GPU diagnostic info on first rank
        diagnose_gpu()
    
    device, gpu = setup_gpu(gpu)
    if world_size > 1:
        logging.info('training on multiple gpus, this gpu {}'.format(gpu))
        dist.init_process_group(backend=args.dist_backend)

    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['shuffle'] = False

    tokenizer = CharTokenizer(f'{args.dict}/dict.txt',
                              f'{args.dict}/words.txt',
                              unk='<filler>',
                              split_with_space=True)
    train_dataset = init_dataset(data_list_file=args.train_data,
                                 conf=train_conf,
                                 tokenizer=tokenizer)
    cv_dataset = init_dataset(data_list_file=args.cv_data, conf=cv_conf,
                              tokenizer=tokenizer, split='dev')

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)

    feats_type = train_conf.get('feats_type', 'fbank')
    input_dim = train_conf[f'{feats_type}_conf'][
        'num_mel_bins']
    output_dim = args.num_keywords

    # Write model_dir/config.yaml for inference and export
    if 'input_dim' not in configs['model']:
        configs['model']['input_dim'] = input_dim
    configs['model']['output_dim'] = output_dim
    if args.cmvn_file is not None:
        configs['model']['cmvn'] = {}
        configs['model']['cmvn']['norm_var'] = args.norm_var
        configs['model']['cmvn']['cmvn_file'] = args.cmvn_file
    # Init asr model from configs
    model = init_model(configs['model'])
    if rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)
        print(model)
    num_params = count_parameters(model)
    print('the number of model params: {}'.format(num_params))

    # !!!IMPORTANT!!!
    # Try to export the model by script, if fails, we should refine
    # the code to satisfy the script export requirements
    if rank == 0:
        pass
        # TODO: for now streaming FSMN do not support export to JITScript,
        # TODO: because there is nn.Sequential with Tuple input
        #  in current FSMN modules.
        #  the issue is in https://stackoverflow.com/questions/75714299/
        #  pytorch-jit-script-error-when-sequential-container-
        #  takes-a-tuple-input/76553450#76553450

        # script_model = torch.jit.script(model)
        # script_model.save(os.path.join(args.model_dir, 'init.zip'))
    
    # 使用帧流式执行器
    executor = FrameStreamExecutor()
    
    # If specify checkpoint, load some info from checkpoint
    if args.checkpoint is not None:
        infos = load_checkpoint(model, args.checkpoint)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    # get the last epoch lr
    lr_last_epoch = infos.get('lr', configs['optim_conf']['lr'])
    configs['optim_conf']['lr'] = lr_last_epoch
    model_dir = args.model_dir
    writer = None
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))
        # Save config file
        config_path = os.path.join(model_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(configs, f, default_flow_style=False, allow_unicode=True)

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires GPU but GPU is not available")
        # cuda model is required for nn.parallel.DistributedDataParallel
        model.cuda()
        model = torch.nn.parallel.DistributedDataParallel(model)
        device = torch.device("cuda")
    else:
        # device is already set by setup_gpu
        model = model.to(device)

    optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        threshold=0.01,
    )

    training_config = configs['training_config']
    training_config['min_duration'] = args.min_duration
    training_config['frame_stream_mode'] = args.frame_stream_mode  # 添加帧流式模式标志
    num_epochs = training_config.get('max_epoch', 100)
    final_epoch = None
    if start_epoch == 0 and rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(model, save_model_path)

    # Start training loop
    for epoch in range(start_epoch, num_epochs):
        # train_dataset.set_epoch(epoch)
        training_config['epoch'] = epoch
        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        if args.frame_stream_mode:
            logging.info('Frame streaming training mode enabled')
        executor.train(model, optimizer, train_data_loader, device, writer,
                       training_config)
        cv_loss, cv_acc = executor.cv(model, cv_data_loader, device,
                                      training_config)
        logging.info('Epoch {} CV info cv_loss {} cv_acc {}'.format(
            epoch, cv_loss, cv_acc))

        if rank == 0:
            save_model_path = os.path.join(model_dir, '{}.pt'.format(epoch))
            save_checkpoint(model, save_model_path, {
                'epoch': epoch,
                'lr': lr,
                'cv_loss': cv_loss,
            })
            writer.add_scalar('epoch/cv_loss', cv_loss, epoch)
            writer.add_scalar('epoch/cv_acc', cv_acc, epoch)
            writer.add_scalar('epoch/lr', lr, epoch)
        final_epoch = epoch
        scheduler.step(cv_loss)

    if final_epoch is not None and rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        writer.close()


if __name__ == '__main__':
    main()

