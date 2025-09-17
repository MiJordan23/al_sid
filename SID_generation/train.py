# encoding: utf-8
"""
@Author: Yingwu.XSW
@Date: 2025/9/4 下午2:14

@Function:
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
from pprint import pprint

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

import utils.lr_sched as lr_sched
from data_loader.local_reader_vqvae_reconstruct import get_data
from rqvae_embed.rqvae_clip import RQVAE_EMBED_CLIP
from utils import dist_utils
from utils import logger
from utils.configs_utils import get_config
from utils.optim_factory import get_param_groups


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser('RQVAE training', add_help=False)
    parser.add_argument('--cfg', default='configs/rqvae_i2v.yml', type=str)
    parser.add_argument('--finetune', default='', help='微调检查点')
    parser.add_argument('--output_dir', default='', help='存储路径')
    parser.add_argument("--tables", default='', help="ODPS表路径")
    parser.add_argument('--resume', default='', help='恢复检查点路径')
    parser.add_argument('--train_root', default='', help='训练数据根目录')
    parser.add_argument('--epochs', default=0, type=int, help='训练周期数')

    # 分布式训练参数
    parser.add_argument('--world_size', default=1, type=int, help='分布式进程数量')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--gpu', default=0, type=int, help='')
    parser.add_argument('--local-rank', default=-1, type=int,
                        help="分布式训练中的本地排名。自动由PAI或XDL启动器输入")
    parser.add_argument('--dist_url', default='env://', help='设置分布式训练的URL')
    parser.add_argument('--distributed', action='store_true', help='是否启用分布式训练')
    parser.add_argument('--save_prefix', default='test', help="保存前缀")

    return parser.parse_args()


def gather_tensors(tensor):
    """收集所有进程上的张量"""
    world_size = dist.get_world_size()
    with torch.no_grad():
        tensors_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensors_list, tensor)
    gathered_tensor = torch.cat(tensors_list, dim=0)
    return gathered_tensor


def initialize_training_environment(args, config):
    """初始化训练环境"""
    dist_utils.init_distributed_mode(config, args)
    pprint(OmegaConf.to_object(config))


def create_model(config):
    """创建模型实例"""
    hps = {
        "bottleneck_type": "rq",
        "embed_dim": config.model.codebook_dim,
        "n_embed": config.model.codebook_size,
        "latent_shape": [8, 8, config.model.codebook_dim],
        "code_shape": [8, 8, config.model.codebook_num],
        "shared_codebook": config.model.shared_codebook,
        "decay": config.model.decay,
        "restart_unused_codes": config.model.restart_unused_codes,
        "loss_type": config.model.loss_type,
        "latent_loss_weight": config.model.latent_loss_weight,
        "masked_dropout": 0.0,
        "use_padding_idx": False,
        "VQ_ema": config.model.VQ_ema,
        "latent_weight": eval(config.model.latent_weight),
        'do_bn': config.model.do_bn,
        'rotation_trick': config.model.rotation_trick
    }

    ddconfig = {
        "double_z": False,
        "z_channels": config.model.codebook_dim,
        "input_dim": config.model.input_dim,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [8],
        "dropout": 0.00
    }

    model_instance = RQVAE_EMBED_CLIP(hps=hps, ddconfig=ddconfig, checkpointing=True)

    return model_instance


def prepare_optimizer_and_scheduler(config, model_without_ddp):
    """准备优化器和学习率调度器"""
    effective_batch_size = config.data.batch_size * config.train.accum_iter * dist_utils.get_world_size()

    config.output_dir += f"{config.data.save_prefix}_ebs{effective_batch_size}_lr{config.train.lr}_ep{config.train.epochs}"

    current_time = int(time.time())
    formatted_time = datetime.datetime.fromtimestamp(current_time).strftime('%Y%m%d_%H%M%S')

    config.output_dir += f'_{formatted_time}'

    if config.train.lr is None:
        config.train.lr = config.train.blr * effective_batch_size / 256

    print(f"基础学习率: {round(config.train.lr * 256 / effective_batch_size, 6)}")
    print(f"实际学习率: {round(config.train.lr, 6)}")

    print(f"梯度累积迭代次数: {config.train.accum_iter}")
    print(f"有效批次大小: {effective_batch_size}")

    param_groups = get_param_groups(config, model_without_ddp)

    optimizer_instance = torch.optim.AdamW(param_groups, lr=config.train.lr, betas=config.train.optimizer.betas)

    return optimizer_instance


def main():
    """主函数入口"""
    args = parse_arguments()
    cfg = get_config(args)

    initialize_training_environment(args=args, config=cfg)

    seed = cfg.seed + dist_utils.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    OmegaConf.set_readonly(cfg, False)
    current_timestamp_seconds = int(time.time())
    formatted_timestamp = datetime.datetime.fromtimestamp(current_timestamp_seconds).strftime('%Y%m%d_%H%M%S')
    cfg.tmp_path += f"{cfg.data.save_prefix}_bs{cfg.data.batch_size}_lr{cfg.train.lr}_ep{cfg.train.epochs}_{formatted_timestamp}"

    if dist_utils.get_rank() == 0 and not cfg.eval:
        os.makedirs(cfg.tmp_path, exist_ok=True)
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        with open(os.path.join(cfg.tmp_path, 'config.json'), 'w') as file_handle:
            json.dump(config_dict, file_handle, indent=4)

    # -----------------------------------------------------------------------------------------------
    # Build model
    print("正在创建模型...")
    model = create_model(cfg)

    # -----------------------------------------------------------------------------------------------
    # Resume training
    OmegaConf.set_readonly(cfg, False)
    if cfg.resume:
        state_dict = torch.load(cfg.resume, map_location='cpu')
        model.load_state_dict(state_dict['model'], strict=True)
        cfg.train.start_epoch = state_dict['epoch'] + 1

    model.cuda(cfg.dist.gpu)

    model_without_ddp = model
    print(f"模型: {str(model_without_ddp)}")
    number_of_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数数量（百万）: {number_of_trainable_params / 1e6:.2f}")

    if cfg.dist.distributed:
        print("模型分布式数据并行化")
        model = torch.nn.parallel.DistributedDataParallel(
            module=model,
            device_ids=[cfg.dist.gpu],
            find_unused_parameters=True
        )

        model_without_ddp = model.module

    # -----------------------------------------------------------------------------------------------
    # Initialize dataset and dataloader
    print("正在创建数据集及数据加载器...")
    assert len(cfg.data.tables) > 0 or cfg.data.FromOSS, '无输入数据！'
    data = get_data(cfg=cfg, epoch_id=0)
    for key in data:
        print(f"训练数据集 {key} 的大小: {len(data[key].dataset)}")

    # -----------------------------------------------------------------------------------------------
    # Initialize optimizer and lr scheduler
    print("正在创建优化器及学习率调度器...")
    optimizer = prepare_optimizer_and_scheduler(config=cfg, model_without_ddp=model_without_ddp)
    print(optimizer)
    OmegaConf.set_readonly(cfg, True)

    # -----------------------------------------------------------------------------------------------
    # training
    print(f"开始训练 {cfg.train.epochs} 周期...")
    print(f"输出目录: {cfg.output_dir}")
    start_time = time.time()

    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        train_stats = train_one_epoch(model, data, optimizer, epoch, cfg=cfg)
        if cfg.output_dir and (epoch % 1 == 0 or epoch + 1 == cfg.train.epochs):
            checkpoint_save_info = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'cfg': OmegaConf.to_container(cfg),
            }

        print('*' * 100)
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': number_of_trainable_params}

        if cfg.output_dir and dist_utils.is_main_process():
            os.makedirs(cfg.tmp_path, exist_ok=True)
            with open(os.path.join(cfg.tmp_path, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"训练时间: {total_time_str}")


def train_one_epoch(model: torch.nn.Module, data: dict, optimizer: torch.optim.Optimizer,
                    epoch: int, cfg=None):
    """
    执行一个训练周期

    参数：
      - `model`: 要训练的模型
      - `data`: 数据加载器字典
      - `optimizer`: 使用的优化器
      - `epoch`: 当前训练周期数
      - `cfg`: 配置对象

    返回值：
      - 训练统计信息字典
    """

    metric_logger = logger.MetricLogger(delimiter="  ")
    header = f'Epoch: [{epoch}]'
    print_freq = cfg.log_step

    model.train(True)
    accum_iter = cfg.train.accum_iter

    dataloader, sampler = data['recon'].dataloader, data['recon'].sampler
    data_iter = iter(dataloader)
    if sampler is not None:
        sampler.set_epoch(epoch)
    num_batches_per_epoch_list = [dataloader.num_batches]
    data_iter_list = [data_iter]
    dataset_names = ['recon']

    for key in data:
        if key != 'recon' and hasattr(data[key], 'dataloader') and hasattr(data[key], 'sampler'):
            dataloader = data[key].dataloader
            sampler = data[key].sampler
            if sampler is not None:
                sampler.set_epoch(epoch)

            num_batches_per_epoch_list.append(dataloader.num_batches)
            data_iter = iter(dataloader)
            data_iter_list.append(data_iter)
            dataset_names.append(key)

    num_batches_per_epoch = sum(num_batches_per_epoch_list)
    optimizer.zero_grad()

    # 开始训练循环
    print('=======>')
    print(f'开始第 {epoch} 周期')

    for data_iter_step, (batch, select_idx, dataset_name) in enumerate(
            metric_logger.log_every_list_with_datasetname(data_iter_list, num_batches_per_epoch_list, dataset_names,
                                                          print_freq, epoch,
                                                          header)):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / num_batches_per_epoch + epoch, cfg)

        # 处理不同类型的训练数据集
        # 重建
        if select_idx == 0:

            _, features = batch
            features = features.cuda(cfg.dist.gpu, non_blocking=True)
            loss, recons, selected_index, loss_dict, feature_norm, quant_norm = model(features)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss /= accum_iter

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr = optimizer.param_groups[0]["lr"]
            lr2 = optimizer.param_groups[-1]["lr"]

            torch.cuda.synchronize()

            log_metrics = {
                f"loss/loss": loss_value,
                f"loss/recon_loss": loss_dict['recon_loss'].item(),
                f"loss/cmt_loss": loss_dict['commitment_loss'].item(),
                "lr": lr,
                "lr2": lr2,
            }
            metric_logger.update(**log_metrics)

        # 对比
        else:
            _, features, _, tar_features = batch
            features = features.cuda(cfg.dist.gpu, non_blocking=True)
            tar_features = tar_features.cuda(cfg.dist.gpu, non_blocking=True)
            output = model(features, tar_features, return_clip_loss=True)
            loss = output['loss']

            # commitment loss
            clip_loss_weight = 1.
            loss = loss * clip_loss_weight + output['commitment_loss']

            loss_ori = output['loss_ori'].item()
            loss_self = output['loss_self'].item()
            loss_cl = output['loss_cl'].item()

            # 计算准确率
            acc = output['clip_acc'].item()

            temperature = model.module.logit_scale.item()
            temperature_self = model.module.logit_scale_self.item()
            temperature_cl = model.module.logit_scale_cl.item()
            loss_value = loss.item()

            commitment_loss = output['commitment_loss'].item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss /= accum_iter

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            lr = optimizer.param_groups[0]["lr"]

            log_metrics = {
                f"loss/loss": loss_value,
                f"loss/loss_ori": loss_ori,
                f"loss/loss_self": loss_self,
                f"loss/loss_cl": loss_cl,
                f"loss/cmt_loss": commitment_loss,
                f"acc": acc,

                "lr": lr,
                f"temp/temperature": temperature,
                f"temp/temperature_self": temperature_self,
                f"temp/temperature_cl": temperature_cl,
            }
            metric_logger.update(**log_metrics)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    main()
