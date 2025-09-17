# encoding: utf-8
"""
@author: Yingwu.XSW
@date: 2022/3/9 下午6:35
"""
import os
import shutil

import torch
import torch.autograd as autograd
import torch.distributed as dist
from omegaconf import OmegaConf


def get_model(model):
    if isinstance(model, torch.nn.DataParallel) \
            or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model


def save_on_master(state, is_best, output_dir):
    if is_main_process():
        ckpt_path = f'{output_dir}/checkpoint.pt'
        best_path = f'{output_dir}/checkpoint_best.pt'
        torch.save(state, ckpt_path)
        if is_best:
            shutil.copyfile(ckpt_path, best_path)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(cfg, args):
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_readonly(cfg, False)

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.dist.rank = int(os.environ["RANK"])
        cfg.dist.world_size = int(os.environ['WORLD_SIZE'])
        cfg.dist.gpu = max(args.local_rank, 0)
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        cfg.dist.distributed = False
        return

    cfg.dist.distributed = True

    torch.cuda.set_device(cfg.dist.gpu)
    cfg.dist.dist_backend = 'nccl'
    cfg.dist.dist_url = args.dist_url
    print('| distributed init (rank {}): {}, gpu {}'.format(cfg.dist.rank, cfg.dist.dist_url, cfg.dist.gpu), flush=True)
    torch.distributed.init_process_group(backend=cfg.dist.dist_backend)
    cfg.dist.rank = dist.get_rank()
    cfg.dist.world_size = dist.get_world_size()
    # setup_for_distributed(args.rank == 0)
    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg, True)


def scaled_all_reduce(tensors, is_scale=True):
    """Performs the scaled all_reduce operation on the provided tensors.
    The input tensors are modified in-place. Currently supports only the sum
    reduction operator. The reduced values are scaled by the inverse size of the
    world size.
    """
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    # Queue the reductions
    reductions = []
    for tensor in tensors:
        reduction = dist.all_reduce(tensor, async_op=True)  # default sum
        reductions.append(reduction)
    # Wait for reductions to finish
    for reduction in reductions:
        reduction.wait()
    # Scale the results
    if is_scale:
        for tensor in tensors:
            tensor.mul_(1.0 / world_size)
    return tensors


def all_gather_batch(tensors):
    """
    Performs all_gather operation on the provided tensors.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []
    for tensor in tensors:
        tensor_all = [torch.ones_like(tensor) for _ in range(world_size)]
        dist.all_gather(
            tensor_all,
            tensor,
            async_op=False  # performance opt
        )

        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


class GatherLayer(autograd.Function):
    """
    Gather tensors from all workers with support for backward propagation:
    This implementation does not cut the gradients as torch.distributed.all_gather does.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def all_gather_batch_with_grad(tensors):
    """
    Performs all_gather operation on the provided tensors.
    Graph remains connected for backward grad computation.
    """
    # Queue the gathered tensors
    world_size = get_world_size()
    # There is no need for reduction in the single-proc case
    if world_size == 1:
        return tensors
    tensor_list = []
    output_tensor = []

    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        tensor_list.append(tensor_all)

    for tensor_all in tensor_list:
        output_tensor.append(torch.cat(tensor_all, dim=0))
    return output_tensor


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def all_reduce_mean_batch(args):
    """
    对输入的多个变量（张量）进行全局均值汇聚
    :param args: 多个输入变量，可以是一个或多个张量
    :return: 每个输入变量的全局均值
    """
    world_size = get_world_size()
    results = []

    for x in args:
        if world_size > 1:
            x_reduce = torch.tensor(x).cuda()
            dist.all_reduce(x_reduce)
            x_reduce /= world_size
            results.append(x_reduce.item())
        else:
            results.append(x)

    return results
