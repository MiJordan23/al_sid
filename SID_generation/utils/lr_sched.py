# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math


def cosine_scheduler(base_value, final_value, epoch, all_epochs, warmup_epochs=0):
    if epoch < warmup_epochs:
        lr = base_value * epoch / warmup_epochs
    else:
        epoch = epoch - warmup_epochs
        all_epochs = all_epochs - warmup_epochs
        lr = final_value + (base_value - final_value) * 0.5 * (1. + math.cos(math.pi * epoch / all_epochs))
    return lr


def polynomial_decay_scheduler_step(base_value, final_value, step, max_steps, warmup_steps=0.0, power=1.0):
    if step < warmup_steps:
        return float(step) / float(max(0.1, warmup_steps)) * base_value
    elif step > max_steps:
        return final_value
    else:
        lr_range = base_value - final_value
        decay_steps = max_steps - warmup_steps
        pct_remaining = 1 - (step - warmup_steps) / decay_steps
        lr = lr_range * pct_remaining ** power + final_value
        return lr


def adjust_learning_rate(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    # calculate now learning rate
    if epoch < cfg.train.warmup_epochs:
        lr = cfg.train.lr * epoch / cfg.train.warmup_epochs
    else:
        lr = cfg.train.min_lr + (cfg.train.lr - cfg.train.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - cfg.train.warmup_epochs) / (cfg.train.epochs - cfg.train.warmup_epochs)))

    # assign learning rate
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def adjust_learning_rate_circle(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    # calculate now learning rate
    if epoch < cfg.train.warmup_epochs:
        lr = cfg.train.lr * epoch / cfg.train.warmup_epochs
    else:
        total_decay_epoch  = cfg.train.epochs - cfg.train.warmup_epochs
        cur_ep = epoch - cfg.train.warmup_epochs
        total_decay_epoch = total_decay_epoch // 3
        cur_ep = cur_ep % total_decay_epoch
        lr = cfg.train.min_lr + (cfg.train.lr - cfg.train.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * cur_ep / total_decay_epoch))

    # assign learning rate
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr



def adjust_learning_rate_wd(optimizer, epoch, cfg):
    """Decay the learning rate with half-cycle cosine after warmup"""
    # calculate now learning rate
    if epoch < cfg.train.warmup_epochs:
        lr = cfg.train.lr * epoch / cfg.train.warmup_epochs
    else:
        lr = cfg.train.min_lr + (cfg.train.lr - cfg.train.min_lr) * 0.5 * \
             (1. + math.cos(math.pi * (epoch - cfg.train.warmup_epochs) / (cfg.train.epochs - cfg.train.warmup_epochs)))

    wd = cosine_scheduler(cfg.train.weight_decay, cfg.train.weight_decay_end, epoch, cfg.train.epochs)

    # assign learning rate
    for idx, param_group in enumerate(optimizer.param_groups):
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
        if idx == 1:
            param_group["weight_decay"] = wd
    return lr


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    x = [x / 20.0 for x in range(0, 100)]
    y = [polynomial_decay_scheduler_step(1.0, 0.01, xx, 5, 0.5) for xx in x]

    plt.plot(x, y)
    plt.show()

