import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import tpa


def get_scheduler(name):
    if hasattr(lr_scheduler, name):
        return getattr(lr_scheduler, name)
    else:
        raise NotImplementedError


def getattr_recursive(m, attr):
    for name in attr.split("."):
        m = getattr(m, name)
    return m


def get_parameters(model, name):
    module = getattr_recursive(model, name)
    if isinstance(module, nn.Module):
        return module.parameters()
    elif isinstance(module, nn.Parameter):
        return module
    return []


def parse_optimizer(config, model):
    # if hasattr(config, "params"):
    #     params = [
    #         {"params": get_parameters(model, name), "name": name, **args}
    #         for name, args in config.params.items()
    #     ]
    #     tpa.debug(f"Specify optimizer params: {config.params}")
    # else:
    #     params = model.parameters()
    params_optimizer = list(model.named_parameters())
    params = []
    for name,args in config.params.items():    
        # params.append({"params": get_parameters(model, name), "name": name, **args})
        params_i = [p for n, p in params_optimizer if name in n]
        params.append({"params": params_i, "name": name, **args})
    params_others = [p for n, p in params_optimizer if all([name not in n for name in config.params.keys()])]
    params.append({"params": params_others, "name": "others", **config.args})
    # clip_params = [p for n, p in params_optimizer if "clip." in n]
    # noclip_params = [p for n, p in params_optimizer if "clip." not in n]
    # optimizer_grouped_params = [
    #     {'params': clip_params, 'lr': config.clip_lr},
    #     {'params': noclip_params, 'lr': config.noclip_lr}
    # ]
    # optimizer = AdamW(optimizer_grouped_params, weight_decay=config.weight_decay)
    # num_training_steps = len(train_data_loader) * config.num_epochs
    # num_warmup_steps = int(config.warmup_proportion * num_training_steps)
    # scheduler = get_cosine_schedule_with_warmup(optimizer,
    #                                             num_warmup_steps=num_warmup_steps,
    #                                             num_training_steps=num_training_steps)
    if config.name in ["FusedAdam"]:
        import apex
        optim = getattr(apex.optimizers, config.name)(params, **config.args)
    elif config.name in ["Adam8bit", "AdamW8bit"]:
        import bitsandbytes as bnb

        optim = bnb.optim.Adam8bit(params, **config.args)
    else:
        optim = getattr(torch.optim, config.name)(params, **config.args)
    return optim


def parse_scheduler_to_instance(config, optimizer):
    if config.name == "ChainedScheduler":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.ChainedScheduler(schedulers)
    elif config.name == "Sequential":
        schedulers = [
            parse_scheduler_to_instance(conf, optimizer) for conf in config.schedulers
        ]
        scheduler = lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=config.milestones
        )
    else:
        scheduler = getattr(lr_scheduler, config.name)(optimizer, **config.args)
    return scheduler


def parse_scheduler(config, optimizer):
    interval = config.get("interval", "epoch")
    assert interval in ["epoch", "step"]
    if config.name == "SequentialLR":
        scheduler = {
            "scheduler": lr_scheduler.SequentialLR(
                optimizer,
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ],
                milestones=config.milestones,
            ),
            "interval": interval,
        }
    elif config.name == "ChainedScheduler":
        scheduler = {
            "scheduler": lr_scheduler.ChainedScheduler(
                [
                    parse_scheduler(conf, optimizer)["scheduler"]
                    for conf in config.schedulers
                ]
            ),
            "interval": interval,
        }
    else:
        scheduler = {
            "scheduler": get_scheduler(config.name)(optimizer, **config.args),
            "interval": interval,
        }
    return scheduler
