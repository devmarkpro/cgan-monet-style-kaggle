from functools import lru_cache
import torch.nn as nn
from typing import Optional
from monet_wandb import MonetWandb
import time


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


@lru_cache(maxsize=1)
def get_run_name(wandb: Optional[MonetWandb] = None) -> str:
    if wandb is not None:
        return wandb.run.name
    # generate random name if wandb is not available
    return f"run_{int(time.time())}"
