import logging
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

__all__ = ["create_folder", "init_torch_seeds", "select_device"]

logger = logging.getLogger(__name__)


def create_folder(folder):
    try:
        os.makedirs(folder)
        logger.info(
            f"Create `{os.path.join(os.getcwd(), folder)}` directory successful."
        )
    except OSError:
        logger.warning(
            f"Directory `{os.path.join(os.getcwd(), folder)}` already exists!"
        )
        pass


# Source from "https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py"
def init_torch_seeds(seed: int = 0):
    r"""Sets the seed for generating random numbers. Returns a
    Args:
        seed (int): The desired seed.
    """

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    logger.info("Initialize random seed.")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device: str = "", batch_size: int = 1) -> torch.device:
    r"""Choose the right equipment.
    Args:
        device (optional, str): Use CPU or CUDA. (Default: ````)
        batch_size (optional, int): Data batch size, cannot be less than the number of devices. (Default: 1).
    Returns:
        torch.device.
    """
    # device = "cpu" or "cuda:0,1,2,3".
    only_cpu = device.lower() == "cpu"
    if device and not only_cpu:  # if device requested other than "cpu".
        os.environ[
            "CUDA_VISIBLE_DEVICES"
        ] = device  # set environment variable.
        assert (
            torch.cuda.is_available()
        ), f"CUDA unavailable, invalid device {device} requested"

    cuda = False if only_cpu else torch.cuda.is_available()
    if cuda:
        c = 1024**2  # bytes to MB.
        gpu_count = torch.cuda.device_count()
        if (
            gpu_count > 1 and batch_size
        ):  # check that batch_size is compatible with device_count.
            assert (
                batch_size % gpu_count == 0
            ), f"batch-size {batch_size} not multiple of GPU count {gpu_count}"
        x = [torch.cuda.get_device_properties(i) for i in range(gpu_count)]
        s = "Using CUDA "
        for i in range(0, gpu_count):
            if i == 1:
                s = " " * len(s)
            logger.info(
                f"{s}\n\t+ device:{i} (name=`{x[i].name}`, total_memory={int(x[i].total_memory / c)}MB)"
            )
    else:
        logger.info("Using CPU.")

    return torch.device("cuda:0" if cuda else "cpu")
