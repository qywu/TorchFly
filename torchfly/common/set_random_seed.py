import torch
import random
import numpy as np
import os

def set_random_seed(random_seed=123):
    # Reproducibility
    if "RANDOM_SEED" not in os.environ:
        random_seed = random_seed
    else:
        random_seed = int(os.environ["RANDOM_SEED"])

    if "RANK" in os.environ:
        random_seed += int(os.environ["RANK"])

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
