import os
import ray
import glob
import time
import torch
from transformers import AutoModel
import torchfly_dev

ray.init(memory=12* 1024**3, object_store_memory=8*1024**3, redis_max_memory=8 * 1024**3)

model = AutoModel.from_pretrained("roberta-large")

device = torch.device("cuda")
model = model.cuda()


for i in range(100):

    start = time.time()

    obj = torchfly_dev.async_save(model.state_dict(), f"tmp.pth")

    time.sleep(4)
    end = time.time()


    print(f"Time takes: {end-start-4}s")

time.sleep(100)