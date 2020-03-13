import os
import ray
import glob
import time
import torch
from transformers import AutoModel
import torchfly_dev

model = AutoModel.from_pretrained("roberta-large")

device = torch.device("cuda")
model = model.cuda()


for i in range(100):

    start = time.time()

    obj = torch.save(model.state_dict(), f"tmp.pth")

    time.sleep(4)
    end = time.time()


    print(f"Time takes: {end-start-4}s")

time.sleep(100)