import json
import logging
from pathlib import Path
import random
import tarfile
import tempfile
import warnings
import fasttext
import torchvision
import torch
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_path  # Path style access for pandas
from tqdm import tqdm
import torchvision.transforms as T

from model import HatefulMemesModel

def img_fgsm_attack(model, test_path, eps) :

    attack_success = 0
    original_success = 0
    test_dataset = model._build_dataset(test_path)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        shuffle=False, 
        batch_size= 4, 
        num_workers= 16)

    # un normalize the tensors before perturbing.
    inverseNorm = torchvision.transforms.Compose([ 
        torchvision.transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        torchvision.transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],std = [ 1., 1., 1. ]),
    ])
    norm = torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)
                )
    
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):

        batch["image"], batch["text"], batch["label"] = batch["image"].to("cuda"), batch["text"].to("cuda"), batch["label"].to("cuda")
        batch["image"].requires_grad =  True
        preds, _ = model.eval().to("cuda")(batch["text"], batch["image"])
        # preds = preds.max(1, keepdim=True)[1]
        preds = torch.nn.functional.softmax(preds)

        model.zero_grad()
        # print(preds, batch["label"])
        l = torch.nn.CrossEntropyLoss()
        output = l(preds,batch["label"]).to("cuda")
        output.backward()
        batch_attack_images = inverseNorm(batch["image"]) + eps*batch["image"].grad.sign()
        batch_attack_images = torch.clamp(batch_attack_images, 0, 1) 
        for i in range(4):
            torchvision.utils.save_image(batch_attack_images[i], '/scratch/mmk9369/ml-cybersec/datasets/hateful_memes/defaults/annotations/perturbed/{}.png'.format((batch['id'][i]).item()))


data_dir = Path.cwd().parent / "datasets" / "hateful_memes" / "defaults" / "annotations"
print(data_dir)

test_path = data_dir / "train.jsonl"
# per_img_path = data_dir / "perturbed_img"

# load_path = Path.cwd()/"model-outputs/epoch=5-step=1134-v1.ckpt"
load_path = "/scratch/mmk9369/ml-cybersec/model-outputs/epoch=10-step=1045.ckpt"
hateful_memes_model = HatefulMemesModel.load_from_checkpoint(load_path)
eps = 10/255
loss = torch.nn.CrossEntropyLoss
print("Starting")
img_fgsm_attack(hateful_memes_model, test_path, eps)
