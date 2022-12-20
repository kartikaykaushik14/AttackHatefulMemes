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

def img_text_fgsm_attack(model, test_path, perturbed_test_path) :

    attack_success = 0
    original_success = 0
    test_dataset = model._build_dataset(test_path)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        shuffle=False, 
        batch_size= 4, 
        num_workers= 16)

    perturbed_test_dataset = model._build_dataset(perturbed_test_path)
    perturbed_test_dataloader = torch.utils.data.DataLoader(
        perturbed_test_dataset,
        shuffle=False,
        batch_size= 4,
        num_workers= 16)
    
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):

        batch["image"], batch["text"], batch["label"] = batch["image"].to("cuda"), batch["text"].to("cuda"), batch["label"].to("cuda")

        preds, _ = model.eval().to("cuda")(batch["text"], batch["image"])
        preds = preds.max(1, keepdim=True)[1]

        batch2 = next(iter(perturbed_test_dataloader))
        batch2["text"], batch2["image"] =  batch2["text"].to("cuda"), batch2["image"] .to("cuda")

        perturbed_preds, _ = model.eval().to("cuda")(batch2["text"], batch2["image"])
        perturbed_preds = perturbed_preds.max(1, keepdim=True)[1]
        
        for i, p in enumerate(preds):
            if p == batch["label"][i]:
                original_success += 1
                if perturbed_preds[i] != batch["label"][i]:
                    attack_success += 1

    print(attack_success, original_success)
    return ((attack_success/original_success)*100)  

test_path = "/scratch/mmk9369/ml-cybersec/datasets/hateful_memes/defaults/annotations/test_unseen.jsonl"
per_img_path = "perturbed_img_text_test.jsonl"

# load_path = Path.cwd()/"model-outputs/epoch=5-step=1134-v1.ckpt"
load_path = "/scratch/mmk9369/ml-cybersec/model-outputs/epoch=10-step=1045.ckpt"
hateful_memes_model = HatefulMemesModel.load_from_checkpoint(load_path)
print("Starting Image and Text Pertubation")
print(img_text_fgsm_attack(hateful_memes_model, test_path, per_img_path))
