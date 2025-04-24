#!/usr/bin/env python
import os
import torch
from torch.hub import load_state_dict_from_url

def download_vgg19_weights():
    # URL for VGG19 weights; this is the official URL used by torchvision.
    url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
    local_filename = "vgg19_pretrained_local.pth"
    
    if os.path.exists(local_filename):
        print(f"File '{local_filename}' already exists. No need to download.")
        return
    
    print("Downloading VGG19 pretrained weights...")
    state_dict = load_state_dict_from_url(url, progress=True)
    torch.save(state_dict, local_filename)
    print(f"Downloaded and saved weights to '{local_filename}'.")

if __name__ == "__main__":
    download_vgg19_weights()