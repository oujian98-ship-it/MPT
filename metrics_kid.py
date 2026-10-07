
import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"
RESULTS_DIR = os.environ.get("RESULTS_DIR", "result")

import glob
import torch 
from PIL import Image
_ = torch.manual_seed(42)
import PIL
import numpy as np
from torchmetrics.multimodal import CLIPScore
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
from torchmetrics.image.kid import KernelInceptionDistance





def kid_score(image_list1, image_list2):
    """Compute mean KID between generated image paths and vanilla reference paths using a fresh metric per call."""
    n1 = len(image_list1)
    n2 = len(image_list2)
    if n1 == 0 or n2 == 0:
        print("[WARNING] kid_score: one or both image lists are empty; returning 0.0")
        return 0.0


    ss = min(n1, n2)

    if ss < 2:
        print(f"[WARNING] kid_score: sample size is too small (n1={n1}, n2={n2}); returning 0.0")
        return 0.0


    kid_metric = KernelInceptionDistance(subset_size=ss).to("cuda")

    for image in image_list1:
        image1 = PIL.Image.open(image).convert("RGB")
        image1 = np.asarray(image1)
        image1 = torch.from_numpy(image1).unsqueeze(0).permute(0, 3, 1, 2)
        kid_metric.update(image1.to("cuda"), real=False)

    for image in image_list2:
        image2 = PIL.Image.open(image).convert("RGB")
        image2 = np.asarray(image2)
        image2 = torch.from_numpy(image2).unsqueeze(0).permute(0, 3, 1, 2)
        kid_metric.update(image2.to("cuda"), real=True)

    score, _ = kid_metric.compute()
    result = score.detach().cpu().item()
    del kid_metric
    torch.cuda.empty_cache()
    return result


kid_total = 0

with open('data/set4.txt', 'r') as f:
    prompts_list = f.readlines()

for i in range(len(prompts_list)):
    prompts = prompts_list[i]
    prompts = prompts.split('\t')
    file_name = prompts[0]
    prompts = prompts[1]
    prompts = prompts[1:-2]
    prompts = prompts.split(',')
    print(prompts)

    savedir_gen = f'./{RESULTS_DIR}/set4/' + file_name # Gen Save Dir
    savedir = f'./{RESULTS_DIR}/set4/' + file_name + '/bs/' # Black Scholes 

    
    image_list = glob.glob(savedir + '*.png')
    image_list_vanilla1 = glob.glob(savedir_gen + '/vanilla/text3/' + '*.png')
    image_list_vanilla2 = glob.glob(savedir_gen + '/vanilla/text4/' + '*.png')

    kid_vanilla1 = kid_score(image_list, image_list_vanilla1)
    kid_vanilla2 = kid_score(image_list, image_list_vanilla2)

    kid_total = kid_total + 0.5 * (kid_vanilla1 + kid_vanilla2)
    
        
print("KID Total: ", kid_total/(len(prompts_list)))
