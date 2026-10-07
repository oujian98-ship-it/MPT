
import os 
import patch_torch
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
from transformers import AutoProcessor, Blip2ForImageTextRetrieval
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
import requests
from transformers import AutoImageProcessor, Dinov2Model, CLIPProcessor, CLIPModel


device = "cuda" if torch.cuda.is_available() else "cpu"

blip2_local_path = r"Model\BLIP-2"
model = Blip2ForImageTextRetrieval.from_pretrained(blip2_local_path, torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained(blip2_local_path)


model.to(device)

# DINO
dino_local_path = r"Model\DINOv2"
image_processor = AutoImageProcessor.from_pretrained(dino_local_path)
dino_model = Dinov2Model.from_pretrained(dino_local_path).cuda()



def blip_score(image, prompt):
    """Compute BLIP image-text matching probability. Apply softmax once and return the matching class (index 1)."""
    text = prompt
    inputs = processor(images=image, text=text, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        itm_out = model(**inputs, use_image_text_matching_head=True)

    score1 = F.softmax(itm_out.logits_per_image, dim=1)[0][1].detach().cpu().float().numpy()
    return score1

def dino_score(image_list1, image_list2):
    """Compute cosine similarity between mean DINOv2 CLS features, normalizing each image list by its own length."""

    cls_feats1 = []
    for image in image_list1:
        image1 = PIL.Image.open(image).convert("RGB")
        image_inputs1 = image_processor(image1, return_tensors="pt")
        image_inputs1['pixel_values'] = image_inputs1['pixel_values'].cuda()
        with torch.no_grad():
            outputs1 = dino_model(**image_inputs1)

        cls_feats1.append(outputs1.last_hidden_state[:, 0, :])


    cls_feats2 = []
    for image in image_list2:
        image2 = PIL.Image.open(image).convert("RGB")
        image_inputs2 = image_processor(image2, return_tensors="pt")
        image_inputs2['pixel_values'] = image_inputs2['pixel_values'].cuda()
        with torch.no_grad():
            outputs2 = dino_model(**image_inputs2)

        cls_feats2.append(outputs2.last_hidden_state[:, 0, :])


    avg_feat1 = torch.stack(cls_feats1, dim=0).mean(dim=0)  # [1, 768]
    avg_feat2 = torch.stack(cls_feats2, dim=0).mean(dim=0)  # [1, 768]


    sim = F.cosine_similarity(avg_feat1, avg_feat2, dim=1)
    return sim.item()


blip_total = 0
dino_total = 0
dino_blip = 0

with open('data/set1.txt', 'r') as f:
    prompts_list = f.readlines()


for i in range(len(prompts_list)):
    prompts = prompts_list[i]
    prompts = prompts.split('\t')
    file_name = prompts[0]
    prompts = prompts[1]
    prompts = prompts[1:-2]
    prompts = prompts.split(',')
    print(prompts)

    savedir_gen = f'./{RESULTS_DIR}/set1/' + file_name # Gen Save Dir
    savedir = f'./{RESULTS_DIR}/set1/' + file_name + '/bs/' # Black Scholes 
    
    image_list = glob.glob(savedir + '*.png')
    image_list_vanilla1 = glob.glob(savedir_gen + '/vanilla/text3/' + '*.png')
    image_list_vanilla2 = glob.glob(savedir_gen + '/vanilla/text4/' + '*.png')

    dino_vanilla1 = dino_score(image_list, image_list_vanilla1)
    dino_vanilla2 = dino_score(image_list, image_list_vanilla2)
    max_dino = 0.5 * (dino_vanilla1 + dino_vanilla2)
    dino_total = dino_total + max_dino





    combined_prompt = ', '.join([p.strip() for p in prompts])

    max_blip_combined = 0
    max_blip_indiv = 0
    for image in image_list:
        image1 = PIL.Image.open(image).convert("RGB")


        blip_comb = blip_score(image1, combined_prompt)


        blip3 = blip_score(image1, prompts[0])
        blip4 = blip_score(image1, prompts[1])

        torch.cuda.empty_cache()

        max_blip_combined += blip_comb
        max_blip_indiv += 0.5 * (blip3 + blip4)

    max_blip_combined = max_blip_combined / len(image_list)
    max_blip_indiv    = max_blip_indiv    / len(image_list)


    max_blip = max_blip_indiv
    blip_total = blip_total + max_blip
    dino_blip = dino_blip + (max_dino * max_blip)
        
        
print("BLIP Total (combined prompt, paper def.): ", blip_total/(len(prompts_list)))
print("DINO Total: ",                               dino_total/(len(prompts_list)))
print("BLIP⊙DINO Score: ",                          dino_blip/(len(prompts_list)))

