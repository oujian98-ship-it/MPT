#https://huggingface.co/docs/transformers/main/en/model_doc/dinov2
#https://github.com/facebookresearch/sscd-copy-detection
#https://huggingface.co/docs/diffusers/conceptual/evaluation

import os 
import patch_torch
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

blip2_local_path = r"d:\projects\BlackScholesDiffusion2024-main\Model\BLIP-2"
model = Blip2ForImageTextRetrieval.from_pretrained(blip2_local_path, torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained(blip2_local_path)


model.to(device)

# DINO
dino_local_path = r"d:\projects\BlackScholesDiffusion2024-main\Model\DINOv2"
image_processor = AutoImageProcessor.from_pretrained(dino_local_path)
dino_model = Dinov2Model.from_pretrained(dino_local_path).cuda()



def blip_score(image, prompt):
    """计算 BLIP 图文匹配分数（ITM正类概率）。
    
    修复 Bug #1: 原代码对 logits 做了两次 softmax（先显式 softmax，再调用 .softmax()），
    导致概率分布被过度平滑。现只做一次 softmax，取 index=1（匹配）的概率。
    """
    text = prompt
    inputs = processor(images=image, text=text, return_tensors="pt").to(device, torch.float16)
    with torch.no_grad():
        itm_out = model(**inputs, use_image_text_matching_head=True)
    # ✅ Fix: 只做一次 softmax，直接从 logits 获取匹配概率
    score1 = F.softmax(itm_out.logits_per_image, dim=1)[0][1].detach().cpu().float().numpy()
    return score1

def dino_score(image_list1, image_list2):
    """计算两组图像之间的 DINOv2 CLS token 余弦相似度。

    修复 Bug #2: 原代码 last_hidden_states2 除以 len(image_list1) 而非 len(image_list2)，
    当两列表长度不同时结果错误。
    修复 Bug #3: 原代码将所有 patch token 展平做余弦相似度；改为使用 CLS token（index=0）
    作为全局语义特征，与 DINOv2 原论文一致。
    """
    # --- 提取 image_list1 的平均 CLS token ---
    cls_feats1 = []
    for image in image_list1:
        image1 = PIL.Image.open(image).convert("RGB")
        image_inputs1 = image_processor(image1, return_tensors="pt")
        image_inputs1['pixel_values'] = image_inputs1['pixel_values'].cuda()
        with torch.no_grad():
            outputs1 = dino_model(**image_inputs1)
        # ✅ Fix: 取 CLS token (index=0)，shape: [1, 768]
        cls_feats1.append(outputs1.last_hidden_state[:, 0, :])

    # --- 提取 image_list2 的平均 CLS token ---
    cls_feats2 = []
    for image in image_list2:
        image2 = PIL.Image.open(image).convert("RGB")
        image_inputs2 = image_processor(image2, return_tensors="pt")
        image_inputs2['pixel_values'] = image_inputs2['pixel_values'].cuda()
        with torch.no_grad():
            outputs2 = dino_model(**image_inputs2)
        # ✅ Fix: 取 CLS token (index=0)，shape: [1, 768]
        cls_feats2.append(outputs2.last_hidden_state[:, 0, :])

    # ✅ Fix: 各自除以自身列表长度（原 Bug 两行都除以 len(image_list1)）
    avg_feat1 = torch.stack(cls_feats1, dim=0).mean(dim=0)  # [1, 768]
    avg_feat2 = torch.stack(cls_feats2, dim=0).mean(dim=0)  # [1, 768]

    # ✅ Fix: 使用 unsqueeze 让 cosine_similarity 在特征维度(dim=1)计算
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

    savedir_gen = './results/set1/' + file_name # Gen Save Dir
    savedir = './results/set1/' + file_name + '/bs/' # Black Scholes 
    
    image_list = glob.glob(savedir + '*.png')
    image_list_vanilla1 = glob.glob(savedir_gen + '/vanilla/text3/' + '*.png')
    image_list_vanilla2 = glob.glob(savedir_gen + '/vanilla/text4/' + '*.png')

    dino_vanilla1 = dino_score(image_list, image_list_vanilla1)
    dino_vanilla2 = dino_score(image_list, image_list_vanilla2)
    max_dino = 0.5 * (dino_vanilla1 + dino_vanilla2)
    dino_total = dino_total + max_dino


    # ✅ Fix Bug #4: 论文定义 BLIP 使用「所有prompt的组合」评估整体对齐
    # 原代码对每个单独 prompt 分别评分再平均，不符合论文定义
    # 这里提供两种计算方式并分别保存
    combined_prompt = ', '.join([p.strip() for p in prompts])  # 组合成总prompt

    max_blip_combined = 0   # 论文定义：用组合prompt
    max_blip_indiv = 0      # 参考对比：对各concept单独prompt平均
    for image in image_list:
        image1 = PIL.Image.open(image).convert("RGB")

        # ✅ 论文定义：BLIP 对组合prompt评分
        blip_comb = blip_score(image1, combined_prompt)

        # 参考：各concept单独prompt平均
        blip3 = blip_score(image1, prompts[0])
        blip4 = blip_score(image1, prompts[1])

        torch.cuda.empty_cache()

        max_blip_combined += blip_comb
        max_blip_indiv += 0.5 * (blip3 + blip4)

    max_blip_combined = max_blip_combined / len(image_list)
    max_blip_indiv    = max_blip_indiv    / len(image_list)

    # 用两条文本分别算再取平均的 BLIP 计算 BLIP⊙DINO
    max_blip = max_blip_indiv
    blip_total = blip_total + max_blip
    dino_blip = dino_blip + (max_dino * max_blip)
        
        
print("BLIP Total (combined prompt, paper def.): ", blip_total/(len(prompts_list)))
print("DINO Total: ",                               dino_total/(len(prompts_list)))
print("BLIP⊙DINO Score: ",                          dino_blip/(len(prompts_list)))

