#https://huggingface.co/docs/transformers/main/en/model_doc/dinov2
#https://github.com/facebookresearch/sscd-copy-detection
#https://huggingface.co/docs/diffusers/conceptual/evaluation

import os 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

# ✅ Fix Bug #5: 原代码在全局创建一个 KID 对象并在所有 prompt 间共享使用，
# 导致每次循环的图像都会被累积到同一个 KID 对象中而不是独立计算。
# 修复: 将 KID 对象移入函数内部创建，保证每次调用相互独立。

def kid_score(image_list1, image_list2):
    """计算两组图像之间的 KID 分数。

    修复 Bug #5: 原代码在全局创建一个 KID 对象，所有循环迭代都累积到同一对象。
    现在在函数内部创建新的 KID 对象，每次调用相互独立。

    参数:
        image_list1: 生成图像路径列表 (fake)
        image_list2: vanilla 参考图像路径列表 (real)
    返回:
        KID 均値 (float)
    """
    n1 = len(image_list1)
    n2 = len(image_list2)
    if n1 == 0 or n2 == 0:
        print("[WARNING] kid_score: 一个或两个图像列表为空，返回 0.0")
        return 0.0

    # ✅ Fix: 在函数内部创建新的 KID 对象，防止跨循环累积
    ss = min(n1, n2)  # subset_size 不能超过最小样本集大小
    # KID 要求 subset_size >= 2
    if ss < 2:
        print(f"[WARNING] kid_score: 样本量过小 (n1={n1}, n2={n2})，返回 0.0")
        return 0.0

    # ✅ Fix: 将 KID 移到 GPU 计算，增大 subset_size 提高统计可靠性
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

    savedir_gen = './results/set4/' + file_name # Gen Save Dir
    savedir = './results/set4/' + file_name + '/bs/' # Black Scholes 

    
    image_list = glob.glob(savedir + '*.png')
    image_list_vanilla1 = glob.glob(savedir_gen + '/vanilla/text3/' + '*.png')
    image_list_vanilla2 = glob.glob(savedir_gen + '/vanilla/text4/' + '*.png')

    kid_vanilla1 = kid_score(image_list, image_list_vanilla1)
    kid_vanilla2 = kid_score(image_list, image_list_vanilla2)

    kid_total = kid_total + 0.5 * (kid_vanilla1 + kid_vanilla2)
    
        
print("KID Total: ", kid_total/(len(prompts_list)))
