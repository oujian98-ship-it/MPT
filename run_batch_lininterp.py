import os 
import patch_torch

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
# os.environ["CUDA_VISIBLE_DEVICES"]="5" # 注释掉硬编码的 GPU ID

# from modelscope import snapshot_download # 移除在线下载

import requests
from PIL import Image
from io import BytesIO
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
import PIL
import cv2
import numpy as np 
from scipy import ndimage 
#import matplotlib.pyplot as plt 

has_cuda = torch.cuda.is_available()

device = torch.device('cuda' if has_cuda else 'cpu')
# torch.hub.set_dir('/scratch0/') # 移除无效的 Linux 路径

SET_NAME = os.environ.get("EXPERIMENT_SET", "set4")

with open(f'data/{SET_NAME}.txt', 'r') as f:
    prompts_list = f.readlines()

# 使用本地模型路径
model_dir = r"d:\projects\BlackScholesDiffusion2024-main\Model\Stable_Diffusion_2.1"


pipe = DiffusionPipeline.from_pretrained(
    model_dir,
    safety_checker=None,
    use_auth_token=False,
    custom_pipeline='./models/linear_interpolation', 
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
).to(device)

generator = torch.Generator(device.type).manual_seed(0)
seed = 0

for i in range(len(prompts_list)):
    prompts = prompts_list[i]
    prompts = prompts.split('\t')
    file_name = prompts[0]
    prompts = prompts[1]
    prompts = prompts[1:-2]
    prompts = prompts.split(',')
    print(prompts)
    
    prompts = prompts[2:4]

    savedir = f'./results/{SET_NAME}/' + file_name + '/lininterp/'
    os.makedirs(savedir, exist_ok=True)
    eval_prompt = prompts
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result1.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result2.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result3.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result4.png')
    res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt = eval_prompt)
    image = res.images[0]
    image.save(savedir+'/result5.png')
