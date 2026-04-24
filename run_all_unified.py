import os 
import requests
from PIL import Image
from io import BytesIO
import torch
from diffusers import DiffusionPipeline, DDIMScheduler
import PIL
import cv2
import numpy as np 
from scipy import ndimage 
import patch_torch
import gc

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
has_cuda = torch.cuda.is_available()
device = torch.device('cuda' if has_cuda else 'cpu')
model_dir = r"d:\projects\BlackScholesDiffusion2024-main\Model\Stable_Diffusion_2.1"

# === 核心设定区 (可以在这里自由开关您想跑的任务) ===
TARGET_SETS = ["set1", "set2", "set3", "set4"] 
ENABLE_STAGE_1 = True 
ENABLE_STAGE_2 = True
ENABLE_STAGE_3 = False 

# 我们把各种方法的差异化参数写死进配置字典，方便一次性循环
METHODS_CONFIG = {
    # vanilla 单独处理：text1 用 prompts[0]，text2 用 prompts[1]（与原 run_batch_vanilla.py 完全一致）
    'vanilla': {'custom': './models/vanilla', 'steps': 50,  'prompts_idx': [0, 1], 'out_dirs': ['text1', 'text2']},
    # bs: prompts[1:4] -> [1,2,3]，与原 run_batch_bs.py 第57行 prompts = prompts[1:4] 一致
    'bs':      {'custom': './models/bs',      'steps': 100, 'prompts_idx': [1,2,3], 'out_dirs': ['']},
    # lininterp: prompts[2:4] -> [2,3]，与原 run_batch_lininterp.py 第54行 prompts = prompts[2:4] 一致
    'lininterp': {'custom': './models/linear_interpolation', 'steps': 50, 'prompts_idx': [2,3], 'out_dirs': ['']},
    # clip_min: prompts[1:4] -> [1,2,3]，与原 run_batch_clip.py 第54行 prompts = prompts[1:4] 一致
    'clip_min':  {'custom': './models/clip_min',  'steps': 100, 'prompts_idx': [1,2,3], 'out_dirs': ['']},
    # alternating_sampling: prompts[2:4] -> [2,3]，与原 run_batch_altsamp.py 第54行 prompts = prompts[2:4] 一致
    # 注意 pipeline 只接受2个 prompt，steps 原版是 50
    'alternating_sampling': {'custom': './models/alternating_sampling', 'steps': 50, 'prompts_idx': [2,3], 'out_dirs': ['']},
    # step(promptmixing_iccv): pipeline 内部只读 eval_prompt[0] 和 [1]，所以同样传 [2,3] 两个独立概念
    'step': {'custom': './models/promptmixing_iccv', 'steps': 100, 'prompts_idx': [2,3], 'out_dirs': ['']}
}

def load_prompts(set_name):
    with open(f'data/{set_name}.txt', 'r') as f:
        return f.readlines()

def safe_load_pipeline(custom_path):
    print(f"\n🚀 正在加载全新 Pipeline: {custom_path}")
    pipe = DiffusionPipeline.from_pretrained(
        model_dir,
        safety_checker=None,
        use_auth_token=False,
        custom_pipeline=custom_path, 
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    ).to(device)
    return pipe

def clear_vram(pipe):
    print("🧹 释放显存，准备加载下一套模型...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

def check_images_exist(folder_path, expected_count=5):
    if not os.path.exists(folder_path):
        return False
    import glob
    imgs = glob.glob(os.path.join(folder_path, '*.png'))
    return len(imgs) == expected_count

def stage1_generate_mixed():
    print(f"\n========== 开始第一阶段：跑满所有方法的混合主图 (text1, text2) ==========")
    
    for method, cfg in METHODS_CONFIG.items():
        pipe = None
        
        # 针对当前这个 method(方法)，一口气跑完所有的 set！(省去了反复加载同一个大模型的成本)
        for cur_set in TARGET_SETS:
            prompts_list = load_prompts(cur_set)
            
            for i in range(len(prompts_list)):
                raw_prompts = prompts_list[i].split('\t')
                file_name = raw_prompts[0]
                
                p_list = raw_prompts[1][1:-2].split(',')
                
                eval_prompt = []
                for idx in cfg['prompts_idx']:
                    if idx < len(p_list):
                        eval_prompt.append(p_list[idx])
                
                if method == 'vanilla':
                    eval_prompt = p_list[0]
                    
                base_savedir = f'./results/{cur_set}/{file_name}/{method}/'
                
                needs_generation = False
                target_folders = []
                
                if method == 'vanilla':
                    target_folders = [f"{base_savedir}text1", f"{base_savedir}text2"]
                else:
                    target_folders = [base_savedir]
                    
                for tf in target_folders:
                    if not check_images_exist(tf, 5):
                        needs_generation = True
                        break
                        
                if not needs_generation:
                    continue # 该目标的该方法已完善，跳过
                    
                if pipe is None:
                    pipe = safe_load_pipeline(cfg['custom'])
                    
                print(f"🔄 正在生成: {cur_set} -> {file_name} -> {method}")
                
                if method == 'vanilla':
                    # Vanilla 特殊处理：text1 用 prompts[0]，text2 用 prompts[1]，分开生成
                    for folder_idx, out_f in enumerate(target_folders):
                        prompt_for_folder = p_list[folder_idx].strip().replace("'", "")
                        os.makedirs(out_f, exist_ok=True)
                        for gen_id in range(1, 6):
                            out_path = f"{out_f}/result{gen_id}.png"
                            if os.path.exists(out_path): continue
                            res = pipe(guidance_scale=7.5, num_inference_steps=cfg['steps'], eval_prompt=prompt_for_folder)
                            res.images[0].save(out_path)
                else:
                    # 其他方法：直接把 eval_prompt 列表传给 pipeline
                    out_f = target_folders[0]
                    os.makedirs(out_f, exist_ok=True)
                    for gen_id in range(1, 6):
                        out_path = f"{out_f}/result{gen_id}.png"
                        if os.path.exists(out_path): continue
                        res = pipe(guidance_scale=7.5, num_inference_steps=cfg['steps'], eval_prompt=eval_prompt)
                        res.images[0].save(out_path)
                
        if pipe is not None:
            clear_vram(pipe)

def stage2_generate_baselines():
    print(f"\n========== 开始第二阶段：单独补齐原生基准分离图 (text3, text4) ==========")
    method = 'vanilla'
    cfg = METHODS_CONFIG[method]
    pipe = None
    
    # 同样地，一个Vanilla模型扛下所有4个set的补图重任
    for cur_set in TARGET_SETS:
        prompts_list = load_prompts(cur_set)
        
        for i in range(len(prompts_list)):
            raw_prompts = prompts_list[i].split('\t')
            file_name = raw_prompts[0]
            p_list = raw_prompts[1][1:-2].split(',')
            
            base_savedir = f'./results/{cur_set}/{file_name}/vanilla/'
            tf3 = f"{base_savedir}text3"
            tf4 = f"{base_savedir}text4"
            
            needs_3 = not check_images_exist(tf3, 5)
            needs_4 = not check_images_exist(tf4, 5)
            
            if not (needs_3 or needs_4):
                continue
                
            if pipe is None:
                pipe = safe_load_pipeline(cfg['custom'])
                
            print(f"🔧 正在缝补基线: {cur_set} -> {file_name}")
            
            if needs_3 and len(p_list) > 2:
                os.makedirs(tf3, exist_ok=True)
                for gen_id in range(1, 6):
                    out_path = f"{tf3}/result{gen_id}.png"
                    if os.path.exists(out_path): continue
                    res = pipe(guidance_scale=7.5, num_inference_steps=cfg['steps'], eval_prompt = p_list[2].strip().replace("'", ""))
                    res.images[0].save(out_path)
                    
            if needs_4 and len(p_list) > 3:
                os.makedirs(tf4, exist_ok=True)
                for gen_id in range(1, 6):
                    out_path = f"{tf4}/result{gen_id}.png"
                    if os.path.exists(out_path): continue
                    res = pipe(guidance_scale=7.5, num_inference_steps=cfg['steps'], eval_prompt = p_list[3].strip().replace("'", ""))
                    res.images[0].save(out_path)
                    
    if pipe is not None:
        clear_vram(pipe)

def stage3_evaluate_results():
    print(f"\n========== 开始第三阶段：跨项目评估打分，生成最终成绩单 ==========")
    import subprocess
    # 调用专门的评测脚本
    result = subprocess.run(["python", "-u", "reproduce_table1.py"])
    if result.returncode == 0:
        print("✅ 成绩单 Table 1 评估完成！请查看 table1_reproduced.md")
    else:
        print("❌ 评测阶段出现异常。")

if __name__ == "__main__":
    print("🌟 强力大一统统筹脚本已启动！")
    if ENABLE_STAGE_1:
        stage1_generate_mixed()
    if ENABLE_STAGE_2:
        stage2_generate_baselines()
    if ENABLE_STAGE_3:
        stage3_evaluate_results()
    
    print("\n✅ 所有配置的工作流已彻底完成！")
