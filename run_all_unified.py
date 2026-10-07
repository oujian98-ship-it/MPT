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
import json
import datetime
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
has_cuda = torch.cuda.is_available()
device = torch.device('cuda' if has_cuda else 'cpu')
model_dir = r"Model\Stable_Diffusion_2.1"


TARGET_SETS = ["set1", "set2", "set3", "set4"] 
ENABLE_STAGE_1 = True 
ENABLE_STAGE_2 = True
ENABLE_STAGE_3 = False 
RESULTS_DIR = os.environ.get("RESULTS_DIR", "result")
LOG_DIR = os.environ.get("LOG_DIR", "logs")
RUN_TS = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_SEED = int(os.environ.get("EXPERIMENT_SEED", "0"))


ALL_METHODS_CONFIG = {

    'vanilla': {'custom': './models/vanilla', 'steps': 50,  'prompts_idx': [0, 1], 'out_dirs': ['text1', 'text2']},

    'bs':      {'custom': './models/bs',      'steps': 100, 'prompts_idx': [1,2,3], 'out_dirs': ['']},

    'lininterp': {'custom': './models/linear_interpolation', 'steps': 50, 'prompts_idx': [2,3], 'out_dirs': ['']},

    'clip_min':  {'custom': './models/clip_min',  'steps': 100, 'prompts_idx': [1,2,3], 'out_dirs': ['']},


    'alternating_sampling': {'custom': './models/alternating_sampling', 'steps': 100, 'prompts_idx': [2,3], 'out_dirs': ['']},

    'step': {'custom': './models/promptmixing_iccv', 'steps': 100, 'prompts_idx': [2,3], 'out_dirs': ['']}
}

# Comma-separated method selection is useful for repeated experiments.  The
# default remains the original behaviour: generate every baseline.
_requested_methods = os.environ.get("METHODS_TO_RUN", "all").strip().lower()
if _requested_methods in ("", "all"):
    METHODS_CONFIG = ALL_METHODS_CONFIG
else:
    _method_names = [name.strip() for name in _requested_methods.split(",") if name.strip()]
    _unknown_methods = sorted(set(_method_names) - set(ALL_METHODS_CONFIG))
    if _unknown_methods:
        raise ValueError(f"Unknown METHODS_TO_RUN value(s): {', '.join(_unknown_methods)}")
    METHODS_CONFIG = {name: ALL_METHODS_CONFIG[name] for name in _method_names}

GEN_PERF = {
    method: {"new_images": 0, "time_s": 0.0, "peak_mem_gb": 0.0}
    for method in METHODS_CONFIG
}

def load_prompts(set_name):
    with open(f'data/{set_name}.txt', 'r') as f:
        return f.readlines()

def safe_load_pipeline(custom_path):
    print(f"\n Loading pipeline: {custom_path}")
    pipe = DiffusionPipeline.from_pretrained(
        model_dir,
        safety_checker=None,
        use_auth_token=False,
        custom_pipeline=custom_path, 
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    ).to(device)
    return pipe

def clear_vram(pipe):
    print(" Releasing GPU memory before loading the next model...")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

def check_images_exist(folder_path, expected_count=5):
    if not os.path.exists(folder_path):
        return False
    import glob
    imgs = glob.glob(os.path.join(folder_path, '*.png'))
    return len(imgs) >= expected_count

def make_generator(set_index, prompt_index, image_index, stream_index=0):
    """Create a reproducible, run-specific generator for one output image."""
    seed = EXPERIMENT_SEED + set_index * 100_000 + prompt_index * 100 + image_index + stream_index * 10_000
    return torch.Generator(device.type).manual_seed(seed)

def start_image_timer():
    if has_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    return time.perf_counter()

def record_generation_perf(method, start_time):
    if has_cuda:
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    GEN_PERF[method]["new_images"] += 1
    GEN_PERF[method]["time_s"] += elapsed
    if has_cuda:
        peak_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        GEN_PERF[method]["peak_mem_gb"] = max(GEN_PERF[method]["peak_mem_gb"], peak_gb)

def write_generation_perf_log():
    os.makedirs(LOG_DIR, exist_ok=True)
    summary = {}
    for method, raw in GEN_PERF.items():
        images = raw["new_images"]
        total_time = raw["time_s"]
        summary[method] = {
            "new_images": images,
            "total_time_s": round(total_time, 4),
            "avg_time_s": round(total_time / images, 4) if images else 0.0,
            "gpu_hrs": round(total_time / 3600.0, 6),
            "peak_mem_gb": round(raw["peak_mem_gb"], 4),
        }

    log_path = os.path.join(LOG_DIR, f"generation_perf_run_all_{RUN_TS}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"\n Generation performance log saved: {log_path}")

def stage1_generate_mixed():
    print(f"\n========== Stage 1: generate main images for all methods (text1, text2) ==========")
    
    for method, cfg in METHODS_CONFIG.items():
        pipe = None
        

        for set_index, cur_set in enumerate(TARGET_SETS):
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
                    
                base_savedir = os.path.join(RESULTS_DIR, cur_set, file_name, method)
                
                needs_generation = False
                target_folders = []
                
                if method == 'vanilla':
                    target_folders = [os.path.join(base_savedir, "text1"), os.path.join(base_savedir, "text2")]
                else:
                    target_folders = [base_savedir]
                    
                for tf in target_folders:
                    if not check_images_exist(tf, 5):
                        needs_generation = True
                        break
                        
                if not needs_generation:
                    continue
                    
                if pipe is None:
                    pipe = safe_load_pipeline(cfg['custom'])
                    
                print(f" Generating: {cur_set} -> {file_name} -> {method}")
                
                if method == 'vanilla':

                    for folder_idx, out_f in enumerate(target_folders):
                        prompt_for_folder = p_list[folder_idx].strip().replace("'", "")
                        os.makedirs(out_f, exist_ok=True)
                        for gen_id in range(1, 6):
                            out_path = os.path.join(out_f, f"result{gen_id}.png")
                            if os.path.exists(out_path): continue
                            timer = start_image_timer()
                            res = pipe(
                                guidance_scale=7.5, num_inference_steps=cfg['steps'],
                                eval_prompt=prompt_for_folder,
                                generator=make_generator(set_index, i, gen_id, folder_idx),
                            )
                            res.images[0].save(out_path)
                            record_generation_perf(method, timer)
                else:

                    out_f = target_folders[0]
                    os.makedirs(out_f, exist_ok=True)
                    for gen_id in range(1, 6):
                        out_path = os.path.join(out_f, f"result{gen_id}.png")
                        if os.path.exists(out_path): continue
                        timer = start_image_timer()
                        res = pipe(
                            guidance_scale=7.5, num_inference_steps=cfg['steps'],
                            eval_prompt=eval_prompt,
                            generator=make_generator(set_index, i, gen_id),
                        )
                        res.images[0].save(out_path)
                        record_generation_perf(method, timer)
                
        if pipe is not None:
            clear_vram(pipe)

def stage2_generate_baselines():
    print(f"\n========== Stage 2: complete vanilla reference images (text3, text4) ==========")
    if 'vanilla' not in METHODS_CONFIG:
        print("[SKIP] vanilla is not selected in METHODS_TO_RUN; cannot generate text3/text4 references.")
        return
    method = 'vanilla'
    cfg = METHODS_CONFIG[method]
    pipe = None
    

    for set_index, cur_set in enumerate(TARGET_SETS):
        prompts_list = load_prompts(cur_set)
        
        for i in range(len(prompts_list)):
            raw_prompts = prompts_list[i].split('\t')
            file_name = raw_prompts[0]
            p_list = raw_prompts[1][1:-2].split(',')
            
            base_savedir = os.path.join(RESULTS_DIR, cur_set, file_name, "vanilla")
            tf3 = os.path.join(base_savedir, "text3")
            tf4 = os.path.join(base_savedir, "text4")
            
            needs_3 = not check_images_exist(tf3, 5)
            needs_4 = not check_images_exist(tf4, 5)
            
            if not (needs_3 or needs_4):
                continue
                
            if pipe is None:
                pipe = safe_load_pipeline(cfg['custom'])
                
            print(f" Completing baseline: {cur_set} -> {file_name}")
            
            if needs_3 and len(p_list) > 2:
                os.makedirs(tf3, exist_ok=True)
                for gen_id in range(1, 6):
                    out_path = os.path.join(tf3, f"result{gen_id}.png")
                    if os.path.exists(out_path): continue
                    timer = start_image_timer()
                    res = pipe(
                        guidance_scale=7.5, num_inference_steps=cfg['steps'],
                        eval_prompt=p_list[2].strip().replace("'", ""),
                        generator=make_generator(set_index, i, gen_id, 2),
                    )
                    res.images[0].save(out_path)
                    record_generation_perf(method, timer)
                    
            if needs_4 and len(p_list) > 3:
                os.makedirs(tf4, exist_ok=True)
                for gen_id in range(1, 6):
                    out_path = os.path.join(tf4, f"result{gen_id}.png")
                    if os.path.exists(out_path): continue
                    timer = start_image_timer()
                    res = pipe(
                        guidance_scale=7.5, num_inference_steps=cfg['steps'],
                        eval_prompt=p_list[3].strip().replace("'", ""),
                        generator=make_generator(set_index, i, gen_id, 3),
                    )
                    res.images[0].save(out_path)
                    record_generation_perf(method, timer)
                    
    if pipe is not None:
        clear_vram(pipe)

def stage3_evaluate_results():
    print(f"\n========== Stage 3: evaluate results and generate the final table ==========")
    import subprocess

    result = subprocess.run(["python", "-u", "reproduce_table1.py"])
    if result.returncode == 0:
        print(" Table 1 evaluation complete. See table1_reproduced.md")
    else:
        print(" Evaluation failed.")

if __name__ == "__main__":
    print(" Unified workflow started.")
    if ENABLE_STAGE_1:
        stage1_generate_mixed()
    if ENABLE_STAGE_2:
        stage2_generate_baselines()
    if ENABLE_STAGE_3:
        stage3_evaluate_results()
    
    write_generation_perf_log()
    print("\n All configured workflows are complete.")
