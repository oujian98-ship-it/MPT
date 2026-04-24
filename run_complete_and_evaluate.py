"""
run_complete_and_evaluate.py

Stage 1: Generate missing vanilla text3/text4 images for set4 prompts 85-90.
Stage 2: Re-run reproduce_table1.py evaluation (all sets, BLIP⊙DINO + KID).
Stage 3: Append timestamped results to reproduce_log.txt.
"""

import os
import sys
import glob
import json
import time
import datetime
import patch_torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SET_NAME = "set4"
BLIP_LOCAL_PATH = r"d:\projects\BlackScholesDiffusion2024-main\Model\BLIP-2"
DINO_LOCAL_PATH = r"d:\projects\BlackScholesDiffusion2024-main\Model\DINOv2"
SETS = ['set1', 'set2', 'set3', 'set4']
METHODS = ['lininterp', 'alternating_sampling', 'clip_min', 'bs']
METHOD_DISPLAY = {
    'lininterp': 'Linear Int.',
    'alternating_sampling': 'Alt. Samp.',
    'clip_min': 'CLIP Min.',
    'bs': 'Black Scholes',
}
CONSTANTS = {
    'lininterp':            {'Steps': 50,  'Time': 6.5, 'GPU': 0.001805, 'Mem': 7.1},
    'alternating_sampling': {'Steps': 100, 'Time': 14,  'GPU': 0.00389,  'Mem': 7.7},
    'clip_min':             {'Steps': 100, 'Time': 14,  'GPU': 0.00389,  'Mem': 7.7},
    'bs':                   {'Steps': 100, 'Time': 14,  'GPU': 0.00389,  'Mem': 7.7},
}
LOG_FILE = "reproduce_log.txt"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_prompts(set_name):
    path = f'data/{set_name}.txt'
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    prompts = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split('\t')
        file_name = parts[0]
        raw_list = parts[1].strip()
        if raw_list.startswith('['):
            raw_list = raw_list[1:]
        if raw_list.endswith(']'):
            raw_list = raw_list[:-1]
        p_list = [p.strip() for p in raw_list.split(',')]
        prompts.append({'file_name': file_name, 'list': p_list})
    return prompts


def get_vanilla_baseline_imgs(pid, set_name):
    base_path = f'results/{set_name}/{pid}/vanilla'
    t3 = glob.glob(os.path.join(base_path, 'text3', '*.png'))
    t4 = glob.glob(os.path.join(base_path, 'text4', '*.png'))
    if t3 and t4:
        return t3 + t4
    t1 = glob.glob(os.path.join(base_path, 'text1', '*.png'))
    t2 = glob.glob(os.path.join(base_path, 'text2', '*.png'))
    return t1 + t2


# ─────────────────────────────────────────────
# Stage 1: Generate missing vanilla text3/text4
# ─────────────────────────────────────────────

def check_missing_vanilla(set_name, prompts):
    missing = []
    for item in prompts:
        pid = item['file_name']
        base = f'results/{set_name}/{pid}/vanilla'
        t3 = glob.glob(os.path.join(base, 'text3', '*.png'))
        t4 = glob.glob(os.path.join(base, 'text4', '*.png'))
        if not t3 or not t4:
            missing.append(item)
    return missing


def generate_missing_vanilla(missing_prompts, set_name):
    if not missing_prompts:
        print("[OK] All vanilla text3/text4 images already exist. Skipping generation.")
        return

    print(f"\n[Stage 1] Generating missing vanilla text3/text4 for {len(missing_prompts)} prompts in {set_name}...")

    from diffusers import DiffusionPipeline, DDIMScheduler
    model_dir = r"d:\projects\BlackScholesDiffusion2024-main\Model\Stable_Diffusion_2.1"
    pipe = DiffusionPipeline.from_pretrained(
        model_dir,
        safety_checker=None,
        use_auth_token=False,
        custom_pipeline='./models/vanilla',
        scheduler=DDIMScheduler(
            beta_start=0.00085, beta_end=0.012,
            beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False
        )
    ).to(DEVICE)

    for item in tqdm(missing_prompts, desc="Generating missing vanilla"):
        pid = item['file_name']
        p_list = item['list']
        base = f'results/{set_name}/{pid}/vanilla'

        for text_idx, sub in [(2, 'text3'), (3, 'text4')]:
            savedir = os.path.join(base, sub)
            existing = glob.glob(os.path.join(savedir, '*.png'))
            if len(existing) >= 5:
                continue  # already done
            os.makedirs(savedir, exist_ok=True)
            eval_prompt = p_list[text_idx] if text_idx < len(p_list) else p_list[0]
            done_nums = {int(os.path.splitext(os.path.basename(f))[0].replace('result',''))
                         for f in existing}
            for n in range(1, 6):
                if n in done_nums:
                    continue
                res = pipe(guidance_scale=7.5, num_inference_steps=50, eval_prompt=eval_prompt)
                res.images[0].save(os.path.join(savedir, f'result{n}.png'))

    del pipe
    torch.cuda.empty_cache()
    print("[OK] Stage 1 complete.")


# ─────────────────────────────────────────────
# Stage 2: Evaluate (BLIP⊙DINO + KID + CLIP)
# ─────────────────────────────────────────────

def run_evaluation():
    print("\n[Stage 2] Running full evaluation (BLIP⊙DINO + KID)...")

    import torch.nn.functional as F
    from transformers import AutoImageProcessor, Dinov2Model, AutoProcessor, Blip2ForImageTextRetrieval
    from torchmetrics.image.kid import KernelInceptionDistance
    import warnings
    warnings.filterwarnings("ignore")

    blip_model = Blip2ForImageTextRetrieval.from_pretrained(
        BLIP_LOCAL_PATH, torch_dtype=torch.float16).to(DEVICE)
    blip_processor = AutoProcessor.from_pretrained(BLIP_LOCAL_PATH)
    dino_processor = AutoImageProcessor.from_pretrained(DINO_LOCAL_PATH)
    dino_model = Dinov2Model.from_pretrained(DINO_LOCAL_PATH).to(DEVICE)

    results = {m: {'dino_blip': [], 'kid': []} for m in METHODS}

    for set_name in SETS:
        prompts = load_prompts(set_name)
        if not prompts:
            continue

        for item in tqdm(prompts, desc=f"Evaluating {set_name}"):
            pid = item['file_name']
            vanilla_imgs = get_vanilla_baseline_imgs(pid, set_name)
            if not vanilla_imgs:
                continue

            v_feats, v_kid_imgs = [], []
            for v_path in vanilla_imgs:
                img = Image.open(v_path).convert("RGB")
                inputs = dino_processor(img, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    feat = dino_model(**inputs).last_hidden_state.mean(dim=1)
                v_feats.append(feat)
                img_t = torch.from_numpy(np.asarray(img)).unsqueeze(0).permute(0, 3, 1, 2)
                v_kid_imgs.append(img_t)

            v_feat_avg = torch.stack(v_feats).mean(dim=0)

            for m in METHODS:
                imgs = glob.glob(os.path.join(f'results/{set_name}/{pid}/{m}', '*.png'))
                if not imgs:
                    continue

                m_feats, m_kid_imgs, blip_scores = [], [], []
                for img_p in imgs:
                    img = Image.open(img_p).convert("RGB")
                    inputs = dino_processor(img, return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        feat = dino_model(**inputs).last_hidden_state.mean(dim=1)
                    m_feats.append(feat)
                    img_t = torch.from_numpy(np.asarray(img)).unsqueeze(0).permute(0, 3, 1, 2)
                    m_kid_imgs.append(img_t)

                    inp1 = blip_processor(images=img, text=item['list'][0], return_tensors="pt").to(DEVICE, torch.float16)
                    inp2 = blip_processor(images=img, text=item['list'][1], return_tensors="pt").to(DEVICE, torch.float16)
                    with torch.no_grad():
                        out1 = blip_model(**inp1, use_image_text_matching_head=True)
                        out2 = blip_model(**inp2, use_image_text_matching_head=True)
                    s1 = torch.nn.functional.softmax(out1.logits_per_image, dim=1)[0][1].item()
                    s2 = torch.nn.functional.softmax(out2.logits_per_image, dim=1)[0][1].item()
                    blip_scores.append(0.5 * (s1 + s2))

                if m_feats:
                    m_feat_avg = torch.stack(m_feats).mean(dim=0)
                    d_sim = F.cosine_similarity(
                        v_feat_avg.view(-1), m_feat_avg.view(-1), dim=0).item()
                    avg_blip = np.mean(blip_scores)
                    results[m]['dino_blip'].append(d_sim * avg_blip)

                    ss = min(5, len(v_kid_imgs), len(m_kid_imgs))
                    if ss >= 2:
                        kid = KernelInceptionDistance(subset_size=ss).to(DEVICE)
                        for v_img in v_kid_imgs:
                            kid.update(v_img.to(DEVICE), real=True)
                        for m_img in m_kid_imgs:
                            kid.update(m_img.to(DEVICE), real=False)
                        k_score, _ = kid.compute()
                        results[m]['kid'].append(k_score.detach().cpu().numpy().item())
                        del kid
                    else:
                        results[m]['kid'].append(0.0)
                    torch.cuda.empty_cache()

            del v_feats, v_kid_imgs
            torch.cuda.empty_cache()

    del blip_model, dino_model
    torch.cuda.empty_cache()
    print("[OK] Stage 2 complete.")
    return results


# ─────────────────────────────────────────────
# Stage 3: Write to log
# ─────────────────────────────────────────────

def build_table(results):
    clip_comp = {m: [] for m in METHODS}
    clip_add  = {m: [] for m in METHODS}
    for set_name in SETS:
        jpath = f'results_{set_name}.json'
        if os.path.exists(jpath):
            with open(jpath, 'r') as f:
                data = json.load(f)
            for m in METHODS:
                if m in data:
                    clip_comp[m].append(data[m]['clip_comp'])
                    clip_add[m].append(data[m].get('clip_indiv', data[m]['clip_comp']))

    header = ("Method | CLIP-combined (↑) | CLIP-add (↑) | BLIP ⊙ DINO (↑) "
              "| KID (↓) | Steps (↓) | Time (s) (↓) | GPU hrs | Memory (GB) (↓)")
    sep = "--- | --- | --- | --- | --- | --- | --- | --- | ---"
    rows = [header, sep]
    for m in METHODS:
        c_comp = np.mean(clip_comp[m]) if clip_comp[m] else 0.0
        c_add  = np.mean(clip_add[m])  if clip_add[m]  else 0.0
        bd     = np.mean(results[m]['dino_blip']) if results[m]['dino_blip'] else 0.0
        k      = np.mean(results[m]['kid'])        if results[m]['kid']        else 0.0
        c      = CONSTANTS[m]
        rows.append(
            f"{METHOD_DISPLAY[m]} | {c_comp:.4f} | {c_add:.4f} | {bd:.4f} "
            f"| {k:.5f} | {c['Steps']} | {c['Time']} | {c['GPU']} | {c['Mem']}"
        )
    return "\n".join(rows)


def append_to_log(table_str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "\n" + "=" * 70 + "\n"
    block = (
        f"{separator}"
        f"[Evaluation Run] {ts}\n"
        f"Sets: {SETS} | Vanilla baseline: text3/text4 (paper baseline)\n\n"
        f"{table_str}\n"
    )
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(block)
    print(f"\n[OK] Stage 3: Results appended to {LOG_FILE}")
    print(block)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()

    # Stage 1
    for sn in SETS:
        prompts = load_prompts(sn)
        if sn == SET_NAME:
            missing = check_missing_vanilla(sn, prompts)
            generate_missing_vanilla(missing, sn)

    # Stage 2
    eval_results = run_evaluation()

    # Stage 3
    table = build_table(eval_results)
    append_to_log(table)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")

