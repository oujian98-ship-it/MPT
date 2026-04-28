"""
eval_per_set.py  (Dual-Track Evaluation Edition)
----------------------------------------------------
For each set in {set1, set2, set3, set4}, we compute two sets of metrics:

[Official Compat Metrics] (To align with original Black-Scholes repo)
  - BLIP⊙DINO (Official): Uses compositional prompts (p0, p1), double-softmax BLIP, separate DINO processing.

[Corrected Metrics] (More rigorous for new methods)
  - CLIP-combined: CLIP(图像, 所有prompt组合成一个总prompt)
  - CLIP-add: 平均 CLIP(图像, 每个单独概念prompt)
  - BLIP-atomic: Uses atomic prompts (p2, p3), single-softmax.
  - Set-Level KID: KernelInceptionDistance computed globally over all images in the set.

Final result = average of per-set means.
Results printed as a table and appended to logs/timestamp.txt in UTF-16LE.
"""

import os, sys, glob, json, datetime, time
import patch_torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.stdout.reconfigure(encoding="utf-8")

import math
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SETS          = ["set1", "set2","set3","set4"]
METHODS       = ["lininterp", "alternating_sampling", "clip_min", "step", "bs", "mpt"]
METHOD_DISPLAY= {
    "lininterp":            "Linear Int.",
    "alternating_sampling": "Alt. Samp.",
    "clip_min":             "CLIP Min.",
    "step":                 "Step.",
    "bs":                   "Black Scholes",
    "mpt":                  "Portfolio Diff.",
}

# 运行时自动测量: 耗时(s) / GPU时(h) / 峰值显存(GB)
method_perf = {m: {"time_s": 0.0, "gpu_hrs": 0.0, "mem_gb": 0.0} for m in METHODS}

BLIP_PATH = r"d:\projects\BlackScholesDiffusion2024-main\Model\BLIP-2"
DINO_PATH = r"d:\projects\BlackScholesDiffusion2024-main\Model\DINOv2"
CLIP_PATH = r"d:\projects\BlackScholesDiffusion2024-main\Model\CLIP"
LOG_DIR  = "logs"
LOG_FILE  = os.path.join(LOG_DIR, f"eval_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# ── Helpers ──────────────────────────────────────────────────────────────

def load_prompts(set_name):
    path = f"data/{set_name}.txt"
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split("\t")
        raw = parts[1].strip().lstrip("[").rstrip("]")
        p_list = [p.strip() for p in raw.split(",")]
        out.append({"file_name": parts[0], "list": p_list})
    return out


def get_vanilla_imgs(pid, set_name):
    base = f"results/{set_name}/{pid}/vanilla"
    t3 = glob.glob(os.path.join(base, "text3", "*.png"))
    t4 = glob.glob(os.path.join(base, "text4", "*.png"))
    if t3 and t4:
        return {"t3": t3, "t4": t4, "all": t3 + t4}
    t1 = glob.glob(os.path.join(base, "text1", "*.png"))
    t2 = glob.glob(os.path.join(base, "text2", "*.png"))
    return {"t3": t1, "t4": t2, "all": t1 + t2}


def get_method_imgs(pid, set_name, method):
    return glob.glob(os.path.join(f"results/{set_name}/{pid}/{method}", "*.png"))


def get_peak_gpu_mem_gb():
    """返回当前进程峰值GPU显存占用(GB)"""
    if DEVICE != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 3)


# ── Load models once ────────────────────────────────────────────────────

print("Loading CLIP model...")
from transformers import CLIPModel, CLIPProcessor
clip_model    = CLIPModel.from_pretrained(CLIP_PATH).to(DEVICE)
clip_proc     = CLIPProcessor.from_pretrained(CLIP_PATH)

print("Loading BLIP-2 model...")
from transformers import AutoProcessor, Blip2ForImageTextRetrieval
blip_model    = Blip2ForImageTextRetrieval.from_pretrained(
    BLIP_PATH, torch_dtype=torch.float16).to(DEVICE)
blip_proc     = AutoProcessor.from_pretrained(BLIP_PATH)

print("Loading DINOv2 model...")
from transformers import AutoImageProcessor, Dinov2Model
dino_proc     = AutoImageProcessor.from_pretrained(DINO_PATH)
dino_model    = Dinov2Model.from_pretrained(DINO_PATH).to(DEVICE)

from torchmetrics.image.kid import KernelInceptionDistance

print("All models loaded.\n")

# ── Metric Helpers ────────────────────────────────────────────────────────

def extract_dino_features(img_paths):
    feats = []
    for path in img_paths:
        img = Image.open(path).convert("RGB")
        inp = dino_proc(img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = dino_model(**inp)
        feats.append(out.last_hidden_state.squeeze(0).view(-1))
    if not feats:
        return None
    return torch.stack(feats, dim=0).mean(dim=0)

def blip_score_official(img, text):
    # Official Double-Softmax
    inp = blip_proc(images=img, text=text, return_tensors="pt").to(DEVICE, torch.float16)
    with torch.no_grad():
        out = blip_model(**inp, use_image_text_matching_head=True)
    probs1 = F.softmax(out.logits_per_image, dim=1)
    probs2 = F.softmax(probs1, dim=1)
    return probs2[0, 1].item()

def blip_score_atomic(img, text):
    # Standard Single-Softmax
    inp = blip_proc(images=img, text=text, return_tensors="pt").to(DEVICE, torch.float16)
    with torch.no_grad():
        out = blip_model(**inp, use_image_text_matching_head=True)
    probs = F.softmax(out.logits_per_image, dim=1)
    return probs[0, 1].item()

# ── Per-set evaluation ────────────────────────────────────────────────────

set_results = {}

for set_name in SETS:
    prompts = load_prompts(set_name)
    if not prompts:
        print(f"[SKIP] {set_name}: no prompts found")
        continue

    print(f"\n{'='*60}")
    print(f"Evaluating {set_name} ({len(prompts)} prompts)...")
    print(f"{'='*60}")

    acc = {m: {"clip_comp": [], "clip_add": [], "blip_dino_official": [], "blip_atomic": []} for m in METHODS}
    
    # Set-level KID objects: subset_size 50 ensures stable kernel calculation over the entire set
    set_kid_objs = {m: KernelInceptionDistance(subset_size=50).to(DEVICE) for m in METHODS}
    
    # 重置每set的计时
    set_method_time = {m: 0.0 for m in METHODS}
    set_method_mem  = {m: 0.0 for m in METHODS}

    for item in tqdm(prompts, desc=set_name):
        pid    = item["file_name"]
        p_list = item["list"]
        if len(p_list) < 4:
            continue

        vanilla_dict = get_vanilla_imgs(pid, set_name)
        if not vanilla_dict["all"]:
            continue

        # Extract DINO features separately for t3 and t4
        v_feat_t3 = extract_dino_features(vanilla_dict["t3"])
        v_feat_t4 = extract_dino_features(vanilla_dict["t4"])
        if v_feat_t3 is None or v_feat_t4 is None:
            continue

        for m in METHODS:
            imgs = get_method_imgs(pid, set_name, m)
            if not imgs:
                continue

            # ═══ 计时开始 ═══
            torch.cuda.reset_peak_memory_stats(device=DEVICE)
            t_start = time.perf_counter()

            clip_comp_scores, clip_add_scores = [], []
            blip_official_scores, blip_atomic_scores = [], []

            # ── Combined prompt text (for CLIP-combined) ──
            all_prompts_combined = " ".join(p_list)
            
            # --- DINO separate aggregation ---
            m_feat_avg = extract_dino_features(imgs)
            dino_sim_3 = F.cosine_similarity(v_feat_t3.unsqueeze(0), m_feat_avg.unsqueeze(0), dim=1).item()
            dino_sim_4 = F.cosine_similarity(v_feat_t4.unsqueeze(0), m_feat_avg.unsqueeze(0), dim=1).item()
            dino_avg = 0.5 * (dino_sim_3 + dino_sim_4)

            for img_p in imgs:
                img = Image.open(img_p).convert("RGB")

                # ═══ 1) CLIP-combined (↑) ═══
                inp_c = clip_proc(text=[all_prompts_combined], images=img, return_tensors="pt", padding=True).to(DEVICE)
                out_c = clip_model(**inp_c)
                clip_comp_scores.append(out_c.logits_per_image.cpu().detach().numpy()[0][0] * 0.01)

                # ═══ 2) CLIP-add (↑) ═══
                individual_scores = []
                for pi in p_list:
                    inp_i = clip_proc(text=[pi], images=img, return_tensors="pt", padding=True).to(DEVICE)
                    out_i = clip_model(**inp_i)
                    individual_scores.append(out_i.logits_per_image.cpu().detach().numpy()[0][0])
                clip_add_scores.append(float(np.mean(individual_scores)) * 0.01)

                # ═══ 3) BLIP Official (↑) (Compositional p0, p1 with double-softmax) ═══
                b_off_0 = blip_score_official(img, p_list[0])
                b_off_1 = blip_score_official(img, p_list[1])
                blip_official_scores.append(0.5 * (b_off_0 + b_off_1))
                
                # ═══ 4) BLIP Atomic (↑) (Atomic p2, p3 with single-softmax) ═══
                b_at_2 = blip_score_atomic(img, p_list[2])
                b_at_3 = blip_score_atomic(img, p_list[3])
                blip_atomic_scores.append(0.5 * (b_at_2 + b_at_3))

                # ═══ 5) Update Set-Level KID ═══
                m_tensor = torch.from_numpy(np.asarray(img)).unsqueeze(0).permute(0, 3,1,2).to(DEVICE)
                set_kid_objs[m].update(m_tensor, real=False)
                
            # Add all vanilla images for this prompt to the set's real distribution
            for v_path in vanilla_dict["all"]:
                v_img = Image.open(v_path).convert("RGB")
                v_tensor = torch.from_numpy(np.asarray(v_img)).unsqueeze(0).permute(0, 3,1,2).to(DEVICE)
                set_kid_objs[m].update(v_tensor, real=True)

            acc[m]["clip_comp"].append(float(np.mean(clip_comp_scores)))
            acc[m]["clip_add"].append(float(np.mean(clip_add_scores)))
            
            # BLIP x DINO (Official)
            blip_official_avg = float(np.mean(blip_official_scores))
            acc[m]["blip_dino_official"].append(blip_official_avg * dino_avg)
            
            # BLIP Atomic (Corrected)
            acc[m]["blip_atomic"].append(float(np.mean(blip_atomic_scores)))

            # ═══ 计时结束 & 记录 ═══
            t_elapsed = time.perf_counter() - t_start
            set_method_time[m] += t_elapsed
            peak_mem = get_peak_gpu_mem_gb()
            if peak_mem > set_method_mem[m]:
                set_method_mem[m] = peak_mem

            torch.cuda.empty_cache()

    # Compute Set-Level KID
    set_kid_scores = {}
    for m in METHODS:
        try:
            k_val, _ = set_kid_objs[m].compute()
            set_kid_scores[m] = k_val.detach().cpu().item()
        except Exception as e:
            print(f"[WARN] Failed to compute Set-Level KID for {m} in {set_name}: {e}")
            set_kid_scores[m] = 0.0
        
        # Cleanup memory
        del set_kid_objs[m]
        torch.cuda.empty_cache()

    # 累积到全局 perf（跨set累加耗时和取最大显存）
    for m in METHODS:
        method_perf[m]["time_s"]  += set_method_time[m]
        method_perf[m]["gpu_hrs"]  += set_method_time[m] / 3600.0
        if set_method_mem[m] > method_perf[m]["mem_gb"]:
            method_perf[m]["mem_gb"] = set_method_mem[m]

    # Compute per-set means and save JSON
    set_means = {}
    for m in METHODS:
        entry = {
            "clip_comp":          float(np.mean(acc[m]["clip_comp"]))          if acc[m]["clip_comp"] else 0.0,
            "clip_add":           float(np.mean(acc[m]["clip_add"]))           if acc[m]["clip_add"]  else 0.0,
            "blip_dino_official": float(np.mean(acc[m]["blip_dino_official"])) if acc[m]["blip_dino_official"] else 0.0,
            "blip_atomic":        float(np.mean(acc[m]["blip_atomic"]))        if acc[m]["blip_atomic"] else 0.0,
            "kid_set":            set_kid_scores[m]
        }
        set_means[m] = entry
    set_results[set_name] = set_means

    print(f"\n--- {set_name} per-set results ---")
    print(f"{'Method':<22} {'BLIPxDINO(Off)':>16} {'BLIP-Atomic':>14} {'Set-KID':>10}")
    print("-" * 66)
    for m in METHODS:
        r = set_means[m]
        print(f"{METHOD_DISPLAY[m]:<22} {r['blip_dino_official']:>16.4f} {r['blip_atomic']:>14.4f} {r['kid_set']:>10.5f}")

    json_path = f"results_{set_name}_full.json"
    with open(json_path, "w") as f:
        json.dump(set_means, f, indent=4)


# ── Final average across sets ─────────────────────────────────────────────

print("\n" + "="*80)
print("FINAL TABLE: Average across all sets")
print("="*80)

final = {}
for m in METHODS:
    keys = ["clip_comp", "clip_add", "blip_dino_official", "blip_atomic", "kid_set"]
    vals = {k: [] for k in keys}
    for sn in SETS:
        if sn in set_results and set_results[sn][m]["clip_comp"] > 0:
            for k in keys:
                vals[k].append(set_results[sn][m][k])
    final[m] = {k: float(np.mean(v)) if v else 0.0 for k, v in vals.items()}

header = (
    f"{'Method':<22} | {'CLIP-comp':>9} | {'CLIP-add':>9} | "
    f"{'BD(Off)':>9} | {'BLIP-At':>9} | {'Set-KID':>9} | "
    f"{'Time(s)':>7} | {'Mem(GB)':>7}"
)
sep = "-" * len(header)
rows = [header, sep]
for m in METHODS:
    p = method_perf[m]
    r = final[m]
    rows.append(
        f"{METHOD_DISPLAY[m]:<22} | {r['clip_comp']:>9.4f} | {r['clip_add']:>9.4f} | "
        f"{r['blip_dino_official']:>9.4f} | {r['blip_atomic']:>9.4f} | {r['kid_set']:>9.5f} | "
        f"{p['time_s']:>7.1f} | {p['mem_gb']:>7.1f}"
    )
table_str = "\n".join(rows)

print("\n" + table_str)


ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
block = (
    f"\n\n{'='*80}\n"
    f"[Evaluation Run (Dual-Track)] {ts}\n"
    f"Official Metrics: BLIPxDINO(Off) uses double-softmax & separate DINO references.\n"
    f"Corrected Metrics: BLIP-At uses single-softmax atomic prompts. Set-KID computed globally per set.\n"
    f"\n{table_str}\n"
)

os.makedirs(LOG_DIR, exist_ok=True)
with open(LOG_FILE, "a", encoding="utf-16-le") as f:
    f.write(block)

print(f"\n[OK] Saved to {LOG_FILE}")
print(f"[OK] Done.")
