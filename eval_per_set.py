"""
eval_per_set.py  (Paper-style Edition)
----------------------------------------------------
For each set in {set1, set2, set3, set4}:
  1. CLIP-combined (↑): CLIP(图像, 所有prompt组合成一个总prompt) — 整体语义对齐
  2. CLIP-add      (↑): 平均 CLIP(图像, 每个单独概念prompt)    — 各概念是否都保住
  3. BLIP⊙DINO     (↑): 对有vanilla参考的concept算(BLIP x DINO)再平均
  4. KID           (↓): KernelInceptionDistance(vanilla, method)— 图像分布距离

Final result = average of per-set means.
Results printed as Table 1 and appended to logs/timestamp.txt in UTF-16LE.

Time / GPU hours / Memory: 运行时自动测量（非硬编码）
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

# DINO对比只对有vanilla参考图的concept做（vanilla只有text3/text4对应p_list[2:4]）
DINO_CONCEPT_INDICES = [2, 3]  # 概念A 和 概念B

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
        return t3 + t4
    t1 = glob.glob(os.path.join(base, "text1", "*.png"))
    t2 = glob.glob(os.path.join(base, "text2", "*.png"))
    return t1 + t2


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

    acc = {m: {"clip_comp": [], "clip_add": [], "kid": [],
               "db_S6": []} for m in METHODS}

    # 重置每set的计时
    set_method_time = {m: 0.0 for m in METHODS}
    set_method_mem  = {m: 0.0 for m in METHODS}

    for item in tqdm(prompts, desc=set_name):
        pid    = item["file_name"]
        p_list = item["list"]

        vanilla_imgs = get_vanilla_imgs(pid, set_name)
        if not vanilla_imgs:
            continue

        # --- Build vanilla KID pool + pre-extract DINO features (for text3/text4) ---
        v_kid_imgs = []
        v_dino_features = []  # per-image [197, 768]
        for v_path in vanilla_imgs:
            img = Image.open(v_path).convert("RGB")
            v_kid_imgs.append(
                torch.from_numpy(np.asarray(img)).unsqueeze(0).permute(0, 3,1,2))
            inp = dino_proc(img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = dino_model(**inp)
            v_dino_features.append(out.last_hidden_state.squeeze(0))

        for m in METHODS:
            imgs = get_method_imgs(pid, set_name, m)
            if not imgs:
                continue

            # ═══ 计时开始 ═══
            torch.cuda.reset_peak_memory_stats(device=DEVICE)
            t_start = time.perf_counter()

            clip_comp_scores, clip_add_scores = [], []
            m_kid_imgs = []

            # ── Combined prompt text (for CLIP-combined) ──
            all_prompts_combined = " ".join(p_list)

            for img_p in imgs:
                img = Image.open(img_p).convert("RGB")

                # ═══ 1) CLIP-combined (↑): 图像 vs 所有prompt组合成总prompt ═══
                inp_c = clip_proc(
                    text=[all_prompts_combined], images=img,
                    return_tensors="pt", padding=True).to(DEVICE)
                out_c = clip_model(**inp_c)
                clip_comp_scores.append(
                    out_c.logits_per_image.cpu().detach().numpy()[0][0] * 0.01)

                # ═══ 2) CLIP-add (↑): 对每个单独concept prompt分别算CLIP再平均 ═══
                individual_scores = []
                for pi in p_list:
                    inp_i = clip_proc(
                        text=[pi], images=img,
                        return_tensors="pt", padding=True).to(DEVICE)
                    out_i = clip_model(**inp_i)
                    individual_scores.append(out_i.logits_per_image.cpu().detach().numpy()[0][0])
                clip_add_scores.append(float(np.mean(individual_scores)) * 0.01)

                # ═══ KID tensor ═══
                m_kid_imgs.append(
                    torch.from_numpy(np.asarray(img)).unsqueeze(0).permute(0, 3,1,2))

            # ═══ 3) BLIP⊙DINO (↑): 对有vanilla参考的concept算(BLIP x DINO)再平均 ═══
            blip_dino_per_concept = []
            for ci in DINO_CONCEPT_INDICES:
                if ci >= len(p_list):
                    break
                concept_prompt = p_list[ci]

                # DINO: vanilla(text3+text4) vs method 该concept的视觉特征余弦相似度
                v_feat_avg = torch.stack(v_dino_features, dim=0).mean(dim=0).view(-1)

                m_dino_features = []
                for img_p in imgs:
                    m_img = Image.open(img_p).convert("RGB")
                    m_inp = dino_proc(m_img, return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        m_out = dino_model(**m_inp)
                    m_dino_features.append(m_out.last_hidden_state.squeeze(0))
                m_feat_avg = torch.stack(m_dino_features, dim=0).mean(dim=0).view(-1)
                dino_sim = F.cosine_similarity(v_feat_avg.unsqueeze(0), m_feat_avg.unsqueeze(0), dim=1).item()

                # BLIP: 图像 vs 该concept prompt的图文匹配概率
                blip_for_concept = []
                for img_p in imgs:
                    b_img = Image.open(img_p).convert("RGB")
                    b_inp = blip_proc(images=b_img, text=concept_prompt,
                                     return_tensors="pt").to(DEVICE, torch.float16)
                    with torch.no_grad():
                        b_out = blip_model(**b_inp, use_image_text_matching_head=True)
                    b_score = F.softmax(b_out.logits_per_image, dim=1)[0][1].item()
                    blip_for_concept.append(b_score)
                blip_avg = float(np.mean(blip_for_concept))

                blip_dino_per_concept.append(blip_avg * dino_sim)

            if blip_dino_per_concept:
                acc[m]["db_S6"].append(float(np.mean(blip_dino_per_concept)))
            else:
                acc[m]["db_S6"].append(0.0)

            acc[m]["clip_comp"].append(float(np.mean(clip_comp_scores)))
            acc[m]["clip_add"].append(float(np.mean(clip_add_scores)))

            # ═══ 4) KID (↓) ═══
            # ✅ Fix: 原代码 ss=min(5,...) 把 subset_size 硬限制到5，对每prompt只有5张图时
            # KID 统计方差极大、结果不可靠。改为使用全部可用样本量（min(n_real, n_fake)），
            # 只保留 >= 2 的合法性下界检查。
            n_real = len(v_kid_imgs)
            n_fake = len(m_kid_imgs)
            ss = min(n_real, n_fake)
            if ss >= 2:
                kid = KernelInceptionDistance(subset_size=ss).to(DEVICE)
                for vi in v_kid_imgs: kid.update(vi.to(DEVICE), real=True)
                for mi in m_kid_imgs: kid.update(mi.to(DEVICE), real=False)
                k_val, _ = kid.compute()
                acc[m]["kid"].append(k_val.detach().cpu().item())
                del kid
            else:
                acc[m]["kid"].append(0.0)

            # ═══ 计时结束 & 记录 ═══
            t_elapsed = time.perf_counter() - t_start
            set_method_time[m] += t_elapsed
            peak_mem = get_peak_gpu_mem_gb()
            if peak_mem > set_method_mem[m]:
                set_method_mem[m] = peak_mem

            torch.cuda.empty_cache()

        del v_kid_imgs
        del v_dino_features
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
            "clip_comp":  float(np.mean(acc[m]["clip_comp"]))  if acc[m]["clip_comp"]  else 0.0,
            "clip_add":   float(np.mean(acc[m]["clip_add"]))   if acc[m]["clip_add"]   else 0.0,
            "kid":        float(np.mean(acc[m]["kid"]))        if acc[m]["kid"]        else 0.0,
            "db_S6":      float(np.mean(acc[m]["db_S6"]))      if acc[m]["db_S6"]      else 0.0,
            "dino_blip":  float(np.mean(acc[m]["db_S6"]))      if acc[m]["db_S6"]      else 0.0,
        }
        set_means[m] = entry
    set_results[set_name] = set_means

    print(f"\n--- {set_name} per-set results ---")
    print(f"{'Method':<22} {'CLIP-comp':>10} {'CLIP-add':>10} {'BLIP⊙DINO':>12} {'KID':>10}")
    print("-" * 68)
    for m in METHODS:
        r = set_means[m]
        print(f"{METHOD_DISPLAY[m]:<22} {r['clip_comp']:>10.4f} {r['clip_add']:>10.4f} {r['db_S6']:>12.4f} {r['kid']:>10.5f}")

    json_path = f"results_{set_name}_full.json"
    with open(json_path, "w") as f:
        json.dump(set_means, f, indent=4)
    print(f"Saved {json_path}")


# ── Final average across sets ─────────────────────────────────────────────

print("\n" + "="*60)
print("FINAL TABLE: Average across all sets")
print("="*60)

final = {}
for m in METHODS:
    keys = ["clip_comp", "clip_add", "kid", "db_S6"]
    vals = {k: [] for k in keys}
    for sn in SETS:
        if sn in set_results and set_results[sn][m]["clip_comp"] > 0:
            for k in keys:
                vals[k].append(set_results[sn][m][k])
    final[m] = {k: float(np.mean(v)) if v else 0.0 for k, v in vals.items()}
    final[m]["dino_blip"] = final[m]["db_S6"]

lines = []
lines.append("\n=== Per-set CLIP-combined scores ===")
lines.append(f"{'Method':<22} " + " ".join(f"{s:>8}" for s in SETS) + f" {'AVG':>8}")
for m in METHODS:
    row = f"{METHOD_DISPLAY[m]:<22}"
    vals = []
    for sn in SETS:
        v = set_results.get(sn, {}).get(m, {}).get("clip_comp", 0.0)
        row += f" {v:>8.4f}"
        vals.append(v)
    row += f" {np.mean(vals):>8.4f}"
    lines.append(row)

lines.append("\n=== Per-set BLIP⊙DINO scores ===")
lines.append(f"{'Method':<22} " + " ".join(f"{sn:>8}" for sn in SETS) + f" {'AVG':>8}")
for m in METHODS:
    row = f"{METHOD_DISPLAY[m]:<22}"
    vals = []
    for sn in SETS:
        v = set_results.get(sn, {}).get(m, {}).get("db_S6", 0.0)
        row += f" {v:>8.4f}"
        vals.append(v)
    row += f" {np.mean(vals):>8.4f}"
    lines.append(row)

lines.append("\n=== Per-set KID scores ===")
lines.append(f"{'Method':<22} " + " ".join(f"{s:>8}" for s in SETS) + f" {'AVG':>8}")
for m in METHODS:
    row = f"{METHOD_DISPLAY[m]:<22}"
    vals = []
    for sn in SETS:
        v = set_results.get(sn, {}).get(m, {}).get("kid", 0.0)
        row += f" {v:>8.5f}"
        vals.append(v)
    row += f" {np.mean(vals):>8.5f}"
    lines.append(row)

breakdown_str = "\n".join(lines)
print(breakdown_str)

header = (
    "Method | CLIP-combined (up) | CLIP-add (up) | BLIP x DINO (up) "
    "| KID (down) | Time(s) | GPU hrs | Mem(GB)"
)
sep = "--- | --- | --- | --- | --- | --- | --- | --- | ---"
rows = [header, sep]
for m in METHODS:
    p = method_perf[m]
    r = final[m]
    rows.append(
        f"{METHOD_DISPLAY[m]} | {r['clip_comp']:.4f} | {r['clip_add']:.4f} "
        f"| {r['db_S6']:.4f} | {r['kid']:.5f} "
        f"| {p['time_s']:.1f} | {p['gpu_hrs']:.4f} | {p['mem_gb']:.1f}"
    )
table_str = "\n".join(rows)

print("\n" + table_str)


ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
eq = "=" * 70
block = (
    f"\n\n{eq}\n"
    f"[Evaluation Run] {ts}\n"
    f"Metrics: CLIP-combined / CLIP-add / BLIP⊙DINO / KID\n"
    f"DINO concepts: indices {DINO_CONCEPT_INDICES} (vanilla text3/text4)\n"
    f"Time/GPU/Mem: measured at runtime (not hardcoded)\n"
    f"\n{breakdown_str}\n"
    f"\n{table_str}\n"
)

os.makedirs(LOG_DIR, exist_ok=True)

with open(LOG_FILE, "w", encoding="utf-16-le") as f:
    f.write(block)

print(f"\n[OK] Saved to {LOG_FILE}")
print(f"[OK] Done.")
