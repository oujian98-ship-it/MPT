"""
run_batch_mpt.py
-----------------
Portfolio Diffusion (MPT) 批量推理脚本 — 自动生成 set1~set4 全部图像

用法:
  python -u run_batch_mpt.py

输出:
  results/{set1~set4}/{file_name}/mpt/result1~5.png

与 run_batch_bs.py 的区别:
  - 使用 custom_pipeline='./models/mpt' (Portfolio Diffusion)
  - 每个 prompt 生成 5 张图（与 BS 一致）
  - 结果保存在 mpt/ 子目录下，便于 eval_per_set.py 识别
"""
import os
import patch_torch
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from PIL import Image
import torch
from diffusers import DiffusionPipeline, DDIMScheduler

has_cuda = torch.cuda.is_available()
device = torch.device("cuda" if has_cuda else "cpu")

# ── Configuration ──────────────────────────────────────────────
ALL_SETS = ["set1", "set2", "set3", "set4"]

# Local model paths
model_dir = r"d:\projects\BlackScholesDiffusion2024-main\Model\Stable_Diffusion_2.1"

# MPT-specific configuration (can override defaults in pipeline.py)
MPT_CONFIG = {
    # Return coefficients — 调大 alpha_bs 以更接近原始BS行为
    "alpha_lookahead": 0.2,
    "alpha_deficiency": 0.3,
    "alpha_history": 0.1,
    "alpha_bs": 0.4,           # BS先验权重更高
    # Risk coefficients
    "beta_attn": 0.5,
    "beta_upd": 0.5,
    "beta_inst": 0.3,
    # Regularization
    "lambda_r": 0.3,           # 降低风险厌恶，允许更多探索
    "gamma_bal": 0.1,
    "tau_ent": 0.1,
    # Time smoothing (FIX: use eta_max + eta_power, PDF formula η_t = η_max*(step/T)^p)
    "eta_max": 0.05,           # was eta_base * eta_growth
    "eta_power": 1.0,
    # Solver
    "rho": 0.3,                # 较小的学习率使权重变化平滑
    "M_inner": 5,              # 减少内循环次数以加速
}

# ── Load Pipeline (once, shared across all sets) ────────────────
print("=" * 70)
print("Loading Portfolio Diffusion Pipeline (MPT)...")
print("=" * 70)

pipe = DiffusionPipeline.from_pretrained(
    model_dir,
    safety_checker=None,
    use_auth_token=False,
    custom_pipeline="./models/mpt",
    scheduler=DDIMScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False
    ),
).to(device)

# P0 FIX: generator is now created per-image inside the loop (see below)
# to guarantee per-image reproducible seeds for fair comparison with BS / baseline.


def is_prompt_complete(set_name, file_name, num_images=5):
    """Check if a prompt already has all images generated."""
    savedir = f"./results/{set_name}/{file_name}/mpt/"
    for idx in range(1, num_images + 1):
        if not os.path.exists(f"{savedir}result{idx}.png"):
            return False
    return True


total_start = time.time()
total_done = 0
total_skipped = 0

# ── Main Loop: iterate ALL sets ─────────────────────────────────
for SET_NAME in ALL_SETS:

    with open(f"data/{SET_NAME}.txt", "r") as f:
        prompts_list = f.readlines()

    # Count how many are already done before starting this set
    pre_done = sum(1 for line in prompts_list if is_prompt_complete(SET_NAME, line.split("\t")[0]))

    print(f"\n{'='*70}")
    print(f"[{ALL_SETS.index(SET_NAME)+1}/{len(ALL_SETS)}] {SET_NAME} ({len(prompts_list)} prompts, {pre_done} already done)")
    print(f"{'='*70}")

    set_start = time.time()
    set_done = 0
    set_skipped = 0

    for i in range(len(prompts_list)):
        prompts_raw = prompts_list[i]
        prompts_parts = prompts_raw.split("\t")

        file_name = prompts_parts[0]
        prompt_full = prompts_parts[1]
        # Extract concept list: format is "{main_prompt},{concept1},{concept2},...}"
        prompt_inner = prompt_full[1:-2]  # strip surrounding quotes
        concept_list = prompt_inner.split(",")

        savedir = f"./results/{SET_NAME}/{file_name}/mpt/"
        os.makedirs(savedir, exist_ok=True)

        # ── Skip if already complete (断点续跑) ──
        if is_prompt_complete(SET_NAME, file_name):
            print(f"  [{i+1}/{len(prompts_list)}] {file_name}  SKIP (already done)")
            set_skipped += 1
            total_skipped += 1
            continue

        eval_prompt = concept_list[:4]  # [main, sub1, sub2]

        print(f"\n  [{i+1}/{len(prompts_list)}] {file_name}")
        print(f"    Concepts: {concept_list[1:4]}")

        for img_idx in range(1, 6):
            # P0 FIX: per-image reproducible seed — prompt_idx * 1000 + img_idx
            # ensures results are deterministic and comparable across methods.
            seed = i * 1000 + img_idx
            generator = torch.Generator(device.type).manual_seed(seed)
            res = pipe(
                guidance_scale=7.5,
                num_inference_steps=100,
                eval_prompt=eval_prompt,
                mpt_config=MPT_CONFIG,
                generator=generator,
            )
            image = res.images[0]
            image.save(savedir + f"/result{img_idx}.png")
        
        set_done += 1
        total_done += 1
        
        # Free memory between prompts
        torch.cuda.empty_cache()

    set_elapsed = time.time() - set_start
    print(f"\n  {SET_NAME}: {set_done} new + {set_skipped} skipped, {set_elapsed:.1f}s")

total_elapsed = time.time() - total_start
print(f"\n{'='*70}")
print(f"ALL SETS COMPLETE! New: {total_done}, Skipped: {total_skipped}, Total time: {total_elapsed/60:.1f} min")
print(f"Results: results/set{{1~4}}/*/mpt/")
print(f"Now run: python -u eval_per_set.py")
print("=" * 70)
