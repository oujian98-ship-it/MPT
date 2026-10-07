# MPT Concept Blending Experiments

This repository runs concept blending experiments with Stable Diffusion 2.1, including Black-Scholes prompt switching, comparison methods, and MPT.

## Dependencies

Use Python 3.9 or later. Install the required packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors huggingface_hub
pip install numpy scipy opencv-python pillow tqdm requests packaging
pip install torchmetrics torch-fidelity
```

## Model Downloads

| Model | Hugging Face repository | Local directory |
| --- | --- | --- |
| Stable Diffusion 2.1 base | `stabilityai/stable-diffusion-2-1-base` | `Model/Stable_Diffusion_2.1/` |
| CLIP | `openai/clip-vit-base-patch32` | `Model/CLIP/` |
| BLIP-2 ITM | `Salesforce/blip2-itm-vit-g` | `Model/BLIP-2/` |
| DINOv2 base | `facebook/dinov2-base` | `Model/DINOv2/` |

Create the directory with `mkdir Model` (PowerShell: `New-Item -ItemType Directory -Force Model`), then download:

```bash
huggingface-cli download stabilityai/stable-diffusion-2-1-base --local-dir Model/Stable_Diffusion_2.1
huggingface-cli download openai/clip-vit-base-patch32 --local-dir Model/CLIP
huggingface-cli download Salesforce/blip2-itm-vit-g --local-dir Model/BLIP-2
huggingface-cli download facebook/dinov2-base --local-dir Model/DINOv2
```

If authentication is required, run `huggingface-cli login` first. Alternatively, use Python:

```python
from huggingface_hub import snapshot_download

snapshot_download("stabilityai/stable-diffusion-2-1-base", local_dir="Model/Stable_Diffusion_2.1")
snapshot_download("openai/clip-vit-base-patch32", local_dir="Model/CLIP")
snapshot_download("Salesforce/blip2-itm-vit-g", local_dir="Model/BLIP-2")
snapshot_download("facebook/dinov2-base", local_dir="Model/DINOv2")
```

## Path Configuration

Model paths are relative to the project root. Run commands from the project root, for example:

```text
Model\Stable_Diffusion_2.1
Model\CLIP
Model\BLIP-2
Model\DINOv2
```

Place downloaded models in the `Model/` directory under the project root, or update the model paths to match your setup.

## Generate Experiment Images

Generate the baseline and comparison methods:

```bash
python -u run_all_unified.py
```

Methods include `vanilla`, `lininterp`, `alternating_sampling`, `clip_min`, `step`, and `bs`. The script also generates `vanilla/text3` and `vanilla/text4` as atomic-concept pseudo-references for DINO/KID evaluation.

Generate MPT images:

```bash
python -u run_batch_mpt.py
```

Default output layout:

```text
result/{set_name}/{prompt_id}/{method}/result1.png ... result5.png
```

For example: `result/set1/prompt1/mpt/result1.png`. Generation supports resuming: prompts with five existing images are skipped automatically.

Generation performance logs are saved to:

```text
logs/generation_perf_run_all_*.json
logs/generation_perf_mpt_*.json
```

`avg_time_s` measures the average time per newly generated image. `peak_mem_gb` records peak GPU memory during generation.

## Evaluation

```bash
python -u eval_per_set.py
```

Default methods:

```python
METHODS = ["lininterp", "alternating_sampling", "clip_min", "step", "bs", "mpt"]
```

To evaluate MPT only, set `METHODS = ["mpt"]` in `eval_per_set.py`.

Metrics include `CLIP-combined`, `CLIP-add`, `BLIP x DINO`, `BLIP-Atomic`, and `Set-KID`. Results are saved to:

```text
results_set1_full.json
results_set2_full.json
results_set3_full.json
results_set4_full.json
logs/eval_*.txt
```

## Repeated Experiments, Raw Results, and Error Bars

The repeated-experiment entry point offers two modes:

- `1`: **All Methods**, including all baselines and MPT.
- `2`: **Only MPT**, with automatic vanilla reference generation for DINO/KID evaluation.

```bash
python -u run_repeated_experiments.py
```

Enter a repetition count such as `3` or `5`. Each run uses a different reproducible seed. Images, generation logs, per-set results, and evaluation snapshots are retained separately:

```text
experiment_results/all_3runs_YYYYMMDD_HHMMSS/
  run_001/images/                 # All generated images from run 1
  run_001/evaluation/evaluation_summary.json
  run_001/logs/
  run_002/...
  repeated_summary.md             # Mean and error for each method
  repeated_summary.json           # Machine-readable summary with all raw run values
```

The default error measure is the standard error of the mean (SEM). Noninteractive examples:

```bash
# All methods, five repetitions, using SEM
python -u run_repeated_experiments.py --mode all --repeats 5

# MPT only, using SEM or a 95% confidence interval
python -u run_repeated_experiments.py --mode mpt --repeats 3 --error sem
python -u run_repeated_experiments.py --mode mpt --repeats 5 --error ci95
```

Regenerate the summary of an existing experiment directory:

```bash
python -u summarize_repeated_runs.py experiment_results/all_3runs_YYYYMMDD_HHMMSS --error sem
```

## Notes

KID follows the original evaluation protocol, using vanilla single-concept generated images as pseudo-references:

```text
reference = vanilla/text3 + vanilla/text4
fake      = images generated by the current method
```

This is vanilla-reference KID. Standard real-image KID instead uses a real-image dataset as the reference.

## Main Files

| File | Purpose |
| --- | --- |
| `run_all_unified.py` | Generate baseline and comparison images |
| `run_batch_mpt.py` | Generate MPT images |
| `eval_per_set.py` | Evaluate generated images |
| `clear_results.py` | Clear previous results before rerunning experiments |
