# MPT 概念融合实验

本仓库用于运行 Stable Diffusion 2.1 上的概念融合实验，包含原始
Black-Scholes prompt switching 方法、多个对比方法，以及本文的 MPT 方法。

## 环境依赖

推荐系统环境：

- Python 3.9+
必需 Python 包：

- `torch`
- `torchvision`
- `diffusers`
- `transformers`
- `accelerate`
- `safetensors`
- `huggingface_hub`
- `numpy`
- `scipy`
- `opencv-python`
- `Pillow`
- `tqdm`
- `requests`
- `packaging`
- `torchmetrics`
- `torch-fidelity`

安装命令：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate safetensors huggingface_hub
pip install numpy scipy opencv-python pillow tqdm requests packaging
pip install torchmetrics torch-fidelity
```



## 模型下载

需要准备以下本地模型：

| 模型 | Hugging Face 仓库 | 放置目录 |
| --- | --- | --- |
| Stable Diffusion 2.1 base | `stabilityai/stable-diffusion-2-1-base` | `Model/Stable_Diffusion_2.1/` |
| CLIP | `openai/clip-vit-base-patch32` | `Model/CLIP/` |
| BLIP-2 ITM | `Salesforce/blip2-itm-vit-g` | `Model/BLIP-2/` |
| DINOv2 base | `facebook/dinov2-base` | `Model/DINOv2/` |

先创建模型目录：

```bash
mkdir Model
```

使用 `huggingface-cli` 下载：

```bash
huggingface-cli download stabilityai/stable-diffusion-2-1-base --local-dir Model/Stable_Diffusion_2.1
huggingface-cli download openai/clip-vit-base-patch32 --local-dir Model/CLIP
huggingface-cli download Salesforce/blip2-itm-vit-g --local-dir Model/BLIP-2
huggingface-cli download facebook/dinov2-base --local-dir Model/DINOv2
```

Windows PowerShell 写法：

```powershell
New-Item -ItemType Directory -Force Model
huggingface-cli download stabilityai/stable-diffusion-2-1-base --local-dir Model/Stable_Diffusion_2.1
huggingface-cli download openai/clip-vit-base-patch32 --local-dir Model/CLIP
huggingface-cli download Salesforce/blip2-itm-vit-g --local-dir Model/BLIP-2
huggingface-cli download facebook/dinov2-base --local-dir Model/DINOv2
```

如果 Stable Diffusion 2.1 下载需要权限，先登录 Hugging Face：

```bash
huggingface-cli login
```

也可以用 Python 下载：

```python
from huggingface_hub import snapshot_download

snapshot_download("stabilityai/stable-diffusion-2-1-base", local_dir="Model/Stable_Diffusion_2.1")
snapshot_download("openai/clip-vit-base-patch32", local_dir="Model/CLIP")
snapshot_download("Salesforce/blip2-itm-vit-g", local_dir="Model/BLIP-2")
snapshot_download("facebook/dinov2-base", local_dir="Model/DINOv2")
```

## 路径设置

当前代码里使用的是本机绝对路径，例如：

```python
d:\projects\BlackScholesDiffusion2024-main\Model\Stable_Diffusion_2.1
d:\projects\BlackScholesDiffusion2024-main\Model\CLIP
d:\projects\BlackScholesDiffusion2024-main\Model\BLIP-2
d:\projects\BlackScholesDiffusion2024-main\Model\DINOv2
```

如果你的仓库路径不同，请修改这些文件里的模型路径：

- `run_all_unified.py`
- `run_batch_mpt.py`
- `eval_per_set.py`
- `run_batch_*.py`

## 正式实验生成图像

先生成 baseline 和对比方法：

```bash
python -u run_all_unified.py
```

该脚本会生成：

- `vanilla`
- `lininterp`
- `alternating_sampling`
- `clip_min`
- `step`
- `bs`

同时会生成 `vanilla/text3` 和 `vanilla/text4`，它们是 DINO/KID 评估时使用的
原子概念 pseudo-reference 图像。

然后生成 MPT 结果：

```bash
python -u run_batch_mpt.py
```

结果默认保存到：

```text
result/{set_name}/{prompt_id}/{method}/result1.png ... result5.png
```

例如：

```text
result/set1/prompt1/mpt/result1.png
```

生成脚本支持断点续跑：如果某个 prompt 已经有 5 张图，会自动跳过。

生成阶段的效率日志会保存到：

```text
logs/generation_perf_run_all_*.json
logs/generation_perf_mpt_*.json
```

其中 `avg_time_s` 是本次新生成图片的平均单图耗时，`peak_mem_gb` 是生成阶段记录到的峰值显存。

## 评估

运行：

```bash
python -u eval_per_set.py
```

默认评估全部方法：

```python
METHODS = ["lininterp", "alternating_sampling", "clip_min", "step", "bs", "mpt"]
```

如果只想评估 MPT，可以把 `eval_per_set.py` 中的 `METHODS` 改成：

```python
METHODS = ["mpt"]
```

评估指标包括：

- `CLIP-combined`
- `CLIP-add`
- `BLIP x DINO`
- `BLIP-Atomic`
- `Set-KID`

评估结果会保存到：

```text
results_set1_full.json
results_set2_full.json
results_set3_full.json
results_set4_full.json
logs/eval_*.txt
```

## 重复实验、保存原始结果与误差条

使用统一入口即可在运行时选择：

- `1`：运行 **All Methods**（全部 baseline 和 MPT）
- `2`：运行 **Only MPT**（只统计 MPT；脚本会自动生成 vanilla 参考图，以保证 DINO/KID 可评估）

```bash
python -u run_repeated_experiments.py
```

随后输入重复次数，例如 `3` 或 `5`。每一轮使用不同且可复现的随机种子，所有图片、生成日志、逐 set 结果和最终评估快照都会保留在独立目录：

```text
experiment_results/all_3runs_YYYYMMDD_HHMMSS/
  run_001/images/                 # 第 1 轮全部生成图片
  run_001/evaluation/evaluation_summary.json
  run_001/logs/
  run_002/...
  repeated_summary.md             # 各方法 Mean ± error
  repeated_summary.json           # 机器可读汇总（含所有轮次原始数值）
```

默认误差是标准误（`mean ± sem`）。无需交互时可直接指定：

```bash
# 所有方法，重复 5 次（默认 mean ± sem）
python -u run_repeated_experiments.py --mode all --repeats 5

# 仅 MPT，重复 3 次；可指定误差为 SEM 或 95% CI
python -u run_repeated_experiments.py --mode mpt --repeats 3 --error sem
python -u run_repeated_experiments.py --mode mpt --repeats 5 --error ci95
```

如需仅重新汇总一个已经完成的实验目录：

```bash
python -u summarize_repeated_runs.py experiment_results/all_3runs_YYYYMMDD_HHMMSS --error sem
```

## 说明

本仓库中的 KID 沿用原始代码的评估方式，使用 vanilla 单概念生成图作为
pseudo-reference：

```text
reference = vanilla/text3 + vanilla/text4
fake      = 当前方法生成图
```

因此这里的 KID 更准确地说是 vanilla-reference KID，不是基于真实图像数据集的
standard real-image KID。

## 主要文件

| 文件 | 作用 |
| --- | --- |
| `run_all_unified.py` | 生成 baseline 和对比方法 |
| `run_batch_mpt.py` | 生成 MPT 图像 |
| `eval_per_set.py` | 评估生成图像 |
| `clear_results.py` | 清理生成结果 |
