"""

Portfolio Diffusion Pipeline  (MPT — Multi-Prompt Trading)
-----------------------------------------------------------
Based on: "Portfolio Diffusion: Risk-Aware Prompt and LoRA Allocation
           for Compositional Generation"

Core idea: At each denoising step, solve an optimization problem on the
simplex to find weight vector w_t that maximizes risk-adjusted return:
    max_w  μᵀ·w  -  λ_r · wᵀ·Σ·w  - γ·Var(q(w)) - τ·H(w) - η·||w-w_{t-1}||²
subject to  Σw_i = 1,  w_i ≥ 0

Then inject weighted text embedding into UNet.

Key differences vs Black-Scholes pipeline:
  - BS: hard switch between prompts each step (min(bs1, bs2))
  - MPT: soft allocation via continuous weights on simplex
"""
import inspect
import warnings
from typing import List, Optional, Union, Dict, Callable
from contextlib import contextmanager

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import math
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer, CLIPModel, CLIPProcessor

from diffusers import DiffusionPipeline
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import deprecate, logging

if hasattr(PIL.Image, "Resampling"):
    _resample = PIL.Image.Resampling
else:
    _resample = PIL.Image

PIL_INTERPOLATION = {
    "linear": getattr(_resample, "BILINEAR", _resample.BILINEAR),
    "bilinear": getattr(_resample, "BILINEAR", _resample.BILINEAR),
    "bicubic": getattr(_resample, "BICUBIC", _resample.BICUBIC),
    "lanczos": getattr(_resample, "LANCZOS", _resample.LANCZOS),
    "nearest": getattr(_resample, "NEAREST", _resample.NEAREST),
}

logger = logging.get_logger(__name__)


# =====================================================================
# Helper functions (reused from BS pipeline)
# =====================================================================

def preprocess(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))
    image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def bs_score(spot, strike, rate, sigma, t):
    """Black-Scholes score.
    FIX: use the strike parameter (was hardcoded to 100); guard edge cases.
    S and K should both be on the ×100 scale (per PDF Page 4).
    """
    spot_val = float(spot[0, 0])
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.cpu().numpy()
    sigma_val = max(float(sigma), 1e-6)
    t_val = max(float(t), 1e-6)
    spot_val = max(spot_val, 1e-6)
    strike_val = max(float(strike), 1e-6)
    d1 = (np.log(spot_val / strike_val) + (rate + (sigma_val ** 2) / 2) * t_val) / (sigma_val * t_val ** 0.5)
    d2 = d1 - sigma_val * t_val ** 0.5
    bs_score_val = (np.exp(-0.5 * d1 * d1)) * spot_val - (
        np.exp(-0.5 * d2 * d2) * strike_val * np.exp(-rate * t_val)
    )
    return bs_score_val


# =====================================================================
# Portfolio Diffusion Configuration
# =====================================================================

DEFAULT_MPT_CONFIG = {
    # --- Return coefficients ---
    "alpha_lookahead": 0.3,   # α₁: 前瞻边际收益
    "alpha_deficiency": 0.3,   # α₂: 当前缺失度
    "alpha_history": 0.2,      # α₃: 历史 EMA 增益
    "alpha_bs": 0.2,           # α₄: BS 紧迫性先验
    # --- Risk coefficients ---
    "beta_attn": 1.0,          # β₁: 注意力重叠风险
    "beta_upd": 1.0,           # β₂: 更新冲突风险
    "beta_inst": 0.5,          # β₃: 不稳定性风险
    # --- Regularization ---
    "lambda_r": 0.5,           # λ_r: 风险厌恶系数（整体风险惩罚缩放）
    "gamma_bal": 0.1,          # γ: 平衡惩罚（概念间方差）
    "tau_ent": 0.05,           # τ: 熵正则（防坍缩）
    # --- Time smoothing ---
    "eta_max": 0.1,            # η_max: 时间平滑最大系数 (PDF: η_t = η_max*(step/T)^p)
    "eta_power": 1.0,          # p: 调度指数
    # --- Solver ---
    "rho": 0.5,                # ρ: 镜像下降学习率
    "M_inner": 10,             # M: 内循环迭代次数
    # --- Warmup for instability risk ---
    "warmup_steps": 10,        # t_warm: 前 N 步不启用 Σ^inst
    "inst_window_size": 5,     # 滑动窗口长度
    # --- State scoring ---
    "lambda_clip_state": 1.0,  # λ_clip: CLIP state score weight
    "lambda_token_state": 0.3, # λ_token: token activation weight
    "lambda_attn_state": 0.3,  # λ_attn: coverage weight
    # --- History EMA ---
    "history_ema_alpha": 0.3,  # history EMA decay factor
}


# =====================================================================
# Attention Feature Extractor (for cross-attention extraction)
# =====================================================================

class PerConceptAttentionCache:
    """
    Captures cross-attention outputs during per-concept UNet forward passes.
    
    Strategy: Instead of fragile global hooks on the main UNet pass,
    we collect attention features from the per-concept forward passes
    that are already computed for Σ^upd (update direction conflict).
    
    For each concept j, running UNet(x_t, t, emb_j) produces cross-attention
    maps that reflect "where concept j wants to attend". We cache these
    and later compute pairwise inner products for Σ^attn.
    
    Also provides data for Ftoken (token activation) and Fcover (spatial coverage).
    """

    def __init__(self):
        self.concept_data = {}
        self._current_concept_idx = None
        self._hook_handles = []

    def start_capture(self, unet, concept_idx):
        """Register temporary hooks before a per-concept forward pass."""
        self._current_concept_idx = concept_idx
        self.concept_data[concept_idx] = {"attn_maps": [], "mid_features": []}
        
        def make_attn_hook(name):
            def hook(module, input, output):
                if self._current_concept_idx is None:
                    return
                if isinstance(output, torch.Tensor) and output.dim() == 4:
                    feat = output.detach().cpu()
                    self.concept_data[self._current_concept_idx]["attn_maps"].append(feat)
            return hook
        
        for name, module in unet.named_modules():
            if name.endswith(".attn2"):
                try:
                    h = module.register_forward_hook(make_attn_hook(name))
                    self._hook_handles.append(h)
                except Exception:
                    pass

    def stop_capture(self):
        """Remove all hooks after the forward pass."""
        self._current_concept_idx = None
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()

    @staticmethod
    def compute_token_activation(attn_maps_list, prompt_indices=None):
        """Compute per-token activation strength from collected attention maps."""
        if not attn_maps_list:
            return 0.0
        
        total_act = 0.0
        count = 0
        for amap in attn_maps_list:
            if amap is None or not isinstance(amap, torch.Tensor):
                continue
            if amap.dim() == 4:
                pooled = amap.mean(dim=0).mean(dim=0)
            elif amap.dim() == 3:
                pooled = amap.mean(dim=0)
            else:
                continue
            
            token_importance = pooled
            
            if prompt_indices is not None and len(prompt_indices) > 0:
                idx_valid = [ii for ii in prompt_indices if ii < token_importance.shape[0]]
                if idx_valid:
                    total_act += token_importance[idx_valid].mean().item()
                else:
                    total_act += token_importance.mean().item()
            else:
                total_act += token_importance.mean().item()
            count += 1
        
        return total_act / max(count, 1)

    @staticmethod
    def compute_spatial_coverage(attn_maps_list):
        """Compute spatial coverage: fraction of spatial positions above threshold."""
        if not attn_maps_list:
            return 0.0
        
        total_cov = 0.0
        count = 0
        for amap in attn_maps_list:
            if amap is None or not isinstance(amap, torch.Tensor):
                continue
            if amap.dim() == 4:
                spatial = amap.mean(dim=[0, 1, 2])
            elif amap.dim() == 3:
                spatial = amap.mean(dim=[0, 1])
            else:
                continue
            
            threshold = spatial.mean()
            coverage = (spatial > threshold).float().mean().item()
            total_cov += coverage
            count += 1
        
        return total_cov / max(count, 1)

    def compute_attn_overlap_matrix(self, n_concepts):
        """Compute Σ^attn: pairwise inner product of attention maps between concepts."""
        import numpy as np
        Sigma_attn = np.zeros((n_concepts, n_concepts))
        
        concept_vectors = {}
        for ci in range(n_concepts):
            if ci not in self.concept_data:
                continue
            maps = self.concept_data[ci].get("attn_maps", [])
            if not maps:
                continue
            parts = []
            for m in maps:
                flat = m.numpy().flatten()
                norm = np.linalg.norm(flat)
                if norm > 1e-8:
                    parts.append(flat / norm)
                else:
                    parts.append(np.zeros_like(flat))
            if parts:
                concept_vectors[ci] = np.concatenate(parts)
        
        for i in range(n_concepts):
            for j in range(n_concepts):
                if i in concept_vectors and j in concept_vectors:
                    vi, vj = concept_vectors[i], concept_vectors[j]
                    min_len = min(len(vi), len(vj))
                    Sigma_attn[i, j] = float(np.dot(vi[:min_len], vj[:min_len]) / max(min_len, 1))
        
        if Sigma_attn.max() > Sigma_attn.min():
            Sigma_attn = (Sigma_attn - Sigma_attn.min()) / (Sigma_attn.max() - Sigma_attn.min() + 1e-8)
        
        return Sigma_attn

    def get_concept_token_activation(self, concept_idx, prompt_indices=None):
        if concept_idx not in self.concept_data:
            return 0.0
        maps = self.concept_data[concept_idx].get("attn_maps", [])
        return self.compute_token_activation(maps, prompt_indices)

    def get_concept_coverage(self, concept_idx):
        if concept_idx not in self.concept_data:
            return 0.0
        maps = self.concept_data[concept_idx].get("attn_maps", [])
        return self.compute_spatial_coverage(maps)

    def clear(self):
        self.concept_data.clear()


class ImagicStableDiffusionPipeline(DiffusionPipeline):
    r"""
    Portfolio Diffusion Pipeline for multi-prompt compositional generation.
    
    Extends standard StableDiffusion with portfolio-theoretic prompt scheduling.
    Instead of hard-switching between prompts (as in Black-Scholes method),
    computes optimal soft-weighted allocation at each denoising step.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: Union[DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler],
        safety_checker: StableDiffusionSafetyChecker,
        feature_extractor: CLIPImageProcessor,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
        )
        # Load CLIP once (shared across all __call__ invocations)
        _clip_dir = r"d:\projects\BlackScholesDiffusion2024-main\Model\CLIP"
        print("[MPT] Loading CLIP model (one-time)...")
        self.clip_model = CLIPModel.from_pretrained(_clip_dir).cuda()
        self.clip_processor = CLIPProcessor.from_pretrained(_clip_dir)
        self.clip_model.eval()
        print("[MPT] CLIP loaded.")

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        if slice_size == "auto":
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        self.enable_attention_slicing(None)

    # -----------------------------------------------------------------
    # Core computation methods
    # -----------------------------------------------------------------

    def _compute_clip_state_scores(
        self, clip_model, clip_processor, image_np, prompts_list
    ) -> List[float]:
        """
        Compute CLIP similarity scores for each concept prompt against current image.
        
        Returns:
            scores: list of floats, one per prompt
        """
        scores = []
        for p in prompts_list:
            clip_inputs = clip_processor(
                text=[p], images=image_np, return_tensors="pt", padding=True
            )
            clip_inputs["pixel_values"] = clip_inputs["pixel_values"].cuda()
            clip_inputs["input_ids"] = clip_inputs["input_ids"].cuda()
            clip_inputs["attention_mask"] = clip_inputs["attention_mask"].cuda()
            clip_out = clip_model(**clip_inputs)
            score = clip_out.logits_per_image.abs().cpu().detach().numpy()
            scores.append(float(score[0, 0]) if score.ndim > 1 else float(score))
        return scores

    def _compute_state_scores(
        self,
        clip_scores: List[float],
        attn_cache: Optional["PerConceptAttentionCache"],
        prompt_token_indices: Optional[List[List[int]]],
        config: dict,
    ) -> List[float]:
        """
        Compute composite state score F_i(x_t) for each prompt i.
        
        F_i(x) = lambda_clip*Fclip_i(x) + lambda_token*Ftoken_i(x) + lambda_attn*Fcover_i(x)
        
        Now uses PerConceptAttentionCache for real per-concept attention data.
        """
        n = len(clip_scores)
        lc = config.get("lambda_clip_state", 1.0)
        lt = config.get("lambda_token_state", 0.3)
        la = config.get("lambda_attn_state", 0.3)

        # Fclip_i: normalized CLIP score
        clip_arr = np.array(clip_scores, dtype=np.float64)
        if clip_arr.max() > clip_arr.min():
            fclip = (clip_arr - clip_arr.min()) / (clip_arr.max() - clip_arr.min() + 1e-8)
        else:
            fclip = np.ones(n) / n

        # Ftoken_i: token activation from per-concept attention cache
        ftoken = np.ones(n)
        if attn_cache is not None:
            for j in range(n):
                pidx = prompt_token_indices[j] if (prompt_token_indices is not None and j < len(prompt_token_indices)) else None
                ftoken[j] = attn_cache.get_concept_token_activation(j, pidx)
            if ftoken.max() > ftoken.min():
                ftoken = (ftoken - ftoken.min()) / (ftoken.max() - ftoken.min() + 1e-8)

        # Fcover_i: spatial coverage from per-concept attention cache
        fcover = np.ones(n)
        if attn_cache is not None:
            for j in range(n):
                fcover[j] = attn_cache.get_concept_coverage(j)
            if fcover.max() > fcover.min():
                fcover = (fcover - fcover.min()) / (fcover.max() - fcover.min() + 1e-8)

        # Composite state score
        F = lc * fclip + lt * ftoken + la * fcover

        # Re-normalize to [0, 1]
        if F.max() > F.min():
            F = (F - F.min()) / (F.max() - F.min() + 1e-8)
        else:
            F = np.ones(n) / n

        return F.tolist()

    def _compute_bs_urgency_prior(
        self, clip_scores, sigma, t_remaining, num_steps, temperature=None,
        kappa_i=None,
    ):
        """
        Compute Black-Scholes urgency prior u^BS via softmax normalization.

        FIX (Bug 4): S_{t,i} = 100·CLIP  (×100 scale per PDF Page 4).
                     K_i     = 100·κ_i   (per-prompt CLIP baseline).
        kappa_i: np.ndarray of per-prompt baselines (captured at step 0).
                 Falls back to 0.25 per concept if not provided.
        """
        n = len(clip_scores)
        if temperature is None:
            temperature = max(t_remaining, 1.0)

        rate = 1.0 / num_steps

        # Per-prompt strike prices K_i = 100·κ_i
        if kappa_i is not None and len(kappa_i) == n:
            strike_prices = [max(float(k) * 100.0, 1.0) for k in kappa_i]
        else:
            strike_prices = [25.0] * n  # 100 × 0.25 default

        bs_vals = []
        for idx, cs in enumerate(clip_scores):
            # S_{t,i} = 100·CLIP  (PDF Page 4)
            spot = np.array([[cs * 100.0]], dtype=np.float32)
            bv = bs_score(
                spot=spot,
                strike=strike_prices[idx],
                rate=rate,
                sigma=sigma,
                t=t_remaining,
            )
            bs_vals.append(bv)

        bs_arr = np.array(bs_vals, dtype=np.float64)
        # Softmax over negative BS values (lower cost → higher urgency)
        bs_arr_clipped = np.clip(bs_arr, -500, 500)
        exp_vals = np.exp(-bs_arr_clipped / max(temperature, 1e-6))
        urgency = exp_vals / (exp_vals.sum() + 1e-8)
        return urgency.tolist()

    def _compute_return_vector(
        self,
        state_scores_now: List[float],
        state_scores_prev: List[float],
        state_scores_hist: List[List[float]],  # FIX: shape [time][n_concepts]
        bs_urgency: List[float],
        config: dict,
        prev_noise_updates: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[float]:
        """
        Compute expected return vector μ for each prompt.

        μ_{t,i} = α₁·lookahead_i + α₂·deficiency_i + α₃·history_i + α₄·u_BS_i

        FIX (Bug 1): history is now a per-concept vector, not a scalar.
        FIX (Bug 2): lookahead uses prev_noise_updates magnitudes as gradient proxy.
        state_scores_hist: list of per-timestep snapshots [[s0_t0,s1_t0,...], [s0_t1,...]]
        """
        a1 = config.get("alpha_lookahead", 0.3)
        a2 = config.get("alpha_deficiency", 0.3)
        a3 = config.get("alpha_history", 0.2)
        a4 = config.get("alpha_bs", 0.2)
        ema_alpha = config.get("history_ema_alpha", 0.3)

        n = len(state_scores_now)
        now = np.array(state_scores_now, dtype=np.float64)
        prev = np.array(state_scores_prev, dtype=np.float64) if state_scores_prev else now.copy()

        # ---- Lookahead (Bug 2 improvement) ----
        # True lookahead needs N extra UNet forward passes (too expensive).
        # Proxy: concepts with larger noise-update magnitude AND higher deficiency
        # are likely to benefit more from extra weight.
        deficiency = 1.0 - now
        if prev_noise_updates and len(prev_noise_updates) == n:
            delta_norms = np.array([
                np.linalg.norm(prev_noise_updates[j]) if j in prev_noise_updates else 0.0
                for j in range(n)
            ], dtype=np.float64)
            delta_norms_norm = delta_norms / (delta_norms.sum() + 1e-8)
            lookahead = delta_norms_norm * deficiency
        else:
            raw_improvement = np.maximum(now - prev, 0)
            lookahead = 0.5 * raw_improvement + 0.5 * deficiency * max(raw_improvement.mean(), 1e-3)

        # ---- History EMA — per-concept (Bug 1 fix) ----
        # state_scores_hist shape: [T][n_concepts] after Bug 5 fix
        history = np.zeros(n, dtype=np.float64)
        if len(state_scores_hist) >= 2:
            hist_arr = np.array(state_scores_hist, dtype=np.float64)  # (T, n_concepts)
            window = hist_arr[-min(len(hist_arr), 5):]                  # (<=5, n_concepts)
            gains = np.diff(window, axis=0)                             # (<=4, n_concepts)
            if len(gains) > 0:
                weights = np.array([(1 - ema_alpha) ** i for i in range(len(gains))[::-1]])
                weights /= weights.sum()
                # dot: (k,) · (k, n_concepts) = (n_concepts,)  — per-concept!
                history = np.dot(weights, gains)

        history = np.clip(history, 0, None)
        bs_u = np.array(bs_urgency, dtype=np.float64)

        mu = a1 * lookahead + a2 * deficiency + a3 * history + a4 * bs_u
        return mu.tolist()

    def _compute_risk_covariance(
        self,
        attn_cache: Optional["PerConceptAttentionCache"],   # FIXED: uses PerConceptAttentionCache
        noise_updates: Dict[int, np.ndarray],
        state_scores_history: List[List[float]],           # FIXED: full history for sliding window
        curr_state_scores: List[float],
        sigma_t: float,
        step_index: int,
        config: dict,
    ) -> np.ndarray:
        """
        Compute risk covariance matrix Sigma_t (FIXED v2).
        
        Sigma_t = beta_1*Sigma_attn + beta_2*Sigma_upd + beta_3*Sigma_inst
        
        FIXES:
          - Sigma^attn: real per-concept attention inner products via PerConceptAttentionCache
          - Sigma^inst: sliding-window variance over last K steps (not rank-1)
        """
        b1 = config.get("beta_attn", 1.0)
        b2 = config.get("beta_upd", 1.0)
        b3 = config.get("beta_inst", 0.5)
        warmup = config.get("warmup_steps", 10)
        inst_window = config.get("inst_window_size", 5)

        n = len(curr_state_scores)
        Sigma = np.zeros((n, n))

        # --- FIX 1: Sigma^attn via PerConceptAttentionCache ---
        if attn_cache is not None and b1 > 0:
            try:
                Sigma_attn = attn_cache.compute_attn_overlap_matrix(n)
                Sigma += b1 * Sigma_attn
            except Exception:
                pass

        # --- Sigma^upd: update direction conflict (unchanged) ---
        if noise_updates and len(noise_updates) >= 2 and b2 > 0:
            Sigma_upd = np.zeros((n, n))
            keys = sorted(noise_updates.keys())
            for i_idx, i in enumerate(keys):
                for j_idx, j in enumerate(keys):
                    di = noise_updates[i].flatten()
                    dj = noise_updates[j].flatten()
                    norm_i = np.linalg.norm(di)
                    norm_j = np.linalg.norm(dj)
                    if norm_i > 1e-8 and norm_j > 1e-8:
                        cos_sim = float(np.dot(di, dj) / (norm_i * norm_j))
                        conflict = norm_i * norm_j * max(0.0, -cos_sim)
                        Sigma_upd[i_idx, j_idx] = conflict
                    else:
                        Sigma_upd[i_idx, j_idx] = 0.0
            
            max_val = Sigma_upd.max() if Sigma_upd.max() > 0 else 1.0
            if max_val > 0:
                Sigma_upd = Sigma_upd / max_val
            Sigma += b2 * Sigma_upd

        # --- Sigma^inst: SLIDING WINDOW covariance (Bug 5 fix) ---
        # state_scores_history shape: [T][n_concepts]  (time-ordered snapshots)
        if step_index >= warmup and b3 > 0 and len(state_scores_history) >= 2:
            hist_len = min(inst_window, len(state_scores_history))
            if hist_len >= 2:
                # Take last hist_len time-step snapshots: shape (hist_len, n_concepts)
                hist_mat = np.array(state_scores_history[-hist_len:], dtype=np.float64)
                Sigma_inst = np.zeros((n, n))
                for ii in range(n):
                    for jj in range(n):
                        col_ii = hist_mat[:, ii]
                        col_jj = hist_mat[:, jj]
                        ci = col_ii - col_ii.mean()
                        cj = col_jj - col_jj.mean()
                        Sigma_inst[ii, jj] = np.dot(ci, cj) / max(hist_len - 1, 1)

                sigma_sq = max(sigma_t ** 2, 1e-8)
                Sigma_inst /= sigma_sq
                Sigma += b3 * Sigma_inst

        Sigma += 1e-6 * np.eye(n)
        return Sigma


    def _compute_regularization(
        self, w: np.ndarray, w_prev: np.ndarray, state_scores: List[float],
        predicted_scores: List[float],
        step_idx: int, num_steps: int, config: dict
    ) -> float:
        """
        Compute regularization terms.

        FIX (Bug 3): R_bal = Var(q_i), not Var(w_i·q_i)  (PDF Page 9).
        FIX (Bug 6): η_t = η_max·(step/T)^p  (power-law, PDF Page 9-10).
        """
        gamma = config.get("gamma_bal", 0.1)
        tau = config.get("tau_ent", 0.05)
        eta_max = config.get("eta_max", 0.1)
        eta_power = config.get("eta_power", 1.0)

        n = len(w)
        eps = 1e-10

        # --- R_bal: Var of per-concept predicted satisfaction q_{t,i} (Bug 3 fix) ---
        if predicted_scores is not None and len(predicted_scores) == n:
            q_arr = np.array(predicted_scores, dtype=np.float64)
        else:
            q_arr = np.array(state_scores, dtype=np.float64)
        R_bal = float(np.var(q_arr))  # Var(q_i), NOT Var(w_i·q_i)

        # --- Entropy: H(w) = -Σ w·log(w) ---
        w_safe = np.clip(w, eps, None)
        H = float(-np.sum(w_safe * np.log(w_safe)))

        # --- Time smoothness: η_t = η_max·(step/T)^p (Bug 6 fix) ---
        progress = step_idx / max(num_steps - 1, 1)
        eta_t = eta_max * (progress ** eta_power)   # 0 early, η_max late
        R_turn = float(np.sum((w - w_prev) ** 2)) if w_prev is not None else 0.0

        reg_total = gamma * R_bal - tau * H + eta_t * R_turn
        return reg_total


    def _mirror_descent_solver(
        self, mu: List[float], Sigma: np.ndarray, state_scores: List[float],
        predicted_scores: List[float],
        w_prev: np.ndarray, step_idx: int, num_steps: int, config: dict
    ) -> np.ndarray:
        """
        Solve portfolio optimization via mirror descent on simplex.
        
        Objective:  max_w  μᵀ·w  -  λ_r·wᵀ·Σ·w  -  regularizers
        Subject to:  w ∈ Δ^{N-1}  (simplex)
        
        Uses exponentiated gradient (mirror descent) with softmax projection.
        """
        rho = config.get("rho", 0.5)
        M = config.get("M_inner", 10)
        lambda_r = config.get("lambda_r", 0.5)

        n = len(mu)
        mu_arr = np.array(mu, dtype=np.float64)
        
        # Initialize: start from previous weights (or uniform)
        if w_prev is not None and w_prev.shape == (n,):
            w = w_prev.copy().astype(np.float64)
        else:
            w = np.ones(n) / n

        # Ensure on simplex
        w = np.clip(w, 1e-8, None)
        w /= w.sum()

        for iteration in range(M):
            # Gradient of objective: ∇L = -μ + 2·λ_r·Σ·w + ∇reg
            grad_return = -mu_arr
            grad_risk = 2.0 * lambda_r * Sigma.dot(w)
            
            # Regularization gradients (approximate for mirror descent)
            reg = self._compute_regularization(w, w_prev, state_scores, predicted_scores, step_idx, num_steps, config)
            
            # Approximate gradient of regularization
            grad_reg = np.zeros(n)
            eps_fd = 1e-5
            for k in range(n):
                w_plus = w.copy()
                w_plus[k] += eps_fd
                w_plus = np.clip(w_plus, 1e-8, None); w_plus /= w_plus.sum()
                reg_plus = self._compute_regularization(w_plus, w_prev, state_scores, predicted_scores, step_idx, num_steps, config)
                grad_reg[k] = (reg_plus - reg) / eps_fd

            total_grad = grad_return + grad_risk + grad_reg

            # Mirror descent step: exponentiated gradient update
            w_new = w * np.exp(-rho * total_grad)
            
            # Project back to simplex
            w_new = np.clip(w_new, 1e-10, None)
            w_new /= w_new.sum()
            
            w = w_new

        return w.astype(np.float32)

    # -----------------------------------------------------------------
    # Main inference call
    # -----------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        height: Optional[int] = 512,
        width: Optional[int] = 512,
        num_inference_steps: Optional[int] = 50,
        generator: Optional[torch.Generator] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        eval_prompt: Union[str, List[str]] = None,
        mpt_config: Optional[dict] = None,
        **kwargs,
    ):
        # Merge default config with user overrides
        cfg = {**DEFAULT_MPT_CONFIG}
        if mpt_config:
            cfg.update(mpt_config)

        # Use pre-loaded CLIP model/processor (loaded once in __init__)
        clip_model = self.clip_model
        clip_processor = self.clip_processor

        # Parse prompts: eval_prompt should be [main_prompt, sub_prompt1, sub_prompt2, ...]
        # For MPT we use sub-prompts as our "portfolio" assets
        prompt_main = eval_prompt[0]  # main compositional prompt (for reference)
        concept_prompts = eval_prompt[1:3]  # 与BS一致：只用[1]交换 + [2]概念A
        n_concepts = len(concept_prompts)

        print(f"[MPT] Portfolio size: {n_concepts} concepts")
        print(f"[MPT] Concepts: {concept_prompts}")

        # Encode all text embeddings
        # Unconditional (empty) embedding for CFG
        uncond_input = self.tokenizer(
            [""] * n_concepts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        # For CFG we need just ONE unconditional embedding, so take first
        uncond_single = uncond_embeddings[:1]  # [1, 77, 768]

        # Encode each concept prompt
        concept_embeddings = []  # list of [1, 77, 768] tensors
        for p in concept_prompts:
            enc = self.tokenizer(
                p,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            emb = self.text_encoder(enc.input_ids.to(self.device))[0].detach()
            concept_embeddings.append(emb)

        # Also encode main prompt (used as fallback reference)
        main_enc = self.tokenizer(
            prompt_main,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        main_embedding = self.text_encoder(main_enc.input_ids.to(self.device))[0].detach()

        do_classifier_free_guidance = guidance_scale > 1.0
        if do_classifier_free_guidance:
            # Pre-concatenate unconditional with each concept embedding
            # Format: for each concept i: cat([uncond_single, emb_i])
            cfg_embeddings = [
                torch.cat([uncond_single, emb], dim=0)  # [2, 77, 768]
                for emb in concept_embeddings
            ]

        # Initialize latents
        latents_shape = (1, self.unet.in_channels, height // 8, width // 8)
        latents_dtype = concept_embeddings[0].dtype
        if self.device.type == "mps":
            latents = torch.randn(latents_shape, generator=generator, device="cpu", dtype=latents_dtype).to(self.device)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=self.device, dtype=latents_dtype)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)
        latents = latents * self.scheduler.init_noise_sigma

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # Initialize portfolio state
        w_prev = np.ones(n_concepts, dtype=np.float32) / n_concepts
        # FIX (Bug 5): store as [time][n_concepts] snapshots, not [concept][time]
        state_scores_history: List[List[float]] = []
        prev_state_scores = [1.0 / n_concepts] * n_concepts  # uniform initial guess
        kappa_i: Optional[np.ndarray] = None   # FIX (Bug 7): per-prompt CLIP baseline
        prev_noise_updates: Dict[int, np.ndarray] = {}  # for lookahead proxy

        # PerConceptAttentionCache: collects attention from per-concept UNet passes
        attn_cache = PerConceptAttentionCache()
        enable_attn = cfg.get("beta_attn", 0) > 0 or cfg.get("lambda_attn_state", 0) > 0

        # FIX (Bug 8): compute actual token indices for each concept prompt
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id
        special_ids = {bos_id, eos_id, pad_id, 0}
        prompt_token_indices = []
        for p in concept_prompts:
            enc = self.tokenizer(
                p, padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            ids = enc.input_ids[0].tolist()
            valid = [idx for idx, tid in enumerate(ids) if tid not in special_ids]
            prompt_token_indices.append(valid if valid else list(range(1, 4)))

        print(f"\n[MPT] Starting {num_inference_steps}-step Portfolio Diffusion...")
        print(f"[MPT] Config: alpha=({cfg['alpha_lookahead']},{cfg['alpha_deficiency']},{cfg['alpha_history']},{cfg['alpha_bs']}), "
              f"beta=({cfg['beta_attn']},{cfg['beta_upd']},{cfg['beta_inst']}), "
              f"reg=({cfg['lambda_r']},{cfg['gamma_bal']},{cfg['tau_ent']}), "
              f"solver: rho={cfg['rho']}, M={cfg['M_inner']}")

        # Main denoising loop
        pbar = self.progress_bar(timesteps_tensor)
        for i, t in enumerate(pbar):
            time_remaining = num_inference_steps - i  # t for BS formula

            # Get variance for BS score — keep as scalar for display, pass raw to BS
            sigma_raw = self.scheduler._get_variance(int(t) + 1, int(t))
            if sigma_raw < 0:
                sigma_raw = -sigma_raw
            sigma_t = float(sigma_raw ** 0.5)
            # Keep sigma as tensor for bs_score (which calls .cpu().numpy())
            sigma_for_bs = torch.tensor(sigma_t, dtype=torch.float32)

            # ---- Forecast x̂_{t-1} for state evaluation ----
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # Use equal-weight combination for initial prediction
            if do_classifier_free_guidance:
                # Build initial combined embedding (equal weights) for prediction
                eq_weights = np.ones(n_concepts, dtype=np.float32) / n_concepts
                cond_part = sum(eq_weights[j] * concept_embeddings[j] for j in range(n_concepts))
                pred_embed = torch.cat([uncond_single, cond_part], dim=0)
            else:
                cond_part = sum((1.0/n_concepts) * concept_embeddings[j] for j in range(n_concepts))
                pred_embed = cond_part.unsqueeze(0)

            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=pred_embed).sample

            # Apply guidance for prediction
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            else:
                noise_pred_guided = noise_pred

            # Step scheduler to get prediction
            sched_out = self.scheduler.step(noise_pred_guided, t, latents, **extra_step_kwargs)
            pred_x0 = sched_out.pred_original_sample

            # Decode to image space for CLIP scoring
            latents_forecast = 1 / 0.18215 * pred_x0
            image_forecast = self.vae.decode(latents_forecast).sample
            image_forecast = (image_forecast / 2 + 0.5).clamp(0, 1)
            image_forecast_np = image_forecast.cpu().permute(0, 2, 3, 1).float().numpy()

            # ---- Step 1: Compute state scores F_i(x_t) for each concept ----
            clip_scores = self._compute_clip_state_scores(
                clip_model, clip_processor, image_forecast_np, concept_prompts
            )

            state_scores = self._compute_state_scores(
                clip_scores, attn_cache if enable_attn else None, prompt_token_indices, cfg
            )

            # FIX (Bug 5): append one snapshot [s0, s1, ...] per timestep
            state_scores_history.append(state_scores[:])

            # FIX (Bug 7): capture per-prompt CLIP baseline κ_i at first step
            if i == 0:
                kappa_i = np.array(clip_scores, dtype=np.float64)
                kappa_i = np.clip(kappa_i, 1e-4, None)

            # ---- Step 2: Compute BS urgency prior ----
            bs_urgency = self._compute_bs_urgency_prior(
                clip_scores, sigma_for_bs, time_remaining, num_inference_steps,
                temperature=max(time_remaining, 1.0),
                kappa_i=kappa_i,
            )

            # ---- Step 3: Compute return vector μ ----
            mu = self._compute_return_vector(
                state_scores_now=state_scores,
                state_scores_prev=prev_state_scores,
                state_scores_hist=state_scores_history,
                bs_urgency=bs_urgency,
                config=cfg,
                prev_noise_updates=prev_noise_updates,
            )

            # ---- Step 4: Collect per-concept noise updates + attention maps ----
            noise_updates = {}
            attn_cache.clear()  # clear previous step data

            for j in range(n_concepts):
                if do_classifier_free_guidance:
                    embed_j = cfg_embeddings[j]
                else:
                    embed_j = concept_embeddings[j].unsqueeze(0)
                
                # Capture cross-attention during this forward pass
                if enable_attn:
                    attn_cache.start_capture(self.unet, j)
                
                noise_j = self.unet(
                    latent_model_input, t, encoder_hidden_states=embed_j
                ).sample
                
                if enable_attn:
                    attn_cache.stop_capture()
                
                if do_classifier_free_guidance:
                    nu_j, nj = noise_j.chunk(2)
                    noise_j_final = nu_j + guidance_scale * (nj - nu_j)
                else:
                    noise_j_final = noise_j
                
                noise_updates[j] = noise_j_final.cpu().numpy().flatten()
            # ---- Step 5: Compute risk covariance Σ ----
            Sigma = self._compute_risk_covariance(
                attn_cache if enable_attn else None,
                noise_updates,
                state_scores_history,
                state_scores,
                sigma_t, i, cfg,
            )

            # ---- Step 6: Solve optimization via mirror descent ----
            w_optimal = self._mirror_descent_solver(
                mu=mu, Sigma=Sigma, state_scores=state_scores,
                predicted_scores=state_scores,
                w_prev=w_prev, step_idx=i, num_steps=num_inference_steps,
                config=cfg,
            )

            # Log progress — update progress bar postfix (no separate print)
            w_str = ", ".join([f"{x:.3f}" for x in w_optimal])
            pbar.set_postfix(
                w=w_str, F=f"{max(state_scores):.3f}", mu=f"{max(mu):.3f}",
                refresh=False,
            )

            # ---- Step 7: Apply weighted condition injection ----
            # Combine concept embeddings with optimal weights
            cond_weighted = sum(
                w_optimal[j] * concept_embeddings[j] for j in range(n_concepts)
            )  # [1, 77, 768]

            if do_classifier_free_guidance:
                final_embed = torch.cat([uncond_single, cond_weighted], dim=0)
            else:
                final_embed = cond_weighted.unsqueeze(0)

            # Final UNet forward pass with optimal weights
            noise_pred_final = self.unet(
                latent_model_input, t, encoder_hidden_states=final_embed
            ).sample

            # Apply guidance
            if do_classifier_free_guidance:
                noise_pred_uncond_f, noise_pred_text_f = noise_pred_final.chunk(2)
                noise_pred_final = noise_pred_uncond_f + guidance_scale * (
                    noise_pred_text_f - noise_pred_uncond_f
                )

            # Scheduler step
            latents = self.scheduler.step(noise_pred_final, t, latents, **extra_step_kwargs).prev_sample

            # Update for next iteration
            w_prev = w_optimal
            prev_state_scores = state_scores[:]
            prev_noise_updates = dict(noise_updates)  # store for next-step lookahead proxy

        # Final decode
        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="pt"
            ).to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(concept_embeddings[0].dtype),
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
