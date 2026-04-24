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
import math
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


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via math.erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_score(spot, strike, rate, sigma, t):
    """Black-Scholes European call option price.
    FIX (P0): Use norm CDF Φ(d) instead of the Gaussian PDF exp(-d²/2).
    FIX: Use per-prompt strike price (not hardcoded); guard edge cases.
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
    # C = S·Φ(d1) − K·e^{−rT}·Φ(d2)  (Black-Scholes call price)
    bs_score_val = spot_val * _norm_cdf(d1) - strike_val * math.exp(-rate * t_val) * _norm_cdf(d2)
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
    "eta_max": 0.1,            # η_max: 时间平滑最大系数
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

class StoreCrossAttnProcessor:
    """
    Custom AttentionProcessor that captures the true cross-attention
    probability map (post-softmax) during a per-concept forward pass.

    Captures attention_probs shape: [batch*heads, query_positions, key_tokens]
    - query_positions: latent spatial positions (for F_cover)
    - key_tokens: text token positions       (for F_token)

    P1 FIX: Added input_ndim==4 reshape, prepare_attention_mask, and
    rescale_output_factor for compatibility across diffusers versions.
    """

    def __init__(self, store: dict, concept_idx: int):
        self.store = store
        self.concept_idx = concept_idx

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        residual = hidden_states
        is_cross = encoder_hidden_states is not None

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        batch_size_orig = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        query = attn.to_q(hidden_states)
        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        
        # to ensure the mask shape aligns with the original batch_size
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size_orig)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Capture TRUE attention probability map [BH, Q, K] for cross-attn only
        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        if is_cross:
            
            # We must only capture the CONDITIONAL attention map to avoid dilution.
            bh, q_len, k_len = attention_probs.shape
            num_heads = bh // batch_size_orig
            
            # Reshape to [batch, heads, Q, K]
            attn_reshaped = attention_probs.view(batch_size_orig, num_heads, q_len, k_len)
            
            # In CFG, the conditional embedding is usually the second half (-1)
            cond_probs = attn_reshaped[-1]  # [heads, Q, K]
            
            # Store only the mean over heads to save memory: [Q, K]
            compressed = cond_probs.mean(dim=0).float().cpu()  # [Q, K]
            self.store.setdefault(self.concept_idx, {}).setdefault("attn_maps", []).append(compressed)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        
        if hasattr(attn, "rescale_output_factor") and attn.rescale_output_factor != 1.0:
            hidden_states = hidden_states / attn.rescale_output_factor

        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if getattr(attn, "residual_connection", False):
            hidden_states = hidden_states + residual
        return hidden_states


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
        self._saved_processors = {}  # stores original processors for restore

    def start_capture(self, unet, concept_idx):
        """Install StoreCrossAttnProcessor on all cross-attn layers before a per-concept forward pass."""
        self.concept_data[concept_idx] = {"attn_maps": [], "mid_features": []}
        self._saved_processors = {}
        for name, module in unet.attn_processors.items():
            self._saved_processors[name] = module  # save originals
        
        new_processors = {}
        for name in unet.attn_processors:
            if "attn2" in name:
                new_processors[name] = StoreCrossAttnProcessor(
                    store=self.concept_data, concept_idx=concept_idx
                )
            else:
                new_processors[name] = self._saved_processors[name]
        unet.set_attn_processor(new_processors)

    def stop_capture(self, unet):
        """Restore original processors after the per-concept forward pass."""
        if self._saved_processors:
            unet.set_attn_processor(self._saved_processors)
            self._saved_processors = {}

    @staticmethod
    def compute_token_activation(attn_maps_list, prompt_indices=None):
        """
        Compute per-token activation strength.

        P0 FIX: attention_probs stored as [Q, K] (mean over heads already done in processor).
        Token activation = average attention over Q positions, for selected K (token) indices.
        """
        if not attn_maps_list:
            return 0.0

        vals = []
        for amap in attn_maps_list:
            if amap is None or not isinstance(amap, torch.Tensor):
                continue
            if amap.dim() == 2:
                # [Q, K] — correct shape from StoreCrossAttnProcessor
                # token_scores[k] = mean attention from all spatial positions to token k
                token_scores = amap.mean(dim=0)  # [K]
            elif amap.dim() == 3:
                # Fallback: [BH, Q, K] — pool over BH and Q to get [K]
                token_scores = amap.mean(dim=(0, 1))  # [K]
            else:
                continue

            if prompt_indices is not None and len(prompt_indices) > 0:
                idx_valid = [ii for ii in prompt_indices if ii < token_scores.shape[0]]
                if idx_valid:
                    vals.append(token_scores[idx_valid].mean().item())
                else:
                    vals.append(token_scores.mean().item())
            else:
                vals.append(token_scores.mean().item())

        return float(sum(vals) / max(len(vals), 1))

    @staticmethod
    def compute_spatial_coverage(attn_maps_list, prompt_indices=None):
        """
        Compute spatial coverage: fraction of query (spatial) positions above mean.

        P0 FIX: attention_probs stored as [Q, K]. Spatial coverage is over Q dimension.
        Optionally restrict to concept token positions in K dimension.
        """
        if not attn_maps_list:
            return 0.0

        vals = []
        for amap in attn_maps_list:
            if amap is None or not isinstance(amap, torch.Tensor):
                continue
            if amap.dim() == 2:
                # [Q, K] — select concept tokens if available, then pool over K
                if prompt_indices is not None and len(prompt_indices) > 0:
                    idx_valid = [ii for ii in prompt_indices if ii < amap.shape[-1]]
                    if idx_valid:
                        spatial_scores = amap[:, idx_valid].mean(dim=1)  # [Q]
                    else:
                        spatial_scores = amap.mean(dim=1)  # [Q]
                else:
                    spatial_scores = amap.mean(dim=1)  # [Q]
            elif amap.dim() == 3:
                # Fallback: [BH, Q, K]
                if prompt_indices is not None and len(prompt_indices) > 0:
                    idx_valid = [ii for ii in prompt_indices if ii < amap.shape[-1]]
                    if idx_valid:
                        spatial_scores = amap[:, :, idx_valid].mean(dim=(0, 2))  # [Q]
                    else:
                        spatial_scores = amap.mean(dim=(0, 2))  # [Q]
                else:
                    spatial_scores = amap.mean(dim=(0, 2))  # [Q]
            else:
                continue

            threshold = spatial_scores.mean()
            coverage = (spatial_scores > threshold).float().mean().item()
            vals.append(coverage)

        return float(sum(vals) / max(len(vals), 1))

    def compute_attn_overlap_matrix(self, n_concepts, prompt_token_indices=None):
        """
        Compute Σ^attn: pairwise inner product of attention maps between concepts.
        preventing dilution by BOS/EOS/padding tokens.
        """
        import numpy as np
        Sigma_attn = np.zeros((n_concepts, n_concepts))
        
        concept_vectors = {}
        for ci in range(n_concepts):
            if ci not in self.concept_data:
                continue
            maps = self.concept_data[ci].get("attn_maps", [])
            if not maps:
                continue
            
            idx = None
            if prompt_token_indices is not None and ci < len(prompt_token_indices):
                idx = prompt_token_indices[ci]

            parts = []
            for m in maps:
                if m.dim() != 2:  # Expected [Q, K] from processor
                    continue
                
                # Filter to only the concept token indices if available
                if idx is not None and len(idx) > 0:
                    idx_valid = [k for k in idx if k < m.shape[-1]]
                    if idx_valid:
                        spatial_vec = m[:, idx_valid].mean(dim=1)  # [Q]
                    else:
                        spatial_vec = m.mean(dim=1)
                else:
                    spatial_vec = m.mean(dim=1)

                spatial_vec = spatial_vec.numpy()
                norm = np.linalg.norm(spatial_vec)
                if norm > 1e-8:
                    parts.append(spatial_vec / norm)
                else:
                    parts.append(np.zeros_like(spatial_vec))

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

    def get_concept_coverage(self, concept_idx, prompt_indices=None):
        """P0 FIX: accept prompt_indices so spatial coverage uses concept tokens only."""
        if concept_idx not in self.concept_data:
            return 0.0
        maps = self.concept_data[concept_idx].get("attn_maps", [])
        return self.compute_spatial_coverage(maps, prompt_indices)

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
        try:
            _clip_device = self._execution_device
        except Exception:
            _clip_device = next(iter(unet.parameters())).device
        self.clip_model = CLIPModel.from_pretrained(_clip_dir).to(_clip_device)
        self.clip_processor = CLIPProcessor.from_pretrained(_clip_dir)
        self.clip_model.eval()
        self._clip_device = _clip_device
        print(f"[MPT] CLIP loaded on {_clip_device}.")

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
        
        _unet_dev = next(self.unet.parameters()).device
        if _unet_dev != self._clip_device:
            self.clip_model = self.clip_model.to(_unet_dev)
            self._clip_device = _unet_dev
        _dev = self._clip_device
        for p in prompts_list:
            clip_inputs = clip_processor(
                text=[p], images=image_np, return_tensors="pt", padding=True
            )
    
            clip_inputs = {k: v.to(_dev) if hasattr(v, "to") else v
                          for k, v in clip_inputs.items()}
            clip_out = clip_model(**clip_inputs)

            # match the prompt. Preserving the sign is critical for correct urgency ranking.
            score = clip_out.logits_per_image.cpu().detach().numpy()
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

        # Fclip_i: calibrated CLIP score relative to per-prompt baseline κ_i

        # Use kappa_i from config if available, else sigmoid calibration.
        clip_arr = np.array(clip_scores, dtype=np.float64)
        kappa_ref = config.get("_kappa_i", None)
        if kappa_ref is not None and len(kappa_ref) == n:
            kappa_arr = np.array(kappa_ref, dtype=np.float64)
            fclip = clip_arr / (kappa_arr + 1e-8)
            fclip = np.clip(fclip, 0.0, 1.5) / 1.5  # normalise to [0,1]
        else:
            # Sigmoid calibration: logit centre ~25, scale 5 (CLIP logits typical range)
            fclip = 1.0 / (1.0 + np.exp(-(clip_arr - 25.0) / 5.0))

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
                
                pidx = prompt_token_indices[j] if (prompt_token_indices is not None and j < len(prompt_token_indices)) else None
                fcover[j] = attn_cache.get_concept_coverage(j, pidx)
            if fcover.max() > fcover.min():
                fcover = (fcover - fcover.min()) / (fcover.max() - fcover.min() + 1e-8)

        # Composite state score — weighted sum, clip to [0,1] (no per-step min-max)
        weight_sum = lc + lt + la + 1e-8
        F = (lc * fclip + lt * ftoken + la * fcover) / weight_sum
        F = np.clip(F, 0.0, 1.0)

        return F.tolist()

    def _compute_bs_urgency_prior(
        self, clip_scores, sigma, t_remaining, num_steps, temperature=None,
        kappa_i=None,
    ):
        """
        Compute Black-Scholes urgency prior u^BS via softmax normalization.

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
        prompt_token_indices: Optional[List[List[int]]] = None,
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
                Sigma_attn = attn_cache.compute_attn_overlap_matrix(n, prompt_token_indices)
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
        # P2 FIX: project to PSD so w^T Sigma w is always non-negative (true risk penalty)
        Sigma = 0.5 * (Sigma + Sigma.T)  # symmetrise
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.maximum(vals, 1e-6)    # clip negative eigenvalues
        Sigma = (vecs * vals) @ vecs.T
        return Sigma


    def _compute_regularization(
        self, w: np.ndarray, w_prev: np.ndarray, state_scores: List[float],
        lookahead_proxy: List[float],
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

        # --- R_bal: Var(q(w)) surrogate — PDF form, but with w-dependent surrogate ---
        # PDF defines q_t,i(w_t) = F_i(x_hat_{t-1}(w_t)), which is expensive to compute.
        # Surrogate: q_i(w) = state_scores_i + w_i * lookahead_proxy_i
        # This preserves the Var(q(w)) form and makes R_bal differentiable in w.
        # P1 FIX: parameter renamed predicted_scores -> lookahead_proxy for clarity.
        # q_i(w) = state_scores_i + w_i * lookahead_proxy_i  (surrogate of PDF q_t,i(w_t))
        state_arr = np.array(state_scores, dtype=np.float64)
        if lookahead_proxy is not None and len(lookahead_proxy) == n:
            marginal = np.array(lookahead_proxy, dtype=np.float64)
        else:
            marginal = np.zeros_like(state_arr)
        w_arr = np.array(w, dtype=np.float64)
        q_surrogate = np.clip(state_arr + w_arr * marginal, 0.0, 1.0)
        R_bal = float(np.var(q_surrogate))  # Var(q(w)) — depends on w via q_surrogate

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
        lookahead_proxy: List[float],
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
            reg = self._compute_regularization(w, w_prev, state_scores, lookahead_proxy, step_idx, num_steps, config)
            
            # Approximate gradient of regularization
            grad_reg = np.zeros(n)
            eps_fd = 1e-5
            for k in range(n):
                w_plus = w.copy()
                w_plus[k] += eps_fd
                w_plus = np.clip(w_plus, 1e-8, None); w_plus /= w_plus.sum()
                reg_plus = self._compute_regularization(w_plus, w_prev, state_scores, lookahead_proxy, step_idx, num_steps, config)
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
        # FIX (P1): Remove hardcoded [1:3] slice — use ALL supplied sub-concepts so
        # the portfolio truly supports N assets, not just 2.
        concept_prompts = eval_prompt[1:]  # all sub-concepts after the main prompt
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

            # P0 FIX: Manually compute pred_x0 to avoid advancing the scheduler's internal step_index!
            # Calling scheduler.step() twice per timestep breaks the sampling trajectory.
            alpha_prod_t = self.scheduler.alphas_cumprod[int(t)].to(latents.device)
            beta_prod_t = 1 - alpha_prod_t
            pred_x0 = (latents - beta_prod_t ** (0.5) * noise_pred_guided) / alpha_prod_t ** (0.5)

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

            # P1 FIX: inject kappa_i into cfg so _compute_state_scores uses baseline calibration
            cfg["_kappa_i"] = kappa_i.tolist() if kappa_i is not None else None

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
            # P1 FIX: Σ_upd should use Δε_i = ε_i - ε_base (not raw ε_i)
            # First compute the base noise prediction with equal-weight embedding.
            noise_updates = {}
            attn_cache.clear()  # clear previous step data

            # Compute base noise (equal-weight composite) for delta subtraction
            if do_classifier_free_guidance:
                base_embed = torch.cat([uncond_single,
                    sum((1.0/n_concepts)*concept_embeddings[j] for j in range(n_concepts))], dim=0)
            else:
                base_embed = sum((1.0/n_concepts)*concept_embeddings[j] for j in range(n_concepts))
            noise_base_raw = self.unet(latent_model_input, t, encoder_hidden_states=base_embed).sample
            if do_classifier_free_guidance:
                nu_base, n_base = noise_base_raw.chunk(2)
                noise_base_final = nu_base + guidance_scale * (n_base - nu_base)
            else:
                noise_base_final = noise_base_raw

            noise_j_tensors = {}  # Store guided noise for step 7 combination
            for j in range(n_concepts):
                if do_classifier_free_guidance:
                    embed_j = cfg_embeddings[j]
                else:
                    embed_j = concept_embeddings[j].unsqueeze(0)

                # Capture true cross-attention probability maps via AttentionProcessor
                # P0 FIX: wrap in try/finally so processors are ALWAYS restored even on error
                if enable_attn:
                    attn_cache.start_capture(self.unet, j)
                try:
                    noise_j = self.unet(
                        latent_model_input, t, encoder_hidden_states=embed_j
                    ).sample
                finally:
                    if enable_attn:
                        attn_cache.stop_capture(self.unet)  # always restore processors

                if do_classifier_free_guidance:
                    nu_j, nj = noise_j.chunk(2)
                    noise_j_final = nu_j + guidance_scale * (nj - nu_j)
                else:
                    noise_j_final = noise_j
                    
                noise_j_tensors[j] = noise_j_final

                # Store Δε_i = ε_i - ε_base (not raw ε_i) — PDF definition
                delta_j = (noise_j_final - noise_base_final).cpu().numpy().flatten()
                noise_updates[j] = delta_j
            # ---- Step 5: Compute risk covariance Σ ----
            Sigma = self._compute_risk_covariance(
                attn_cache if enable_attn else None,
                noise_updates,
                state_scores_history,
                state_scores,
                sigma_t, i, cfg,
                prompt_token_indices=prompt_token_indices,
            )

            # ---- Step 6: Solve optimization via mirror descent ----
            # P1 FIX: pass mu as lookahead_proxy surrogate to R_bal (mu already encodes
            # per-concept expected gain including lookahead_proxy component)
            w_optimal = self._mirror_descent_solver(
                mu=mu, Sigma=Sigma, state_scores=state_scores,
                lookahead_proxy=list(mu),
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
            # P0 FIX: MPT linearly combines the GUIDED noise predictions, NOT the text embeddings!
            # Combining text embeddings shrinks their L2 norm (due to quasi-orthogonality),
            # causing the UNet to collapse to unconditional generation (e.g. random humans/landscapes).
            # This also saves 1 full UNet forward pass per step!
            noise_pred_final = sum(
                w_optimal[j] * noise_j_tensors[j] for j in range(n_concepts)
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
