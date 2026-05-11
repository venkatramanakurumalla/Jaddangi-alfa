
# ============================================================
# JADDANGI‑ALFA · 350M FRONTIER SLM · v1.6.1 (DEFINITIVE)
# ======================================================
# ============================================================

import os, gc, math, random, warnings, numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
warnings.filterwarnings("ignore")

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32        = True
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# --------- optional fused kernels ---------
try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    from flash_attn.losses.cross_entropy import CrossEntropyLoss as FusedCrossEntropyLoss
    HAS_FUSED_CE = True
except ImportError:
    HAS_FUSED_CE = False

try:
    from flash_attn.ops.rms_norm import RMSNorm as FusedRMSNorm
    HAS_FUSED_RMSNORM = True
except ImportError:
    HAS_FUSED_RMSNORM = False

try:
    from flash_attn.ops.fused_dense import FusedMLP
    HAS_FUSED_SWIGLU = True
except ImportError:
    HAS_FUSED_SWIGLU = False

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

from transformers import (
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PretrainedConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    GenerationMixin,
)
from torch.utils.data import IterableDataset

@dataclass
class JaddangiCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    past_position_ids: Optional[torch.LongTensor] = None

# ============================================================
# USER CONFIGURATION – ADJUST PATHS & HYPERPARAMETERS
# ============================================================
OUTPUT_DIR            = "/content/jaddangi-alfa-350m-v16"
TOKENIZER_NAME        = "YOUR_NEW_32K_TOKENIZER_PATH"      # replace with real path
MMAP_DATA_FILE        = "train_tokens_packed.bin"          # packed uint16 mmap

VOCAB_SIZE            = 32000                               # fallback
MAX_LENGTH            = 4096
HIDDEN_SIZE           = 1024
NUM_LAYERS            = 24
NUM_HEADS             = 16
NUM_KV_HEADS          = 4
INTERMEDIATE_SIZE     = 3584
MAX_POSITION_EMBEDDINGS = 32768
ROPE_THETA            = 100000.0
ATTN_DROPOUT          = 0.0
RMSNORM_EPS           = 1e-5

MAX_STEPS             = 50000
LEARNING_RATE         = 2e-4          # 2e-4 for 350M + DeepNet + bf16
WARMUP_STEPS          = 1000          # 2% of max steps
WEIGHT_DECAY          = 0.01
MAX_GRAD_NORM         = 1.0
BETA1                 = 0.9
BETA2                 = 0.98
BATCH_SIZE            = 1
GRAD_ACCUM            = 16
SAVE_STEPS            = 1000
LOGGING_STEPS         = 100
SAMPLE_EVERY          = 500
SEED                  = 42

ENABLE_COMPILE        = False         # DO NOT ENABLE – dynamic FA varlen breaks compile
SOFTCAP_VALUE         = 50.0          # attention logit soft‑capping (bf16 safety)
EMA_DECAY             = 0.999         # Exponential Moving Average decay (0 = disabled)

# ============================================================
# SEED
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)

# ============================================================
# ARCHITECTURE: JADDANGI‑ALFA‑1.6.1
# ============================================================

class JaddangiRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        if HAS_FUSED_RMSNORM:
            self.norm = FusedRMSNorm(hidden_size, eps=eps)
        else:
            self.norm = None
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

    def forward(self, x):
        if self.norm is not None:
            return self.norm(x)
        x_fp32 = x.to(torch.float32)
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        return self.weight.to(x.dtype) * (x_fp32 * torch.rsqrt(variance + self.eps)).to(x.dtype)

class JaddangiRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=100000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Pre‑compute full cache [max_seq, dim]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        emb = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        cos = self.cos_cached[position_ids]   # [B, S, dim]
        sin = self.sin_cached[position_ids]
        return cos.to(x.dtype), sin.to(x.dtype)

def rotate_every_two(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view_as(x)

def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos.unsqueeze(1)) + (rotate_every_two(q) * sin.unsqueeze(1)), \
           (k * cos.unsqueeze(1)) + (rotate_every_two(k) * sin.unsqueeze(1))

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1: return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

class JaddangiAttention(nn.Module):
    def __init__(self, config, rotary_emb):
        super().__init__()
        self.hidden_size  = config.hidden_size
        self.num_heads    = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim     = self.hidden_size // self.num_heads
        self.attn_dropout = config.attn_dropout
        self.softcap      = config.attn_logit_softcapping if hasattr(config, "attn_logit_softcapping") else 0.0

        self.qkv_size = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(self.hidden_size, self.qkv_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj.is_residual_proj = True
        self.q_norm = JaddangiRMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = JaddangiRMSNorm(self.num_kv_heads * self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = rotary_emb

        # ✅ Correct scale for SDPA (1/√d)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=False, cu_seqlens=None, max_seqlen=None):
        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)

        q = self.q_norm(q.contiguous()).view(bsz, q_len, self.num_heads, self.head_dim)
        k = self.k_norm(k.contiguous()).view(bsz, q_len, self.num_kv_heads, self.head_dim)
        v = v.contiguous().view(bsz, q_len, self.num_kv_heads, self.head_dim)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        dropout_p = self.attn_dropout if self.training else 0.0

        # ---- FlashAttention varlen path ----
        if HAS_FLASH_ATTN and cu_seqlens is not None:
            q_unpad = q.reshape(-1, self.num_heads, self.head_dim)
            k_unpad = k.reshape(-1, self.num_kv_heads, self.head_dim)
            v_unpad = v.reshape(-1, self.num_kv_heads, self.head_dim)
            attn_output_unpad = flash_attn_varlen_func(
                q_unpad, k_unpad, v_unpad,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
                dropout_p=dropout_p,
                causal=True,
                window_size=(-1, -1),
                deterministic=True,
                softcap=self.softcap if self.softcap > 0 else 0.0,
            )
            attn_output = attn_output_unpad.view(bsz, q_len, self.num_heads, self.head_dim)
        else:
            # ---- fallback SDPA (with correct scale) ----
            q = q.transpose(1, 2)  # [B, H, S, d]
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if use_cache and past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            past_key_value = (k, v) if use_cache else None

            k_att = repeat_kv(k, self.num_heads // self.num_kv_heads)
            v_att = repeat_kv(v, self.num_heads // self.num_kv_heads)
            is_causal = q_len > 1

            if attention_mask is not None:
                sdpa_mask = attention_mask[:, None, None, :] == 1.0
                if is_causal:
                    causal_mask = (torch.arange(q_len, device=q.device)[:, None]
                                   >= (torch.arange(k_att.size(2), device=q.device)[None, :]
                                       - (k_att.size(2) - q_len)))
                    sdpa_mask = sdpa_mask & causal_mask.unsqueeze(0).unsqueeze(0)
                attn_output = F.scaled_dot_product_attention(
                    q, k_att, v_att, attn_mask=sdpa_mask, dropout_p=dropout_p,
                    is_causal=False, scale=self.scale)   # ✅ 1/√d
            else:
                attn_output = F.scaled_dot_product_attention(
                    q, k_att, v_att, attn_mask=None, dropout_p=dropout_p,
                    is_causal=is_causal, scale=self.scale)   # ✅ 1/√d

            # Softcapping path also uses correct scale (separate from SDPA)
            if self.softcap > 0:
                attn_weights = torch.matmul(q, k_att.transpose(-2, -1)) * self.scale
                attn_weights = self.softcap * torch.tanh(attn_weights / self.softcap)
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
                attn_output = torch.matmul(attn_weights, v_att)

            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        return self.o_proj(attn_output), past_key_value

class JaddangiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        if HAS_FUSED_SWIGLU:
            self.fused_mlp = FusedMLP(
                in_features=config.hidden_size,
                hidden_features=config.intermediate_size,
                activation="swiglu",
                bias=False
            )
        else:
            self.fused_mlp = None
            self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
            self.down_proj.is_residual_proj = True

    def forward(self, x):
        if self.fused_mlp is not None:
            return self.fused_mlp(x)
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class JaddangiDecoderLayer(nn.Module):
    def __init__(self, config, rotary_emb):
        super().__init__()
        self.self_attn = JaddangiAttention(config, rotary_emb)
        self.mlp = JaddangiMLP(config)
        self.input_layernorm = JaddangiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = JaddangiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, use_cache=False, cu_seqlens=None, max_seqlen=None):
        if self.gradient_checkpointing and self.training:
            if not hidden_states.requires_grad:
                hidden_states = hidden_states.requires_grad_(True)
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, hidden_states, attention_mask, position_ids,
                past_key_value, use_cache, cu_seqlens, max_seqlen,
                use_reentrant=False, preserve_rng_state=False
            )
        return self._forward_impl(
            hidden_states, attention_mask, position_ids,
            past_key_value, use_cache, cu_seqlens, max_seqlen)

    def _forward_impl(self, hidden_states, attention_mask=None, position_ids=None,
                      past_key_value=None, use_cache=False, cu_seqlens=None, max_seqlen=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states, attention_mask, position_ids,
            past_key_value, use_cache, cu_seqlens, max_seqlen)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states, present_key_value

class JaddangiAuxiliaryRegularizer(nn.Module):
    """Predicts token at t+2 from hidden state at t."""
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.SiLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

    def forward(self, hidden_states, lm_head):
        return lm_head(self.proj(hidden_states))

# ============================================================
# CONFIGURATION
# ============================================================
class JaddangiIndependentConfig(PretrainedConfig):
    model_type = "jaddangi_independent"
    def __init__(self,
                 vocab_size=32000,
                 hidden_size=1024,
                 num_layers=24,
                 num_attention_heads=16,
                 num_key_value_heads=4,
                 intermediate_size=3584,
                 max_position_embeddings=32768,
                 rope_theta=100000.0,
                 attn_dropout=0.0,
                 rms_norm_eps=1e-5,
                 tie_word_embeddings=True,
                 eos_token_id=None,
                 attn_logit_softcapping=50.0,
                 **kwargs):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        assert hidden_size % num_attention_heads == 0
        assert num_attention_heads % num_key_value_heads == 0

        self.architecture_version = "jaddangi_alfa_v1.6.1"
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.attn_dropout = attn_dropout
        self.rms_norm_eps = rms_norm_eps
        self.eos_token_id = eos_token_id
        self.attn_logit_softcapping = attn_logit_softcapping

# ============================================================
# MODEL CLASSES (unchanged core logic)
# ============================================================
class JaddangiIndependentModel(PreTrainedModel):
    config_class = JaddangiIndependentConfig
    def __init__(self, config):
        super().__init__(config)
        assert config.architecture_version == "jaddangi_alfa_v1.6.1", "Architecture version mismatch!"
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = JaddangiRotaryEmbedding(
            config.hidden_size // config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        self.layers = nn.ModuleList([JaddangiDecoderLayer(config, self.rotary_emb) for _ in range(config.num_layers)])
        self.norm = JaddangiRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, JaddangiDecoderLayer):
            module.gradient_checkpointing = value

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, cu_seqlens=None, max_seqlen=None):
        hidden_states = self.embed_tokens(input_ids)
        next_decoder_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            layer_outputs = layer(
                hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                past_key_value=past_key_value, use_cache=use_cache,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

        return self.norm(hidden_states), next_decoder_cache

class JaddangiIndependentForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = JaddangiIndependentConfig
    supports_gradient_checkpointing = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = JaddangiIndependentModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.aux_reg = JaddangiAuxiliaryRegularizer(config.hidden_size)
        self.post_init()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if getattr(module, "is_residual_proj", False):
                std /= math.sqrt(2 * self.config.num_layers)
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def get_input_embeddings(self): return self.model.embed_tokens
    def set_input_embeddings(self, v): self.model.embed_tokens = v
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, v): self.lm_head = v

    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, *args, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(outputs, model_kwargs, *args, **kwargs)
        if "past_position_ids" in outputs:
            model_kwargs["past_position_ids"] = outputs["past_position_ids"]
        if "logits" in outputs:
            model_kwargs["last_token"] = outputs["logits"].argmax(dim=-1)[:, -1:]
        return model_kwargs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("past_position_ids", None)

        if position_ids is None:
            eos_mask = (input_ids == self.config.eos_token_id)
            if eos_mask.any():
                start_mask = torch.zeros_like(eos_mask)
                start_mask[:, 1:] = eos_mask[:, :-1]
                start_mask[:, 0] = True
                global_pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
                reset_anchors = torch.cummax(global_pos * start_mask.long(), dim=1)[0]
                position_ids = global_pos - reset_anchors
            else:
                past_length = past_key_values[0][0].shape[2] if past_key_values else 0
                position_ids = past_length + torch.arange(input_ids.shape[1], dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
        else:
            last_token = kwargs.get("last_token", None)
            if last_token is not None:
                eos_mask = (last_token[:, -1] == self.config.eos_token_id)
                if eos_mask.any():
                    position_ids = torch.where(eos_mask.unsqueeze(-1), torch.zeros_like(position_ids), position_ids + 1)
                    new_past = []
                    for k, v in past_key_values:
                        k_masked = k.masked_fill(eos_mask[:, None, None, None], 0.0)
                        v_masked = v.masked_fill(eos_mask[:, None, None, None], 0.0)
                        new_past.append((k_masked, v_masked))
                    past_key_values = tuple(new_past)
                    if attention_mask is not None:
                        attention_mask = attention_mask.masked_fill(eos_mask.unsqueeze(-1), 0.0)
                        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)], dim=-1)
                else:
                    position_ids = position_ids + 1
                    if attention_mask is not None:
                        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)], dim=-1)
            else:
                position_ids = position_ids + 1
                if attention_mask is not None:
                    attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)], dim=-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
            "position_ids": position_ids
        }

    def forward(self, input_ids, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=None, labels=None,
                cu_seqlens=None, max_seqlen=None, **kwargs):
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if position_ids is None and self.training and cu_seqlens is None:
            eos_mask_2d = (input_ids == self.config.eos_token_id)
            start_mask = torch.zeros_like(eos_mask_2d)
            start_mask[:, 1:] = eos_mask_2d[:, :-1]
            start_mask[:, 0] = True
            global_pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
            reset_anchors = torch.cummax(global_pos * start_mask.long(), dim=1)[0]
            position_ids = global_pos - reset_anchors

        outputs = self.model(
            input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
            cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            flat_logits = logits[..., :-1, :].contiguous().view(-1, logits.size(-1))
            flat_labels = labels[..., 1:].contiguous().view(-1)
            if HAS_FUSED_CE:
                ce_loss = FusedCrossEntropyLoss(ignore_index=-100)(flat_logits, flat_labels)
            else:
                ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(flat_logits, flat_labels)

            valid_mask = flat_labels != -100
            if valid_mask.any():
                lse = torch.logsumexp(flat_logits[valid_mask].float(), dim=-1)
                z_loss = 1e-5 * torch.square(lse).mean()
            else:
                z_loss = 0.0

            aux_logits = self.aux_reg(hidden_states, self.lm_head)
            flat_aux_logits = aux_logits[..., :-2, :].contiguous().view(-1, aux_logits.size(-1))
            flat_aux_labels = labels[..., 2:].contiguous().view(-1)
            shifted_labels_t1 = labels[..., 1:-1].contiguous().view(-1)
            valid_aux_mask = (flat_aux_labels != -100) & (shifted_labels_t1 != self.config.eos_token_id)

            if valid_aux_mask.any():
                if HAS_FUSED_CE:
                    aux_loss = FusedCrossEntropyLoss(ignore_index=-100)(
                        flat_aux_logits[valid_aux_mask], flat_aux_labels[valid_aux_mask])
                else:
                    aux_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                        flat_aux_logits[valid_aux_mask], flat_aux_labels[valid_aux_mask])
                loss = ce_loss + z_loss + 0.1 * aux_loss
            else:
                loss = ce_loss + z_loss

        return JaddangiCausalLMOutput(
            loss=loss, logits=logits, past_key_values=outputs[1],
            past_position_ids=position_ids
        )

# ============================================================
# MMAP DATA LOADER
# ============================================================
class BlockShuffledMemmapDataset(IterableDataset):
    def __init__(self, bin_path, seq_len, eos_id):
        super().__init__()
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.eos_id = eos_id
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.num_blocks = len(self.data) // seq_len
        self.indices = np.arange(self.num_blocks)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            self.indices = self.indices[worker_info.id::worker_info.num_workers]
        while True:
            np.random.shuffle(self.indices)
            for idx in self.indices:
                start = idx * self.seq_len
                chunk = self.data[start : start + self.seq_len].astype(np.int64)

                eos_indices = np.where(chunk == self.eos_id)[0]
                boundaries = np.concatenate([[0], eos_indices + 1, [self.seq_len]])
                cu_seqlens = np.unique(boundaries).astype(np.int32)
                max_seqlen = int(np.max(cu_seqlens[1:] - cu_seqlens[:-1]))
                assert max_seqlen > 0, "Zero-length segment encountered"

                labels = chunk.copy()
                for seg_start in cu_seqlens[1:-1]:
                    labels[seg_start] = -100

                yield {
                    "input_ids": chunk,
                    "labels": labels,
                    "attention_mask": np.ones(self.seq_len, dtype=np.float32),
                    "cu_seqlens": cu_seqlens,
                    "max_seqlen": max_seqlen
                }

# ============================================================
# COLLATOR
# ============================================================
def simple_collator(features):
    input_ids = torch.stack([torch.tensor(f["input_ids"], dtype=torch.long) for f in features])
    labels = torch.stack([torch.tensor(f["labels"], dtype=torch.long) for f in features])
    batch = {"input_ids": input_ids, "labels": labels}

    if "attention_mask" in features[0]:
        batch["attention_mask"] = torch.stack([torch.tensor(f["attention_mask"], dtype=torch.float32) for f in features])

    if "cu_seqlens" in features[0]:
        seqlens_list = []
        offset = 0
        max_seqlen = 0
        for f in features:
            cu_seq = np.array(f["cu_seqlens"])
            max_seqlen = max(max_seqlen, f["max_seqlen"])
            if len(seqlens_list) > 0:
                cu_seq = cu_seq[1:]
            seqlens_list.append(cu_seq + offset)
            offset += int(cu_seq[-1])
        batch["cu_seqlens"] = torch.tensor(np.concatenate(seqlens_list), dtype=torch.int32)
        batch["max_seqlen"] = max_seqlen
    return batch

# ============================================================
# TRAINER WITH NAN GUARD, WEIGHT DECAY, EMA
# ============================================================
class JaddangiTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        if torch.isnan(loss) or torch.isinf(loss) or (loss.abs() > 1e6):
            print(f"⚠️ Invalid loss ({loss.item():.2f}). Skipping batch.")
            return (torch.tensor(0.0, device=loss.device, requires_grad=True), outputs) if return_outputs else torch.tensor(0.0, device=loss.device, requires_grad=True)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        decay_params = []
        no_decay_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            if p.ndim < 2 or "norm" in n or "bias" in n:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        optimizer_grouped_parameters = [
            {"params": decay_params, "weight_decay": self.args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer_cls = bnb.optim.AdamW8bit if bnb is not None else torch.optim.AdamW
        self.optimizer = optimizer_cls(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2)
        )
        return self.optimizer

class EMACallback(TrainerCallback):
    """Maintains an Exponential Moving Average of model weights."""
    def __init__(self, decay=0.999):
        self.decay = decay
        self.ema_state = {}
        self.enabled = decay > 0

    def on_train_begin(self, args, state, control, model, **kwargs):
        if self.enabled:
            self.ema_state = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

    def on_step_end(self, args, state, control, model, **kwargs):
        if self.enabled and state.global_step % args.gradient_accumulation_steps == 0:
            with torch.no_grad():
                for n, p in model.named_parameters():
                    if p.requires_grad and n in self.ema_state:
                        self.ema_state[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def on_train_end(self, args, state, control, model, **kwargs):
        if self.enabled:
            print("🔄 Loading EMA weights into model...")
            for n, p in model.named_parameters():
                if n in self.ema_state:
                    p.data.copy_(self.ema_state[n])

# ============================================================
# SAMPLE CALLBACK (optional)
# ============================================================
class SampleCallback(TrainerCallback):
    def __init__(self, tokenizer, prompt="The magic forest was", every_n_steps=500):
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if self.every_n_steps <= 0 or state.global_step % self.every_n_steps != 0 or state.global_step == 0:
            return
        model = kwargs["model"]
        if hasattr(model, "module"): model = model.module
        model.eval()
        was_chkpt = getattr(model.config, "gradient_checkpointing", False)
        if was_chkpt: model.gradient_checkpointing_disable()
        try:
            inputs = self.tokenizer(self.prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=60, do_sample=True, temperature=0.7, use_cache=True)
            print(f"\n🔹 STEP {state.global_step}\n{self.tokenizer.decode(out[0], skip_special_tokens=True)}\n")
        except Exception as e: print(f"\n⚠️ Generation error: {e}\n")
        finally:
            if was_chkpt: model.gradient_checkpointing_enable()
            model.train()

# ============================================================
# MAIN PIPELINE
# ============================================================
print("Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({"eos_token": "<|endoftext|>"})
    ACTUAL_VOCAB_SIZE = len(tokenizer)
    eos_id = tokenizer.eos_token_id
    print(f"Tokenizer loaded. Vocab size = {ACTUAL_VOCAB_SIZE}, EOS id = {eos_id}")
except Exception:
    print(f"⚠️ Cannot load tokenizer from '{TOKENIZER_NAME}'. Using fallback vocab size {VOCAB_SIZE}.")
    ACTUAL_VOCAB_SIZE = VOCAB_SIZE
    eos_id = 2

assert ACTUAL_VOCAB_SIZE < 65536, f"Vocab size {ACTUAL_VOCAB_SIZE} exceeds uint16 limit – data corruption risk!"

config = JaddangiIndependentConfig(
    vocab_size=ACTUAL_VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_attention_heads=NUM_HEADS,
    num_key_value_heads=NUM_KV_HEADS,
    intermediate_size=INTERMEDIATE_SIZE,
    max_position_embeddings=MAX_POSITION_EMBEDDINGS,
    rope_theta=ROPE_THETA,
    attn_dropout=ATTN_DROPOUT,
    rms_norm_eps=RMSNORM_EPS,
    eos_token_id=eos_id,
    attn_logit_softcapping=SOFTCAP_VALUE,
)

print("Initialising JADDANGI-ALFA-1.6.1 model (350M scale)...")
model = JaddangiIndependentForCausalLM(config)
model.resize_token_embeddings(ACTUAL_VOCAB_SIZE)
model.tie_weights()
model.config.use_cache = False
model.gradient_checkpointing_enable()

if ENABLE_COMPILE:
    print("⚠️ torch.compile requested but may break with dynamic FA. Use with caution.")
    model = torch.compile(model, mode="reduce-overhead")

model.generation_config = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id if tokenizer else 0,
    eos_token_id=eos_id,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    max_new_tokens=80,
)

if not os.path.exists(MMAP_DATA_FILE):
    raise FileNotFoundError(f"Data file '{MMAP_DATA_FILE}' not found.")

print(f"Loading block‑shuffled MMAP dataset: {MMAP_DATA_FILE}")
dataset = BlockShuffledMemmapDataset(MMAP_DATA_FILE, MAX_LENGTH, config.eos_token_id)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    max_grad_norm=MAX_GRAD_NORM,
    bf16=use_bf16,
    fp16=not use_bf16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    dataloader_num_workers=0,
    remove_unused_columns=False,
    report_to="none",
)

callbacks = []
if EMA_DECAY > 0:
    callbacks.append(EMACallback(decay=EMA_DECAY))
# Uncomment to enable text generation samples
# callbacks.append(SampleCallback(tokenizer, every_n_steps=SAMPLE_EVERY))

trainer = JaddangiTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=simple_collator,
    callbacks=callbacks,
)

print("🚀 Starting JADDANGI-ALFA-1.6.1 training (350M)...")
try:
    trainer.train()
except KeyboardInterrupt:
    print("🛑 Interrupted. Saving final state...")
finally:
    unwrapped_model = trainer.accelerator.unwrap_model(model)
    unwrapped_model.config.use_cache = True
    unwrapped_model.save_pretrained(OUTPUT_DIR)
    if tokenizer is not None:
        tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Model saved to {OUTPUT_DIR}")
