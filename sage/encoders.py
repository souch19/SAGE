"""Encodeurs multi-vecteurs. Interface commune :
    enc.encode_doc(image) -> [N, dim] fp16 (CPU, L2-normalisé)
    enc.encode_query(text) -> [Lq, dim] fp16

colpali, colqwen, colnomic : colpali-engine        jinav4 : transformers (trust_remote_code)
colbert, AGC               : omni-col-press (cloné dans $OMNI_COL_PRESS, transformers 4.54.1)
"""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from sage.config import AGC_ID, MODELS
from sage.utils import DEVICE

MAX_PIXELS, MIN_PIXELS = 1003520, 614656      # résolution officielle Qwen-VL (ColBERT, AGC)
QUERY_PREFIX, PASSAGE_PREFIX = "Query: ", "Passage: "


def _attn():
    try:
        from transformers.utils.import_utils import is_flash_attn_2_available
        return "flash_attention_2" if is_flash_attn_2_available() else "sdpa"
    except Exception:
        return "sdpa"


def _out(e):
    return F.normalize(e.float(), dim=-1).cpu().half()


class ColPaliEncoder:
    def __init__(self, cfg):
        from colpali_engine import models
        self.model = getattr(models, cfg["cls"]).from_pretrained(
            cfg["id"], torch_dtype=torch.bfloat16, device_map=DEVICE,
            attn_implementation=_attn()).eval()
        self.proc = getattr(models, cfg["proc"]).from_pretrained(cfg["id"])

    @torch.inference_mode()
    def _run(self, batch):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            return _out(self.model(**batch.to(DEVICE))[0])

    def encode_doc(self, image):
        return self._run(self.proc.process_images([image.convert("RGB")]))

    def encode_query(self, text):
        return self._run(self.proc.process_queries([text]))


class JinaEncoder:
    def __init__(self, cfg):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(
            cfg["id"], trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation=_attn()).to(DEVICE).eval()
        try:
            self.model.task = "retrieval"
        except Exception:
            pass
        self.dim = cfg["dim"]

    @torch.inference_mode()
    def encode_doc(self, image):
        out = self.model.encode_image(images=[image.convert("RGB")],
                                      return_multivector=True, truncate_dim=self.dim)
        return _out(out[0])

    @torch.inference_mode()
    def encode_query(self, text):
        out = self.model.encode_text(texts=[text], task="retrieval",
                                     return_multivector=True, truncate_dim=self.dim)
        return _out(out[0])


def _load_omni():
    """Rend omni-col-press importable et contourne un bug de transformers 4.54.1
    (to_json_string échoue sur les torch.dtype)."""
    sys.path.insert(0, os.environ.get("OMNI_COL_PRESS", str(Path.home() / "omni-col-press")))
    import transformers.configuration_utils as cu
    import transformers.generation.configuration_utils as gcu

    def clean(d):
        if isinstance(d, dict):
            return {k: clean(v) for k, v in d.items()}
        if isinstance(d, (list, tuple)):
            return [clean(v) for v in d]
        return str(d).replace("torch.", "") if isinstance(d, torch.dtype) else d

    def to_json_string(self, *a, **k):
        d = {k: v for k, v in self.to_dict().items() if k != "transformers_version"}
        return json.dumps(clean(d), indent=2, sort_keys=True) + "\n"

    gcu.GenerationConfig.to_json_string = to_json_string
    cu.PretrainedConfig.to_json_string = to_json_string


class OmniEncoder:
    """ColBERT Qwen2.5-VL (tous les patchs) ou, avec agc=True, AGC (k vecteurs par page,
    k réglable à l'encodage grâce à la transférabilité de compression)."""

    def __init__(self, model_id, agc=False, n_proxy=64):
        _load_omni()
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor
        from src.arguments import ModelArguments
        from src.models.qwen2_5_vl_embed.qwen2_5_vl_embed import Qwen2_5ForEmbedding
        attn = _attn()
        if agc:
            from src.encoder.select_encoder import AttentionSelectEncoder as Encoder
            from src.utils import get_appending_token_strings
            args = ModelArguments(model_name_or_path=model_id, pooling="select", normalize=True,
                                  num_appending_token=n_proxy, use_cluster_pooling=True,
                                  use_attn_weight_cluster_pooling=True, attn_implementation=attn)
            self.suffix = "".join(get_appending_token_strings(n_proxy))
        else:
            from src.encoder.multivec_encoder import MultiVecEncoder as Encoder
            args = ModelArguments(model_name_or_path=model_id, pooling="colbert", normalize=True,
                                  num_appending_token=0, attn_implementation=attn)
            self.suffix = ""
        self.model = Encoder.load(Qwen2_5ForEmbedding, args, attn_implementation=attn,
                                  dtype=torch.bfloat16).to(DEVICE).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.vision = process_vision_info

    @torch.inference_mode()
    def embed(self, inputs, is_query=False):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            emb, mask = self.model.encode(inputs.to(DEVICE), is_query=is_query)
        return _out(emb[0][mask[0].bool()] if mask is not None else emb[0])

    def doc_inputs(self, image):
        msg = [{"role": "user", "content": [
            {"type": "text", "text": PASSAGE_PREFIX},
            {"type": "image", "image": image, "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS}]}]
        text = self.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        images, videos = self.vision(msg)
        return self.proc(text=[text + self.suffix], images=images, videos=videos,
                         padding=True, return_tensors="pt")

    def encode_doc(self, image, k=None):
        if k is not None:
            self.model.num_select_tokens = int(max(k, 1))
        return self.embed(self.doc_inputs(image))

    def encode_query(self, text):
        msg = [{"role": "user", "content": [{"type": "text", "text": QUERY_PREFIX + text}]}]
        text = self.proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
        return self.embed(self.proc(text=[text], padding=True, return_tensors="pt"), is_query=True)


def build_encoder(model):
    """Encodeur d'un modèle du registre, ou d'AGC avec model='agc'."""
    if model == "agc":
        return OmniEncoder(AGC_ID, agc=True)
    cfg = MODELS[model]
    if cfg["loader"] == "colpali":
        return ColPaliEncoder(cfg)
    if cfg["loader"] == "jina":
        return JinaEncoder(cfg)
    return OmniEncoder(cfg["id"])
