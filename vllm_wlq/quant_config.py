from typing import Any, Dict, List, Optional
import torch

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from .linear_method import WLQLinearMethod


@register_quantization_config("wlq_fastshot")
@register_quantization_config("wlq_fastShot")
class WLQFastShotConfig(QuantizationConfig):
    """Konfigurationsklasse für die Metalog FastShot Quantisierung (wlq_fastShot)."""

    def __init__(
        self,
        bits: int = 4,
        outlier_fraction: float = 0.05,
        chunk_m: int = 512,
        use_dora: bool = True,
        model_id: Optional[str] = None,
    ):
        self.bits = bits
        self.outlier_fraction = outlier_fraction
        self.chunk_m = 640 if bits == 3 else chunk_m
        self.use_dora = use_dora
        self.model_id = model_id or "wichtel-labs/Qwen3.8-27B-WLQ-FastShot"

    def get_name(self) -> str:
        return "wlq_fastShot"

    def get_supported_act_dtypes(self) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["/config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "WLQFastShotConfig":
        q_cfg = config.get("quantization_config")
        if not q_cfg:
            q_cfg = config

        bits = q_cfg.get("bits", 4)
        outlier_frac = q_cfg.get("outlier_fraction", 0.05)
        chunk_m = 640 if bits == 3 else q_cfg.get("chunk_m", 512)
        use_dora = q_cfg.get("use_dora", True)
        model_id = config.get("_name_or_path", "wichtel-labs/Qwen3.8-27B-WLQ-FastShot")
        return cls(
            bits=bits,
            outlier_fraction=outlier_frac,
            chunk_m=chunk_m,
            use_dora=use_dora,
            model_id=model_id,
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> Optional[Any]:
        if not isinstance(layer, LinearBase):
            return None

        if (
            prefix.startswith("model.visual")
            or "visual" in prefix
            or "blocks." in prefix
            or "lm_head" in prefix
        ):
            return UnquantizedLinearMethod()

        unquantized_substrings = ("conv1d", "router", "shared_expert_gate")
        if any(s in prefix for s in unquantized_substrings):
            return UnquantizedLinearMethod()

        layer.vllm_prefix = prefix
        return WLQLinearMethod(self)

    def get_scaled_act_names(self) -> List[str]:
        return []
