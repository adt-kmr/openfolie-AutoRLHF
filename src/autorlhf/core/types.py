"""Shared value types used across the pipeline, training engine, and TUI.

Kept deliberately small and JSON-serializable: everything here crosses a
SQLite/JSONL boundary or a thread boundary at some point.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from autorlhf.core.enums import TrainingMethod

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    """One chat turn.  ``role`` is restricted to the three roles HF chat
    templates understand."""

    role: Role
    content: str

    def validate_for_training(self) -> None:
        if not self.content.strip():
            raise ValueError(f"empty {self.role} message content")


# --- hardware --------------------------------------------------------------- #


class HardwareInfo(BaseModel):
    """Result of a runtime preflight inspection.

    Fields set from measurement carry the measured value; fields that are
    estimates are documented as such (see :attr:`estimate_notes`) so the UI can
    label them honestly.
    """

    #: ``cuda`` | ``rocm`` | ``mps`` | ``xpu`` | ``cpu``
    device_type: str = "cpu"
    device_name: str = ""
    #: Number of accelerator devices usable by torch.
    device_count: int = 0
    #: Total VRAM in MiB, summed over devices.  ``None`` when not queryable.
    total_vram_mib: int | None = None
    #: Free VRAM in MiB at inspection time.  ``None`` when not queryable.
    free_vram_mib: int | None = None
    #: System RAM in MiB (measured via the OS).
    total_ram_mib: int | None = None
    available_ram_mib: int | None = None
    cpu_count: int = 1
    platform: str = "linux"
    #: e.g. ``"2.5.1+cu124"``
    torch_version: str | None = None
    cuda_version: str | None = None
    bf16_supported: bool = False
    fp16_supported: bool = False
    #: True when bitsandbytes imported *and* can quantize on this device.
    bitsandbytes_available: bool = False
    bitsandbytes_version: str | None = None
    #: True when flash-attn (or an equivalent fused-attention kernel) is usable.
    flash_attention_available: bool = False
    #: Libraries that were importable, e.g. ``{"trl": "0.13.0"}``.
    packages: dict[str, str] = Field(default_factory=dict)
    #: Human-readable caveats, e.g. "VRAM reported by torch; other processes
    #: may already be using it", or "RAM figures are measured, VRAM estimates
    #: are derived".
    estimate_notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def has_accelerator(self) -> bool:
        return self.device_type not in ("cpu",) and self.device_count > 0

    def supports(self, method: TrainingMethod) -> tuple[bool, str]:
        """Whether ``method`` is runnable here, with a specific reason if not."""

        if method is TrainingMethod.QLORA:
            if not self.has_accelerator:
                return False, "QLoRA requires a CUDA/ROCm GPU; none detected."
            if not self.bitsandbytes_available:
                return False, (
                    "QLoRA requires the optional `bitsandbytes` extra "
                    "(`uv sync --extra qlora`) and a compatible GPU."
                )
            if self.device_type == "mps":
                return False, "4-bit quantization is not supported on Apple MPS."
            return True, ""
        if method is TrainingMethod.FFT:
            if not self.has_accelerator:
                return False, "Full fine-tuning requires a GPU; none detected."
            if (self.total_vram_mib or 0) < 16 * 1024 and not self.estimate_notes:
                return False, "Full fine-tuning needs at least ~16 GiB VRAM."
            return True, ""
        return True, ""


class ResourceEstimate(BaseModel):
    """Approximate resource requirement for a training configuration.

    Every field here is a *model-derived estimate*, never a measurement.
    """

    required_vram_mib: int
    required_ram_mib: int
    trainable_parameters: int | None = None
    total_parameters: int | None = None
    #: Which assumptions the estimate rests on, e.g. ``"optimizer=adamw_8bit"``.
    assumptions: list[str] = Field(default_factory=list)


# --- generation ------------------------------------------------------------- #


class GenerationRequest(BaseModel):
    """A single local-model generation to run."""

    model_id: str
    prompt_messages: list[ChatMessage]
    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    seed: int | None = None
    #: Stable id so cached generations can be matched across resume.
    request_id: str = ""


class GenerationResult(BaseModel):
    request_id: str = ""
    model_id: str = ""
    text: str = ""
    #: ``local`` (HF model) or the OpenRouter model slug.
    source: str = "local"
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_s: float | None = None
    cached: bool = False
    error: str | None = None
    #: Whether the generation was truncated at ``max_new_tokens``.
    truncated: bool = False
    meta: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ChatMessage",
    "GenerationRequest",
    "GenerationResult",
    "HardwareInfo",
    "ResourceEstimate",
    "Role",
]
