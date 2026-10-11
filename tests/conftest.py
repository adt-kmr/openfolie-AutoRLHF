"""Shared pytest fixtures for autorlhf.

Everything here is offline: no network access, no model downloads, no GPU.  The
tiny model and tokenizer are built from scratch in a temp directory so that real
``AutoTokenizer.from_pretrained`` / ``AutoModelForCausalLM.from_pretrained`` calls
(and therefore the real training path) can be exercised in unit tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from autorlhf.config.character import DEFAULT_CHARACTER, CharacterSpec, SeedConversation
from autorlhf.config.paths import AppPaths
from autorlhf.config.rubric import default_rubric
from autorlhf.config.run_config import RunConfig
from autorlhf.core.enums import FeedbackMode, TrainingMethod, TrainingObjective
from autorlhf.core.events import EventBus
from autorlhf.core.types import ChatMessage
from autorlhf.storage.db import RunStore

# --------------------------------------------------------------------------- #
# paths / storage
# --------------------------------------------------------------------------- #


@pytest.fixture()
def app_paths(tmp_path: Path) -> AppPaths:
    """An isolated config/data layout rooted in ``tmp_path``."""

    return AppPaths.for_tests(tmp_path).ensure()


@pytest.fixture()
def store(tmp_path: Path) -> Iterator[RunStore]:
    store = RunStore(tmp_path / "autorlhf.db")
    yield store
    store.close()


@pytest.fixture()
def bus() -> EventBus:
    return EventBus()


# --------------------------------------------------------------------------- #
# config builders
# --------------------------------------------------------------------------- #


@pytest.fixture()
def character() -> CharacterSpec:
    return DEFAULT_CHARACTER.model_copy(deep=True)


@pytest.fixture()
def minimal_character() -> CharacterSpec:
    return CharacterSpec(
        name="Test",
        system_prompt="You are a terse test assistant.",
        behavioral_constraints=["Never mention being an AI."],
        seed_conversations=[
            SeedConversation(
                messages=[
                    ChatMessage(role="user", content="Say hi."),
                    ChatMessage(role="assistant", content="Hi."),
                ]
            )
        ],
    )


@pytest.fixture()
def rubric():
    return default_rubric()


@pytest.fixture()
def run_config(minimal_character: CharacterSpec) -> RunConfig:
    config = RunConfig(
        slug="test-run",
        feedback_mode=FeedbackMode.AUTOMATED,
        character=minimal_character,
        rubric=default_rubric(),
    )
    config.generation.num_prompts = 10
    config.generation.candidates_per_prompt = 2
    config.training.objective = TrainingObjective.DPO
    config.training.method = TrainingMethod.LORA
    config.training.lora_target_modules = ["c_attn"]
    config.training.max_steps = 2
    return config


# --------------------------------------------------------------------------- #
# offline tiny model + tokenizer
# --------------------------------------------------------------------------- #

#: Vocabulary size of the hand-built tokenizer (printable ASCII plus room for
#: the special tokens).
TINY_VOCAB_SIZE = 260


def build_tiny_tokenizer(out_dir: Path) -> Path:
    """Write a real, loadable tokenizer directory with no downloads.

    Uses a character-level ``WordLevel`` backend wrapped in
    :class:`transformers.PreTrainedTokenizerFast`, which needs no
    ``tokenizer.json`` from the hub.  A chat template is included so the chat
    path can be exercised for real.  Model context is 512 tokens — still tiny,
    but large enough that DPO's prompt+completion concatenation and local
    generation never overrun the position embeddings (a 128-token context does).
    """

    from tokenizers import Tokenizer, models, pre_tokenizers

    out_dir.mkdir(parents=True, exist_ok=True)
    alphabet = [chr(i) for i in range(32, 127)]
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    vocab["?"] = len(vocab)  # unk / pad
    vocab["~"] = len(vocab)  # eos
    vocab["^"] = len(vocab)  # bos

    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="?"))
    # Empty pattern with "isolated" behaviour splits on every character.  No
    # decoder is installed: single-character tokens concatenate back exactly.
    backend.pre_tokenizer = pre_tokenizers.Split(pattern="", behavior="isolated")
    tokenizer_path = out_dir / "tokenizer.json"
    backend.save(str(tokenizer_path))

    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token="?",
        pad_token="?",
        eos_token="~",
        bos_token="^",
        model_max_length=512,
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: "
        "{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] }}{% endgeneration %}"
        "{% else %}{{ message['content'] }}{% endif %}"
        "\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}assistant: {% endif %}"
    )
    tokenizer.save_pretrained(str(out_dir))
    return out_dir


def build_tiny_model(
    out_dir: Path,
    *,
    architecture: str = "gpt2",
    num_hidden_layers: int = 2,
    hidden_size: int = 32,
    num_attention_heads: int = 2,
    vocab_size: int = TINY_VOCAB_SIZE,
) -> Path:
    """Save a tiny causal LM of the requested family to ``out_dir``.

    ``architecture="gpt2"`` exercises the fused ``c_attn``/``c_proj`` target-module
    path; ``architecture="llama"`` exercises the ``q_proj``/``k_proj``/... path.
    """

    out_dir.mkdir(parents=True, exist_ok=True)
    if architecture == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=512,
            n_embd=hidden_size,
            n_layer=num_hidden_layers,
            n_head=num_attention_heads,
            bos_token_id=vocab_size - 1,
            eos_token_id=vocab_size - 2,
        )
        model = GPT2LMHeadModel(config)
    else:
        from transformers import LlamaConfig, LlamaForCausalLM

        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_attention_heads,
            max_position_embeddings=512,
            pad_token_id=vocab_size - 1,
            eos_token_id=vocab_size - 2,
        )
        model = LlamaForCausalLM(config)

    model.save_pretrained(str(out_dir), safe_serialization=False)
    return out_dir


@pytest.fixture(scope="session")
def tiny_tokenizer_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_tokenizer(tmp_path_factory.mktemp("tiny-tok"))


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_model(tmp_path_factory.mktemp("tiny-gpt2"))


@pytest.fixture(scope="session")
def tiny_llama_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return build_tiny_model(tmp_path_factory.mktemp("tiny-llama"), architecture="llama")


@pytest.fixture(scope="session")
def tiny_chat_model_dir(tmp_path_factory: pytest.TempPathFactory, tiny_tokenizer_dir: Path) -> Path:
    """A tiny model directory that also contains the tokenizer files.

    This is what a real HF repo looks like, so ``AutoTokenizer.from_pretrained``
    and ``AutoModelForCausalLM.from_pretrained`` can both be pointed at it.
    """

    out = tmp_path_factory.mktemp("tiny-chat-model")
    build_tiny_model(out)
    for source in tiny_tokenizer_dir.iterdir():
        if source.is_file():
            (out / source.name).write_bytes(source.read_bytes())
    return out


@pytest.fixture()
def hf_model_config(tiny_chat_model_dir: Path):
    from autorlhf.config.run_config import HFModelConfig

    return HFModelConfig(
        model_id=str(tiny_chat_model_dir),
        torch_dtype="float32",
        attn_implementation="eager",
    )


# --------------------------------------------------------------------------- #
# Jev wire fixtures (the exact documented shapes)
# --------------------------------------------------------------------------- #


@pytest.fixture()
def jev_score_response() -> dict:
    return {
        "id": "gen-dec-1790265859-EaXKSt7hul1Wcqots1ZK",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": {
            "character_consistency": {
                "type": "score",
                "score": 3.2,
                "legend": {
                    "0": "Badly violates the specification.",
                    "4": "Excellent: fully consistent.",
                },
                "probabilities": {"0": 0.0, "1": 0.05, "2": 0.2, "3": 0.4, "4": 0.35},
                "confidence": 0.72,
            },
            "voice_style": {
                "type": "score",
                "score": 4.0,
                "probabilities": {"0": 0.0, "1": 0.0, "2": 0.02, "3": 0.13, "4": 0.85},
                "confidence": 0.91,
            },
        },
        "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992},
    }


@pytest.fixture()
def jev_choice_response() -> dict:
    return {
        "id": "gen-dec-1",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": {
            "preference": {
                "type": "choice",
                "choice": "b",
                "probabilities": {"a": 0.19, "b": 0.81, "tie": 0.0},
                "confidence": 0.67,
            }
        },
        "usage": {"input_tokens": 512, "output_tokens": 12, "cost": 0.00002},
    }


@pytest.fixture()
def jev_noul_response() -> dict:
    return {
        "id": "gen-dec-2",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "answers": {"condition": {"type": "noul", "noul": 0.96}},
        "usage": {"input_tokens": 100, "output_tokens": 4, "cost": 0.000004},
    }
