"""Course Lesson 7: the design view of FaultLine's S2 model.

There is no model in this file. The real model is the decoder in
``src/faultline/model/transformer.py`` (``TelemetryDecoder``, built from ``ModelSpec``), and this
dataclass does not build it. This file writes the S2 decisions down in the course's eight fields,
with enough arithmetic to show what they cost.

    Architecture
          |
    Parameter count
          |
    Memory requirement
          |
    Training feasibility

**Which rung, and why S2.** The size ladder (S0 to S3, by d_model, n_layer and n_head) has no ADR
of its own. Its source of record is ``configs/model/ladder_v0.yaml`` (M1e, with its 12·L·d² and
measured-backbone table), ``docs/ROADMAP.md`` and ADR-0018. S2 is the rung ADR-0018's resource
ruling fixes for all three M3 arms, "for comparability, not performance". It is the only rung
trained at the full per-arm budget and gated: the ADR-0021 gate run, ``tel_only`` seed 1,
50,003,968 tokens.

Every default below is the value the gate run's backbone was built with, and each comment names
its source. ``tests/model/test_course_design.py`` pins them to those sources.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

#: The ladder's source of record (M1e); ``ladder_table`` reads it.
LADDER_CONFIG = Path("configs/model/ladder_v0.yaml")

# -- recorded figures, each read from the artefact named beside it ------------------------------

#: Telemetry training steps, ``reports/data/shards_v2_20260912.md`` ("training steps").
TELEMETRY_TRAIN_STEPS = 4_738_594
#: Telemetry training tokens, ``docs/ROADMAP.md`` M1b step 12 and ADR-0018's stream table.
TELEMETRY_TRAIN_TOKENS = 61_601_722
#: Text training tokens, ``reports/data/m2_gate6_20260916.md`` and ADR-0018.
TEXT_TRAIN_TOKENS = 10_043_874
#: Tokens the gate run pretrained on, ADR-0021 (erratum: 763 steps x 32 windows x 2,048).
GATE_RUN_TOKENS = 50_003_968
#: Parameters in the gate run's saved backbone, counted from its state dict
#: (``checkpoints/gate_check_v0_9697a266/S2_tel_only_seed1.pt``).
MEASURED_TOTAL_PARAMETERS = 10_454_208
#: The same backbone excluding embeddings, ``configs/model/ladder_v0.yaml`` rung table (S2).
MEASURED_BACKBONE_PARAMETERS = 3_542_208
#: Gate run wall clock, pretraining and probe, ``reports/data/gate_check_v0_20260916.md``.
GATE_RUN_GPU_HOURS = 0.27
#: A 50M-token S2 arm's pretraining alone, validation excluded, measured in ADR-0018.
PRETRAINING_ONLY_GPU_HOURS = 0.15
#: The card, ``docs/ENVIRONMENT.md``.
GPU = "NVIDIA GeForce RTX 4060, 8 GB"
#: Grid steps a site reports per turbine per day: ten-minute steps (ADR-0006 canonical grid).
STEPS_PER_DAY = 144


@dataclass
class ModelConfig:
    """The eight numbers that define the S2 model, as the ADR-0021 gate run built it."""

    # Identifiers the embedding covers: the full joint vocabulary of ADR-0003 v2, fixed-capacity
    # blocks specials [0, 32) · channel [32, 96) · bin [96, 1120) · time [1120, 1184) ·
    # text [1184, 33952) (src/faultline/tokenizers/layout.py). It is the layout's capacity, not a
    # count of tokens observed. The gate checkpoint's spec records vocab_size 33,952, even though
    # tel_only never emits a text id.
    vocab_size: int = 33_952

    # Context in tokens: configs/train/joint_v0.yaml `context_tokens`. That file says: "2,048: the
    # M2 text context (text_shards_v1.yaml), used for every stream so a window is the same length
    # whatever it is drawn from. Pure telemetry fills it in 157.5 steps, more than M1's 144-step
    # context." The 144 x 13 = 1,872 tokens of configs/model/ladder_v0.yaml (`context_steps`) is
    # the separate M1 ladder context and the risk probe's window, not this backbone's context.
    block_size: int = 2_048

    # Residual width: configs/model/ladder_v0.yaml, rung S2 `d_model`.
    n_embd: int = 192

    # Blocks: configs/model/ladder_v0.yaml, rung S2 `n_layer`.
    n_layer: int = 8

    # Attention heads: configs/model/ladder_v0.yaml, rung S2 `n_head` (head dimension 24, which
    # clears the fused kernel's divisible-by-8 requirement that file documents).
    n_head: int = 8

    # configs/model/ladder_v0.yaml `dropout`.
    dropout: float = 0.0

    # Not a configuration key: src/faultline/model/transformer.py builds the feed-forward layer
    # as Linear(d_model, 4 * d_model) and back.
    ffn_mult: int = 4

    # Not a configuration key: src/faultline/model/transformer.py ties the output head to the
    # token embedding ("The output head is tied to the token embedding").
    tie_weights: bool = True

    def head_dim(self) -> int:
        """Size of one attention head."""
        return self.n_embd // self.n_head

    def check(self) -> None:
        """Catch the two mistakes that would break the model on line one.

        The ADR-0003 v2 layout's 33,952 identifiers fit the ``uint16`` token files.

        Raises:
            ValueError: If the width does not split across the heads, or the vocabulary does not
                fit ``uint16``.
        """
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must divide evenly by n_head ({self.n_head})")
        if self.vocab_size > 65536:
            raise ValueError("vocab_size above 65536 does not fit in the uint16 token files")

    def estimate_parameters(self) -> dict[str, int]:
        """A rough parameter count. Deliberately rough.

        Non-embedding parameters, per transformer block, come out at about 12 * n_embd^2:

            attention  Q, K, V and the output projection  ->  4 * n_embd^2
            feed-forward  n_embd -> 4*n_embd -> n_embd    ->  8 * n_embd^2

        so the stack of blocks is roughly 12 * n_layer * n_embd^2. Biases and normalisation layers
        add a few thousand more. The real decoder has no biases, and its RMSNorm weights are
        exactly the gap to the measured count.

        The embedding table is a separate cost: vocab_size * n_embd. With tie_weights=True the
        output layer reuses it, so it is counted once.

        Returns:
            The non-embedding, embedding, position and total counts.
        """
        non_embedding = 12 * self.n_layer * self.n_embd**2
        embedding = self.vocab_size * self.n_embd
        position = self.block_size * self.n_embd  # learned position embeddings

        total = non_embedding + embedding + position
        if not self.tie_weights:
            total += self.vocab_size * self.n_embd  # a separate output layer

        return {
            "non_embedding": non_embedding,
            "embedding": embedding,
            "position": position,
            "total": total,
        }

    def estimate_training_memory_mb(self) -> float:
        """Very rough GPU memory for the weights during training.

        Each parameter is stored four times over: the weight itself, its gradient, and the two
        running averages AdamW keeps. At 4 bytes each that is 16 bytes per parameter.

        Activations cost more on top of this and depend on batch size, so treat this as a floor,
        not a budget.

        Returns:
            Megabytes.
        """
        total = self.estimate_parameters()["total"]
        return total * 16 / (1024**2)

    def ladder_table(self, path: Path = LADDER_CONFIG) -> list[dict[str, Any]]:
        """The size ladder's rungs, read from ``configs/model/ladder_v0.yaml``.

        That file is the ladder's source of record (with ``docs/ROADMAP.md`` and ADR-0018); no ADR
        defines the rungs. The rows carry the file's own d_model, n_layer and n_head. The rough
        counts come from ``estimate_parameters`` at this config's vocabulary and context, so they
        are the rungs as the joint-vocabulary backbone would size them. No number is added.

        Args:
            path: The ladder configuration.

        Returns:
            One row a rung, in the file's order: rung, d, L, heads, and the rough non-embedding
            and total parameter counts.
        """
        with open(path, encoding="utf-8") as handle:
            ladder = yaml.safe_load(handle)
        rows = []
        for rung in ladder["rungs"]:
            view = ModelConfig(
                vocab_size=self.vocab_size,
                block_size=self.block_size,
                n_embd=int(rung["d_model"]),
                n_layer=int(rung["n_layer"]),
                n_head=int(rung["n_head"]),
                dropout=float(ladder["dropout"]),
                ffn_mult=self.ffn_mult,
                tie_weights=self.tie_weights,
            )
            estimate = view.estimate_parameters()
            rows.append(
                {
                    "rung": str(rung["name"]),
                    "d": view.n_embd,
                    "L": view.n_layer,
                    "heads": view.n_head,
                    "non_embedding": estimate["non_embedding"],
                    "total": estimate["total"],
                }
            )
        return rows

    def to_dict(self) -> dict[str, Any]:
        """The eight fields as a dictionary."""
        return asdict(self)


def load_config(path: Path) -> ModelConfig:
    """Read a ModelConfig from a YAML file.

    The config lives in a file and not inside a .py file for one reason: six weeks from now, the
    only question that matters is "which settings produced this checkpoint?", and a filename can
    answer it. This course view names the files the real checkpoint was produced from.

    Args:
        path: The YAML file.

    Returns:
        The checked configuration.
    """
    with open(path, encoding="utf-8") as handle:
        values = yaml.safe_load(handle)

    config = ModelConfig(**values)
    config.check()
    return config
