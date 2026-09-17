"""Course Lesson 7: FaultLine's S2 model design, and what the record says it costs.

    python scripts/design_model.py

This script builds nothing and trains nothing. It reads configs/model/course/run_01.yaml, a
course view of the ADR-0021 gate run's S2 backbone, and multiplies a few numbers. It prints them
beside what the project actually measured. The model itself already exists: it is
``TelemetryDecoder`` in src/faultline/model/transformer.py.
"""

from __future__ import annotations

from pathlib import Path

from faultline.model.design import (
    GATE_RUN_GPU_HOURS,
    GATE_RUN_TOKENS,
    GPU,
    MEASURED_BACKBONE_PARAMETERS,
    MEASURED_TOTAL_PARAMETERS,
    PRETRAINING_ONLY_GPU_HOURS,
    STEPS_PER_DAY,
    TELEMETRY_TRAIN_STEPS,
    TELEMETRY_TRAIN_TOKENS,
    TEXT_TRAIN_TOKENS,
    load_config,
)

CONFIG_FILE = Path("configs/model/course/run_01.yaml")
GATE_REPORT = "reports/data/gate_check_v0_20260916.md"
GATE_CHECKPOINT = "checkpoints/gate_check_v0_9697a266/S2_tel_only_seed1.pt"
TEXT_LADDER = "docs/ROADMAP.md correction 2026-09-16; reports/data/m2_gate6_20260916.md"


def main() -> None:
    """Print the design, its estimates, the measured figures and the reasons."""
    config = load_config(CONFIG_FILE)
    params = config.estimate_parameters()
    d, layers = config.n_embd, config.n_layer

    print("=== FaultLine S2 model design (course view of the ADR-0021 gate run) ===")
    print()
    print("Vocabulary size    :", config.vocab_size, "(ADR-0003 v2 joint layout capacity)")
    print("Context length     :", config.block_size, "(configs/train/joint_v0.yaml)")
    print("Embedding dimension:", config.n_embd)
    print("Layers             :", config.n_layer)
    print("Attention heads    :", config.n_head, "(head dim", config.head_dim(), ")")
    print("Dropout            :", config.dropout)
    print("FFN multiplier     :", config.ffn_mult, "(src/faultline/model/transformer.py)")
    print("Tied weights       :", config.tie_weights, "(src/faultline/model/transformer.py)")
    print()

    total = params["total"]
    gap = MEASURED_TOTAL_PARAMETERS - total
    norms = 2 * d * layers + d
    print("Estimated parameters (rough formula)")
    print(f"  non-embedding :  {params['non_embedding']:>12,}")
    print(f"  embedding     :  {params['embedding']:>12,}")
    print(f"  position      :  {params['position']:>12,}")
    print(f"  TOTAL         :  {total:>12,}   (~{total / 1e6:.1f}M)")
    print(f"  measured      :  {MEASURED_TOTAL_PARAMETERS:>12,}   (state dict of the gate")
    print(f"                                   checkpoint, {GATE_CHECKPOINT})")
    print(f"  difference    :  {gap:>12,}   = RMSNorm weights: 2 x {d} per block x {layers}")
    print(f"                                   blocks + the final {d} = {norms:,}")
    print()

    memory = config.estimate_training_memory_mb()
    print(f"Rough training memory for weights + gradients + AdamW: {memory:.0f} MB (estimate)")
    print("(a floor: activations add more on top, and they grow with batch size)")
    print(f"measured: S2, {GATE_RUN_TOKENS:,} tokens, {GATE_RUN_GPU_HOURS} GPU-h wall clock,")
    print(f"          pretraining + probe, on an {GPU} ({GATE_REPORT})")
    print(f"measured: ~{PRETRAINING_ONLY_GPU_HOURS} GPU-h for a 50M-token S2 arm's pretraining")
    print("          alone, without validation (ADR-0018)")
    print()

    tokens = TELEMETRY_TRAIN_TOKENS
    total_measured = MEASURED_TOTAL_PARAMETERS
    backbone = MEASURED_BACKBONE_PARAMETERS
    days = TELEMETRY_TRAIN_STEPS / STEPS_PER_DAY
    print("Tokens per parameter")
    print(f"  telemetry training tokens: {tokens:,}")
    print("    (reports/data/shards_v2_20260912.md; docs/ROADMAP.md M1b)")
    print(f"  {tokens:,} / {total_measured:,} total parameters    = {tokens / total_measured:.1f}")
    print(f"  {tokens:,} / {backbone:,} backbone parameters  = {tokens / backbone:.1f}")
    print("    (backbone count: configs/model/ladder_v0.yaml)")
    print(
        f"  {params['embedding'] / 1e6:.1f}M of the {total_measured / 1e6:.2f}M parameters are "
        f"the {config.vocab_size:,} x {d} joint embedding table,"
    )
    print("  sized for a text region tel_only never emits:")
    print('  "The configured capacities are sized for a corpus several times larger than the')
    print('  licence-clean route could produce."')
    print("    (docs/ROADMAP.md; reports/data/m2_gate6_20260916.md)")
    print("  text ladder, recorded: S2 saw 0.98 and S3 0.50 tokens per parameter;")
    print('  "a ladder over a corpus of about 10M tokens cannot demonstrate scaling"')
    print(f"    ({TEXT_LADDER})")
    print(
        f"  arithmetic on shards_v2 step count: {TELEMETRY_TRAIN_STEPS:,} ten-minute steps "
        f"= {days:,.0f} turbine-days ~ {days / 365:.2f} turbine-years"
    )
    print("  The ~20 tokens-per-parameter figure is a natural-language heuristic; no project")
    print("  record establishes the right ratio for quantised telemetry. Context, not a target.")
    print("  Budget policy: arms are compared at equal tokens seen; GPU-hours are an")
    print("  observation, never the bound (ADR-0018).")
    print()

    print("The size ladder (configs/model/ladder_v0.yaml; docs/ROADMAP.md; ADR-0018)")
    print("  rung |   d |  L | heads | 12*L*d^2 (rough) | rough total at this vocab and context")
    for row in config.ladder_table():
        print(
            f"  {row['rung']:>4} | {row['d']:>3} | {row['L']:>2} | {row['heads']:>5} | "
            f"{row['non_embedding']:>16,} | {row['total']:>12,}"
        )
    print()

    print("Why this model?")
    print()
    print(f"- The corpus: {tokens:,} telemetry and {TEXT_TRAIN_TOKENS:,} text training tokens")
    print("  are all the arms can draw from.")
    print(f"- The GPU: one {GPU}, the card every run in this record was made on")
    print("  (docs/ENVIRONMENT.md).")
    print("- Comparability: S2 is fixed for all three arms for comparability, not performance")
    print("  (ADR-0018).")
    print(f"- The cost: a full-budget arm measured at {GATE_RUN_GPU_HOURS} GPU-h wall clock,")
    print(f"  pretraining and probe ({GATE_REPORT}).")
    print()
    print("A bigger model is not automatically a better model.")
    print("The model is already built: TelemetryDecoder in src/faultline/model/transformer.py.")


if __name__ == "__main__":
    main()
