from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run inference, evaluation, and embedding extraction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required ---
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to CSV mapping samples to image file paths.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory for all predict.py outputs.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file (same format as train.py).")

    # --- Weight source (mutually exclusive) ---
    weight_group = parser.add_mutually_exclusive_group(required=True)
    weight_group.add_argument(
        "--weights", type=str, default=None,
        help=(
            "Path to a single .h5 weights file. "
            "Runs inference on the full CSV (no fold splitting)."
        ),
    )
    weight_group.add_argument(
        "--training_dir", type=str, default=None,
        help=(
            "Path to a train.py output directory. "
            "Discovers per-fold weights and re-evaluates each fold on its "
            "original test set using the seed from run_manifest.json."
        ),
    )

    # --- Model overrides ---
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["BinarySFCN", "RegressionSFCN"],
                        help="Overrides config.model.backbone if set.")
    parser.add_argument("--backbone_weights", type=str, default=None,
                        help=(
                            "Pre-trained backbone weights to initialise "
                            "BEFORE loading the fine-tuned --weights. "
                            "Usually not needed — omit unless you need it."
                        ))

    # --- Embedding layer override ---
    parser.add_argument(
        "--embedding_layer", type=str, default=None,
        help=(
            "Substring of the layer name to use for embedding extraction. "
            "Overrides config.inference.embedding_layer_name. "
            "Pass 'none' to disable embedding extraction."
        ),
    )

    # --- Misc ---
    parser.add_argument("--seed", type=int, default=None,
                        help=(
                            "Random seed for dataset construction. "
                            "Overrides the seed in run_manifest.json when "
                            "using --training_dir. Auto-generated if None "
                            "and no manifest is found."
                        ))
    parser.add_argument("--verbose", type=int, default=2,
                        choices=[0, 1, 2],
                        help="Keras verbosity level for predict/evaluate calls.")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = build_parser()
    args   = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Output directory
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from aispycore.utils.logging import setup_logging, write_log
    log_file = setup_logging(output_dir, args.model_type or "predict")

    # ------------------------------------------------------------------
    # 2. Config
    # ------------------------------------------------------------------
    from aispycore.config.schema import PipelineConfig, HyperparameterSet
    config = PipelineConfig.from_yaml(args.config)

    cli_overrides: dict = {}
    if args.model_type is not None:
        cli_overrides["model.backbone"] = args.model_type
    if args.backbone_weights is not None:
        cli_overrides["model.backbone_weights"] = args.backbone_weights
    if cli_overrides:
        config = config.resolve(HyperparameterSet(values=cli_overrides))

    # Resolve embedding layer name: CLI > config > None
    embedding_layer = args.embedding_layer
    if embedding_layer is None:
        embedding_layer = config.inference.embedding_layer_name
    if isinstance(embedding_layer, str) and embedding_layer.lower() == "none":
        embedding_layer = None

    write_log(log_file,
        f"predict.py starting\n"
        f"  config           : {args.config}\n"
        f"  backbone         : {config.model.backbone}\n"
        f"  task_type        : {config.model.task_type}\n"
        f"  embedding_layer  : {embedding_layer or '(disabled)'}"
    )

    # ------------------------------------------------------------------
    # 3. Data
    # ------------------------------------------------------------------
    import pandas as pd
    df = pd.read_csv(args.csv_file)
    write_log(log_file, f"Loaded {len(df)} rows from {args.csv_file}")

    # ------------------------------------------------------------------
    # 4. Dispatch to the right mode
    # ------------------------------------------------------------------
    if args.weights is not None:
        _run_single(
            config=config,
            df=df,
            weights_path=Path(args.weights),
            output_dir=output_dir,
            log_file=log_file,
            embedding_layer=embedding_layer,
            seed=args.seed or 42,
            verbose=args.verbose,
        )
    else:
        _run_from_training_dir(
            config=config,
            df=df,
            training_dir=Path(args.training_dir),
            output_dir=output_dir,
            log_file=log_file,
            embedding_layer=embedding_layer,
            seed_override=args.seed,
            verbose=args.verbose,
        )

    write_log(log_file, f"All outputs written to: {output_dir}")


# ---------------------------------------------------------------------------
# Single-weights mode
# ---------------------------------------------------------------------------

def _run_single(
    config,
    df,
    weights_path: Path,
    output_dir: Path,
    log_file: str,
    embedding_layer: str | None,
    seed: int,
    verbose: int,
) -> None:
    """Evaluate a single weights file on the full df."""
    from aispycore.utils.logging import write_log

    write_log(log_file, f"Single-weights mode: {weights_path}")

    model, batch_size = _load_model(config, weights_path, log_file)

    ds, steps = _make_dataset(
        df=df,
        output_dir=output_dir,
        name="predict",
        config=config,
        batch_size=batch_size,
        seed=seed,
    )

    metrics, predictions_df, embeddings_df = _evaluate_and_embed(
        model=model,
        dataset=ds,
        df=df,
        steps=steps,
        config=config,
        embedding_layer=embedding_layer,
        log_file=log_file,
        verbose=verbose,
    )

    _save_fold_outputs(
        fold_dir=output_dir,
        metrics=metrics,
        predictions_df=predictions_df,
        embeddings_df=embeddings_df,
        log_file=log_file,
    )


# ---------------------------------------------------------------------------
# Training-directory mode
# ---------------------------------------------------------------------------

def _run_from_training_dir(
    config,
    df,
    training_dir: Path,
    output_dir: Path,
    log_file: str,
    embedding_layer: str | None,
    seed_override: int | None,
    verbose: int,
) -> None:
    """
    Discover fold weight files in training_dir, re-create the same splits,
    and evaluate each fold on its original test set.
    """
    import pandas as pd
    from aispycore.evaluation.results import aggregate_fold_results
    from aispycore.training.schemes.base import FoldResult
    from aispycore.training.trainer import TrainingResult
    from aispycore.utils.logging import write_log

    # ── Read seed from run manifest ────────────────────────────────────
    seed = seed_override
    manifest_path = training_dir / "run_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest_seed = manifest.get("cli_args", {}).get("seed_used")
        if seed is None and manifest_seed is not None:
            seed = int(manifest_seed)
            write_log(log_file, f"Seed read from run manifest: {seed}")
    if seed is None:
        import random
        seed = random.randint(0, 2**31 - 1)
        write_log(log_file, f"No seed found — using random seed: {seed}")

    # ── Discover fold weight files ─────────────────────────────────────
    fold_dirs = sorted(training_dir.glob("fold_*/"))
    if not fold_dirs:
        raise FileNotFoundError(
            f"No fold_XX/ directories found under {training_dir}. "
            "Is --training_dir pointing to the right place?"
        )

    write_log(log_file, f"Found {len(fold_dirs)} fold directories in {training_dir}")

    # ── Re-create the same stratified splits ──────────────────────────
    from aispycore.data.stratification import StratificationManager
    label_col = config.data.target_col
    stratifier = StratificationManager.from_config(
        config.cross_validation, target_col=label_col, random_state=seed
    )

    fold_results = []
    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(stratifier.split(df)):
        fold_dir_src = training_dir / f"fold_{fold_idx:02d}"
        weights_path = fold_dir_src / "training_checkpoint.weights.h5"

        if not weights_path.exists():
            write_log(log_file,
                f"WARNING: No weights found at {weights_path} — skipping fold {fold_idx}")
            continue

        write_log(log_file,
            f"\nFold {fold_idx + 1}/{len(fold_dirs)}: "
            f"test={len(test_idx)} samples  weights={weights_path}")

        test_df = df.iloc[test_idx].reset_index(drop=True)
        fold_out = output_dir / f"fold_{fold_idx:02d}"
        fold_out.mkdir(parents=True, exist_ok=True)

        model, batch_size = _load_model(config, weights_path, log_file)

        ds, steps = _make_dataset(
            df=test_df,
            output_dir=fold_out,
            name=f"fold_{fold_idx}_test",
            config=config,
            batch_size=batch_size,
            seed=seed,
        )

        metrics, predictions_df, embeddings_df = _evaluate_and_embed(
            model=model,
            dataset=ds,
            df=test_df,
            steps=steps,
            config=config,
            embedding_layer=embedding_layer,
            log_file=log_file,
            verbose=verbose,
        )

        metrics["fold"]    = fold_idx
        metrics["n_train"] = len(train_idx)
        metrics["n_val"]   = len(val_idx)
        metrics["n_test"]  = len(test_idx)

        _save_fold_outputs(
            fold_dir=fold_out,
            metrics=metrics,
            predictions_df=predictions_df,
            embeddings_df=embeddings_df,
            log_file=log_file,
        )

        # Wrap in FoldResult so aggregate_fold_results() can consume it
        import pandas as _pd
        fold_results.append(FoldResult(
            fold_index=fold_idx,
            training_result=TrainingResult(
                history_df=_pd.DataFrame(),
                best_weights_path=weights_path,
                final_epoch=0,
            ),
            metrics=metrics,
            predictions_df=predictions_df,
        ))

    if not fold_results:
        raise RuntimeError("No folds were successfully evaluated.")

    write_log(log_file, "All folds evaluated. Aggregating results...")
    aggregate_fold_results(
        fold_results,
        output_dir,
        id_col=config.data.id_col,
    )


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _load_model(config, weights_path: Path, log_file: str):
    """Build model architecture, load fine-tuned weights, return (model, batch_size)."""
    from aispycore.models.builder import build_model
    from aispycore.utils.logging import write_log

    write_log(log_file, f"Building model ({config.model.backbone})...")
    model = build_model(config)

    write_log(log_file, f"Loading weights from: {weights_path}")
    model.load_weights(str(weights_path))

    # Inference batch size: use inference override if set, else training default
    batch_size = config.inference.batch_size or config.data.batch_size

    return model, batch_size


def _make_dataset(df, output_dir: Path, name: str, config, batch_size: int, seed: int):
    """Build a non-repeating, non-shuffled tf.data.Dataset for inference."""
    from aispycore.data.dataset import make_dataset_from_df

    steps = math.ceil(len(df) / batch_size)
    ds = make_dataset_from_df(
        sub_df=df,
        dir=str(output_dir),
        name=name,
        target_col=config.data.target_col,
        batch_size=batch_size,
        shuffle=False,
        repeat=False,
        seed=seed,
    )
    return ds, steps


def _evaluate_and_embed(
    model,
    dataset,
    df,
    steps: int,
    config,
    embedding_layer: str | None,
    log_file: str,
    verbose: int,
) -> tuple[dict, object, object | None]:
    """
    Run evaluation and (optionally) embedding extraction.

    Returns:
        (metrics, predictions_df, embeddings_df | None)
    """
    from aispycore.evaluation.evaluators import get_evaluator
    from aispycore.training.schemes.base import task_from_config
    from aispycore.utils.logging import write_log

    task = task_from_config(config)
    evaluator = get_evaluator(task)

    write_log(log_file, "Running evaluation...")
    metrics, predictions_df = evaluator.evaluate(model, dataset, df, steps)
    write_log(log_file, f"Metrics: {metrics}")

    # ── Embedding extraction ───────────────────────────────────────────
    embeddings_df = None
    if embedding_layer:
        from aispycore.evaluation.embeddings import EmbeddingExtractor
        try:
            extractor = EmbeddingExtractor(model, layer_name_substr=embedding_layer)
            # Dataset must be re-iterable — make_dataset_from_df with repeat=False is.
            embeddings_df = extractor.extract(dataset, df, steps, config)
            write_log(log_file,
                f"Embeddings extracted: {len(embeddings_df)} samples, "
                f"{sum(1 for c in embeddings_df.columns if c.startswith('embedding_'))} dims")
        except (ValueError, RuntimeError) as e:
            write_log(log_file, f"WARNING: Embedding extraction failed: {e}")

    return metrics, predictions_df, embeddings_df


def _save_fold_outputs(
    fold_dir: Path,
    metrics: dict,
    predictions_df,
    embeddings_df,
    log_file: str,
) -> None:
    """Write predictions.csv, metrics.csv, and embeddings.csv to fold_dir."""
    import pandas as pd
    from aispycore.utils.logging import write_log

    predictions_df.to_csv(fold_dir / "predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(fold_dir / "metrics.csv", index=False)

    if embeddings_df is not None:
        embeddings_df.to_csv(fold_dir / "embeddings.csv", index=False)
        write_log(log_file, f"  embeddings.csv → {fold_dir / 'embeddings.csv'}")

    write_log(log_file,
        f"  predictions.csv → {fold_dir / 'predictions.csv'}\n"
        f"  metrics.csv     → {fold_dir / 'metrics.csv'}"
    )


if __name__ == "__main__":
    main()
