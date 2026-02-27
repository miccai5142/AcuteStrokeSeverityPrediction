import argparse
import sys
from pathlib import Path

from aispycore.utils.logging import setup_logging, write_log

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an MRI deep learning model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Required ---
    parser.add_argument("--csv_file", type=str, required=True,
                        help="Path to CSV mapping patient data to image filepaths.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root directory for all outputs.")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file.")

    # --- Model overrides (CLI takes precedence over config file) ---
    parser.add_argument(
        "--model_type", type=str, default=None,
        choices=["BinarySFCN", "RegressionSFCN"],
        help="Pyment backbone. Overrides config.model.backbone if set.",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to pre-trained pyment weights. "
             "Overrides config.model.backbone_weights if set.",
    )

    # --- Training scheme ---
    parser.add_argument("--scheme", type=str, default="kfold",
                        choices=["kfold", "holdout"],
                        help="Training scheme to use.")

    # --- Parallelism / reproducibility ---
    parser.add_argument("--repeat_number", type=int, default=None,
                        help="Repeat index for parallelised multi-repeat runs.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Global random seed. Auto-generated and logged if None.")

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Seed — must be set before importing TensorFlow
    # ------------------------------------------------------------------
    import random as _random
    seed = args.seed if args.seed is not None else _random.randint(0, 2**31 - 1)

    from aispycore.utils.reproducibility import set_global_seed
    set_global_seed(seed)
    
    # ------------------------------------------------------------------
    # 2. Output directory
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    if args.repeat_number is not None:
        output_dir = output_dir / f"repeat_{args.repeat_number:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)


    log_file = setup_logging(output_dir, args.model_type)
    write_log(log_file, f"Global seed: {seed}")

    # ------------------------------------------------------------------
    # 3. Load config
    # ------------------------------------------------------------------
    from aispycore.config.schema import PipelineConfig, HyperparameterSet
    config = PipelineConfig.from_yaml(args.config)

    # Apply CLI overrides via resolve() so the config stays immutable
    cli_overrides: dict = {}
    cli_overrides["training.seed"] = seed  # always log the seed used, even if auto-generated
    if args.model_type is not None:
        cli_overrides["model.backbone"] = args.model_type
    if args.weights is not None:
        cli_overrides["model.backbone_weights"] = args.weights
    if cli_overrides:
        config = config.resolve(HyperparameterSet(values=cli_overrides))

    write_log(
        log_file,
        f"Config loaded from: {args.config}\n"
        f"  backbone    : {config.model.backbone}\n"
        f"  task_type   : {config.model.task_type}\n"
        f"  weights     : {config.model.backbone_weights}\n"
        f"  loss_fn     : {config.training.loss_fn}\n"
        f"  optimizer   : {config.training.optimizer}"
    )

    # ------------------------------------------------------------------
    # 4. Load data
    # ------------------------------------------------------------------
    import pandas as pd
    df = pd.read_csv(args.csv_file)
    write_log(log_file, f"Loaded {len(df)} rows from {args.csv_file}")
    write_log(log_file, df.head())
    # ------------------------------------------------------------------
    # 5. Save run manifest (before training starts)
    # ------------------------------------------------------------------
    from aispycore.utils.logging import save_run_manifest
    save_run_manifest(output_dir, config, vars(args) | {"seed_used": seed})

    # ------------------------------------------------------------------
    # 6. Run training scheme
    # ------------------------------------------------------------------
    from aispycore.training.schemes import get_scheme
    scheme = get_scheme(args.scheme)(
        config=config,
        output_dir=output_dir,
        log_file=log_file,
        seed=seed,
    )
    fold_results = scheme.run(df)
    write_log(log_file, "Training and evaluation complete. Aggregating results...")
    # ------------------------------------------------------------------
    # 7. Aggregate results
    # ------------------------------------------------------------------
    from aispycore.evaluation import aggregate_fold_results
    summary_df = aggregate_fold_results(
        fold_results,
        output_dir,
        id_col=config.data.id_col,
        repeat=args.repeat_number,
    )

    write_log(log_file, f"\nAll outputs written to: {output_dir}")
    # return summary_df


if __name__ == "__main__":
    main()