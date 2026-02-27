from __future__ import annotations

import math

import pandas as pd

from aispycore.data.dataset import make_dataset_from_df, check_no_overlap
from aispycore.data.stratification import StratificationManager
from aispycore.evaluation.evaluators import get_evaluator
from aispycore.models.builder import build_model
from aispycore.training.schemes.base import BaseTrainingScheme, FoldResult
from aispycore.training.trainer import get_trainer
from aispycore.training.tuning import get_tuner, get_pretuned_hps
from aispycore.utils.logging import write_log


class HoldoutScheme(BaseTrainingScheme):

    def run(self, df: pd.DataFrame) -> list[FoldResult]:
        config = self.config
        label_col = config.data.target_col
        output_dir = self.output_dir

        stratifier = StratificationManager.from_config(
            config.cross_validation, target_col=label_col, random_state=self.seed
        )
        train_idx, val_idx, test_idx = stratifier.single_split(df)

        print(f"\n{'='*60}")
        print(f"  Holdout split")
        print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
        print(f"{'='*60}")

        fold_dir = self._make_fold_output_dir(0)

        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]
        test_df  = df.iloc[test_idx]

        # Guard against data leakage before any data is loaded
        check_no_overlap(train_df, val_df,  id_col="id", fold_num=0, log_file=self.log_file)
        check_no_overlap(train_df, test_df, id_col="id", fold_num=0, log_file=self.log_file)
        check_no_overlap(val_df,   test_df, id_col="id", fold_num=0, log_file=self.log_file)

        # Step counts — required because train_ds repeats indefinitely
        steps_per_epoch  = math.ceil(len(train_df) / config.data.batch_size)
        validation_steps = math.ceil(len(val_df)   / config.data.batch_size)
        test_steps       = math.ceil(len(test_df)  / config.data.batch_size)
        write_log(
            self.log_file,
            f"Training Steps: {steps_per_epoch} | "
            f"Val Steps: {validation_steps} | "
            f"Test Steps: {test_steps}"
        )
        write_log(self.log_file, f"Train class distribution: {train_df[label_col].value_counts().to_dict()}")
        write_log(self.log_file, f"Val class distribution:   {val_df[label_col].value_counts().to_dict()}")
        write_log(self.log_file, f"Test class distribution:  {test_df[label_col].value_counts().to_dict()}")

        train_ds = make_dataset_from_df(
            sub_df=train_df,
            dir=output_dir,
            name="holdout_train",
            target_col=config.data.target_col,
            batch_size=config.data.batch_size,
            shuffle=True,
            repeat=True,
            seed=config.training.seed,
        )
        val_ds = make_dataset_from_df(
            sub_df=val_df,
            dir=output_dir,
            name="holdout_val",
            target_col=config.data.target_col,
            batch_size=config.data.batch_size,
            shuffle=False,
            repeat=False,
            seed=config.training.seed,
        )
        test_ds = make_dataset_from_df(
            sub_df=test_df,
            dir=output_dir,
            name="holdout_test",
            target_col=config.data.target_col,
            batch_size=config.data.batch_size,
            shuffle=False,
            repeat=False,
            seed=config.training.seed,
        )

        # Optional tuning — test set is never seen by the tuner
        run_config = config
        if config.tuning.enabled:
            # Holdout has only one split, so fold_idx is always 0.
            best_hps = get_pretuned_hps(config, fold_idx=0)
            if best_hps is not None:
                write_log(self.log_file,
                    f"Holdout: using pre-tuned HPs from config: {best_hps.values}")
            else:
                write_log(self.log_file, "Tuning model")
                tuner = get_tuner(config, fold_dir, self.log_file)
                best_hps = tuner.tune(
                    train_dataset=train_ds,
                    train_steps=steps_per_epoch,
                    val_dataset=val_ds,
                    val_steps=validation_steps,
                )
                write_log(self.log_file, f"  Tuning complete. Best HPs: {best_hps.values}")
            run_config = config.resolve(best_hps)

        model = build_model(run_config)

        trainer_cls = get_trainer(run_config.training.trainer_type)
        trainer = trainer_cls(run_config, fold_dir)
        training_result = trainer.train(
            model=model,
            train_dataset=train_ds,
            train_steps=steps_per_epoch,
            val_dataset=val_ds,
            val_steps=validation_steps,
        )
        write_log(self.log_file, "Training complete. Running evaluation.")

        # Reload best weights saved by ModelCheckpoint callback
        model.load_weights(str(training_result.best_weights_path))

        evaluator = get_evaluator(self.task)
        metrics, predictions_df = evaluator.evaluate(model, test_ds, test_df, test_steps)

        write_log(self.log_file, "Evaluation complete:")
        write_log(self.log_file, metrics)

        metrics["fold"]    = 0
        metrics["n_train"] = len(train_idx)
        metrics["n_val"]   = len(val_idx)
        metrics["n_test"]  = len(test_idx)

        training_result.history_df.to_csv(fold_dir / "history.csv", index=False)
        predictions_df.to_csv(fold_dir / "predictions.csv", index=False)
        pd.DataFrame([metrics]).to_csv(fold_dir / "metrics.csv", index=False)

        return [FoldResult(
            fold_index=0,
            training_result=training_result,
            metrics=metrics,
            predictions_df=predictions_df,
        )]