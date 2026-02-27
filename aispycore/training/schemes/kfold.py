from __future__ import annotations

import pandas as pd
import math

from aispycore.data.augmentation import AugmentationPipeline
from aispycore.data.dataset import make_dataset_from_df, check_no_overlap
from aispycore.data.stratification import StratificationManager
from aispycore.evaluation.evaluators import get_evaluator
from aispycore.models.builder import build_model
from aispycore.training.schemes.base import BaseTrainingScheme, FoldResult
from aispycore.training.trainer import get_trainer
from aispycore.training.tuning import get_tuner, get_pretuned_hps
from aispycore.utils.logging import write_log

class KFoldScheme(BaseTrainingScheme):

    def run(self, df: pd.DataFrame) -> list[FoldResult]:
        config = self.config
        label_col = config.data.target_col
        output_dir = self.output_dir
        augmentation = AugmentationPipeline.from_config(config.augmentation)

        stratifier = StratificationManager.from_config(
            config.cross_validation, random_state=self.seed, target_col=label_col
        )

        fold_results: list[FoldResult] = []
        n_folds = config.cross_validation.n_folds

        for fold_idx, (train_idx, val_idx, test_idx) in enumerate(stratifier.split(df)):
            print(f"\n{'='*60}")
            print(f"  Fold {fold_idx + 1} / {n_folds}")
            print(f"  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")
            print(f"{'='*60}")

            fold_dir = self._make_fold_output_dir(fold_idx)

            train_df = df.iloc[train_idx]
            val_df   = df.iloc[val_idx]
            test_df  = df.iloc[test_idx]

            # Guard against data leakage before any data is loaded
            check_no_overlap(train_df, val_df,  id_col="id", fold_num=fold_idx, log_file=self.log_file)
            check_no_overlap(train_df, test_df, id_col="id", fold_num=fold_idx, log_file=self.log_file)
            check_no_overlap(val_df,   test_df, id_col="id", fold_num=fold_idx, log_file=self.log_file)

            # steps
            steps_per_epoch = math.ceil(len(train_df) / config.data.batch_size)
            validation_steps = math.ceil(len(val_df) / config.data.batch_size)
            test_steps = math.ceil(len(test_df) / config.data.batch_size)
            write_log(self.log_file, f"Training Steps: {steps_per_epoch} | Val Steps: {validation_steps} | Test Steps: {test_steps}")
            write_log(self.log_file, f"Train class distribution: {train_df[label_col].value_counts().to_dict()}")
            write_log(self.log_file, f"Val class distribution: {val_df[label_col].value_counts().to_dict()}")
            write_log(self.log_file, f"est class distribution: {test_df[label_col].value_counts().to_dict()}")
                
            
            train_ds = make_dataset_from_df(
                sub_df=train_df,
                dir=output_dir,
                name=f"fold_{fold_idx}_train",
                target_col=config.data.target_col,
                batch_size=config.data.batch_size,
                shuffle=True,
                repeat=True,
                seed=config.training.seed,
            )
            val_ds = make_dataset_from_df(
                sub_df=val_df,
                dir=output_dir,
                name=f"fold_{fold_idx}_val",
                target_col=config.data.target_col,
                batch_size=config.data.batch_size,
                shuffle=False,
                repeat=False,
                seed=config.training.seed,
            )
            test_ds = make_dataset_from_df(
                sub_df=test_df,
                dir=output_dir,
                name=f"fold_{fold_idx}_test",
                target_col=config.data.target_col,
                batch_size=config.data.batch_size,
                shuffle=False,
                repeat=False,
                seed=config.training.seed,
            )

            # Optional tuning — test set is never seen by the tuner
            fold_config = config
            if config.tuning.enabled:
                best_hps = get_pretuned_hps(config, fold_idx)
                if best_hps is not None:
                    write_log(self.log_file,
                        f"Fold {fold_idx}: using pre-tuned HPs from config: {best_hps.values}")
                else:
                    write_log(self.log_file, f"Tuning model")
                    tuner = get_tuner(config, fold_dir, self.log_file)
                    best_hps = tuner.tune(train_dataset=train_ds, 
                                        train_steps=steps_per_epoch,
                                        val_dataset=val_ds, 
                                        val_steps=validation_steps)
                    fold_config = config.resolve(best_hps)
                    write_log(self.log_file, f"  Tuning complete. Best HPs: {best_hps.values}")

            # build_model reads backbone, task_type, weights, loss_fn, etc.
            # from fold_config — no model_cls or input_shape needed
            model = build_model(fold_config)

            trainer_cls = get_trainer(fold_config.training.trainer_type)
            trainer = trainer_cls(fold_config, fold_dir)
            training_result = trainer.train(model=model, 
                                            train_dataset=train_ds, 
                                            train_steps=steps_per_epoch, 
                                            val_dataset=val_ds,
                                            val_steps=validation_steps)
            write_log(self.log_file, "Training Complete. Running evaluation.")
            # Reload best weights saved by ModelCheckpoint callback
            model.load_weights(str(training_result.best_weights_path))

            evaluator = get_evaluator(self.task)
            metrics, predictions_df = evaluator.evaluate(model, test_ds, test_df, test_steps)

            write_log(self.log_file, "Evaluation Complete:")
            write_log(self.log_file, metrics)

            metrics["fold"]    = fold_idx
            metrics["n_train"] = len(train_idx)
            metrics["n_val"]   = len(val_idx)
            metrics["n_test"]  = len(test_idx)

            training_result.history_df.to_csv(fold_dir / "history.csv", index=False)
            predictions_df.to_csv(fold_dir / "predictions.csv")
            pd.DataFrame([metrics]).to_csv(fold_dir / "metrics.csv", index=False)

            fold_results.append(FoldResult(
                fold_index=fold_idx,
                training_result=training_result,
                metrics=metrics,
                predictions_df=predictions_df,
            ))

        return fold_results