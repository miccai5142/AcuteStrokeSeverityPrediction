import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    balanced_accuracy_score, roc_auc_score,
    precision_score, recall_score, log_loss
)
from iterstrat.ml_stratifiers import (
    MultilabelStratifiedKFold,
    MultilabelStratifiedShuffleSplit
)
import xgboost as xgb
from itertools import product


def train_multistrat_kfold(
    df,
    target_col,
    stratify_col=None,
    stratify_cols=None,
    drop_stratify_cols=None,
    n_splits=5,
    random_state=42,
    param_grid=None,
    weight_classes=False,
    binning_threshold=10,
    tuning_metric="loss",
    val_size=0.2,
    verbose=False,
    tune_model=False
):

    # ------------------------------------------------------------
    # Validate metric
    # ------------------------------------------------------------
    valid_metrics = {
        "loss", "accuracy", "balanced_accuracy",
        "precision", "recall", "f1_score",
        "auc", "sensitivity", "specificity"
    }

    if tuning_metric not in valid_metrics:
        raise ValueError(f"tuning_metric must be one of {valid_metrics}")

    maximize_metrics = {
        "accuracy", "balanced_accuracy",
        "precision", "recall", "f1_score",
        "auc", "sensitivity", "specificity"
    }

    # ------------------------------------------------------------
    # Default parameter grid
    # ------------------------------------------------------------
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }

    # ------------------------------------------------------------
    # Resolve stratification columns
    # ------------------------------------------------------------
    if stratify_col is not None and stratify_cols is None:
        stratify_cols = stratify_col

    if stratify_cols is None:
        stratify_cols = [target_col]

    if isinstance(stratify_cols, str):
        stratify_cols = [stratify_cols]

    is_multilabel = len(stratify_cols) > 1

    # ------------------------------------------------------------
    # Feature matrix
    # ------------------------------------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if drop_stratify_cols is not None:
        if isinstance(drop_stratify_cols, str):
            drop_stratify_cols = [drop_stratify_cols]
        X = X.drop(columns=[c for c in drop_stratify_cols if c in X.columns])

    # ------------------------------------------------------------
    # Build stratification matrix (mirrors StratificationManager)
    # ------------------------------------------------------------
    df_work = df.copy()
    binned_col_map = {}

    for col in stratify_cols:
        if df[col].nunique() > binning_threshold:
            bin_col = f"__bin_{col}"
            series = df[col]
            zero_mask = series == 0

            df_work[bin_col] = np.nan
            df_work.loc[zero_mask, bin_col] = 0

            if (~zero_mask).any():
                non_zero_bins = pd.qcut(
                    series[~zero_mask],
                    q=5,
                    labels=False,
                    duplicates="drop"
                )
                df_work.loc[~zero_mask, bin_col] = non_zero_bins + 1

            df_work[bin_col] = df_work[bin_col].astype(int)
            binned_col_map[col] = bin_col

    if not is_multilabel:
        col = stratify_cols[0]
        use_col = binned_col_map.get(col, col)
        stratify_data = df_work[use_col].to_numpy()
    else:
        encoded_blocks = []
        for col in stratify_cols:
            use_col = binned_col_map.get(col, col)
            one_hot = pd.get_dummies(df_work[use_col], prefix=col)
            encoded_blocks.append(one_hot)
        stratify_data = pd.concat(encoded_blocks, axis=1).to_numpy()

    # ------------------------------------------------------------
    # Outer K-Fold (rotating test set)
    # ------------------------------------------------------------
    if is_multilabel:
        outer_kf = MultilabelStratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )
    else:
        outer_kf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state
        )

    param_keys = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combos = list(product(*param_values))

    fold_predictions = {}
    metrics_list = []
    best_params_tracker = {}
    
    all_true = []
    all_pred = []
    all_proba = []

    # ------------------------------------------------------------
    # K-Fold loop
    # ------------------------------------------------------------
    for fold, (train_outer_idx, test_idx) in enumerate(
        outer_kf.split(np.arange(len(df)), stratify_data), 1
    ):

        # ---- Carve validation from outer train (mirrors manager) ----
        if is_multilabel:
            splitter = MultilabelStratifiedShuffleSplit(
                n_splits=1,
                test_size=val_size,
                random_state=random_state
            )
        else:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=val_size,
                random_state=random_state
            )

        rel_idx = np.arange(len(train_outer_idx))
        outer_labels = stratify_data[train_outer_idx]

        rel_train, rel_val = next(splitter.split(rel_idx, outer_labels))

        train_inner_idx = train_outer_idx[rel_train]
        val_idx = train_outer_idx[rel_val]

        # ---- Construct datasets ----
        X_train = X.iloc[train_inner_idx]
        y_train = y.iloc[train_inner_idx]

        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]

        X_train_full = X.iloc[train_outer_idx]
        y_train_full = y.iloc[train_outer_idx]

        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        
        

        # ------------------------------------------------------------
        # Hyperparameter tuning (single validation split)
        # ------------------------------------------------------------
        best_score = -np.inf if tuning_metric in maximize_metrics else np.inf
        best_params = None
        scale_pos_weight = 1.0

        if weight_classes:
            # compute from INNER training split (for tuning)
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()

            if n_pos > 0:
                scale_pos_weight = n_neg / n_pos
            else:
                scale_pos_weight = 1.0

            if verbose:
                print(f"Class counts (train): neg={n_neg}, pos={n_pos}")
                print(f"Using scale_pos_weight={scale_pos_weight:.4f}")
        
        
        if tune_model:
            
            for combo in param_combos:
                params = dict(zip(param_keys, combo))
                params["eval_metric"] = "logloss"
                params["random_state"] = random_state
                if weight_classes:
                    params["scale_pos_weight"] = scale_pos_weight
                    
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train)

                y_val_pred = model.predict(X_val)
                y_val_proba = model.predict_proba(X_val)[:, 1]

                if tuning_metric == "loss":
                    score = log_loss(y_val, model.predict_proba(X_val))
                elif tuning_metric == "accuracy":
                    score = accuracy_score(y_val, y_val_pred)
                elif tuning_metric == "balanced_accuracy":
                    score = balanced_accuracy_score(y_val, y_val_pred)
                elif tuning_metric == "precision":
                    score = precision_score(y_val, y_val_pred, zero_division=0)
                elif tuning_metric == "recall":
                    score = recall_score(y_val, y_val_pred, zero_division=0)
                elif tuning_metric == "f1_score":
                    score = f1_score(y_val, y_val_pred, zero_division=0)
                elif tuning_metric == "auc":
                    score = roc_auc_score(y_val, y_val_proba)
                elif tuning_metric == "sensitivity":
                    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
                    score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                elif tuning_metric == "specificity":
                    tn, fp, fn, tp = confusion_matrix(y_val, y_val_pred).ravel()
                    score = tn / (tn + fp) if (tn + fp) > 0 else 0.0

                if tuning_metric in maximize_metrics:
                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                else:
                    if score < best_score:
                        best_score = score
                        best_params = params.copy()

            best_params_tracker[fold] = best_params
        else:
            best_params = {}
            best_params_tracker[fold] = best_params
        
        # ------------------------------------------------------------
        # Recompute class weight using full outer train
        # ------------------------------------------------------------
        
        final_scale_pos_weight = 1.0

        if weight_classes:
            n_pos_full = (y_train_full == 1).sum()
            n_neg_full = (y_train_full == 0).sum()

            if n_pos_full > 0:
                final_scale_pos_weight = n_neg_full / n_pos_full
            else:
                final_scale_pos_weight = 1.0

            if verbose:
                print(f"Full train class counts: neg={n_neg_full}, pos={n_pos_full}")
                print(f"Final scale_pos_weight={final_scale_pos_weight:.4f}")

            best_params["scale_pos_weight"] = final_scale_pos_weight
        # ------------------------------------------------------------
        # Final model trained on full outer train
        # ------------------------------------------------------------
        
        if tune_model:
            final_model = xgb.XGBClassifier(**best_params)
        else:
            final_model = xgb.XGBClassifier(eval_metric='logloss', random_state=random_state, scale_pos_weight=final_scale_pos_weight)
        final_model.fit(X_train_full, y_train_full)

        y_pred = final_model.predict(X_test)
        y_proba = final_model.predict_proba(X_test)[:, 1]

        all_pred.extend(y_pred)
        all_proba.extend(y_proba)
        all_true.extend(y_test.values)
        
        fold_predictions[fold] = pd.DataFrame({
            "true_mRS": y_test.values,
            "predicted_mRS_binary": y_pred,
            "predicted_mRS": y_proba
        }, index=y_test.index)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        metrics_list.append({
            "fold": fold,
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            # "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "auc": roc_auc_score(y_test, y_proba),
            # "n_test": len(y_test),
            # "n_train": len(y_train_full)
        })

    
    
    
    # Overall metrics across all folds
    acc_all = accuracy_score(all_true, all_pred)
    bal_acc_all = balanced_accuracy_score(all_true, all_pred)
    f1_all = f1_score(all_true, all_pred, average='binary')
    precision_all = precision_score(all_true, all_pred, average='binary', zero_division=0)
    recall_all = recall_score(all_true, all_pred, average='binary', zero_division=0)
    auc_all = roc_auc_score(all_true, all_proba)
    tn, fp, fn, tp = confusion_matrix(all_true, all_pred).ravel()
    sensitivity_all = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity_all = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("Overall Results:")
    print(f"  Accuracy          : {acc_all:.4f}")
    print(f"  Balanced Accuracy : {bal_acc_all:.4f}")
    print(f"  Precision         : {precision_all:.4f}")
    print(f"  Recall            : {recall_all:.4f}")
    print(f"  F1 Score          : {f1_all:.4f}")
    # print(f"  Sensitivity       : {sensitivity_all:.4f}")
    print(f"  Specificity       : {specificity_all:.4f}")
    print(f"  AUC               : {auc_all:.4f}")
    print("=" * 50)

    metrics_list.append({
        'fold': 'overall',
        'accuracy': acc_all,
        'balanced_accuracy': bal_acc_all,
        'precision': precision_all,
        'recall': recall_all,
        'f1_score': f1_all,
        # 'sensitivity': sensitivity_all,
        'specificity': specificity_all,
        'auc': auc_all
    })
    metrics_df = pd.DataFrame(metrics_list).set_index("fold")

    

    # summary_df = pd.DataFrame(metrics_list).set_index('fold')
    

    return fold_predictions, metrics_df, best_params_tracker