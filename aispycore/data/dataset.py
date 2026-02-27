from typing import Dict
import numpy as np
import pandas as pd
import nibabel as nib

import tensorflow as tf

EXPECTED_SHAPE = (167, 212, 160)

def load_niftis(path):
    if isinstance(path, tf.Tensor):
        path = path.numpy()[0]
    elif isinstance(path, np.ndarray):
        path = path[0]

    path = path.decode("utf-8")
    img = nib.load(path).get_fdata()
    img = np.expand_dims(img, axis=-1)
    return img.astype(np.float32)


def load_row(row: Dict[str, tf.Tensor], target_col: str):
    image = tf.py_function(load_niftis, [row['path']], tf.float32)
    image.set_shape([167, 212, 160, 1])
    return image, row[target_col]


def configure_nifti_dataset(dataset: tf.Tensor, 
                            target_col: str, 
                            num_threads: int = 1, 
                            batch_size: int = 6, 
                            shuffle: bool = False, 
                            repeat: bool = True, 
                            seed: int = 42
                            ) -> tf.Tensor:
    dataset = dataset.map(lambda row: load_row(row, target_col), num_parallel_calls=num_threads)
    
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=4 * batch_size, reshuffle_each_iteration=True, seed=seed)
    dataset = dataset.batch(batch_size)
    # dataset = dataset.cache() # This was causing OOM errors with large datasets - commented out
    if repeat:
        dataset = dataset.repeat()

    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset



def make_dataset_from_df(sub_df: pd.DataFrame, 
                         dir: str, 
                         name: str, 
                         target_col: str, 
                         batch_size: int = 6, 
                         shuffle: bool = False, 
                         repeat: bool = True, 
                         seed: int = 42):
    fold_csv = f"{dir}/{name}_datapoints.csv"
    sub_df.to_csv(fold_csv, index=False)

    dataset = tf.data.experimental.make_csv_dataset(
        fold_csv,
        batch_size=1,
        shuffle=shuffle,
        select_columns=['path', target_col],
        column_defaults=[tf.string, tf.float32],
    )

    dataset = configure_nifti_dataset(dataset, target_col=target_col, batch_size=batch_size, shuffle=shuffle, repeat=repeat, seed=seed)
    
    return dataset





from aispycore.utils.logging import write_log

def check_no_overlap(trainVal_df, test_df, id_col="path", fold_num=None, log_file=None):
    """
    Ensures no repeated samples between trainVal_df and test_df.
    Raises ValueError if overlap detected.

    Args:
        trainVal_df (pd.DataFrame): Combined training + validation set.
        test_df (pd.DataFrame): Test set.
        id_col (str): Column identifying each sample uniquely (default: 'path').
        fold_num (int, optional): Fold number for clearer logging.
        log_file (str, optional): Log file path to record results.
    """
    overlap = set(trainVal_df[id_col]) & set(test_df[id_col])
    if overlap:
        msg = f"Overlap detected in fold {fold_num}: {len(overlap)} repeated samples."
        if log_file:
            write_log(log_file, msg)
        raise ValueError(msg)
    else:
        msg = f"Fold {fold_num}: No overlap between trainVal and test sets. Verified."
        if log_file:
            write_log(log_file, msg)