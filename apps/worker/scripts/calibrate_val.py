import os
import sys
from pathlib import Path
from typing import Dict
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.calibration import calibration_manager
from src.inference import inference_engine

def compute_gt_volumes(seg: np.ndarray, voxel_vol_cm3: float) -> Dict[str, float]:
    seg = np.asarray(seg)
    if seg.ndim == 4 and seg.shape[0] == 3:
        return {
            "WT": float(np.sum(seg[0] == 1) * voxel_vol_cm3),
            "TC": float(np.sum(seg[1] == 1) * voxel_vol_cm3),
            "ET": float(np.sum(seg[2] == 1) * voxel_vol_cm3)
        }
    elif seg.ndim == 3:
        unique_labels = set(np.unique(seg))
        if 4 in unique_labels:
            mask_et = (seg == 4)
            mask_tc = np.isin(seg, [1, 4])
            mask_wt = (seg > 0)
        else:
            mask_et = (seg == 1)
            mask_tc = np.isin(seg, [1, 2])
            mask_wt = np.isin(seg, [1, 2, 3])
        return {
            "WT": float(np.sum(mask_wt) * voxel_vol_cm3),
            "TC": float(np.sum(mask_tc) * voxel_vol_cm3),
            "ET": float(np.sum(mask_et) * voxel_vol_cm3)
        }
    raise ValueError(f"Invalid mask shape: {seg.shape}")

def load_val_case_npy(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(str(filepath), allow_pickle=True)
    if isinstance(data, np.ndarray) and data.dtype == object and data.shape == ():
        item = data.item()
        image = np.asarray(item["image"], dtype=np.float32)
        seg = np.asarray(item.get("seg", item.get("mask")), dtype=np.float32)
    elif isinstance(data, np.ndarray) and data.shape[0] >= 5:
        image = data[:4].astype(np.float32)
        seg = data[4].astype(np.float32)
    else:
        raise ValueError(f"Unsupported npy format in {filepath}")
    return image, seg

def main():
    inference_engine.load_model()
    device = inference_engine.device
    print(f"Device: {device}")

    val_dir = Path(__file__).resolve().parent.parent / "val_calibration"
    files = sorted(list(val_dir.glob("*.npy")))
    print(f"Loaded {len(files)} validation cases.")

    voxel_vol_cm3 = 0.001
    num_passes = 20

    wt_cdfs_all = []
    tc_cdfs_all = []
    et_cdfs_all = []

    wt_gt_all = []
    tc_gt_all = []
    et_gt_all = []

    for idx, fpath in enumerate(files):
        image, seg = load_val_case_npy(fpath)
        gt_vols = compute_gt_volumes(seg, voxel_vol_cm3)

        wt_gt_all.append(gt_vols["WT"])
        tc_gt_all.append(gt_vols["TC"])
        et_gt_all.append(gt_vols["ET"])

        mc_probs = inference_engine.predict_mc_dropout_batched(image, num_passes=num_passes, batch_t=2)

        wt_cdf, _, _, _, _ = inference_engine.compute_calibrated_quantiles(mc_probs["WT"], voxel_vol_cm3, a_scale=1.0)
        tc_cdf, _, _, _, _ = inference_engine.compute_calibrated_quantiles(mc_probs["TC"], voxel_vol_cm3, a_scale=1.0)
        et_cdf, _, _, _, _ = inference_engine.compute_calibrated_quantiles(mc_probs["ET"], voxel_vol_cm3, a_scale=1.0)

        wt_cdfs_all.append(wt_cdf)
        tc_cdfs_all.append(tc_cdf)
        et_cdfs_all.append(et_cdf)
        print(f"[{idx + 1}/{len(files)}] {fpath.name} processed.")

    a_wt, wt_folds = calibration_manager.run_three_fold_cv(wt_cdfs_all, wt_gt_all, modality="WT")
    a_tc, tc_folds = calibration_manager.run_three_fold_cv(tc_cdfs_all, tc_gt_all, modality="TC")
    a_et, et_folds = calibration_manager.run_three_fold_cv(et_cdfs_all, et_gt_all, modality="ET")

    print("\nCalibration Results:")
    print(f"a_wt: {a_wt}, folds: {wt_folds}")
    print(f"a_tc: {a_tc}, folds: {tc_folds}")
    print(f"a_et: {a_et}, folds: {et_folds}")

    factors = {"a_wt": a_wt, "a_tc": a_tc, "a_et": a_et}
    details = {
        "dataset": "val_calibration",
        "total_cases": len(files),
        "mc_dropout_passes": num_passes,
        "method": "Conformal Prediction Uncertainty Scaling",
        "folds": {
            "WT": wt_folds,
            "TC": tc_folds,
            "ET": et_folds
        }
    }
    calibration_manager.save_cache(factors, details)
    print("Saved cache to calibration_cache.json")

if __name__ == "__main__":
    main()
