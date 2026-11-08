import os
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.calibration import calibration_manager
from src.inference import inference_engine

def main():
    inference_engine.load_model()
    device = inference_engine.device
    print(f"Device: {device}")

    val_dir = Path(__file__).resolve().parent.parent / "val_calibration"
    files = sorted(list(val_dir.glob("*.npy")))
    print(f"Loaded {len(files)} validation cases.")

    wt_cdfs_all = []
    tc_cdfs_all = []
    et_cdfs_all = []

    wt_gt_all = []
    tc_gt_all = []
    et_gt_all = []

    num_passes = 20

    for idx, fpath in enumerate(files):
        data = np.load(str(fpath), allow_pickle=True).item()
        image = data["image"]
        gt_mask = data["mask"]

        wt_gt_all.append(float(np.sum(gt_mask[0] > 0)))
        tc_gt_all.append(float(np.sum(gt_mask[1] > 0)))
        et_gt_all.append(float(np.sum(gt_mask[2] > 0)))

        mc_probs = inference_engine.predict_mc_dropout_batched(image, num_passes=num_passes, batch_t=2)
        wt_cdf, _, _, _ = inference_engine._compute_fast_gpu_quantiles(mc_probs["WT"], 1.0)
        tc_cdf, _, _, _ = inference_engine._compute_fast_gpu_quantiles(mc_probs["TC"], 1.0)
        et_cdf, _, _, _ = inference_engine._compute_fast_gpu_quantiles(mc_probs["ET"], 1.0)

        wt_cdfs_all.append(wt_cdf)
        tc_cdfs_all.append(tc_cdf)
        et_cdfs_all.append(et_cdf)
        print(f"[{idx + 1}/{len(files)}] {fpath.name} processed.")

    a_wt, wt_folds = calibration_manager.run_three_fold_cv(wt_cdfs_all, wt_gt_all)
    a_tc, tc_folds = calibration_manager.run_three_fold_cv(tc_cdfs_all, tc_gt_all)
    a_et, et_folds = calibration_manager.run_three_fold_cv(et_cdfs_all, et_gt_all)

    print("\nCalibration Results:")
    print(f"a_wt: {a_wt}, folds: {wt_folds}")
    print(f"a_tc: {a_tc}, folds: {tc_folds}")
    print(f"a_et: {a_et}, folds: {et_folds}")

    factors = {"a_wt": a_wt, "a_tc": a_tc, "a_et": a_et}
    details = {
        "dataset": "val_calibration",
        "total_cases": len(files),
        "mc_dropout_passes": num_passes,
        "method": "OLS Uniform Theoretical Quantile Matching",
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
