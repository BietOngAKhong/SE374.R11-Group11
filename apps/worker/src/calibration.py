import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

class CalibrationManager:
    PERCENTILES_LEVELS = np.linspace(0.01, 0.99, 99)
    MODALITY_CONFIG = {
        "WT": (0.25, 1.0),
        "TC": (0.25, 0.8),
        "ET": (0.30, 0.6)
    }

    def __init__(self, cache_path: str | Path | None = None):
        if cache_path is None:
            self.cache_path = Path(__file__).resolve().parent / "calibration_cache.json"
        else:
            self.cache_path = Path(cache_path)
        self.cached_factors: Dict[str, float] = self._load_cache()

    def _load_cache(self) -> Dict[str, float]:
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if "calibration_factors" in data:
                    return data["calibration_factors"]
            except Exception:
                pass

        return {
            "a_wt": 1.75,
            "a_tc": 2.867,
            "a_et": 2.6526
        }

    @classmethod
    def compute_conformal_scores(
        cls,
        cdfs_vols: np.ndarray,
        gt_vols: np.ndarray,
        modality: str = "WT",
        percentiles_levels: np.ndarray | None = None
    ) -> np.ndarray:
        if percentiles_levels is None:
            percentiles_levels = cls.PERCENTILES_LEVELS

        idx_025 = int(np.argmin(np.abs(percentiles_levels - 0.025)))
        idx_500 = int(np.argmin(np.abs(percentiles_levels - 0.500)))
        idx_975 = int(np.argmin(np.abs(percentiles_levels - 0.975)))

        rel_factor, abs_floor = cls.MODALITY_CONFIG.get(modality, (0.25, 0.8))

        scores = []
        for cdf_vol, gt_vol in zip(cdfs_vols, gt_vols):
            mono_cdf = np.maximum.accumulate(cdf_vol)
            v_low = mono_cdf[idx_025]
            v_med = mono_cdf[idx_500]
            v_high = mono_cdf[idx_975]

            raw_half_width = max((v_high - v_low) / 2.0, 0.0)
            baseline_half_width = max(raw_half_width, rel_factor * v_med + abs_floor)

            residual = abs(gt_vol - v_med)
            s = residual / baseline_half_width
            scores.append(s)

        return np.array(scores, dtype=np.float64)

    @classmethod
    def fit_conformal_scale(cls, scores_train: np.ndarray, target_coverage: float = 0.95) -> float:
        n = len(scores_train)
        if n == 0:
            return 1.05
        q_level = min(1.0, float(np.ceil((n + 1) * target_coverage) / n))
        scale = float(np.quantile(scores_train, min(q_level, 0.95)))
        return float(np.clip(scale, 1.05, 3.50))

    @classmethod
    def evaluate_conformal_coverage(
        cls,
        cdfs_vols: np.ndarray,
        gt_vols: np.ndarray,
        scale_factor: float,
        modality: str = "WT",
        percentiles_levels: np.ndarray | None = None
    ) -> float:
        if percentiles_levels is None:
            percentiles_levels = cls.PERCENTILES_LEVELS

        idx_025 = int(np.argmin(np.abs(percentiles_levels - 0.025)))
        idx_500 = int(np.argmin(np.abs(percentiles_levels - 0.500)))
        idx_975 = int(np.argmin(np.abs(percentiles_levels - 0.975)))

        rel_factor, abs_floor = cls.MODALITY_CONFIG.get(modality, (0.25, 0.8))

        covered = 0
        n = len(gt_vols)
        if n == 0:
            return 1.0

        for cdf_vol, gt_vol in zip(cdfs_vols, gt_vols):
            mono_cdf = np.maximum.accumulate(cdf_vol)
            v_low = mono_cdf[idx_025]
            v_med = mono_cdf[idx_500]
            v_high = mono_cdf[idx_975]

            raw_half_width = max((v_high - v_low) / 2.0, 0.0)
            baseline_half_width = max(raw_half_width, rel_factor * v_med + abs_floor)

            ci_low = max(0.0, v_med - scale_factor * baseline_half_width)
            ci_high = v_med + scale_factor * baseline_half_width

            if ci_low <= gt_vol <= ci_high:
                covered += 1

        return float(covered / n)

    @classmethod
    def run_three_fold_cv(
        cls,
        predicted_cdfs: List[List[float]] | np.ndarray,
        gt_volumes: List[float] | np.ndarray,
        modality: str = "WT"
    ) -> Tuple[float, List[Dict[str, Any]]]:
        cdfs = np.asarray(predicted_cdfs, dtype=np.float64)
        gts = np.asarray(gt_volumes, dtype=np.float64)
        n = len(gts)

        if n < 3:
            all_scores = cls.compute_conformal_scores(cdfs, gts, modality=modality)
            scale = cls.fit_conformal_scale(all_scores)
            return round(scale, 4), [{"fold": 1, "train_cases": n, "val_cases": n, "slope": round(scale, 4), "coverage_95": 1.0}]

        indices = np.arange(n)
        rng = np.random.RandomState(42)
        rng.shuffle(indices)

        fold_sizes = np.full(3, n // 3, dtype=int)
        fold_sizes[:n % 3] += 1

        current = 0
        folds_details = []

        for fold_idx, fold_size in enumerate(fold_sizes):
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            current = stop

            train_cdfs, train_gts = cdfs[train_idx], gts[train_idx]
            test_cdfs, test_gts = cdfs[test_idx], gts[test_idx]

            train_scores = cls.compute_conformal_scores(train_cdfs, train_gts, modality=modality)
            a_fold = cls.fit_conformal_scale(train_scores, target_coverage=0.95)
            cov_95 = cls.evaluate_conformal_coverage(test_cdfs, test_gts, a_fold, modality=modality)

            folds_details.append({
                "fold": fold_idx + 1,
                "train_cases": int(len(train_idx)),
                "val_cases": int(len(test_idx)),
                "slope": round(float(a_fold), 4),
                "coverage_95": round(float(cov_95), 4)
            })

        all_scores = cls.compute_conformal_scores(cdfs, gts, modality=modality)
        final_scale = cls.fit_conformal_scale(all_scores, target_coverage=0.95)
        return round(float(final_scale), 4), folds_details

    def save_cache(self, factors: Dict[str, float], validation_details: Dict[str, Any] | None = None) -> None:
        payload = {
            "calibration_factors": {
                "a_wt": round(float(factors.get("a_wt", 1.75)), 4),
                "a_tc": round(float(factors.get("a_tc", 2.867)), 4),
                "a_et": round(float(factors.get("a_et", 2.6526)), 4)
            },
            "validation_details": validation_details or {}
        }
        self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.cached_factors = payload["calibration_factors"]

calibration_manager = CalibrationManager()
