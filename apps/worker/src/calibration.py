import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np

class CalibrationManager:
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
            "a_wt": 1.15,
            "a_tc": 1.12,
            "a_et": 1.08
        }

    @staticmethod
    def fit_slope_ols(alphas: np.ndarray) -> float:
        n = len(alphas)
        if n < 2:
            return 1.0

        sorted_alphas = np.sort(alphas)
        k = np.arange(1, n + 1)
        u_k = (k - 0.5) / n

        x = sorted_alphas - 0.5
        y = u_k - 0.5

        denominator = float(np.sum(x ** 2))
        if denominator < 1e-9:
            return 1.0

        numerator = float(np.sum(x * y))
        slope = numerator / denominator
        return max(0.1, min(slope, 10.0))

    @classmethod
    def calculate_alphas(cls, predicted_samples: List[List[float]], gt_volumes: List[float]) -> np.ndarray:
        if len(predicted_samples) > 0 and len(predicted_samples[0]) == 99:
            return cls.calculate_alphas_from_cdfs(predicted_samples, gt_volumes)
        alphas = []
        for samples, gt in zip(predicted_samples, gt_volumes):
            samples_arr = np.asarray(samples, dtype=np.float32)
            t = len(samples_arr)
            count_le = float(np.sum(samples_arr <= gt))
            alpha = (count_le + 0.5) / (t + 1.0)
            alphas.append(float(np.clip(alpha, 1e-4, 1.0 - 1e-4)))
        return np.asarray(alphas, dtype=np.float32)

    @classmethod
    def calculate_alphas_from_cdfs(cls, cdf_curves: List[List[float]], gt_volumes: List[float], q_levels: np.ndarray | None = None) -> np.ndarray:
        if q_levels is None:
            q_levels = np.linspace(0.01, 0.99, 99)
        alphas = []
        for cdf, gt in zip(cdf_curves, gt_volumes):
            cdf_arr = np.asarray(cdf, dtype=np.float32)
            alpha = float(np.interp(gt, cdf_arr, q_levels))
            alphas.append(float(np.clip(alpha, 1e-4, 1.0 - 1e-4)))
        return np.asarray(alphas, dtype=np.float32)

    @classmethod
    def run_three_fold_cv(
        cls,
        predicted_samples: List[List[float]],
        gt_volumes: List[float]
    ) -> Tuple[float, List[Dict[str, Any]]]:
        n = len(gt_volumes)
        if n < 3:
            alphas = cls.calculate_alphas(predicted_samples, gt_volumes)
            slope = cls.fit_slope_ols(alphas)
            return slope, [{"fold": 1, "slope": slope, "coverage_95": 1.0}]

        alphas = cls.calculate_alphas(predicted_samples, gt_volumes)
        fold_size = n // 3
        fold_results = []
        slopes = []

        for fold_idx in range(3):
            val_start = fold_idx * fold_size
            val_end = (fold_idx + 1) * fold_size if fold_idx < 2 else n

            test_alphas = alphas[val_start:val_end]
            train_alphas = np.concatenate([alphas[:val_start], alphas[val_end:]])

            fold_slope = cls.fit_slope_ols(train_alphas)
            slopes.append(fold_slope)

            half_width = 0.475 / fold_slope
            alpha_low = max(0.0, 0.5 - half_width)
            alpha_high = min(1.0, 0.5 + half_width)

            covered = np.sum((test_alphas >= alpha_low) & (test_alphas <= alpha_high))
            coverage = float(covered / len(test_alphas)) if len(test_alphas) > 0 else 1.0

            fold_results.append({
                "fold": fold_idx + 1,
                "train_cases": len(train_alphas),
                "val_cases": len(test_alphas),
                "slope": round(float(fold_slope), 4),
                "coverage_95": round(coverage, 4)
            })

        overall_slope = float(np.mean(slopes))
        return round(overall_slope, 4), fold_results

    def save_cache(self, factors: Dict[str, float], validation_details: Dict[str, Any] | None = None) -> None:
        payload = {
            "calibration_factors": {
                "a_wt": round(float(factors.get("a_wt", 1.0)), 4),
                "a_tc": round(float(factors.get("a_tc", 1.0)), 4),
                "a_et": round(float(factors.get("a_et", 1.0)), 4)
            },
            "validation_details": validation_details or {}
        }
        self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.cached_factors = payload["calibration_factors"]

calibration_manager = CalibrationManager()

