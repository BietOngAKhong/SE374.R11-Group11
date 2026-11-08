import io
import os
import uuid
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import nibabel as nib
import numpy as np
from scipy import ndimage
import torch

from .config import settings
from .model import ResidualUNet3D
from .calibration import calibration_manager

def enable_mc_dropout(model: torch.nn.Module) -> None:
    model.eval()
    for m in model.modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            m.train()

class InferenceEngine:
    def __init__(self):
        self.model: ResidualUNet3D | None = None
        self.device = self._resolve_device()
        self.temp_dir = Path(__file__).resolve().parent.parent / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_device(self) -> torch.device:
        target = settings.DEVICE.lower()
        if target == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if target == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load_model(self) -> None:
        model = ResidualUNet3D(
            in_channels=4,
            num_classes=4,
            features=(16, 32, 64, 128),
            bottleneck=256,
            dropout=0.2
        )
        weights_path = Path(settings.MODEL_WEIGHTS_PATH)
        if not weights_path.exists():
            root_fallback = Path(__file__).resolve().parents[3] / "working_unet3d_final_fp32.pth"
            if root_fallback.exists():
                weights_path = root_fallback

        if weights_path.exists():
            checkpoint = torch.load(str(weights_path), map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                model.load_state_dict(checkpoint)
        
        model.to(self.device)
        enable_mc_dropout(model)
        self.model = model

    @staticmethod
    def p95_normalize(volume: np.ndarray) -> np.ndarray:
        vol = volume.astype(np.float32, copy=False)
        nonzero = vol[vol != 0]
        if nonzero.size == 0:
            return np.zeros_like(vol, dtype=np.float32)

        p95 = np.percentile(nonzero, 95.0)
        out = np.zeros_like(vol, dtype=np.float32)
        if p95 > 0:
            mask = vol != 0
            out[mask] = vol[mask] / p95
        return out

    def predict_mc_dropout_batched(self, image: np.ndarray, num_passes: int = 20, batch_t: int = 2) -> Dict[str, np.ndarray]:
        c, d, h, w = image.shape
        pd, ph, pw = settings.PATCH_DEPTH, settings.PATCH_HEIGHT, settings.PATCH_WIDTH
        overlap = settings.OVERLAP

        step_d = max(1, int(pd * (1 - overlap)))
        step_h = max(1, int(ph * (1 - overlap)))
        step_w = max(1, int(pw * (1 - overlap)))

        steps_d = list(range(0, max(1, d - pd + 1), step_d))
        if steps_d[-1] + pd < d:
            steps_d.append(d - pd)

        steps_h = list(range(0, max(1, h - ph + 1), step_h))
        if steps_h[-1] + ph < h:
            steps_h.append(h - ph)

        steps_w = list(range(0, max(1, w - pw + 1), step_w))
        if steps_w[-1] + pw < w:
            steps_w.append(w - pw)

        prob_wt_list, prob_tc_list, prob_et_list = [], [], []
        is_cuda = self.device.type == "cuda"
        enable_mc_dropout(self.model)

        with torch.no_grad():
            for t_step in range(0, num_passes, batch_t):
                cur_b = min(batch_t, num_passes - t_step)
                prob_accum = np.zeros((cur_b, 4, d, h, w), dtype=np.float32)
                count_accum = np.zeros((d, h, w), dtype=np.float32)

                for sd in steps_d:
                    for sh in steps_h:
                        for sw in steps_w:
                            sub_vol = image[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]
                            tensor = torch.from_numpy(sub_vol.astype(np.float32)).unsqueeze(0).repeat(cur_b, 1, 1, 1, 1).to(self.device)

                            if is_cuda:
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    logits = self.model(tensor)
                            else:
                                logits = self.model(tensor)

                            probs = torch.softmax(logits, dim=1).cpu().numpy()
                            prob_accum[:, :, sd:sd + pd, sh:sh + ph, sw:sw + pw] += probs
                            count_accum[sd:sd + pd, sh:sh + ph, sw:sw + pw] += 1.0

                count_accum = np.maximum(count_accum, 1.0)
                prob_accum /= count_accum[None, None, ...]

                for b in range(cur_b):
                    p_et = prob_accum[b, 1]
                    p_tc = prob_accum[b, 1] + prob_accum[b, 2]
                    p_wt = prob_accum[b, 1] + prob_accum[b, 2] + prob_accum[b, 3]

                    prob_wt_list.append(p_wt.astype(np.float16))
                    prob_tc_list.append(p_tc.astype(np.float16))
                    prob_et_list.append(p_et.astype(np.float16))

        return {
            "WT": np.stack(prob_wt_list, axis=0),
            "TC": np.stack(prob_tc_list, axis=0),
            "ET": np.stack(prob_et_list, axis=0)
        }

    def _compute_fast_gpu_quantiles(self, prob_arr: np.ndarray, voxel_vol_cm3: float):
        t_prob = torch.from_numpy(prob_arr.astype(np.float32)).to(self.device)
        d, h, w = prob_arr.shape[1], prob_arr.shape[2], prob_arr.shape[3]

        valid_mask = (t_prob.max(dim=0).values > 0.05)
        q_levels = torch.linspace(0.01, 0.99, 99, device=self.device)

        if not valid_mask.any():
            return [0.0] * 99, np.zeros((d, h, w), dtype=bool), np.zeros((d, h, w), dtype=bool), np.zeros((d, h, w), dtype=bool)

        valid_voxels = t_prob[:, valid_mask]
        q_voxels = torch.quantile(valid_voxels, q_levels, dim=0)

        vol_cdf = (q_voxels >= 0.5).sum(dim=1).cpu().numpy().astype(np.float64) * voxel_vol_cm3

        m05_valid = (q_voxels[4] >= 0.5).cpu().numpy()
        m50_valid = (q_voxels[49] >= 0.5).cpu().numpy()
        m95_valid = (q_voxels[94] >= 0.5).cpu().numpy()

        valid_mask_np = valid_mask.cpu().numpy()

        mask_05 = np.zeros((d, h, w), dtype=bool)
        mask_50 = np.zeros((d, h, w), dtype=bool)
        mask_95 = np.zeros((d, h, w), dtype=bool)

        mask_05[valid_mask_np] = m05_valid
        mask_50[valid_mask_np] = m50_valid
        mask_95[valid_mask_np] = m95_valid

        del t_prob, valid_voxels, q_voxels
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return [round(float(v), 2) for v in vol_cdf], mask_05, mask_50, mask_95

    @staticmethod
    def postprocess_volume(pred: np.ndarray, second_lcc_min: int, et_min: int) -> np.ndarray:
        output = pred.copy()
        tumor = output > 0
        labeled, count = ndimage.label(tumor)

        if count > 0:
            sizes = ndimage.sum(tumor, labeled, index=np.arange(1, count + 1))
            order = np.argsort(sizes)[::-1]
            keep_ids = [int(order[0] + 1)]

            if len(order) >= 2 and sizes[order[1]] >= second_lcc_min:
                keep_ids.append(int(order[1] + 1))

            keep_mask = np.isin(labeled, keep_ids)
            output[~keep_mask] = 0

        et_count = int((output == 1).sum())
        if 0 < et_count < et_min:
            output[output == 1] = 2

        return output

    @staticmethod
    def _load_nii_from_bytes(data: bytes) -> nib.Nifti1Image:
        with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            img = nib.load(tmp_path)
            data_arr = img.get_fdata(dtype=np.float32)
            affine = img.affine.copy()
            header = img.header.copy()
            return nib.Nifti1Image(data_arr, affine, header)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    @staticmethod
    def _save_nii_to_file(mask_arr: np.ndarray, affine: np.ndarray, header: Any, out_path: Path) -> None:
        spatial = mask_arr.transpose(1, 2, 0).astype(np.uint8)
        nii_obj = nib.Nifti1Image(spatial, affine, header)
        nib.save(nii_obj, str(out_path))

    def run_uq_inference(
        self,
        flair_bytes: bytes,
        t1_bytes: bytes,
        t1ce_bytes: bytes,
        t2_bytes: bytes,
        num_passes: int = 20
    ) -> Dict[str, Any]:
        flair_nii = self._load_nii_from_bytes(flair_bytes)
        t1_nii = self._load_nii_from_bytes(t1_bytes)
        t1ce_nii = self._load_nii_from_bytes(t1ce_bytes)
        t2_nii = self._load_nii_from_bytes(t2_bytes)

        affine = flair_nii.affine
        header = flair_nii.header
        zooms = header.get_zooms()[:3]
        voxel_vol_cm3 = float(abs(zooms[0] * zooms[1] * zooms[2])) / 1000.0

        flair_data = self.p95_normalize(flair_nii.get_fdata(dtype=np.float32))
        t1_data = self.p95_normalize(t1_nii.get_fdata(dtype=np.float32))
        t1ce_data = self.p95_normalize(t1ce_nii.get_fdata(dtype=np.float32))
        t2_data = self.p95_normalize(t2_nii.get_fdata(dtype=np.float32))

        stacked = np.stack([flair_data, t1_data, t1ce_data, t2_data], axis=0).transpose(0, 3, 1, 2)
        d, h, w = stacked.shape[1], stacked.shape[2], stacked.shape[3]

        mc_probs = self.predict_mc_dropout_batched(stacked, num_passes=num_passes, batch_t=2)

        wt_cdf, wt_05, wt_50, wt_95 = self._compute_fast_gpu_quantiles(mc_probs["WT"], voxel_vol_cm3)
        tc_cdf, tc_05, tc_50, tc_95 = self._compute_fast_gpu_quantiles(mc_probs["TC"], voxel_vol_cm3)
        et_cdf, et_05, et_50, et_95 = self._compute_fast_gpu_quantiles(mc_probs["ET"], voxel_vol_cm3)

        def create_multilabel_mask(wt_m: np.ndarray, tc_m: np.ndarray, et_m: np.ndarray) -> np.ndarray:
            m = np.zeros((d, h, w), dtype=np.uint8)
            m[wt_m] = 3
            m[tc_m] = 2
            m[et_m] = 1
            return self.postprocess_volume(m, settings.SECOND_LCC_MIN_VOXELS, settings.ET_MIN_VOXELS)

        mask_p05 = create_multilabel_mask(wt_05, tc_05, et_05)
        mask_p50 = create_multilabel_mask(wt_50, tc_50, et_50)
        mask_p95 = create_multilabel_mask(wt_95, tc_95, et_95)

        task_id = str(uuid.uuid4())
        p05_path = self.temp_dir / f"{task_id}_p05.nii.gz"
        p50_path = self.temp_dir / f"{task_id}_p50.nii.gz"
        p95_path = self.temp_dir / f"{task_id}_p95.nii.gz"

        self._save_nii_to_file(mask_p05, affine, header, p05_path)
        self._save_nii_to_file(mask_p50, affine, header, p50_path)
        self._save_nii_to_file(mask_p95, affine, header, p95_path)

        factors = calibration_manager.cached_factors
        percentiles_range = list(range(1, 100))

        return {
            "task_id": task_id,
            "masks": {
                "p05_url": f"/api/mask/{task_id}?p=5",
                "p50_url": f"/api/mask/{task_id}?p=50",
                "p95_url": f"/api/mask/{task_id}?p=95"
            },
            "volumes_cm3": {
                "WT": {
                    "median": wt_cdf[49],
                    "percentiles": percentiles_range,
                    "volumes_cm3": wt_cdf
                },
                "TC": {
                    "median": tc_cdf[49],
                    "percentiles": percentiles_range,
                    "volumes_cm3": tc_cdf
                },
                "ET": {
                    "median": et_cdf[49],
                    "percentiles": percentiles_range,
                    "volumes_cm3": et_cdf
                }
            },
            "calibration_factors": {
                "a_wt": float(factors.get("a_wt", 1.1524)),
                "a_tc": float(factors.get("a_tc", 1.1218)),
                "a_et": float(factors.get("a_et", 1.0845))
            }
        }

    def get_mask_file_path(self, task_id: str, p: int) -> Path | None:
        p_str = f"p0{p}" if p < 10 else f"p{p}"
        target = self.temp_dir / f"{task_id}_{p_str}.nii.gz"
        if target.exists():
            return target
        return None

inference_engine = InferenceEngine()
