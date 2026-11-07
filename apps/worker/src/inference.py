import io
import math
import os
from pathlib import Path
from typing import Tuple, Dict, Any
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch

from .config import settings
from .model import ResidualUNet3D

class InferenceEngine:
    def __init__(self):
        self.model: ResidualUNet3D | None = None
        self.device = self._resolve_device()

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
        model.eval()
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

    def predict_sliding_window(self, image: np.ndarray) -> np.ndarray:
        c, d, h, w = image.shape
        pd = settings.PATCH_DEPTH
        ph = settings.PATCH_HEIGHT
        pw = settings.PATCH_WIDTH
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

        prob_accum = np.zeros((4, d, h, w), dtype=np.float32)
        count_accum = np.zeros((d, h, w), dtype=np.float32)

        is_cuda = self.device.type == "cuda"
        amp_enabled = is_cuda

        with torch.no_grad():
            for sd in steps_d:
                for sh in steps_h:
                    for sw in steps_w:
                        sub_vol = image[:, sd:sd + pd, sh:sh + ph, sw:sw + pw]
                        tensor = torch.from_numpy(sub_vol.astype(np.float32)).unsqueeze(0).to(self.device)

                        if amp_enabled:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                logits = self.model(tensor)
                        else:
                            logits = self.model(tensor)

                        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                        prob_accum[:, sd:sd + pd, sh:sh + ph, sw:sw + pw] += probs
                        count_accum[sd:sd + pd, sh:sh + ph, sw:sw + pw] += 1.0

        prob_accum /= np.maximum(count_accum, 1.0)
        return prob_accum.argmax(axis=0).astype(np.uint8)

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

    def infer_volumes(
        self,
        flair_file: io.BytesIO,
        t1_file: io.BytesIO,
        t1ce_file: io.BytesIO,
        t2_file: io.BytesIO
    ) -> Tuple[bytes, Dict[str, Any]]:
        flair_nii = nib.Nifti1Image.from_stream(flair_file)
        t1_nii = nib.Nifti1Image.from_stream(t1_file)
        t1ce_nii = nib.Nifti1Image.from_stream(t1ce_file)
        t2_nii = nib.Nifti1Image.from_stream(t2_file)

        affine = flair_nii.affine
        header = flair_nii.header
        zooms = header.get_zooms()[:3]
        voxel_vol_mm3 = float(abs(zooms[0] * zooms[1] * zooms[2]))

        flair_data = self.p95_normalize(flair_nii.get_fdata(dtype=np.float32))
        t1_data = self.p95_normalize(t1_nii.get_fdata(dtype=np.float32))
        t1ce_data = self.p95_normalize(t1ce_nii.get_fdata(dtype=np.float32))
        t2_data = self.p95_normalize(t2_nii.get_fdata(dtype=np.float32))

        stacked = np.stack([flair_data, t1_data, t1ce_data, t2_data], axis=0).transpose(0, 3, 1, 2)
        raw_pred = self.predict_sliding_window(stacked)

        refined_pred = self.postprocess_volume(
            raw_pred,
            settings.SECOND_LCC_MIN_VOXELS,
            settings.ET_MIN_VOXELS
        )

        final_spatial = refined_pred.transpose(1, 2, 0)
        output_nii = nib.Nifti1Image(final_spatial, affine, header)

        buffer = io.BytesIO()
        file_map = output_nii.make_file_map({'image': buffer, 'header': buffer})
        output_nii.to_file_map(file_map)
        nii_bytes = buffer.getvalue()

        et_voxels = int((final_spatial == 1).sum())
        ncr_voxels = int((final_spatial == 2).sum())
        ed_voxels = int((final_spatial == 3).sum())
        tc_voxels = et_voxels + ncr_voxels
        wt_voxels = et_voxels + ncr_voxels + ed_voxels

        metadata = {
            "success": True,
            "input_shape": list(flair_data.shape),
            "output_shape": list(final_spatial.shape),
            "voxel_dimensions": [float(z) for z in zooms],
            "metrics": {
                "whole_tumor": {
                    "voxel_count": wt_voxels,
                    "volume_mm3": round(wt_voxels * voxel_vol_mm3, 2)
                },
                "tumor_core": {
                    "voxel_count": tc_voxels,
                    "volume_mm3": round(tc_voxels * voxel_vol_mm3, 2)
                },
                "enhancing_tumor": {
                    "voxel_count": et_voxels,
                    "volume_mm3": round(et_voxels * voxel_vol_mm3, 2)
                },
                "peritumoral_edema": {
                    "voxel_count": ed_voxels,
                    "volume_mm3": round(ed_voxels * voxel_vol_mm3, 2)
                },
                "necrotic_core": {
                    "voxel_count": ncr_voxels,
                    "volume_mm3": round(ncr_voxels * voxel_vol_mm3, 2)
                }
            }
        }

        return nii_bytes, metadata

inference_engine = InferenceEngine()

