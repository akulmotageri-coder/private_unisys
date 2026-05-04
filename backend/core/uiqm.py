"""
Underwater Image Quality Measure (UIQM).

Three sub-metrics combined into one score:
  UIChroM — Chroma (color richness)
  UIConM  — Contrast (structural visibility)
  UISM    — Sharpness (edge crispness via Laplacian variance)

Reference:
  Panetta, Gao, Agaian (2016) — Human-Visual-System-Inspired
  Underwater Image Quality Measures. IEEE Journal of Oceanic Engineering.

All sub-metrics are normalised to [0, 1] before combining.
Higher UIQM = better perceptual quality.
"""

import numpy as np
from scipy.ndimage import laplace, uniform_filter
from dataclasses import dataclass
from typing import Tuple
from loguru import logger


# ---------------------------------------------------------------------------
# Weighting coefficients from Panetta et al. (2016) Table II
# ---------------------------------------------------------------------------

_W_CHROMM  = 0.0282   # chroma weight
_W_CONM    = 0.2953   # contrast weight
_W_SHARPNESS = 3.5753  # sharpness weight


@dataclass
class UIQMResult:
    """
    Full UIQM breakdown for one frame.
    All sub-scores normalised to [0, 1].
    Combined UIQM is weighted sum — higher is better.
    """
    uiqm:       float   # combined score
    uichrom:    float   # chroma richness      0→1
    uiconm:     float   # contrast             0→1
    uism:       float   # sharpness            0→1
    frame_mean: float   # mean pixel intensity 0→1
    frame_std:  float   # pixel std dev        0→1
    red_mean:   float   # per-channel means
    green_mean: float
    blue_mean:  float

    def to_dict(self) -> dict:
        return {
            "uiqm":       round(self.uiqm,    4),
            "uichrom":    round(self.uichrom,  4),
            "uiconm":     round(self.uiconm,   4),
            "uism":       round(self.uism,     4),
            "frame_mean": round(self.frame_mean, 4),
            "frame_std":  round(self.frame_std,  4),
            "red_mean":   round(self.red_mean,   4),
            "green_mean": round(self.green_mean, 4),
            "blue_mean":  round(self.blue_mean,  4),
        }

    def quality_label(self) -> str:
        """Human-readable quality tier."""
        if self.uiqm >= 0.75:
            return "Excellent"
        elif self.uiqm >= 0.55:
            return "Good"
        elif self.uiqm >= 0.35:
            return "Fair"
        elif self.uiqm >= 0.15:
            return "Poor"
        else:
            return "Critical"


class UIQMEngine:
    """
    Computes UIQM and all sub-metrics for a given frame.

    Input : float32 numpy array (H, W, 3), RGB, values in [0, 1]
    Output: UIQMResult dataclass

    Designed to run on every frame in the pipeline — kept fast
    by avoiding Python loops and using vectorised numpy ops.
    """

    def __init__(
        self,
        w_chrom:     float = _W_CHROMM,
        w_con:       float = _W_CONM,
        w_sharp:     float = _W_SHARPNESS,
        patch_size:  int   = 5,
    ):
        """
        w_chrom, w_con, w_sharp : combination weights (Panetta 2016 defaults)
        patch_size               : local neighbourhood for contrast metric
        """
        self.w_chrom    = w_chrom
        self.w_con      = w_con
        self.w_sharp    = w_sharp
        self.patch_size = patch_size
        logger.info(
            f"UIQMEngine ready — weights: "
            f"chrom={w_chrom} con={w_con} sharp={w_sharp}"
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def compute(self, frame: np.ndarray) -> UIQMResult:
        """
        Compute full UIQM for one frame.
        frame: float32 (H, W, 3) RGB in [0, 1].
        """
        self._validate(frame)

        r = frame[:, :, 0]
        g = frame[:, :, 1]
        b = frame[:, :, 2]

        uichrom = self._uichrom(r, g, b)
        uiconm  = self._uiconm(r, g, b)
        uism    = self._uism(r, g, b)

        # Weighted combination (Panetta 2016 Eq. 1)
        uiqm = (
            self.w_chrom  * uichrom +
            self.w_con    * uiconm  +
            self.w_sharp  * uism
        )
        # Normalise combined score to [0, 1]
        # Theoretical max ≈ w_chrom + w_con + w_sharp = 3.8988
        uiqm_norm = float(np.clip(uiqm / (_W_CHROMM + _W_CONM + _W_SHARPNESS), 0.0, 1.0))

        return UIQMResult(
            uiqm        = uiqm_norm,
            uichrom     = float(uichrom),
            uiconm      = float(uiconm),
            uism        = float(uism),
            frame_mean  = float(frame.mean()),
            frame_std   = float(frame.std()),
            red_mean    = float(r.mean()),
            green_mean  = float(g.mean()),
            blue_mean   = float(b.mean()),
        )

    def compute_delta(
        self,
        before: np.ndarray,
        after:  np.ndarray,
    ) -> dict:
        """
        Compute UIQM before and after enhancement.
        Returns both results plus improvement deltas.
        Useful for the dashboard enhancement gain display.
        """
        q_before = self.compute(before)
        q_after  = self.compute(after)

        return {
            "before":        q_before.to_dict(),
            "after":         q_after.to_dict(),
            "delta_uiqm":    round(q_after.uiqm    - q_before.uiqm,    4),
            "delta_uichrom": round(q_after.uichrom  - q_before.uichrom, 4),
            "delta_uiconm":  round(q_after.uiconm   - q_before.uiconm,  4),
            "delta_uism":    round(q_after.uism      - q_before.uism,    4),
            "label_before":  q_before.quality_label(),
            "label_after":   q_after.quality_label(),
            "improved":      q_after.uiqm > q_before.uiqm,
        }

    # ------------------------------------------------------------------ #
    #  Sub-metric 1 — UIChroM (Chroma)                                   #
    # ------------------------------------------------------------------ #

    def _uichrom(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Measures colour richness using the RG and YB opponent channels.
        Underwater images lose red first — low chroma = degraded.

        RG = R − G  (red-green opponent)
        YB = 0.5(R+G) − B  (yellow-blue opponent)

        UIChroM = sqrt(σ_RG² + σ_YB²) + 0.3×sqrt(μ_RG² + μ_YB²)
        (Panetta 2016 Eq. 3)
        """
        RG = r.astype(np.float32) - g.astype(np.float32)
        YB = (0.5 * (r + g) - b).astype(np.float32)

        mu_RG,  sigma_RG  = float(RG.mean()), float(RG.std())
        mu_YB,  sigma_YB  = float(YB.mean()), float(YB.std())

        chroma = (
            np.sqrt(sigma_RG ** 2 + sigma_YB ** 2) +
            0.3 * np.sqrt(mu_RG ** 2 + mu_YB ** 2)
        )
        # Normalise: typical clear-water range 0→0.35
        return float(np.clip(chroma / 0.35, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  Sub-metric 2 — UIConM (Contrast)                                  #
    # ------------------------------------------------------------------ #

    def _uiconm(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Measures local contrast using log-AME (Agaian Measure of Enhancement).

        For each channel:
          Compute local max and min in a patch_size neighbourhood.
          AME = mean of log((Imax + ε) / (Imin + ε)) over all patches.

        UIConM = mean(AME_R, AME_G, AME_B)  normalised to [0, 1].
        (Panetta 2016 Eq. 5-6)
        """
        ame_scores = []
        for ch in [r, g, b]:
            ame = self._log_ame(ch)
            ame_scores.append(ame)

        uiconm = float(np.mean(ame_scores))
        # Normalise: typical range 0→2.5
        return float(np.clip(uiconm / 2.5, 0.0, 1.0))

    def _log_ame(self, channel: np.ndarray) -> float:
        """
        Log-AME for a single channel.
        Uses uniform_filter as an efficient local max/min approximation.
        """
        eps = 1e-6
        p   = self.patch_size

        # Local maximum approximation via dilation proxy
        # (full morphological dilation is slower; uniform filter captures
        #  local spread well enough for a quality metric)
        local_max = uniform_filter(
            np.maximum(channel, uniform_filter(channel, size=p)),
            size=p,
        )
        local_min = uniform_filter(
            np.minimum(channel, uniform_filter(channel, size=p)),
            size=p,
        )

        ratio = (local_max + eps) / (local_min + eps)
        log_ratio = np.log(ratio + eps)

        # Only count patches with meaningful contrast
        mask = local_max > (local_min + 0.01)
        if mask.sum() == 0:
            return 0.0

        return float(log_ratio[mask].mean())

    # ------------------------------------------------------------------ #
    #  Sub-metric 3 — UISM (Sharpness)                                   #
    # ------------------------------------------------------------------ #

    def _uism(
        self,
        r: np.ndarray,
        g: np.ndarray,
        b: np.ndarray,
    ) -> float:
        """
        Measures edge crispness using the Variance of the Laplacian.
        Sharp edges produce high Laplacian variance.
        Blurry / backscatter-fogged images → near-zero variance.

        UISM = weighted mean of per-channel Laplacian variance.
        Blue channel weighted less (most degraded by water).
        (Panetta 2016 Eq. 8)
        """
        # Per-channel Laplacian
        lap_r = laplace(r.astype(np.float64))
        lap_g = laplace(g.astype(np.float64))
        lap_b = laplace(b.astype(np.float64))

        var_r = float(np.var(lap_r))
        var_g = float(np.var(lap_g))
        var_b = float(np.var(lap_b))

        # Weight: R=0.299  G=0.587  B=0.114 (luminance weights)
        sharpness = 0.299 * var_r + 0.587 * var_g + 0.114 * var_b

        # Normalise: typical range 0 (blurred) → 0.004 (sharp edges)
        return float(np.clip(sharpness / 0.004, 0.0, 1.0))

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate(frame: np.ndarray) -> None:
        if frame.dtype != np.float32:
            raise TypeError(
                f"Frame must be float32, got {frame.dtype}. "
                "Convert: frame = frame.astype(np.float32) / 255.0"
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame must be (H, W, 3), got {frame.shape}"
            )