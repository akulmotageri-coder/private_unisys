"""
Jaffe-McGlamery Underwater Image Formation Model.

Light hitting an underwater camera splits into three components:
  Ed — Direct component    : actual signal, decays exponentially with depth
  Ef — Forward scatter     : blur from particles deflecting light
  Eb — Backscatter         : veiling luminance / fog from suspended sediment

Reference:
  Jaffe (1990), McGlamery (1980), Akkaynak & Treibitz (2018),
  Mobley (1994) — Light and Water.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass, field
from typing import Tuple, Dict
from loguru import logger

from backend.core.ocean_lookup import WaterType


# ---------------------------------------------------------------------------
# Jerlov spectral profiles — one per WaterType.
# All coefficients are per-metre, for (R, G, B) channels.
# Values derived from Jerlov (1976) Tables 3-5 and Mobley (1994) Appendix B.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JerlovProfile:
    """
    Spectral optical properties for a named water body.

    attenuation_rgb     : beam attenuation coefficient c = a + b  [m⁻¹]
    backscatter_rgb     : volume backscattering coefficient        [m⁻¹ sr⁻¹]
    forward_scatter_sig : PSF sigma for forward scatter blur       [px/10m]
    dominant_nm         : peak transmission wavelength            [nm]
    base_turbidity      : intrinsic turbidity of this water type  [0-1]
    chlorophyll_mgl     : chlorophyll-a concentration             [mg/L]
    cdom_abs            : CDOM absorption at 440nm                [m⁻¹]
    """
    name:                str
    attenuation_rgb:     Tuple[float, float, float]
    backscatter_rgb:     Tuple[float, float, float]
    forward_scatter_sig: float
    dominant_nm:         int
    base_turbidity:      float
    chlorophyll_mgl:     float
    cdom_abs:            float


def _build_profiles() -> Dict[WaterType, JerlovProfile]:
    """
    Build the Jerlov profile table from optical constants.
    Kept in a factory function so it is easy to extend
    without touching the engine class.
    """
    return {
        WaterType.OPEN_OCEAN: JerlovProfile(
            name                = "Open Ocean — Jerlov Type IA",
            attenuation_rgb     = (0.22, 0.04, 0.015),
            backscatter_rgb     = (0.010, 0.004, 0.001),
            forward_scatter_sig = 0.4,
            dominant_nm         = 475,
            base_turbidity      = 0.10,
            chlorophyll_mgl     = 0.05,
            cdom_abs            = 0.01,
        ),
        WaterType.ARABIAN_SEA: JerlovProfile(
            name                = "Arabian Sea — Jerlov Type I",
            attenuation_rgb     = (0.35, 0.06, 0.020),
            backscatter_rgb     = (0.030, 0.008, 0.002),
            forward_scatter_sig = 0.8,
            dominant_nm         = 490,
            base_turbidity      = 0.30,
            chlorophyll_mgl     = 0.40,
            cdom_abs            = 0.05,
        ),
        WaterType.BAY_OF_BENGAL: JerlovProfile(
            name                = "Bay of Bengal — Jerlov Type II",
            attenuation_rgb     = (0.52, 0.12, 0.040),
            backscatter_rgb     = (0.080, 0.025, 0.006),
            forward_scatter_sig = 1.4,
            dominant_nm         = 560,
            base_turbidity      = 0.60,
            chlorophyll_mgl     = 1.20,
            cdom_abs            = 0.18,
        ),
        WaterType.COASTAL_TURBID: JerlovProfile(
            name                = "Coastal Turbid — Jerlov Type III",
            attenuation_rgb     = (0.65, 0.18, 0.070),
            backscatter_rgb     = (0.140, 0.050, 0.012),
            forward_scatter_sig = 2.2,
            dominant_nm         = 570,
            base_turbidity      = 0.75,
            chlorophyll_mgl     = 3.50,
            cdom_abs            = 0.40,
        ),
        WaterType.INDUSTRIAL_PORT: JerlovProfile(
            name                = "Industrial Port — Extreme",
            attenuation_rgb     = (0.85, 0.28, 0.120),
            backscatter_rgb     = (0.220, 0.090, 0.025),
            forward_scatter_sig = 3.5,
            dominant_nm         = 580,
            base_turbidity      = 0.92,
            chlorophyll_mgl     = 8.00,
            cdom_abs            = 0.90,
        ),
    }


JERLOV_PROFILES: Dict[WaterType, JerlovProfile] = _build_profiles()


# ---------------------------------------------------------------------------
# Physics state — driven live by the simulator / GPS
# ---------------------------------------------------------------------------

@dataclass
class PhysicsState:
    """
    Complete physical state of the underwater environment.
    Every field maps to a measurable real-world parameter.
    """
    depth_m:           float     = 10.0   # metres below surface
    turbidity:         float     = 0.50   # user-adjusted 0→1
    water_type:        WaterType = WaterType.BAY_OF_BENGAL
    ambient_light:     float     = 0.80   # surface irradiance 0→1
    salinity_ppt:      float     = 35.0   # parts per thousand
    temperature_c:     float     = 28.0   # °C
    current_speed_ms:  float     = 0.20   # m/s — affects particle suspension
    lat:               float     = 13.08
    lon:               float     = 80.27

    def effective_turbidity(self, profile: JerlovProfile) -> float:
        """
        Combine water-type base turbidity with the user slider value.
        Current speed increases suspended particles.
        """
        current_factor = 1.0 + (self.current_speed_ms / 2.0) * 0.3
        combined = (
            profile.base_turbidity * 0.4 +
            self.turbidity * 0.6
        ) * current_factor
        return float(np.clip(combined, 0.0, 1.0))

    def light_at_depth(self) -> float:
        """
        Beer-Lambert law: irradiance decays exponentially with depth.
        Diffuse attenuation coefficient Kd ≈ 0.04 m⁻¹ for clear water,
        increases with turbidity.
        """
        kd = 0.04 + self.turbidity * 0.08
        return float(self.ambient_light * np.exp(-kd * self.depth_m))


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class UnderwaterPhysicsEngine:
    """
    Applies the full Jaffe-McGlamery image formation model to a clean frame,
    producing a physically-accurate degraded underwater image.

    Input  : float32 numpy array (H, W, 3), RGB, values in [0, 1]
    Output : float32 numpy array (H, W, 3), RGB, values in [0, 1]
    """

    def __init__(self):
        self._profiles = JERLOV_PROFILES
        logger.info("UnderwaterPhysicsEngine ready — "
                    f"{len(self._profiles)} Jerlov profiles loaded")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def degrade(self, frame: np.ndarray, state: PhysicsState) -> np.ndarray:
        """
        Full Jaffe-McGlamery degradation pipeline.
        frame must be float32, shape (H, W, 3), values 0-1.
        """
        self._validate(frame)
        profile  = self._profiles[state.water_type]
        eff_turb = state.effective_turbidity(profile)
        light    = state.light_at_depth()

        # Three-component model
        Ed = self._direct_component(frame, state.depth_m, profile, eff_turb)
        Ef = self._forward_scatter(Ed, state.depth_m, profile, eff_turb)
        Eb = self._backscatter(frame.shape, state, profile, eff_turb)

        # Compose: I(x,y) = Ef(x,y) + Eb(x,y)
        composed = Ef + Eb

        # Apply depth-dependent ambient light
        composed *= light

        # CDOM yellow-brown tint (absorbs blue)
        composed = self._apply_cdom(composed, profile)

        return np.clip(composed, 0.0, 1.0).astype(np.float32)

    def compute_turbidity_estimate(self, frame: np.ndarray) -> float:
        """
        No-reference turbidity estimate from pixel variance.
        Low contrast  →  high backscatter  →  high turbidity.
        Returns float 0.0 (crystal clear) → 1.0 (completely opaque).
        """
        self._validate(frame)
        gray     = (0.299 * frame[:, :, 0] +
                    0.587 * frame[:, :, 1] +
                    0.114 * frame[:, :, 2])
        variance = float(np.var(gray))
        # Empirically calibrated range: 0.002 (muddy) → 0.06 (clear)
        estimate = 1.0 - np.clip((variance - 0.002) / 0.058, 0.0, 1.0)
        return float(estimate)

    def get_profile(self, water_type: WaterType) -> JerlovProfile:
        return self._profiles[water_type]

    def all_profiles(self) -> Dict[WaterType, JerlovProfile]:
        return dict(self._profiles)

    def profile_summary(self) -> list[dict]:
        """Serialisable summary for the frontend dropdown."""
        return [
            {
                "water_type":    wt.value,
                "name":          p.name,
                "dominant_nm":   p.dominant_nm,
                "base_turbidity": p.base_turbidity,
                "chlorophyll":   p.chlorophyll_mgl,
            }
            for wt, p in self._profiles.items()
        ]

    # ------------------------------------------------------------------ #
    #  Private — three Jaffe-McGlamery components                         #
    # ------------------------------------------------------------------ #

    def _direct_component(
        self,
        frame:    np.ndarray,
        depth:    float,
        profile:  JerlovProfile,
        turbidity: float,
    ) -> np.ndarray:
        """
        Ed = frame × exp(−c_eff × depth)   per channel.
        Turbidity amplifies the beam attenuation coefficient c.
        """
        c     = np.array(profile.attenuation_rgb, dtype=np.float32)
        c_eff = c * (1.0 + turbidity * 2.8)
        decay = np.exp(-c_eff * depth)                  # shape (3,)
        return (frame * decay[np.newaxis, np.newaxis, :]).astype(np.float32)

    def _forward_scatter(
        self,
        Ed:       np.ndarray,
        depth:    float,
        profile:  JerlovProfile,
        turbidity: float,
    ) -> np.ndarray:
        """
        Ef = Gaussian blur of Ed.
        Sigma grows with depth and turbidity — deeper & murkier = more blur.
        Applied per-channel because shorter wavelengths scatter more.
        """
        base_sigma = profile.forward_scatter_sig
        depth_factor = depth / 10.0
        sigma_rgb = np.array([
            base_sigma * (1.0 + turbidity * 1.2) * depth_factor * 1.0,   # R — least scatter
            base_sigma * (1.0 + turbidity * 1.4) * depth_factor * 1.1,   # G
            base_sigma * (1.0 + turbidity * 1.8) * depth_factor * 1.3,   # B — most scatter
        ], dtype=np.float32)

        sigma_rgb = np.clip(sigma_rgb, 0.0, 9.0)
        blurred   = np.zeros_like(Ed)

        for ch in range(3):
            if sigma_rgb[ch] < 0.1:
                blurred[:, :, ch] = Ed[:, :, ch]
            else:
                blurred[:, :, ch] = gaussian_filter(
                    Ed[:, :, ch], sigma=float(sigma_rgb[ch])
                )
        return blurred

    def _backscatter(
        self,
        shape:    tuple,
        state:    PhysicsState,
        profile:  JerlovProfile,
        turbidity: float,
    ) -> np.ndarray:
        """
        Eb = spatially-varying veiling luminance.

        Spatial structure:
          - Radial gradient: brighter near lens center (forward scatter from
            the drone's own lights reflecting back)
          - Vertical gradient: stronger near bottom (sediment stir-up from
            current and ROV prop wash)
          - Random particle noise: fine-grain texture from suspended particles
        """
        H, W, _ = shape
        rng      = np.random.default_rng(seed=42)   # deterministic grain

        b     = np.array(profile.backscatter_rgb, dtype=np.float32)
        b_eff = b * turbidity * (1.0 + state.depth_m / 25.0)

        # Radial mask — peaks at lens centre
        cy, cx   = H / 2.0, W / 2.0
        y_idx, x_idx = np.mgrid[0:H, 0:W]
        radial   = 1.0 - np.clip(
            np.sqrt(((y_idx - cy) / cy) ** 2 +
                    ((x_idx - cx) / cx) ** 2),
            0.0, 1.0
        )
        radial = (radial * 0.5 + 0.5).astype(np.float32)   # range 0.5→1.0

        # Vertical gradient — sediment heavier near bottom
        vert     = np.linspace(0.7, 1.3, H, dtype=np.float32).reshape(H, 1)

        # Particle grain — fine random noise scaled by turbidity
        grain    = rng.normal(0.0, turbidity * 0.012, (H, W)).astype(np.float32)
        grain    = np.clip(grain, -0.04, 0.04)

        spatial  = radial * vert    # (H, W)

        Eb       = np.zeros((H, W, 3), dtype=np.float32)
        for ch in range(3):
            Eb[:, :, ch] = b_eff[ch] * spatial + grain

        return np.clip(Eb, 0.0, 1.0)

    def _apply_cdom(
        self,
        frame:   np.ndarray,
        profile: JerlovProfile,
    ) -> np.ndarray:
        """
        Colored Dissolved Organic Matter (CDOM) absorbs blue light
        and gives water a yellow-brown tint in coastal/estuarine environments.
        Absorption follows an exponential spectral slope:
          a_cdom(λ) = a_cdom(440) × exp(−S × (λ − 440))
        S ≈ 0.014 nm⁻¹ (typical coastal value, Bricaud et al. 1981).
        We approximate for RGB channels at 650, 550, 450 nm.
        """
        S    = 0.014
        a440 = profile.cdom_abs
        # Absorption per channel (relative, normalised to G channel)
        a_r  = a440 * np.exp(-S * (650 - 440))
        a_g  = a440 * np.exp(-S * (550 - 440))
        a_b  = a440 * np.exp(-S * (450 - 440))

        absorb = np.array([a_r, a_g, a_b], dtype=np.float32)
        absorb = np.clip(absorb, 0.0, 0.6)

        result = frame.copy()
        result[:, :, 0] *= (1.0 - absorb[0])
        result[:, :, 1] *= (1.0 - absorb[1])
        result[:, :, 2] *= (1.0 - absorb[2])
        return result

    # ------------------------------------------------------------------ #
    #  Validation                                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate(frame: np.ndarray) -> None:
        if frame.dtype != np.float32:
            raise TypeError(
                f"Frame must be float32, got {frame.dtype}. "
                "Convert with: frame = frame.astype(np.float32) / 255.0"
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Frame must be shape (H, W, 3), got {frame.shape}"
            )
        if frame.max() > 1.0 + 1e-5 or frame.min() < -1e-5:
            raise ValueError(
                f"Frame values must be in [0, 1], "
                f"got min={frame.min():.4f} max={frame.max():.4f}"
            )