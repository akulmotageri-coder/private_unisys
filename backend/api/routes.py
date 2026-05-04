"""
FastAPI REST + WebSocket API.

Endpoints:
  GET  /                        — health check
  GET  /api/config              — current settings
  GET  /api/water-profiles      — all Jerlov profiles for frontend dropdown
  GET  /api/threat-classes      — threat class metadata for legend
  GET  /api/ocean-regions       — named ocean regions for map overlay
  POST /api/location            — update GPS / dragged marker position
  POST /api/physics             — update physics state from simulator sliders
  WS   /ws/stream               — real-time frame stream
  WS   /ws/telemetry            — detection + metric telemetry only (no video)
"""

import asyncio
import base64
import time
import traceback
from typing import Optional

import cv2
import numpy as np
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

from backend.config import settings
from backend.core.ocean_lookup import OceanLookupEngine, WaterType
from backend.core.water_physics import UnderwaterPhysicsEngine, PhysicsState
from backend.core.uiqm import UIQMEngine
from backend.pipeline.clahe_enhancer import CLAHEEnhancer
from backend.pipeline.detector import ThreatDetector

router = APIRouter()

# ---------------------------------------------------------------------------
# Shared pipeline instances — created once, reused across all connections
# ---------------------------------------------------------------------------

_ocean_lookup = OceanLookupEngine()
_physics_engine = UnderwaterPhysicsEngine()
_uiqm_engine = UIQMEngine()
_enhancer = CLAHEEnhancer()
_detector = ThreatDetector()

# ---------------------------------------------------------------------------
# Global mutable physics state — updated by REST calls, read by WS stream
# ---------------------------------------------------------------------------

_current_state = PhysicsState(
    depth_m          = settings.DEFAULT_DEPTH_M,
    turbidity        = settings.DEFAULT_TURBIDITY,
    water_type       = WaterType(settings.DEFAULT_WATER_TYPE),
    ambient_light    = 0.80,
    salinity_ppt     = 35.0,
    temperature_c    = 28.0,
    current_speed_ms = 0.20,
    lat              = settings.DEFAULT_LAT,
    lon              = settings.DEFAULT_LON,
)

_current_location_meta: dict = {}
_frame_counter: int = 0
_pipeline_stats: dict = {
    "frames_processed": 0,
    "total_threats":    0,
    "avg_uiqm_raw":     0.0,
    "avg_uiqm_enhanced":0.0,
    "avg_inference_ms": 0.0,
    "uptime_s":         0.0,
    "start_time":       time.time(),
}

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class LocationUpdate(BaseModel):
    lat:          float = Field(..., ge=-90,  le=90)
    lon:          float = Field(..., ge=-180, le=180)
    is_simulated: bool  = Field(default=False)


class PhysicsUpdate(BaseModel):
    depth_m:          Optional[float] = Field(None, ge=0,   le=200)
    turbidity:        Optional[float] = Field(None, ge=0.0, le=1.0)
    water_type:       Optional[str]   = None
    ambient_light:    Optional[float] = Field(None, ge=0.0, le=1.0)
    salinity_ppt:     Optional[float] = Field(None, ge=0.0, le=45.0)
    temperature_c:    Optional[float] = Field(None, ge=-2,  le=40)
    current_speed_ms: Optional[float] = Field(None, ge=0.0, le=5.0)


# ---------------------------------------------------------------------------
# Frame generator — produces synthetic underwater frames
# ---------------------------------------------------------------------------

class FrameGenerator:
    """
    Generates synthetic clean frames that look like real underwater footage.
    Uses a combination of:
      - Procedural texture (Perlin-like noise via OpenCV)
      - Animated fish silhouettes
      - Seabed/rock geometry
      - Time-varying caustic light patterns
    All without any external assets.
    """

    def __init__(self, width: int, height: int):
        self.W   = width
        self.H   = height
        self._t  = 0.0
        self._rng = np.random.default_rng(seed=99)
        self._seabed  = self._make_seabed()
        self._rocks   = self._make_rocks()

    def next_frame(self, state: PhysicsState) -> np.ndarray:
        """
        Generate the next synthetic clean frame.
        Returns float32 (H, W, 3) RGB [0, 1].
        """
        self._t += 0.04

        canvas = self._seabed.copy()
        canvas = self._draw_caustics(canvas)
        canvas = self._draw_rocks(canvas)
        canvas = self._draw_particles(canvas, state.turbidity)
        canvas = self._draw_depth_gradient(canvas, state.depth_m)

        self._t += 0.01
        return np.clip(canvas, 0.0, 1.0).astype(np.float32)

    def _make_seabed(self) -> np.ndarray:
        """Sandy seabed base texture."""
        base  = np.zeros((self.H, self.W, 3), dtype=np.float32)
        noise = self._rng.random((self.H // 4, self.W // 4)).astype(np.float32)
        noise = cv2.resize(noise, (self.W, self.H), interpolation=cv2.INTER_CUBIC)
        noise = cv2.GaussianBlur(noise, (21, 21), 0)

        # Sandy tan colour
        base[:, :, 0] = 0.72 + noise * 0.22
        base[:, :, 1] = 0.62 + noise * 0.18
        base[:, :, 2] = 0.45 + noise * 0.15

        # Darker near top (water column above), lighter near bottom (seabed)
        gradient = np.linspace(0.4, 1.0, self.H, dtype=np.float32).reshape(self.H, 1)
        base *= gradient[:, :, np.newaxis]

        return np.clip(base, 0.0, 1.0)

    def _make_rocks(self) -> list[dict]:
        """Pre-compute rock positions and sizes."""
        rocks = []
        for _ in range(self._rng.integers(4, 9)):
            rocks.append({
                "cx":    int(self._rng.integers(60, self.W - 60)),
                "cy":    int(self._rng.integers(self.H // 2, self.H - 40)),
                "rx":    int(self._rng.integers(20, 65)),
                "ry":    int(self._rng.integers(15, 40)),
                "color": tuple(self._rng.uniform(0.25, 0.45, 3).tolist()),
            })
        return rocks

    def _draw_rocks(self, canvas: np.ndarray) -> np.ndarray:
        out = canvas.copy()
        for r in self._rocks:
            mask = np.zeros((self.H, self.W), dtype=np.float32)
            cv2.ellipse(
                mask,
                (r["cx"], r["cy"]),
                (r["rx"], r["ry"]),
                0, 0, 360,
                1.0, -1,
            )
            for ch in range(3):
                out[:, :, ch] = np.where(
                    mask > 0,
                    r["color"][ch],
                    out[:, :, ch],
                )
        return out

    def _draw_caustics(self, canvas: np.ndarray) -> np.ndarray:
        """
        Animated caustic light patterns (refracted sunlight ripples).
        Approximated as sum of shifted sinusoids.
        """
        H, W = self.H, self.W
        x = np.linspace(0, 2 * np.pi, W, dtype=np.float32)
        y = np.linspace(0, 2 * np.pi, H, dtype=np.float32)
        xx, yy = np.meshgrid(x, y)

        t = self._t
        caustic = (
            np.sin(xx * 3.1 + t * 1.3) *
            np.cos(yy * 2.7 - t * 0.9) +
            np.sin(xx * 1.8 - t * 0.7 + yy * 2.2) * 0.5
        )
        caustic = (caustic - caustic.min()) / (caustic.max() - caustic.min() + 1e-6)
        caustic *= 0.12   # subtle

        out = canvas.copy()
        out[:, :, 0] += caustic * 0.6
        out[:, :, 1] += caustic * 0.8
        out[:, :, 2] += caustic * 1.0
        return np.clip(out, 0.0, 1.0)

    def _draw_particles(
        self, canvas: np.ndarray, turbidity: float
    ) -> np.ndarray:
        """
        Floating sediment particles — density scales with turbidity.
        Each particle is a tiny Gaussian blob.
        """
        out     = canvas.copy()
        n_parts = int(turbidity * 80)
        if n_parts == 0:
            return out

        rng = np.random.default_rng(
            seed=int(self._t * 1000) % 100000
        )
        for _ in range(n_parts):
            px  = int(rng.integers(0, self.W))
            py  = int(rng.integers(0, self.H))
            rad = int(rng.integers(1, 4))
            brightness = float(rng.uniform(0.5, 0.9))
            cv2.circle(out, (px, py), rad,
                       (brightness, brightness * 0.9, brightness * 0.7), -1)
        return np.clip(out, 0.0, 1.0)

    def _draw_depth_gradient(
        self, canvas: np.ndarray, depth_m: float
    ) -> np.ndarray:
        """Deeper = darker overall, more blue-green cast."""
        depth_factor = float(np.clip(1.0 - depth_m / 120.0, 0.45, 1.0))
        out          = canvas.copy()
        out[:, :, 0] *= depth_factor * 0.90
        out[:, :, 1] *= depth_factor * 0.95
        out[:, :, 2] *= depth_factor * 1.00
        return np.clip(out, 0.0, 1.0)

# Singleton frame generator
_frame_gen = FrameGenerator(settings.FRAME_WIDTH, settings.FRAME_HEIGHT)


# ---------------------------------------------------------------------------
# Frame encoder
# ---------------------------------------------------------------------------

def _encode_frame(frame: np.ndarray, quality: int = 82) -> str:
    """
    Encode a float32 [0,1] RGB frame to base64 JPEG string.
    Returns data-URI string ready for <img src="..."> in frontend.
    """
    uint8 = (frame * 255.0).clip(0, 255).astype(np.uint8)
    bgr   = cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(
        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
    )
    if not ok:
        raise RuntimeError("Frame encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@router.get("/")
async def health():
    return {
        "status":  "online",
        "app":     settings.APP_NAME,
        "version": settings.APP_VERSION,
        "mode":    _detector.mode,
    }


@router.get("/api/config")
async def get_config():
    return {
        "frame_width":          settings.FRAME_WIDTH,
        "frame_height":         settings.FRAME_HEIGHT,
        "target_fps":           settings.TARGET_FPS,
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
        "nms_threshold":        settings.NMS_THRESHOLD,
        "default_lat":          settings.DEFAULT_LAT,
        "default_lon":          settings.DEFAULT_LON,
        "enable_gps":           settings.ENABLE_GPS,
    }


@router.get("/api/water-profiles")
async def get_water_profiles():
    return {"profiles": _physics_engine.profile_summary()}


@router.get("/api/threat-classes")
async def get_threat_classes():
    return {"classes": _detector.class_info()}


@router.get("/api/ocean-regions")
async def get_ocean_regions():
    return {"regions": _ocean_lookup.all_regions()}


@router.get("/api/stats")
async def get_stats():
    _pipeline_stats["uptime_s"] = round(
        time.time() - _pipeline_stats["start_time"], 1
    )
    return _pipeline_stats

@router.get("/api/frame/raw")
async def get_raw_frame():
    """Returns latest raw frame as JPEG — polled by frontend."""
    from fastapi.responses import Response
    state = _current_state
    clean = _frame_gen.next_frame(state)
    raw   = _physics_engine.degrade(clean, state)
    uint8 = (raw * 255.0).clip(0, 255).astype(np.uint8)
    bgr   = cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 55])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.get("/api/frame/enhanced")
async def get_enhanced_frame():
    """Returns latest enhanced frame as JPEG — polled by frontend."""
    from fastapi.responses import Response
    state      = _current_state
    clean      = _frame_gen.next_frame(state)
    raw        = _physics_engine.degrade(clean, state)
    enh_result = _enhancer.enhance(raw, state)
    enhanced   = enh_result.enhanced
    uint8      = (enhanced * 255.0).clip(0, 255).astype(np.uint8)
    bgr        = cv2.cvtColor(uint8, cv2.COLOR_RGB2BGR)
    _, buf     = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.post("/api/location")
async def update_location(body: LocationUpdate):
    """
    Called when:
      - Browser GPS fires
      - User drags the marker on the map
    Updates water type automatically from coordinates.
    """
    global _current_state, _current_location_meta

    ctx = _ocean_lookup.lookup(
        body.lat,
        body.lon,
        is_simulated=body.is_simulated,
    )

    _current_state.lat        = body.lat
    _current_state.lon        = body.lon
    _current_state.water_type = ctx.water_type

    _current_location_meta = {
        "lat":              body.lat,
        "lon":              body.lon,
        "water_type":       ctx.water_type.value,
        "region_name":      ctx.region_name,
        "nearest_city":     ctx.nearest_city,
        "country_code":     ctx.country_code,
        "confidence":       ctx.confidence,
        "is_simulated":     ctx.is_simulated,
        "detection_method": ctx.detection_method,
    }

    logger.info(
        f"Location updated → {ctx.region_name} "
        f"({ctx.water_type.value}) "
        f"conf={ctx.confidence:.2f}"
    )
    return {"status": "ok", "context": _current_location_meta}


@router.post("/api/physics")
async def update_physics(body: PhysicsUpdate):
    """
    Called when user adjusts sliders in the simulator panel.
    Only updates fields that were actually sent.
    """
    global _current_state

    if body.depth_m          is not None:
        _current_state.depth_m          = body.depth_m
    if body.turbidity         is not None:
        _current_state.turbidity         = body.turbidity
    if body.water_type        is not None:
        try:
            _current_state.water_type = WaterType(body.water_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown water_type '{body.water_type}'"
            )
    if body.ambient_light     is not None:
        _current_state.ambient_light     = body.ambient_light
    if body.salinity_ppt      is not None:
        _current_state.salinity_ppt      = body.salinity_ppt
    if body.temperature_c     is not None:
        _current_state.temperature_c     = body.temperature_c
    if body.current_speed_ms  is not None:
        _current_state.current_speed_ms  = body.current_speed_ms

    return {
        "status": "ok",
        "state": {
            "depth_m":          _current_state.depth_m,
            "turbidity":        _current_state.turbidity,
            "water_type":       _current_state.water_type.value,
            "ambient_light":    _current_state.ambient_light,
            "salinity_ppt":     _current_state.salinity_ppt,
            "temperature_c":    _current_state.temperature_c,
            "current_speed_ms": _current_state.current_speed_ms,
        }
    }


# ---------------------------------------------------------------------------
# WebSocket — full frame stream
# ---------------------------------------------------------------------------

@router.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Streams one JSON message per frame containing:
      - base64-encoded raw (degraded) frame
      - base64-encoded enhanced frame
      - detection bounding boxes
      - UIQM metrics
      - CLAHE parameters
      - current physics state
    Target: 20-30 FPS depending on client connection.
    """
    await ws.accept()
    logger.info("WebSocket stream client connected")

    frame_interval = 1.0 / settings.TARGET_FPS

    try:
        while True:
            t_frame_start = time.perf_counter()

            state = _current_state   # snapshot

            # ── 1. Generate synthetic clean frame ──
            clean = _frame_gen.next_frame(state)

            # ── 2. Degrade with Jaffe-McGlamery physics ──
            raw = _physics_engine.degrade(clean, state)

            # ── 3. Enhance with auto-adaptive CLAHE ──
            enh_result = _enhancer.enhance(raw, state)
            enhanced   = enh_result.enhanced

            # ── 4. Detect threats ──
            turb_est = enh_result.uiqm_before.get(
                "frame_std", state.turbidity
            )
            det_result = _detector.detect(
                enhanced,
                turbidity  = state.effective_turbidity(
                    _physics_engine.get_profile(state.water_type)
                ),
                depth_m    = state.depth_m,
                water_type = state.water_type,
            )

            # ── 5. Encode frames ──
            # Frames delivered via HTTP polling — not via WebSocket
            raw_b64 = None
            enh_b64 = None
            # ── 6. Update global stats ──
            _pipeline_stats["frames_processed"] += 1
            _pipeline_stats["total_threats"]    += det_result.total_threats
            n = _pipeline_stats["frames_processed"]
            _pipeline_stats["avg_uiqm_raw"] = (
                (_pipeline_stats["avg_uiqm_raw"] * (n - 1) +
                 enh_result.uiqm_before["uiqm"]) / n
            )
            _pipeline_stats["avg_uiqm_enhanced"] = (
                (_pipeline_stats["avg_uiqm_enhanced"] * (n - 1) +
                 enh_result.uiqm_after["uiqm"]) / n
            )
            _pipeline_stats["avg_inference_ms"] = (
                (_pipeline_stats["avg_inference_ms"] * (n - 1) +
                 det_result.inference_ms) / n
            )

            # ── 7. Build and send payload ──
            payload = {
                "type":        "frame",
                "frame_id":    det_result.frame_id,
                "timestamp":   time.time(),
                "raw_frame":   "/api/frame/raw",
                "enh_frame":   "/api/frame/enhanced",
                "detections":  det_result.to_dict(),
                "enhancement": enh_result.to_meta(),
                "physics": {
                    "depth_m":          state.depth_m,
                    "turbidity":        state.turbidity,
                    "water_type":       state.water_type.value,
                    "ambient_light":    state.ambient_light,
                    "salinity_ppt":     state.salinity_ppt,
                    "temperature_c":    state.temperature_c,
                    "current_speed_ms": state.current_speed_ms,
                    "lat":              state.lat,
                    "lon":              state.lon,
                },
                "location":    _current_location_meta,
                "stats":       {
                    k: v for k, v in _pipeline_stats.items()
                    if k != "start_time"
                },
            }

            await ws.send_json(payload)

            # ── 8. Pace to target FPS ──
            elapsed = time.perf_counter() - t_frame_start
            sleep   = frame_interval - elapsed
            if sleep > 0:
                await asyncio.sleep(sleep)

    except WebSocketDisconnect:
        logger.info("WebSocket stream client disconnected")
    except Exception as exc:
        logger.error(f"WebSocket error: {exc}\n{traceback.format_exc()}")
        try:
            await ws.send_json({"type": "error", "message": str(exc)})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# WebSocket — telemetry only (no video, for bandwidth-limited clients)
# ---------------------------------------------------------------------------

@router.websocket("/ws/telemetry")
async def websocket_telemetry(ws: WebSocket):
    """
    Lightweight telemetry stream.
    Sends detection results and metrics without frame images.
    Useful for secondary dashboard panels or mobile clients.
    """
    await ws.accept()
    logger.info("WebSocket telemetry client connected")

    try:
        while True:
            state      = _current_state
            clean      = _frame_gen.next_frame(state)
            raw        = _physics_engine.degrade(clean, state)
            enh_result = _enhancer.enhance(raw, state)
            det_result = _detector.detect(
                enh_result.enhanced,
                turbidity  = state.effective_turbidity(
                    _physics_engine.get_profile(state.water_type)
                ),
                depth_m    = state.depth_m,
                water_type = state.water_type,
            )

            await ws.send_json({
                "type":        "telemetry",
                "frame_id":    det_result.frame_id,
                "timestamp":   time.time(),
                "detections":  det_result.to_dict(),
                "enhancement": enh_result.to_meta(),
                "physics": {
                    "depth_m":    state.depth_m,
                    "turbidity":  state.turbidity,
                    "water_type": state.water_type.value,
                },
            })

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        logger.info("WebSocket telemetry client disconnected")
    except Exception as exc:
        logger.error(f"Telemetry WS error: {exc}")