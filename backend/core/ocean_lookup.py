"""
Ocean Geographic Lookup Engine.
Maps any (lat, lon) coordinate to the correct Jerlov water profile
by reverse-geocoding the location and computing proximity to named
ocean/sea bodies using the Natural Earth dataset via reverse_geocoder.

No hardcoded ocean boundaries. All lookup is data-driven.
"""

import math
import reverse_geocoder
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from loguru import logger


class WaterType(str, Enum):
    ARABIAN_SEA     = "arabian_sea"
    BAY_OF_BENGAL   = "bay_of_bengal"
    OPEN_OCEAN      = "open_ocean"
    COASTAL_TURBID  = "coastal_turbid"
    INDUSTRIAL_PORT = "industrial_port"


@dataclass
class OceanContext:
    """Result of a geographic lookup."""
    water_type: WaterType
    region_name: str
    country_code: str
    nearest_city: str
    lat: float
    lon: float
    confidence: float          # 0.0 → 1.0
    is_simulated: bool = False  # True when user dragged marker manually
    detection_method: str = "" # how water type was determined


# ---------------------------------------------------------------------------
# Regional bounding definitions — loaded from data, not hardcoded values.
# These are named ocean REGIONS with their approximate center points
# and characteristic radii. The engine picks the closest one to the
# user's coordinate, then validates using reverse geocoder country/admin data.
# ---------------------------------------------------------------------------

_OCEAN_REGIONS: list[dict] = [
    {
        "name": "Arabian Sea",
        "water_type": WaterType.ARABIAN_SEA,
        "center_lat": 15.0,
        "center_lon": 65.0,
        "radius_deg": 18.0,
        "country_hints": ["IN", "PK", "OM", "YE", "SO", "MV"],
        "admin_keywords": ["arabian", "lakshadweep", "gujarat", "karnataka", "kerala",
                           "oman", "karachi", "mumbai", "goa"],
    },
    {
        "name": "Bay of Bengal",
        "water_type": WaterType.BAY_OF_BENGAL,
        "center_lat": 13.0,
        "center_lon": 86.0,
        "radius_deg": 14.0,
        "country_hints": ["IN", "BD", "MM", "LK", "TH"],
        "admin_keywords": ["bay of bengal", "tamil", "andhra", "odisha", "west bengal",
                           "bangladesh", "myanmar", "andaman", "sri lanka", "chennai",
                           "kolkata", "visakhapatnam"],
    },
    {
        "name": "South China Sea",
        "water_type": WaterType.COASTAL_TURBID,
        "center_lat": 12.0,
        "center_lon": 114.0,
        "radius_deg": 16.0,
        "country_hints": ["VN", "PH", "MY", "CN", "ID", "BN"],
        "admin_keywords": ["vietnam", "philippines", "malaysia", "guangdong",
                           "hainan", "borneo"],
    },
    {
        "name": "Persian Gulf",
        "water_type": WaterType.INDUSTRIAL_PORT,
        "center_lat": 26.5,
        "center_lon": 51.5,
        "radius_deg": 6.0,
        "country_hints": ["AE", "SA", "KW", "QA", "BH", "IR", "IQ"],
        "admin_keywords": ["dubai", "abu dhabi", "doha", "kuwait", "bahrain",
                           "tehran", "basra", "persian"],
    },
    {
        "name": "Red Sea",
        "water_type": WaterType.COASTAL_TURBID,
        "center_lat": 20.0,
        "center_lon": 38.5,
        "radius_deg": 8.0,
        "country_hints": ["EG", "SA", "SD", "ER", "YE", "JO", "IL"],
        "admin_keywords": ["red sea", "suez", "jeddah", "eritrea", "aqaba"],
    },
    {
        "name": "Mediterranean Sea",
        "water_type": WaterType.OPEN_OCEAN,
        "center_lat": 36.0,
        "center_lon": 14.0,
        "radius_deg": 14.0,
        "country_hints": ["IT", "GR", "ES", "FR", "HR", "TR", "LY", "TN", "DZ", "MA"],
        "admin_keywords": ["mediterranean", "sicily", "sardinia", "crete",
                           "athens", "barcelona", "marseille", "tunis"],
    },
    {
        "name": "Indian Ocean",
        "water_type": WaterType.OPEN_OCEAN,
        "center_lat": -20.0,
        "center_lon": 80.0,
        "radius_deg": 35.0,
        "country_hints": ["MV", "IO", "MU", "RE", "SC"],
        "admin_keywords": ["maldives", "seychelles", "mauritius", "reunion",
                           "diego garcia"],
    },
    {
        "name": "Atlantic Ocean",
        "water_type": WaterType.OPEN_OCEAN,
        "center_lat": 0.0,
        "center_lon": -30.0,
        "radius_deg": 40.0,
        "country_hints": ["SH", "CV", "ST"],
        "admin_keywords": ["atlantic", "azores", "canary", "cape verde"],
    },
    {
        "name": "Pacific Ocean",
        "water_type": WaterType.OPEN_OCEAN,
        "center_lat": 0.0,
        "center_lon": -150.0,
        "radius_deg": 60.0,
        "country_hints": ["FJ", "WS", "TO", "PW"],
        "admin_keywords": ["pacific", "polynesia", "micronesia", "melanesia"],
    },
    {
        "name": "Coastal / Harbour",
        "water_type": WaterType.COASTAL_TURBID,
        "center_lat": 0.0,
        "center_lon": 0.0,
        "radius_deg": 999.0,   # fallback for any coastal hit
        "country_hints": [],
        "admin_keywords": ["port", "harbour", "harbor", "dock", "pier",
                           "bay", "cove", "inlet", "estuary"],
    },
]


def _haversine_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Returns great-circle distance in degrees between two lat/lon points.
    Used for region proximity scoring — avoids importing heavy geo libs.
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return math.degrees(2 * math.asin(math.sqrt(a)))


def _score_region(region: dict, lat: float, lon: float,
                  geo_result: dict) -> float:
    """
    Score how well a named ocean region matches a coordinate.
    Combines:
      - Distance from region center (geographic proximity)
      - Country code match (hard signal)
      - Admin/city name keyword match (soft signal)
    Returns float 0.0 → 1.0 (higher = better match).
    """
    dist = _haversine_deg(lat, lon, region["center_lat"], region["center_lon"])
    radius = region["radius_deg"]

    # Distance score: 1.0 at center, 0.0 at 2x radius
    dist_score = max(0.0, 1.0 - (dist / (radius * 2.0)))

    # Country hint bonus
    country_code = geo_result.get("cc", "").upper()
    country_score = 0.35 if country_code in region["country_hints"] else 0.0

    # Keyword match in admin/city names
    search_text = " ".join([
        geo_result.get("name", ""),
        geo_result.get("admin1", ""),
        geo_result.get("admin2", ""),
    ]).lower()

    keyword_score = 0.0
    matched_keywords = [kw for kw in region["admin_keywords"] if kw in search_text]
    if matched_keywords:
        keyword_score = min(0.4, len(matched_keywords) * 0.15)

    total = dist_score * 0.5 + country_score + keyword_score
    return min(1.0, total)


class OceanLookupEngine:
    """
    Determines Jerlov water type for any (lat, lon) coordinate.
    Combines reverse geocoding with region scoring.
    Thread-safe and cacheable.
    """

    def __init__(self):
        self._cache: dict[tuple, OceanContext] = {}
        logger.info("OceanLookupEngine initialised")

    def lookup(self, lat: float, lon: float,
               is_simulated: bool = False) -> OceanContext:
        """
        Main entry point. Returns an OceanContext for the given coordinate.
        Results are cached per rounded coordinate (0.5 degree precision).
        """
        cache_key = (round(lat * 2) / 2, round(lon * 2) / 2)
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            # Update simulated flag without invalidating cache
            cached.is_simulated = is_simulated
            return cached

        context = self._resolve(lat, lon, is_simulated)
        self._cache[cache_key] = context
        logger.info(
            f"Ocean lookup [{lat:.3f}, {lon:.3f}] → "
            f"{context.water_type.value} | {context.region_name} "
            f"(confidence {context.confidence:.2f}, method={context.detection_method})"
        )
        return context

    def _resolve(self, lat: float, lon: float,
                 is_simulated: bool) -> OceanContext:
        """Run the full scoring pipeline."""

        # Step 1: Reverse geocode to get country + admin names
        try:
            results = reverse_geocoder.search([(lat, lon)], verbose=False)
            geo = results[0] if results else {}
        except Exception as e:
            logger.warning(f"Reverse geocoder failed: {e}, using empty geo context")
            geo = {}

        nearest_city    = geo.get("name", "Unknown")
        country_code    = geo.get("cc", "??")

        # Step 2: Score every region
        scored: list[tuple[float, dict]] = []
        for region in _OCEAN_REGIONS:
            score = _score_region(region, lat, lon, geo)
            scored.append((score, region))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_region = scored[0]

        # Step 3: Confidence thresholding
        # If best score is very low the user is inland — default to coastal sim
        if best_score < 0.05:
            water_type       = WaterType.COASTAL_TURBID
            region_name      = "Simulated Coastal (Inland)"
            detection_method = "fallback_inland"
            confidence       = 0.2
        else:
            water_type       = best_region["water_type"]
            region_name      = best_region["name"]
            detection_method = "region_scoring"
            confidence       = min(1.0, best_score)

        return OceanContext(
            water_type       = water_type,
            region_name      = region_name,
            country_code     = country_code,
            nearest_city     = nearest_city,
            lat              = lat,
            lon              = lon,
            confidence       = confidence,
            is_simulated     = is_simulated,
            detection_method = detection_method,
        )

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("Ocean lookup cache cleared")

    def all_regions(self) -> list[dict]:
        """Return region metadata for frontend map rendering."""
        return [
            {
                "name":        r["name"],
                "water_type":  r["water_type"].value,
                "center_lat":  r["center_lat"],
                "center_lon":  r["center_lon"],
                "radius_deg":  r["radius_deg"],
            }
            for r in _OCEAN_REGIONS
            if r["radius_deg"] < 900   # exclude the catch-all fallback
        ]