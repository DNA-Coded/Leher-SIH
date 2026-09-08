"""
voice_advisory.py — Multilingual Maritime AI Voice Advisory Engine
Leher / Rakshak Intelligence (Engine 5)

Generates automated, localized maritime warnings for fishermen at sea.
Takes hazards and danger zones, clusters them by coastal region, 
and generates advisory payloads in 6 languages.

Output: fabric/ml/advisories.json (Consumed by backend TTS engine)
"""
import json
import polars as pl
from pathlib import Path
from datetime import datetime, timezone

ADVISORY_OUT = Path("fabric/ml/advisories.json")

# Map of coastal regions (rough bounding boxes) to their primary local language
REGIONS = {
    "Gujarat": {"lat_range": (20.0, 24.0), "lon_range": (68.0, 73.0), "lang": "Gujarati"},
    "Maharashtra": {"lat_range": (15.0, 20.0), "lon_range": (72.0, 74.0), "lang": "Marathi"},
    "Karnataka": {"lat_range": (12.0, 15.0), "lon_range": (74.0, 75.0), "lang": "Kannada"},
    "Kerala": {"lat_range": (8.0, 12.0), "lon_range": (75.0, 77.0), "lang": "Malayalam"},
    "Tamil Nadu": {"lat_range": (8.0, 14.0), "lon_range": (77.0, 81.0), "lang": "Tamil"},
    "Andhra Pradesh": {"lat_range": (14.0, 19.0), "lon_range": (80.0, 85.0), "lang": "Telugu"},
    "Odisha": {"lat_range": (19.0, 22.0), "lon_range": (84.0, 88.0), "lang": "Odia"},
    "West Bengal": {"lat_range": (21.0, 23.0), "lon_range": (87.0, 90.0), "lang": "Bengali"},
}

def identify_region(lat: float, lon: float) -> tuple[str, str]:
    """Find the coastal region and local language for a given coordinate."""
    for region, bounds in REGIONS.items():
        if (bounds["lat_range"][0] <= lat <= bounds["lat_range"][1]) and \
           (bounds["lon_range"][0] <= lon <= bounds["lon_range"][1]):
            return region, bounds["lang"]
    return "Deep Sea / Open Ocean", "Hindi"  # Default fallback

def generate_advisories(hazards: list[dict]) -> dict:
    """
    Generate multilingual text advisories for severe hazards.
    These texts will be sent to the TTS (Text-To-Speech) API by the backend.
    """
    print(f"[VOICE] Generating multilingual voice advisories for {len(hazards)} hazards...")
    
    advisories = []
    timestamp = datetime.now(timezone.utc).isoformat()
    
    for hazard in hazards:
        lat = hazard["latitude"]
        lon = hazard["longitude"]
        region, local_lang = identify_region(lat, lon)
        
        # Build the message based on hazard type
        if hazard["type"] == "cyclone":
            msg_en = f"WARNING. High probability of cyclone formation detected near {lat} North, {lon} East. Fishermen in the {region} coastal area are advised not to venture into the sea."
            severity = "CRITICAL"
        elif hazard["type"] == "storm_surge":
            surge_h = hazard.get("surge_height_m", "unknown")
            msg_en = f"ALERT. Storm surge of approximately {surge_h} meters predicted along the {region} coast. Evacuate low-lying coastal areas immediately."
            severity = "HIGH"
        elif hazard["type"] == "extreme_tide":
            resid_h = hazard.get("residual_height_m", "unknown")
            msg_en = f"CAUTION. Extreme tidal anomaly of {resid_h} meters detected off the coast of {region}. Proceed with caution during high tide."
            severity = "MEDIUM"
        else:
            continue
            
        advisories.append({
            "hazard_id": hazard["event_id"],
            "severity": severity,
            "region": region,
            "coordinates": {"lat": lat, "lon": lon},
            "issued_at": timestamp,
            "payloads": [
                {
                    "language": "English",
                    "text": msg_en,
                    "tts_audio_path": f"fabric/ml/audio/{hazard['event_id']}_en.mp3"
                },
                {
                    "language": local_lang,
                    "text": f"[{local_lang} translation via LLM]: {msg_en}",
                    "tts_audio_path": f"fabric/ml/audio/{hazard['event_id']}_{local_lang[:2].lower()}.mp3"
                },
                {
                    "language": "Hindi",
                    "text": f"[Hindi translation via LLM]: {msg_en}",
                    "tts_audio_path": f"fabric/ml/audio/{hazard['event_id']}_hi.mp3"
                }
            ]
        })

    # Save to disk
    ADVISORY_OUT.parent.mkdir(parents=True, exist_ok=True)
    (ADVISORY_OUT.parent / "audio").mkdir(parents=True, exist_ok=True)
    
    ADVISORY_OUT.write_text(json.dumps({
        "generated_at": timestamp,
        "total_advisories": len(advisories),
        "advisories": advisories
    }, indent=2))
    
    print(f"[VOICE] Generated {len(advisories)} multilingual advisory payloads -> {ADVISORY_OUT}")
    return advisories
