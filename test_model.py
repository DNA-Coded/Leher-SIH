"""
Interactive ML Model Tester
Run this script in your terminal to manually input ocean parameters and see the Rakshak models' predictions!
"""
import numpy as np
import xgboost as xgb
from pathlib import Path
import pyttsx3
import sys, os
import importlib.util

_e_path = Path(__file__).parent / "services" / "jal-chakra" / "core" / "ecosystem.py"
_e_spec = importlib.util.spec_from_file_location("ecosystem", _e_path)
ecosystem = importlib.util.module_from_spec(_e_spec)
_e_spec.loader.exec_module(ecosystem)

compute_coral_bleaching_risk = ecosystem.compute_coral_bleaching_risk
compute_hab_risk = ecosystem.compute_hab_risk
compute_fish_stress = ecosystem.compute_fish_stress
compute_hypoxia_risk = ecosystem.compute_hypoxia_risk

# Paths to the trained models and features
CYCLONE_MODEL_PATH = Path("fabric/ml/anomaly_model.json")
SURGE_MODEL_PATH = Path("fabric/ml/surge_model.json")
FEATURES_PATH = Path("fabric/ml/features_unlabeled.parquet")

def load_data_and_models():
    cyclone_model = xgb.XGBClassifier()
    surge_model = xgb.XGBRegressor()
    
    if not CYCLONE_MODEL_PATH.exists() or not SURGE_MODEL_PATH.exists() or not FEATURES_PATH.exists():
        print("Error: Models or feature data not found in fabric/ml/. Please run the training pipeline first.")
        return None, None, None
        
    cyclone_model.load_model(CYCLONE_MODEL_PATH)
    surge_model.load_model(SURGE_MODEL_PATH)
    
    print("Loading ocean data (this takes a few seconds)...")
    import polars as pl
    features_df = pl.read_parquet(FEATURES_PATH)
    
    return cyclone_model, surge_model, features_df

def get_float_input(prompt: str, default: float) -> float:
    user_input = input(f"{prompt} [default: {default}]: ").strip()
    if not user_input:
        return default
    try:
        return float(user_input)
    except ValueError:
        print("Invalid input, using default.")
        return default

def main():
    print("=" * 60)
    print("🌊 LEHER RAKSHAK - MANUAL ML TESTER 🌊")
    print("=" * 60)
    
    cyclone_model, surge_model, features_df = load_data_and_models()
    if not cyclone_model:
        return

    # Convert lat/lon columns to numpy for fast distance calculation
    lats = features_df["latitude"].to_numpy()
    lons = features_df["longitude"].to_numpy()

    while True:
        print("\n--- Enter Coordinates (Mimicking Map Click) ---")
        target_lat = get_float_input("1. Latitude (-20.0 to 25.0)", 15.0)
        target_lon = get_float_input("2. Longitude (50.0 to 100.0)", 85.0)
        
        # 1. FETCH DATA: Find the closest grid cell in the dataset
        dist = np.sqrt((lats - target_lat)**2 + (lons - target_lon)**2)
        closest_idx = np.argmin(dist)
        closest_dist_deg = dist[closest_idx]
        
        if closest_dist_deg > 2.0:
            print("\n⚠️ Warning: That location is far outside the Indian Ocean data grid. Results may be inaccurate.")
            
        row = features_df.row(closest_idx, named=True)
        
        # Extract fetched parameters
        sst = row["sst"]
        temp_100m = row["temp_100m"] if row["temp_100m"] is not None else 25.0
        thermal_contrast = row["thermal_contrast"] if row["thermal_contrast"] is not None else 0.0
        current_speed = row["current_speed"]
        surface_uo = row["surface_uo"]
        surface_vo = row["surface_vo"]
        vorticity = row["vorticity"]
        surface_salinity = row["surface_salinity"]  # Fetched! (so)
        
        # Placeholders for variables that require the live Copernicus API connection
        zos = row.get("zos", np.nan) if "zos" in features_df.columns else np.nan
        mlotst = row.get("mlotst", np.nan) if "mlotst" in features_df.columns else np.nan
        bottomT = row.get("bottomT", np.nan) if "bottomT" in features_df.columns else np.nan
        chl = row.get("chl", np.nan) if "chl" in features_df.columns else np.nan
        siconc = np.nan    # Sea-ice concentration (Not applicable in Indian Ocean)
        sithick = np.nan   # Sea-ice thickness
        
        # Calculate coastal proximity
        if 8.0 <= target_lat <= 22.0 and 80.0 <= target_lon <= 100.0:
            coastal_prox = 1.0
        else:
            coastal_prox = max(0.0, 1.0 - (abs(target_lat - 15.0) / 30.0 + abs(target_lon - 90.0) / 50.0))

        # 1. Test Cyclone Model
        # Features: sst, temp_100m, thermal_contrast, current_speed, surface_uo, surface_vo, vorticity, zos, mlotst
        cyclone_features = np.array([[
            sst, temp_100m, thermal_contrast, current_speed, surface_uo, surface_vo, vorticity, zos, mlotst
        ]], dtype=np.float32)
        
        cyclone_prob = cyclone_model.predict_proba(cyclone_features)[0][1]
        
        # 2. Test Surge Model
        # Features: sst, current_speed, surface_uo, surface_vo, coastal_proximity, zos
        surge_features = np.array([[
            sst, current_speed, surface_uo, surface_vo, coastal_prox, zos
        ]], dtype=np.float32)
        
        surge_height = surge_model.predict(surge_features)[0]

        # Determine safe zone risk
        risk = 0.0
        if current_speed > 1.5: risk += 0.5
        elif current_speed > 0.5: risk += 0.2
        if sst > 29.0: risk += 0.2
        if cyclone_prob > 0.65: risk += 0.5
        if surge_height > 0.5: risk += 0.3
        
        zone = "🔴 DANGER" if risk >= 0.7 else ("🟡 CAUTION" if risk >= 0.3 else "🟢 SAFE")

        print("\n" + "=" * 60)
        print(f"📍 FETCHED OCEAN DATA FOR {target_lat}°N, {target_lon}°E")
        print("=" * 60)
        print(f"Data Point Matched  : {row['latitude']:.2f}°N, {row['longitude']:.2f}°E (Distance: {closest_dist_deg*111:.0f} km)")
        print(f"Surface Temp (thetao): {sst:.2f} °C")
        print(f"Salinity (so)       : {surface_salinity:.2f} PSU")
        print(f"Current Speed       : {current_speed:.2f} m/s")
        print(f"Eastward Curr (uo)  : {surface_uo:.2f} m/s")
        print(f"Northward Curr (vo) : {surface_vo:.2f} m/s")
        print(f"Thermal Contrast    : {thermal_contrast:.2f} °C")
        print(f"Sea Sfc Height (zos): {zos:.2f} m" if not np.isnan(zos) else "Sea Sfc Height (zos): [Awaiting Live Copernicus API]")
        print(f"Mixed Layer (mlotst): {mlotst:.2f} m" if not np.isnan(mlotst) else "Mixed Layer (mlotst): [Awaiting Live Copernicus API]")
        print(f"Bottom Temp (bottomT): {bottomT:.2f} °C" if not np.isnan(bottomT) else "Bottom Temp (bottomT): [Awaiting Live Copernicus API]")
        print(f"Chlorophyll (chl)   : {chl:.2f} mg/m³" if not np.isnan(chl) else "Chlorophyll (chl)   : [Awaiting Live Copernicus Bio API]")
        print(f"Sea Ice (siconc)    : [Not Applicable in Indian Ocean]")
        
        print("\n" + "=" * 60)
        print(f"🤖 RAKSHAK INTELLIGENCE PREDICTIONS")
        print("=" * 60)
        print(f"Coastal Proximity   : {coastal_prox:.2f}")
        print(f"Cyclone Probability : {cyclone_prob * 100:.2f}%")
        print(f"Predicted Surge     : {surge_height:.2f} meters")
        print(f"Fishing Zone Status : {zone}")

        # --- Engine 6: Ecosystem Health ---
        coral = compute_coral_bleaching_risk(sst)
        hab = compute_hab_risk(sst, current_speed, surface_salinity)
        fish = compute_fish_stress(sst, thermal_contrast, current_speed)
        hypoxia = compute_hypoxia_risk(sst, current_speed)
        eco_score = round(
            coral["score"] * 0.30 + hab["score"] * 0.25 +
            fish["score"] * 0.25 + hypoxia["score"] * 0.20
        )
        if eco_score >= 70:
            eco_status = "🔴 CRITICAL"
        elif eco_score >= 50:
            eco_status = "🟠 STRESSED"
        elif eco_score >= 30:
            eco_status = "🟡 MODERATE"
        else:
            eco_status = "🟢 HEALTHY"

        print("\n" + "=" * 60)
        print(f"🏝️  MARINE ECOSYSTEM HEALTH ANALYSIS")
        print("=" * 60)
        print(f"Ecosystem Score     : {eco_score}/100 ({eco_status})")
        print(f"Coral Bleaching     : {coral['score']}/100 ({coral['level']})")
        print(f"Algal Bloom Risk    : {hab['score']}/100 ({hab['level']})")
        print(f"Fish Stress         : {fish['score']}/100 ({fish['level']})")
        print(f"Hypoxia Risk        : {hypoxia['score']}/100 ({hypoxia['level']})")
        print("=" * 60)

        # TTS Audio Support
        try:
            tts_engine = pyttsx3.init()
            if "DANGER" in zone:
                speech = f"Danger Zone Warning. High risk detected at {target_lat} North, {target_lon} East. Cyclone probability is {cyclone_prob * 100:.0f} percent."
            elif "CAUTION" in zone:
                speech = "Caution Zone. Proceed with care. Ocean currents are elevated."
            else:
                speech = "Safe Zone. No immediate ocean hazards detected."
            
            print(f"🔊 Playing Audio Advisory...")
            tts_engine.say(speech)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"[Audio Error: {e}]")

        again = input("\nTest another coordinate? (y/n): ").strip().lower()
        if again != 'y':
            break

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
