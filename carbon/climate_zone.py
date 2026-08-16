import requests
import logging

logger = logging.getLogger("ClimateZone")

def get_koppen_classification(lat: float, lon: float) -> str:
    """
    Calls the MAPresso Climate API to get the Köppen-Geiger classification for a given lat/lon.
    Returns the Köppen-Geiger code (e.g. 'Af', 'Am', 'Aw', 'Cfb') or None if it fails.
    """
    if lat is None or lon is None:
        return None
        
    url = f"https://climate.mapresso.com/api/koeppen/?lat={lat}&lon={lon}"
    try:
        logger.info(f"Querying climate zone for coordinates: ({lat}, {lon})")
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            logger.warning(f"MAPresso API returned status code {response.status_code}")
            return None
            
        data = response.json()
        if data.get("status") == "OK" and "data" in data:
            for item in data["data"]:
                if item.get("type") == "Köppen-Geiger" and "code" in item:
                    code = item["code"]
                    logger.info(f"Detected Köppen climate code: {code} ({item.get('text', '')})")
                    return code
    except Exception as e:
        logger.error(f"Error querying climate zone from MAPresso: {e}")
        
    return None

def classify_koppen_to_forest_type(koppen_code: str) -> str:
    """
    Maps Köppen climate code to a Chave forest type: "dry", "moist", or "wet".
    Handles composite codes such as "As/Aw", "Aw/As", "Af/Am".
    Tropical climates start with "A":
      - "Af": Wet forest
      - "Am": Moist forest
      - "Aw" / "As": Dry forest
      - Other "A" codes fallback to "moist"
    Non-tropical climates fallback to "moist" (default generic tropical assumption).
    """
    if not koppen_code:
        return "moist"
        
    code_raw = koppen_code.strip()
    sub_codes = [c.strip() for c in code_raw.replace("-", "/").replace(",", "/").split("/") if c.strip()]
    if not sub_codes:
        return "moist"
        
    all_tropical = all(c.startswith("A") for c in sub_codes)
    if not all_tropical:
        return "moist"
        
    has_wet = any(c == "Af" for c in sub_codes)
    all_dry = all(c in ("Aw", "As") for c in sub_codes)
    has_dry = any(c in ("Aw", "As") for c in sub_codes)

    if has_wet:
        return "wet"
    if all_dry or (has_dry and not any(c in ("Af", "Am") for c in sub_codes)):
        return "dry"
    if any(c == "Am" for c in sub_codes):
        return "moist"
        
    return "moist"
