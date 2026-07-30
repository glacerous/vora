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
    Tropical climates start with "A":
      - "Af": Wet forest
      - "Am": Moist forest
      - "Aw" / "As": Dry forest
      - Other "A" codes fallback to "moist"
    Non-tropical climates fallback to "moist" (default generic tropical assumption).
    """
    if not koppen_code:
        return "moist"
        
    code = koppen_code.strip()
    if not code.startswith("A"):
        # Not a tropical climate zone, default to moist for tropical carbon models
        return "moist"
        
    if code == "Af":
        return "wet"
    elif code == "Am":
        return "moist"
    elif code in ("Aw", "As"):
        return "dry"
        
    return "moist"
