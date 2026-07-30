import os
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Optional, Tuple

def get_exif_gps(image_path: str) -> Optional[Tuple[float, float]]:
    """
    Extracts GPS coordinates (latitude, longitude) from JPEG EXIF metadata.
    Returns: (latitude, longitude) as floats, or None if not found/invalid.
    """
    try:
        if not os.path.exists(image_path):
            return None
            
        img = Image.open(image_path)
        exif_data = img._getexif()
        if not exif_data:
            return None
            
        gps_info = {}
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
                    
        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            lat = gps_info["GPSLatitude"]
            lat_ref = gps_info.get("GPSLatitudeRef", "N")
            lon = gps_info["GPSLongitude"]
            lon_ref = gps_info.get("GPSLongitudeRef", "E")
            
            def to_degrees(val):
                # Pillow GPS tags can be tuples of (degrees, minutes, seconds)
                # or direct numbers or tuples of rational numbers
                if isinstance(val, (list, tuple)) and len(val) >= 3:
                    def convert_rational(r):
                        if isinstance(r, tuple) and len(r) == 2:
                            return float(r[0]) / float(r[1]) if r[1] != 0 else 0.0
                        # Pillow 10.0+ might parse it directly to float or Fraction
                        try:
                            return float(r)
                        except Exception:
                            return 0.0
                    
                    degrees = convert_rational(val[0])
                    minutes = convert_rational(val[1])
                    seconds = convert_rational(val[2])
                    return degrees + (minutes / 60.0) + (seconds / 3600.0)
                try:
                    return float(val)
                except Exception:
                    return 0.0
                
            lat_deg = to_degrees(lat)
            if lat_ref == "S":
                lat_deg = -lat_deg
            lon_deg = to_degrees(lon)
            if lon_ref == "W":
                lon_deg = -lon_deg
                
            if lat_deg != 0.0 or lon_deg != 0.0:
                return lat_deg, lon_deg
    except Exception as e:
        print(f"[GPS EXIF] Failed to parse EXIF GPS: {e}")
        
    return None
