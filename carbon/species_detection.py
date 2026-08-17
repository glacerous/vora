import os
import requests
import logging

logger = logging.getLogger("SpeciesDetection")

def detect_species_with_status(image_paths):
    """
    Sends 1 to 3 images to the Pl@ntNet API for species identification.
    Returns:
      (predictions: list | None, status_reason: str)
      status_reason can be: 'success', 'missing_api_key', 'no_images_provided',
      'no_valid_images_found', 'api_error_HTTP_<status>', 'no_matches_found', 'api_exception_<msg>'
    """
    api_key = os.environ.get("PLANTNET_API_KEY")
    if not api_key:
        logger.info("Pl@ntNet API key not set (PLANTNET_API_KEY env missing). Skipping species detection.")
        return None, "missing_api_key"

    if not image_paths:
        logger.warning("No images provided for species detection.")
        return None, "no_images_provided"

    # Limit to maximum 3 images
    target_images = image_paths[:3]
    url = f"https://my-api.plantnet.org/v2/identify/all?api-key={api_key}"
    
    files = []
    try:
        from PIL import Image
        import io

        total_bytes_sent = 0
        for path in target_images:
            if os.path.exists(path):
                try:
                    img = Image.open(path)
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    # Maintain aspect ratio, max 800px dimension
                    if img.width > 800 or img.height > 800:
                        img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                    
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=80)
                    img_bytes = buf.getvalue()
                    total_bytes_sent += len(img_bytes)
                    
                    files.append(('images', (os.path.basename(path), img_bytes, 'image/jpeg')))
                except Exception as img_err:
                    logger.warning(f"Failed to process image {path} for Pl@ntNet: {img_err}")
                
        if not files:
            logger.warning("No valid images found on disk for species detection.")
            return None, "no_valid_images_found"

        data = {
            'organs': ['bark'] * len(files)
        }

        logger.info(f"Sending {len(files)} downscaled images ({total_bytes_sent / 1024:.1f} KB total outbound) to Pl@ntNet...")
        response = requests.post(url, files=files, data=data, timeout=15)
        
        if response.status_code != 200:
            logger.warning(f"Pl@ntNet API returned status code {response.status_code}: {response.text}")
            return None, f"api_error_HTTP_{response.status_code}"

        res_json = response.json()
        results = res_json.get("results", [])
        if not results:
            logger.info("Pl@ntNet found no matching species.")
            return None, "no_matches_found"

        # Parse top-3 results
        predictions = []
        for match in results[:3]:
            species = match.get("species", {})
            scientific_name = species.get("scientificNameWithoutAuthor", "")
            common_names = species.get("commonNames", [])
            common_name = common_names[0] if common_names else ""
            score = match.get("score", 0.0)
            
            predictions.append({
                "scientific_name": scientific_name,
                "common_name": common_name,
                "confidence": float(round(score * 100.0, 1))
            })

        logger.info(f"Species detection success. Top prediction: {predictions[0] if predictions else 'None'}")
        return predictions, "success"

    except Exception as e:
        logger.error(f"Error during species detection API call: {e}")
        return None, f"api_exception_{type(e).__name__}"


def detect_species(image_paths):
    """
    Backward-compatible wrapper returning only predictions (list or None).
    """
    preds, _ = detect_species_with_status(image_paths)
    return preds
