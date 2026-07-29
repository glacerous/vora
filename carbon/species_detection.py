import os
import requests
import logging

logger = logging.getLogger("SpeciesDetection")

def detect_species(image_paths):
    """
    Sends 1 to 3 images to the Pl@ntNet API for species identification.
    Returns:
      A list of top-3 predictions:
      [
        {"scientific_name": "...", "common_name": "...", "confidence": 85.4},
        ...
      ]
      Or None if API key is missing, request fails, or no matches found (graceful fallback).
    """
    api_key = os.environ.get("PLANTNET_API_KEY")
    if not api_key:
        logger.info("Pl@ntNet API key not set (PLANTNET_API_KEY env missing). Skipping species detection.")
        return None

    if not image_paths:
        logger.warning("No images provided for species detection.")
        return None

    # Limit to maximum 3 images
    target_images = image_paths[:3]
    url = f"https://my-api.plantnet.org/v2/identify/all?api-key={api_key}"
    
    files = []
    opened_files = []
    try:
        # Prepare multipart files
        # We classify tree closeups primarily as 'bark' (or fallback to 'leaf' if not specified)
        for path in target_images:
            if os.path.exists(path):
                f = open(path, 'rb')
                opened_files.append(f)
                files.append(('images', (os.path.basename(path), f, 'image/jpeg')))
                
        if not files:
            logger.warning("No valid images found on disk for species detection.")
            return None

        # Pl@ntNet requires 'organs' parameter for each image in matching order
        # Defaulting all frames from tree trunk video to 'bark'
        data = {
            'organs': ['bark'] * len(files)
        }

        logger.info(f"Sending {len(files)} images to Pl@ntNet for identification...")
        response = requests.post(url, files=files, data=data, timeout=15)
        
        if response.status_code != 200:
            logger.warning(f"Pl@ntNet API returned status code {response.status_code}: {response.text}")
            return None

        res_json = response.json()
        results = res_json.get("results", [])
        if not results:
            logger.info("Pl@ntNet found no matching species.")
            return None

        # Parse top-3 results
        predictions = []
        for match in results[:3]:
            species = match.get("species", {})
            scientific_name = species.get("scientificNameWithoutAuthor", "")
            common_names = species.get("commonNames", [])
            # Use the first common name if available, otherwise fallback to empty string
            common_name = common_names[0] if common_names else ""
            score = match.get("score", 0.0)
            
            predictions.append({
                "scientific_name": scientific_name,
                "common_name": common_name,
                "confidence": float(round(score * 100.0, 1))
            })

        logger.info(f"Species detection success. Top prediction: {predictions[0] if predictions else 'None'}")
        return predictions

    except Exception as e:
        logger.error(f"Error during species detection API call: {e}")
        return None
    finally:
        # Ensure all opened files are closed
        for f in opened_files:
            try:
                f.close()
            except Exception:
                pass
