"""
Global Wood Density Lookup Module
Integrates the Global Wood Density Database (GWDD - Zanne et al. 2009 / Chave et al. / Fischer et al.)
Provides a 3-tier hierarchical lookup:
  Tier 1: Exact Species Match (Binomial)
  Tier 2: Genus-Level Average Match
  Tier 3: Global Generic Default (0.60 g/cm³)
"""

import os
import csv
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("WoodDensity")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BINOMIAL_CSV = os.path.join(DATA_DIR, "gwdd_binomial.csv")
GENUS_CSV = os.path.join(DATA_DIR, "gwdd_genus.csv")

# In-memory lookup tables (populated lazily or at module load)
_SPECIES_MAP: Dict[str, float] = {}
_SPECIES_META: Dict[str, Dict[str, Any]] = {}
_GENUS_MAP: Dict[str, float] = {}
_GENUS_META: Dict[str, Dict[str, Any]] = {}
_INITIALIZED: bool = False

# Fallback basic dictionary (curated tropical species) if CSVs are unavailable
_FALLBACK_LOOKUP: Dict[str, float] = {
    "swietenia macrophylla": 0.53,      # Mahogany
    "tectona grandis": 0.55,            # Teak
    "hevea brasiliensis": 0.50,         # Rubber tree
    "acacia mangium": 0.53,             # Acacia
    "pinus merkusii": 0.45,             # Sumatran pine
    "eucalyptus globulus": 0.65,        # Blue gum
    "cocos nucifera": 0.35,             # Coconut
    "mangifera indica": 0.58,           # Mango
    "artocarpus heterophyllus": 0.55,   # Jackfruit / Nangka
    "paraserianthes falcataria": 0.33,  # Sengon / Albasia
    "spathodea campanulata": 0.28,      # African Tulip
    "pterocarpus indicus": 0.56,        # Angsana
    "ficus elastica": 0.41,             # Rubber Fig
    "roystonea regia": 0.30,            # Royal Palm
    "gmelina arborea": 0.42,            # Gmelina
    "delonix regia": 0.40,              # Flamboyant
    "terminalia catappa": 0.48,         # Ketapang / Indian Almond
    "leucaena leucocephala": 0.60,      # Lamtoro
    "albizia chinensis": 0.35,          # Sengon Jawa
    "manilkara zapota": 0.85,           # Sawo / Sapodilla
    "tamarindus indica": 0.75,          # Asam Jawa / Tamarind
}


def _load_gwdd_database():
    global _SPECIES_MAP, _SPECIES_META, _GENUS_MAP, _GENUS_META, _INITIALIZED
    if _INITIALIZED:
        return

    # Load species/binomial CSV
    if os.path.exists(BINOMIAL_CSV):
        try:
            with open(BINOMIAL_CSV, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    binomial = row.get("binomial", "").strip()
                    wsg_str = row.get("wsg_est") or row.get("wsg_raw")
                    if binomial and wsg_str:
                        try:
                            val = float(wsg_str)
                            k = binomial.lower()
                            _SPECIES_MAP[k] = val
                            _SPECIES_META[k] = {
                                "binomial": binomial,
                                "genus": row.get("genus", ""),
                                "family": row.get("family", ""),
                                "nb_samples": row.get("nb", "1"),
                                "wsg": val
                            }
                        except ValueError:
                            pass
            logger.info(f"Loaded {len(_SPECIES_MAP)} species wood density records from GWDD.")
        except Exception as e:
            logger.warning(f"Failed to load GWDD binomial CSV: {e}")

    # Load genus CSV
    if os.path.exists(GENUS_CSV):
        try:
            with open(GENUS_CSV, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    genus = row.get("genus", "").strip()
                    wsg_str = row.get("wsg_est") or row.get("wsg_raw")
                    if genus and wsg_str:
                        try:
                            val = float(wsg_str)
                            k = genus.lower()
                            _GENUS_MAP[k] = val
                            _GENUS_META[k] = {
                                "genus": genus,
                                "family": row.get("family", ""),
                                "nb_samples": row.get("nb", "1"),
                                "wsg": val
                            }
                        except ValueError:
                            pass
            logger.info(f"Loaded {len(_GENUS_MAP)} genus wood density records from GWDD.")
        except Exception as e:
            logger.warning(f"Failed to load GWDD genus CSV: {e}")

    # Populate fallback map if species map is empty
    if not _SPECIES_MAP:
        _SPECIES_MAP.update(_FALLBACK_LOOKUP)

    _INITIALIZED = True


# Initialize immediately upon import
_load_gwdd_database()


def get_wood_density_with_metadata(scientific_name: str) -> Tuple[float, str, Dict[str, Any]]:
    """
    Tiered wood density lookup:
      Tier 1: Exact species match (returns density, "exact_species", meta)
      Tier 2: Genus average match (returns density, "genus_average", meta)
      Tier 3: Global generic default 0.60 g/cm3 (returns 0.60, "default_fallback", meta)
    """
    _load_gwdd_database()
    
    if not scientific_name:
        return 0.60, "default_fallback", {"reason": "no_scientific_name_provided", "database": "None"}

    clean_name = scientific_name.strip()
    norm_name = clean_name.lower()
    
    # 1. Tier 1: Exact Species Match
    if norm_name in _SPECIES_MAP:
        val = _SPECIES_MAP[norm_name]
        meta = _SPECIES_META.get(norm_name, {"binomial": clean_name, "wsg": val})
        meta["database"] = "Global Wood Density Database (Zanne et al. 2009 / Chave et al.)"
        meta["tier"] = "exact_species"
        return val, "exact_species", meta

    # 2. Tier 2: Genus-Level Average Match
    parts = clean_name.split()
    if parts:
        genus_key = parts[0].lower()
        if genus_key in _GENUS_MAP:
            val = _GENUS_MAP[genus_key]
            meta = _GENUS_META.get(genus_key, {"genus": parts[0], "wsg": val})
            meta["database"] = "Global Wood Density Database (Zanne et al. 2009 / Chave et al.)"
            meta["tier"] = "genus_average"
            meta["matched_genus"] = parts[0]
            return val, "genus_average", meta

    # Check fallback lookup
    if norm_name in _FALLBACK_LOOKUP:
        val = _FALLBACK_LOOKUP[norm_name]
        return val, "exact_species", {"binomial": clean_name, "wsg": val, "database": "Curated Tropical List", "tier": "exact_species"}

    # 3. Tier 3: Global Generic Fallback (0.60 g/cm3)
    return 0.60, "default_fallback", {
        "reason": f"Species '{clean_name}' and Genus '{parts[0] if parts else ''}' not found in GWDD",
        "database": "Global Generic Tropical Prior (Chave et al. 2005 / IPCC)",
        "tier": "default_fallback",
        "wsg": 0.60
    }


def get_wood_density(scientific_name: str) -> Optional[float]:
    """
    Backward-compatible helper: returns specific wood density float or None if fallback.
    """
    density, tier, _ = get_wood_density_with_metadata(scientific_name)
    if tier in ("exact_species", "genus_average"):
        return density
    return None
