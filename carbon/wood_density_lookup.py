# Wood Density Lookup Table (g/cm3)
# Based on Global Wood Density Database (Zanne et al. 2009)
# Format: "Scientific Name" -> wood density value in g/cm3

WOOD_DENSITY_LOOKUP = {
    "Swietenia macrophylla": 0.53,      # Mahogany
    "Tectona grandis": 0.55,            # Teak
    "Hevea brasiliensis": 0.50,         # Rubber tree
    "Acacia mangium": 0.53,             # Acacia
    "Pinus merkusii": 0.45,             # Sumatran pine
    "Eucalyptus globulus": 0.65,        # Blue gum
    "Cocos nucifera": 0.35,             # Coconut
    "Mangifera indica": 0.58,           # Mango
    "Artocarpus heterophyllus": 0.55,   # Jackfruit / Nangka
    "Paraserianthes falcataria": 0.33,  # Sengon / Albasia
    "Spathodea campanulata": 0.28,      # African Tulip
    "Pterocarpus indicus": 0.56,        # Angsana
    "Ficus elastica": 0.41,             # Rubber Fig
    "Roystonea regia": 0.30,            # Royal Palm
    "Gmelina arborea": 0.42,            # Gmelina
    "Delonix regia": 0.40,              # Flamboyant
    "Terminalia catappa": 0.48,         # Ketapang / Indian Almond
    "Leucaena leucocephala": 0.60,      # Lamtoro
    "Albizia chinensis": 0.35,          # Sengon Jawa
    "Manilkara zapota": 0.85,           # Sawo / Sapodilla
    "Tamarindus indica": 0.75,          # Asam Jawa / Tamarind
}

def get_wood_density(scientific_name: str) -> float:
    """
    Returns the specific wood density (g/cm3) if the species is in the lookup table.
    Otherwise, returns None.
    """
    if not scientific_name:
        return None
    
    # Normalize scientific name (strip whitespace and matching case-insensitively just in case)
    name_norm = scientific_name.strip()
    
    # Try exact match first
    if name_norm in WOOD_DENSITY_LOOKUP:
        return WOOD_DENSITY_LOOKUP[name_norm]
        
    # Try case-insensitive lookup
    for key, val in WOOD_DENSITY_LOOKUP.items():
        if key.lower() == name_norm.lower():
            return val
            
    return None
