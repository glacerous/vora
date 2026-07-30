import math
import logging

logger = logging.getLogger("Allometric")

def estimate_biomass(dbh_cm, height_m=None, wood_density=0.6, forest_type="moist"):
    """
    Estimates the Above-Ground Biomass (AGB) in kg for a tree.
    
    Equations used:
    Chave et al. (2005) forest-type specific equations.
    Source: Chave et al. (2005) "Tree allometry and improved estimation of carbon stocks 
            and balance in tropical forests." Oecologia.
            
    Parameters:
      - dbh_cm: Diameter at Breast Height in centimeters
      - height_m: Tree height in meters (optional)
      - wood_density: Wood density in g/cm3 (default: 0.6 for generic tropical trees)
      - forest_type: "dry", "moist", or "wet" (default: "moist")
    """
    if dbh_cm <= 0:
        logger.warning(f"Invalid DBH: {dbh_cm}. Biomass set to 0.0.")
        return 0.0, "None"
        
    ft = forest_type.lower().strip()
    if ft not in ("dry", "moist", "wet"):
        ft = "moist"
        
    # Check if height is available and reliable
    if height_m is not None and height_m > 0:
        logger.info(f"Using Chave et al. (2005) height-enabled equation for {ft} forest. DBH={dbh_cm}cm, Height={height_m}m")
        val = wood_density * (dbh_cm ** 2) * height_m
        ln_val = math.log(val)
        
        if ft == "dry":
            agb = math.exp(-2.187 + 1.016 * ln_val)
            formula = "Chave 2005 (dry forest with height)"
        elif ft == "wet":
            agb = math.exp(-1.239 + 0.947 * ln_val)
            formula = "Chave 2005 (wet forest with height)"
        else: # moist
            agb = math.exp(-1.499 + 0.976 * ln_val)
            formula = "Chave 2005 (moist forest with height)"
    else:
        logger.info(f"Using Chave et al. (2005) DBH-only equation for {ft} forest. DBH={dbh_cm}cm")
        ln_d = math.log(dbh_cm)
        
        if ft == "dry":
            exponent = -0.667 + 1.784 * ln_d + 0.207 * (ln_d ** 2) - 0.0281 * (ln_d ** 3)
            formula = "Chave 2005 (dry forest, DBH-only)"
        elif ft == "wet":
            exponent = -1.239 + 1.980 * ln_d + 0.207 * (ln_d ** 2) - 0.0281 * (ln_d ** 3)
            formula = "Chave 2005 (wet forest, DBH-only)"
        else: # moist
            exponent = -1.499 + 2.148 * ln_d + 0.207 * (ln_d ** 2) - 0.0281 * (ln_d ** 3)
            formula = "Chave 2005 (moist forest, DBH-only)"
            
        agb = wood_density * math.exp(exponent)
        
    return agb, formula

def estimate_carbon(dbh_cm, height_m=None, wood_density=0.6, forest_type="moist"):
    """
    Computes biomass, carbon stock, and CO2 equivalent from tree dimensions.
    
    IPCC Standards used:
    1. Root-to-Shoot Ratio (BGB = AGB * 0.24) to find Below-Ground Biomass.
       Source: IPCC (2006) Guidelines for National Greenhouse Gas Inventories.
    2. Carbon Fraction (Carbon = Total Biomass * 0.47).
       Source: IPCC default value of 47% carbon content in dry wood of tropical trees.
    3. CO2 Equivalent factor (CO2e = Carbon * 3.67) based on molecular weights (44/12).
    
    Returns:
      A dictionary containing biomass, carbon, co2e estimation, and carbon disclaimer.
    """
    # 1. Above-Ground Biomass (AGB)
    agb, formula = estimate_biomass(dbh_cm, height_m, wood_density, forest_type)
    
    # 2. Below-Ground Biomass (BGB) - Root-to-Shoot ratio of 0.24 for tropical forests
    bgb = agb * 0.24
    
    # 3. Total Biomass (Dry Weight)
    total_biomass = agb + bgb
    
    # 4. Carbon Stock (47% of dry biomass)
    carbon = total_biomass * 0.47
    
    # 5. CO2 Equivalent (CO2 factor 44/12 = 3.67)
    co2e = carbon * 3.67
    
    disclaimer = (
        "DISCLAIMER: Estimasi ini merupakan screening awal berdasarkan model allometric volumetrik "
        "(Chave et al., 2005) dan standar IPCC, BUKAN merupakan hasil pengukuran bersertifikasi "
        "resmi untuk perdagangan karbon (carbon offsets)."
    )
    
    logger.info(f"Carbon estimation computed ({formula}): AGB={agb:.2f}kg, Carbon={carbon:.2f}kg, CO2e={co2e:.2f}kg")
    
    return {
        "above_ground_biomass_kg": float(round(agb, 2)),
        "below_ground_biomass_kg": float(round(bgb, 2)),
        "total_biomass_kg": float(round(total_biomass, 2)),
        "carbon_kg": float(round(carbon, 2)),
        "co2e_kg": float(round(co2e, 2)),
        "wood_density_used": wood_density,
        "formula_used": formula,
        "disclaimer": disclaimer
    }
