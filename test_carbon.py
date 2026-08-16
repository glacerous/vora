import os
import sys

# Ensure project root is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from carbon.dbh_extractor import extract_dbh
from carbon.allometric import estimate_carbon

def test_carbon_pipeline(ply_path, scale_factor):
    print("=" * 60)
    print(f"Testing Carbon Pipeline with Scale Factor: {scale_factor}")
    print("=" * 60)
    
    # 1. DBH Extraction
    print("[1/2] Running DBH extraction on point cloud...")
    dbh_result = extract_dbh(
        ply_path=ply_path, 
        scale_factor=scale_factor, 
        vertical_axis='z', 
        breast_height=1.3, 
        tolerance=0.05
    )
    
    if "error" in dbh_result:
        print(f"Error extracting DBH: {dbh_result['error']}")
        return
        
    print(f"Extraction Method: {dbh_result['method']}")
    print(f"Points in Slice:   {dbh_result['slice_points_count']}")
    print(f"Mean Fit Error:    {dbh_result['mean_fit_error_cm']:.2f} cm")
    print(f"Estimated DBH:     {dbh_result['dbh_cm']:.2f} cm")
    print(f"Estimated Height:  {dbh_result['height_m']:.2f} m")
    print(f"Confidence Note:   {dbh_result['confidence_note']}")
    print()
    
    # 2. Carbon Estimation
    if dbh_result['dbh_cm'] <= 0:
        print("[2/2] Skipped carbon estimation because DBH was not successfully computed.")
        return
        
    print("[2/2] Running allometric calculation...")
    # Calculate carbon. We pass height_m as well to utilize the Chave et al. (2005) height-based equation
    carbon_result = estimate_carbon(
        dbh_cm=dbh_result['dbh_cm'], 
        height_m=dbh_result['height_m'], 
        wood_density=0.6
    )
    
    print(f"Wood Density:     {carbon_result['wood_density_used']} g/cm3")
    print(f"Above-Ground Biomass (AGB): {carbon_result['above_ground_biomass_kg']:.2f} kg")
    print(f"Below-Ground Biomass (BGB): {carbon_result['below_ground_biomass_kg']:.2f} kg")
    print(f"Total Biomass:              {carbon_result['total_biomass_kg']:.2f} kg")
    print(f"Carbon Stock:               {carbon_result['carbon_kg']:.2f} kg")
    print(f"CO2 Equivalent (CO2e):      {carbon_result['co2e_kg']:.2f} kg")
    print("-" * 60)
    print(carbon_result['disclaimer'])
    print("=" * 60)
    print()

if __name__ == "__main__":
    ply_file = os.path.join("output", "result.ply")
    if not os.path.exists(ply_file) or os.path.getsize(ply_file) < 100:
        if os.path.exists("pohon_4497_pts.ply"):
            ply_file = "pohon_4497_pts.ply"
        elif os.path.exists("temp_cleaned.ply"):
            ply_file = "temp_cleaned.ply"
        else:
            print(f"Error: No valid .ply point cloud file found.")
            sys.exit(1)
        
    print(f"Using point cloud file: {ply_file} ({os.path.getsize(ply_file) / 1024 / 1024:.2f} MB)")
    scale_factors = [1.0, 10.0, 15.0, 20.0]
    for sf in scale_factors:
        test_carbon_pipeline(ply_file, sf)
