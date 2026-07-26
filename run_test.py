import os
import sys
import time
import modal

def main():
    # Define local directories
    image_dir = "./test_images"
    output_dir = "./output"
    
    # 1. Verify input directory contains images
    if not os.path.exists(image_dir):
        print(f"Error: Directory '{image_dir}' does not exist.")
        print("Please create it and place your input photos inside.")
        sys.exit(1)
        
    supported_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [
        f for f in os.listdir(image_dir) 
        if f.lower().endswith(supported_extensions)
    ]
    
    if not image_files:
        print(f"Error: No images found in '{image_dir}'.")
        print(f"Please put some photos (supported: {', '.join(supported_extensions)}) in this directory.")
        sys.exit(1)
        
    print(f"Found {len(image_files)} images in '{image_dir}':")
    for f in image_files:
        print(f"  - {f}")
        
    # Read files into bytes
    images_bytes = []
    for f in image_files:
        path = os.path.join(image_dir, f)
        with open(path, "rb") as file:
            images_bytes.append(file.read())
            
    # 2. Connect to the deployed Modal app
    app_name = "instantsplat-app"
    function_name = "run_reconstruction"
    
    print(f"\nLooking up Modal function '{function_name}' in app '{app_name}'...")
    try:
        run_reconst = modal.Function.from_name(app_name, function_name)
    except Exception as e:
        print(f"Error: Could not find the deployed Modal app. Did you run 'modal deploy modal_app.py'?")
        print(f"Detail error: {e}")
        sys.exit(1)
        
    # 3. Trigger remote inference and measure time
    print(f"\nSending images to Modal and starting 3D reconstruction...")
    start_time = time.time()
    
    try:
        # Call the remote function
        result_bytes = run_reconst.remote(images_bytes)
        
        duration = time.time() - start_time
        print(f"\nSuccess! Reconstruction finished in {duration:.2f} seconds.")
        
        # 4. Save output
        os.makedirs(output_dir, exist_ok=True)
        # Determine the file extension (typically .ply or .splat)
        # Since we load it recursively, let's name it result.ply by default or match what was received
        output_filename = "result.ply"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, "wb") as f:
            f.write(result_bytes)
            
        print(f"Saved reconstruction output to: {output_path} ({len(result_bytes)} bytes)")
        print("\nNext step: Open 'viewer.html' in your browser to view the 3D Gaussian Splat.")
        
    except Exception as e:
        duration = time.time() - start_time
        print(f"\nError occurred during remote execution (after {duration:.2f} seconds):")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    main()
