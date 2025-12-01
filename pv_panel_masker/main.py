import os
import glob
from sam3_wrapper import SAM3Wrapper
from solar import PanelMasker, PanelMaskerConfig, PanelVisualizer, PanelIO, setup_logger

def main():
    # 1. Setup Universal Logger (Info to Console, Debug to File)
    # This must be called before initializing other classes
    logger = setup_logger(log_dir="logs")
    
    logger.info("=== Starting Solar Panel Segmentation ===")

    # 2. Configuration
    cfg = PanelMaskerConfig()

    # 3. Initialize Model
    # Assumes checkpoints are in the root directory or specified path
    logger.info("Initializing SAM3 model...")
    try:
        sam = SAM3Wrapper(checkpoint_path="checkpoints/sam3.pt", bpe_path="assets/bpe_simple_vocab_16e6.txt.gz")
    except Exception as e:
        logger.critical(f"Failed to load SAM3 Model: {e}")
        return

    # 4. Initialize Masker
    masker = PanelMasker(sam, cfg)

    # 5. Process Images
    image_folder = "/path/to/images"  # Update this path
    output_dir = "final_results"
    
    # Example for single image or loop
    # For now, let's pretend we are processing one specific file for testing
    image_path = "/mnt/d/S/ANTS/Data/spectra_thermal/132FTASK_1761021720/IRX_4569.JPG" # "D:\S\ANTS\Data\spectra_thermal\132FTASK_1761021720\IRX_4552.JPG"
    output_dir = "/mnt/d/S/ANTS/Data/spectra_thermal/132FTASK_1761021720_v3/1"
    
    if os.path.exists(image_path):
        results = masker.process_image(image_path)
        
        # Visualize and Save
        if results.get('final_boxes'):
            # vis_path = os.path.join(output_dir, "visualization.png")
            # masker.panel_visualizer.visualize(image_path, masker.state, show=True)
            masker.panel_io.save_results(masker.state, output_dir=output_dir, only_boxes=True)
        else:
            logger.warning(f"No panels detected in {image_path}")
    else:
        logger.warning(f"Image not found: {image_path}")

    logger.info("=== Processing Complete ===")

if __name__ == "__main__":
    main()