import os
import glob
from sam3_wrapper import SAM3Wrapper
from solar import PanelMasker, PanelMaskerConfig, PanelVisualizer, PanelIO, setup_logger

def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    # 1. Setup Universal Logger (Info to Console, Debug to File)
    # This must be called before initializing other classes
    logger = setup_logger(log_dir="logs")
    logger.info("=== Starting Solar Panel Segmentation (Batch Mode) ===")

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
    image_folder = "/mnt/d/S/ANTS/Data/spectra_thermal/132FTASK_1761021720"  # Update this path
    output_dir = "/mnt/d/S/ANTS/Data/spectra_thermal/132FTASK_1761021720/v3/batch/1"
    
    # Get all images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    image_files = sorted(image_files)

    if not image_files:
        logger.warning("No images found.")
        return

    # Batch Processing
    BATCH_SIZE = cfg.batch_size # Adjust based on GPU VRAM (e.g., 2, 4, 8)
    
    total_chunks = (len(image_files) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Processing {len(image_files)} images in {total_chunks} batches (Batch Size: {BATCH_SIZE}).")

    for i, batch_paths in enumerate(chunk_list(image_files, BATCH_SIZE)):
        logger.info(f"--- Batch {i+1}/{total_chunks} ---")
        
        # Run Batch Pipeline
        batch_results = masker.process_batch(batch_paths)
        
        # Save Results
        for result_state in batch_results:
            path = result_state['image_path']
            if result_state.get('final_boxes'):
                # Save
                masker.panel_io.save_results(result_state, output_dir)
                
                # Optional: Visualize specific image for debug
                masker.panel_visualizer.visualize(path, result_state, show=True)
            else:
                logger.debug(f"No panels found in {path}")

    logger.info("=== Processing Complete ===")

if __name__ == "__main__":
    main()