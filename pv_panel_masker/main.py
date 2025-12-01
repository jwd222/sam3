# main.py

import os
from sam3_wrapper import SAM3Wrapper

# Import from your new package structure
from solar import PanelMasker, PanelMaskerConfig, PanelVisualizer, PanelIO

def main():
    # 1. Config
    cfg = PanelMaskerConfig()

    # 2. Init Model
    sam = SAM3Wrapper(checkpoint_path="checkpoints/sam3.pt", bpe_path="assets/bpe_simple_vocab_16e6.txt.gz")
    
    # 3. Init Masker6
    masker = PanelMasker(sam, cfg)

    # 4. Run
    image_path="/mnt/d/S/ANTS/Data/spectra_thermal/132FTASK_1761021720/IRX_4570.JPG" # "D:\S\ANTS\Data\spectra_thermal\132FTASK_1761021720\IRX_4552.JPG"
    if os.path.exists(image_path):
        # Process
        results = masker.process_image(image_path)
        
        # Access internals via state
        print(f"Found {len(masker.state['final_boxes'])} panels")
        
        # Visualize & Save using the separated modules
        masker.panel_visualizer.visualize(image_path, masker.state, show=True)
        masker.panel_io.save_results(masker.state, output_dir="results", only_boxes=True)

if __name__ == "__main__":
    main()