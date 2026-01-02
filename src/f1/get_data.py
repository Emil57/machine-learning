import os
import kaggle

DATASET = "rohanrao/formula-1-world-championship-1950-2020"
OUTPUT_DIR = os.path.join(
        os.path.abspath("../../"), 
        "artifacts", 
        "f1", 
        "data",
        "raw"
    )    
try: 
    # Make sure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load the latest version
    kaggle.api.dataset_download_files(DATASET, path=OUTPUT_DIR, unzip=True)
    print(f"Downloaded {DATASET} to {OUTPUT_DIR}")
except Exception as e: 
    print(f'Error: {e}')