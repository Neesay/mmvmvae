import os
import urllib.request

def download_file(url, save_path):
    print(f"Downloading {url}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    urllib.request.urlretrieve(url, save_path)
    print(f"Saved to {save_path}")

def main():
    # Target directory in the repo
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "scMNC"))
    
    # Files needed for scMNC preprocessing (visual_cortex)
    files = {
        "geneExp_filtered.csv": "https://raw.githubusercontent.com/daifengwanglab/scMNC/main/mouse_visual_cortex/data/geneExp_filtered.csv",
        "efeature_filtered.csv": "https://raw.githubusercontent.com/daifengwanglab/scMNC/main/mouse_visual_cortex/data/efeature_filtered.csv",
        "20200711_patchseq_metadata_mouse.csv": "https://raw.githubusercontent.com/daifengwanglab/scMNC/main/mouse_visual_cortex/data/20200711_patchseq_metadata_mouse.csv"
    }
    
    for filename, url in files.items():
        save_path = os.path.join(target_dir, filename)
        download_file(url, save_path)
        
    print("\nAll raw files downloaded successfully!")
    print(f"Location: {target_dir}")
    print("When you run the model, the dataset loader will automatically pre-process these files.")

if __name__ == "__main__":
    main()
