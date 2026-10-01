import os
import yaml
import cv2

def check_dataset():
    with open("dataset/data.yaml", "r") as f:
        data = yaml.safe_load(f)
        
    print("Dataset Validation Report")
    print("=========================")
    print(f"Classes ({data['nc']}): {data['names']}")
    
    # Check splits
    for split in ['train', 'val', 'test']:
        if split in data:
            # Handle potential absolute paths vs relative
            path = data[split]
            if not path.startswith("dataset"):
                pass 
                
            folder_name = split
            if split == 'val':
                folder_name = 'valid'
                
            img_dir = os.path.join("dataset", folder_name, "images")
            lbl_dir = os.path.join("dataset", folder_name, "labels")
            
            if os.path.exists(img_dir):
                num_imgs = len(os.listdir(img_dir))
                num_lbls = len(os.listdir(lbl_dir)) if os.path.exists(lbl_dir) else 0
                print(f"[{split.upper()}] Images: {num_imgs} | Labels: {num_lbls}")
            else:
                print(f"[{split.upper()}] Not found or directory structure differs.")
                
if __name__ == "__main__":
    check_dataset()
