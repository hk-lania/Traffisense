import os
import random
import cv2
import yaml
import matplotlib.pyplot as plt

def visualize():
    with open("dataset/data.yaml", "r") as f:
        data = yaml.safe_load(f)
    classes = data['names']
    
    os.makedirs("reports/dataset_visualizations", exist_ok=True)
    
    # Pick a random image from the training set
    img_dir = "dataset/train/images"
    lbl_dir = "dataset/train/labels"
    
    if not os.path.exists(img_dir):
        print("Images not found!")
        return
        
    images = os.listdir(img_dir)
    sample_imgs = random.sample(images, min(3, len(images)))
    
    for img_name in sample_imgs:
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.rsplit('.', 1)[0] + '.txt')
        
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    # YOLO format is normalized center_x, center_y, width, height
                    cx, cy, bw, bh = map(float, parts[1:])
                    
                    # Convert to pixel coordinates
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, classes[class_id], (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
        save_path = os.path.join("reports/dataset_visualizations", f"vis_{img_name}")
        cv2.imwrite(save_path, img)
        print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    visualize()
