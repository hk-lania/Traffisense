import yaml
import os
import argparse
from ultralytics import YOLO

def train_model():
    # Load config
    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Loading base model {config['base_model']}...")
    model = YOLO(config['base_model'])
    
    # Train
    print("Starting local training...")
    results = model.train(
        data=config['data_yaml'],
        epochs=config['epochs'],
        imgsz=config['imgsz'],
        batch=config['batch'],
        workers=config['workers'],
        device=config['device'] if config['device'] else None,
        project=config['project_name'],
        name="train",
        exist_ok=True
    )
    
    print("\nTraining complete!")
    print("Your best model is saved at: runs/traffisense_yolo/train/weights/best.pt")
    
    # Copy best.pt to models/
    # Determine the correct path for the best model (Ultralytics saves under runs/detect/...)
    best_src = "runs/detect/traffisense_yolo/train/weights/best.pt"
    if not os.path.exists(best_src):
        # fallback to previous assumed location (in case project config changes)
        best_src = "runs/traffisense_yolo/train/weights/best.pt"
    shutil.copy(best_src, "models/best.pt")
    print(f"Copied best.pt from {best_src} to models/best.pt")

if __name__ == "__main__":
    train_model()
