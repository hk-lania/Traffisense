import os
import yaml
from ultralytics import YOLO

def evaluate_model():
    with open("config/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    print(f"Loading model from {config['model_path']}...")
    model = YOLO(config['model_path'])
    
    print("Evaluating model on the test dataset...")
    metrics = model.val(data=config['data_yaml'])
    
    print("\n--- Evaluation Results ---")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    
    os.makedirs("reports", exist_ok=True)
    with open("reports/model_evaluation.md", "w") as f:
        f.write("# Model Evaluation\n")
        f.write(f"- **mAP50-95**: {metrics.box.map:.4f}\n")
        f.write(f"- **mAP50**: {metrics.box.map50:.4f}\n")

if __name__ == "__main__":
    evaluate_model()
