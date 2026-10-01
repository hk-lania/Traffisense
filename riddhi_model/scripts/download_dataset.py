import os
from roboflow import Roboflow
from dotenv import load_dotenv

def download_dataset():
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    
    if not api_key or api_key == "your_api_key_here":
        print("Error: ROBOFLOW_API_KEY is not set in .env")
        print("Please add it to download the dataset.")
        return

    print("Initializing Roboflow...")
    rf = Roboflow(api_key=api_key)
    
    # The user should fill these in based on their project
    workspace_name = input("Enter your Roboflow workspace name (e.g., riddhi-savla): ").strip()
    project_name = input("Enter your Roboflow project name: ").strip()
    version_num = input("Enter the dataset version number (e.g., 1): ").strip()
    
    try:
        project = rf.workspace(workspace_name).project(project_name)
        version = project.version(int(version_num))
        
        print(f"Downloading dataset {project_name} v{version_num}...")
        dataset = version.download("yolov8")
        
        print("\nDataset downloaded successfully!")
        print(f"Location: {dataset.location}")
        print("Please move the contents of this folder into 'riddhi_model/dataset/'")
        
    except Exception as e:
        print(f"Failed to download dataset: {e}")

if __name__ == "__main__":
    download_dataset()
