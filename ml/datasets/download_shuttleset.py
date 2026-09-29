import os
import json
import urllib.request
from ml.config import DATA_DIR

SHUTTLESET_INFO = {
    "name": "ShuttleSet & TrackNet Badminton Benchmark Datasets",
    "description": "Public research dataset for badminton video analysis, shot classification, and rally trajectory tracking.",
    "citations": [
        "Wang et al., 'ShuttleSet: A High-Resolution Stroke-Level Badminton Dataset', CVPR Workshop / ACM MM 2023.",
        "Sun et al., 'TrackNet: A Deep Learning Network for Tracking High-speed Small Objects in Badminton Videos', IEEE Access 2020."
    ],
    "public_urls": {
        "shuttleset_github": "https://github.com/Darkness-In-The-Light/ShuttleSet",
        "tracknet_github": "https://github.com/yanyongjie/TrackNetv2"
    }
}

def prepare_public_dataset_info(output_path: str = None):
    if output_path is None:
        output_path = os.path.join(DATA_DIR, "public_dataset_metadata.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(SHUTTLESET_INFO, f, indent=2)
    print(f"[Dataset Metadata] Saved public dataset specs to {output_path}")

if __name__ == "__main__":
    prepare_public_dataset_info()
