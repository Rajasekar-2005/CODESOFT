"""
Script to download pretrained models
"""

import torch
import torchvision.models as models
import os


def download_pretrained_models():
    """Download pretrained CNN models"""
    print("="*70)
    print(" "*15 + "DOWNLOADING PRETRAINED MODELS")
    print("="*70)
    
    models_to_download = {
        'ResNet50': models.resnet50,
        'VGG16': models.vgg16,
        'Inception V3': models.inception_v3
    }
    
    for name, model_fn in models_to_download.items():
        print(f"\nDownloading {name}...")
        try:
            model = model_fn(pretrained=True)
            print(f"✓ {name} downloaded successfully")
        except Exception as e:
            print(f"✗ Error downloading {name}: {e}")
    
    print("\n" + "="*70)
    print("Download completed!")
    print("="*70)


if __name__ == "__main__":
    download_pretrained_models()