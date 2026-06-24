"""
Configuration file for Image Captioning model
"""

import torch

class Config:
    # Device configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    EMBED_SIZE = 256
    HIDDEN_SIZE = 512
    NUM_LAYERS = 1
    LEARNING_RATE = 0.001
    
    # Image parameters
    IMAGE_SIZE = 224
    
    # Training parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    
    # Vocabulary
    VOCAB_SIZE = 5000
    MAX_SEQ_LENGTH = 20
    
    # Special tokens
    PAD_TOKEN = '<PAD>'
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'
    UNK_TOKEN = '<UNK>'
    
    # Paths
    MODEL_SAVE_PATH = 'saved_models/'
    IMAGES_PATH = 'sample_images/'
    
    # Pretrained model
    ENCODER_MODEL = 'resnet50'  # Options: 'resnet50', 'vgg16', 'inception_v3'
    
    @staticmethod
    def print_config():
        """Print configuration"""
        print("="*60)
        print("IMAGE CAPTIONING CONFIGURATION")
        print("="*60)
        print(f"Device: {Config.DEVICE}")
        print(f"Encoder Model: {Config.ENCODER_MODEL}")
        print(f"Embed Size: {Config.EMBED_SIZE}")
        print(f"Hidden Size: {Config.HIDDEN_SIZE}")
        print(f"Learning Rate: {Config.LEARNING_RATE}")
        print(f"Batch Size: {Config.BATCH_SIZE}")
        print(f"Max Sequence Length: {Config.MAX_SEQ_LENGTH}")
        print("="*60)