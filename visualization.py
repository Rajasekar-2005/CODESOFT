"""
Visualization utilities for displaying images and captions
"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch


def denormalize_image(image):
    """
    Denormalize image for display
    
    Args:
        image: Normalized image tensor
    
    Returns:
        image: Denormalized image
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    image = image.numpy().transpose((1, 2, 0))
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    return image


def show_image_with_caption(image_path, caption, predicted_caption=None):
    """
    Display image with caption
    
    Args:
        image_path: Path to image or image tensor
        caption: Ground truth caption
        predicted_caption: Predicted caption (optional)
    """
    plt.figure(figsize=(12, 8))
    
    # Load image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
        plt.imshow(image)
    elif isinstance(image_path, torch.Tensor):
        image = denormalize_image(image_path.cpu())
        plt.imshow(image)
    
    plt.axis('off')
    
    # Add captions
    title = f"Ground Truth: {caption}"
    if predicted_caption:
        title += f"\n\nPredicted: {predicted_caption}"
    
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    plt.show()


def plot_attention(image, caption, attention_weights):
    """
    Visualize attention weights (for attention-based models)
    
    Args:
        image: Input image
        caption: Generated caption
        attention_weights: Attention weights
    """
    words = caption.split()
    n_words = len(words)
    
    fig, axes = plt.subplots(1, n_words, figsize=(20, 4))
    
    for i, (word, ax) in enumerate(zip(words, axes)):
        # Reshape attention to image size
        attention = attention_weights[i].reshape(7, 7)
        
        ax.imshow(image)
        ax.imshow(attention, alpha=0.6, cmap='jet')
        ax.set_title(word)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def display_results(images, true_captions, predicted_captions, num_samples=5):
    """
    Display multiple images with their captions
    
    Args:
        images: List of images
        true_captions: List of ground truth captions
        predicted_captions: List of predicted captions
        num_samples: Number of samples to display
    """
    num_samples = min(num_samples, len(images))
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        if isinstance(images[i], torch.Tensor):
            img = denormalize_image(images[i].cpu())
        else:
            img = Image.open(images[i])
        
        axes[i].imshow(img)
        axes[i].axis('off')
        
        title = f"True: {true_captions[i]}\n\nPredicted: {predicted_captions[i]}"
        axes[i].set_title(title, fontsize=12, pad=10)
    
    plt.tight_layout()
    plt.show()