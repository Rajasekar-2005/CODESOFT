"""
Data loader for image captioning
"""

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class ImageCaptionDataset(Dataset):
    """
    Dataset for image captioning
    """
    
    def __init__(self, image_dir, captions_dict, vocab, transform=None):
        """
        Initialize dataset
        
        Args:
            image_dir: Directory containing images
            captions_dict: Dictionary mapping image IDs to captions
            vocab: Vocabulary object
            transform: Image transforms
        """
        self.image_dir = image_dir
        self.captions_dict = captions_dict
        self.vocab = vocab
        self.transform = transform
        
        # Create list of (image_id, caption) pairs
        self.samples = []
        for img_id, captions in captions_dict.items():
            for caption in captions:
                self.samples.append((img_id, caption))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample"""
        img_id, caption = self.samples[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_id)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Encode caption
        tokens = self.vocab.encode_caption(caption)
        target = torch.LongTensor(tokens)
        
        return image, target


def collate_fn(data):
    """
    Custom collate function for batching
    Handles variable length captions
    """
    # Sort by caption length (descending)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    
    images, captions = zip(*data)
    
    # Stack images
    images = torch.stack(images, 0)
    
    # Get lengths
    lengths = [len(cap) for cap in captions]
    
    # Pad captions
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    
    return images, targets, lengths


def get_data_loader(image_dir, captions_dict, vocab, transform=None,
                    batch_size=32, shuffle=True, num_workers=0):
    """
    Get data loader
    
    Args:
        image_dir: Directory containing images
        captions_dict: Dictionary of captions
        vocab: Vocabulary object
        transform: Image transforms
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        data_loader: DataLoader object
    """
    dataset = ImageCaptionDataset(image_dir, captions_dict, vocab, transform)
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return data_loader