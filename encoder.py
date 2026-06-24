"""
CNN Encoder for extracting image features
Uses pre-trained models (ResNet, VGG, etc.)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, model_name='resnet50'):
        """
        Initialize the encoder
        
        Args:
            embed_size: Dimension of the embedding
            model_name: Name of pretrained model ('resnet50', 'vgg16', 'inception_v3')
        """
        super(EncoderCNN, self).__init__()
        self.model_name = model_name
        self.embed_size = embed_size
        
        # Load pretrained model
        if model_name == 'resnet50':
            resnet = models.resnet50(pretrained=True)
            # Remove the last fully connected layer
            modules = list(resnet.children())[:-1]
            self.model = nn.Sequential(*modules)
            self.feature_size = resnet.fc.in_features
            
        elif model_name == 'vgg16':
            vgg = models.vgg16(pretrained=True)
            # Remove the classifier
            self.model = vgg.features
            self.feature_size = 512 * 7 * 7  # VGG16 output
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            
        elif model_name == 'inception_v3':
            inception = models.inception_v3(pretrained=True)
            # Remove the last fully connected layer
            inception.fc = nn.Identity()
            self.model = inception
            self.feature_size = 2048
        
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        # Freeze pretrained model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Linear layer to transform features to embed_size
        self.linear = nn.Linear(self.feature_size, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, images):
        """
        Forward pass
        
        Args:
            images: Input images (batch_size, 3, 224, 224)
        
        Returns:
            features: Encoded features (batch_size, embed_size)
        """
        with torch.no_grad():
            features = self.model(images)
        
        # Flatten features
        features = features.reshape(features.size(0), -1)
        
        # Transform to embedding size
        features = self.linear(features)
        features = self.bn(features)
        features = self.dropout(features)
        
        return features
    
    def fine_tune(self, fine_tune=True):
        """
        Allow fine-tuning of encoder
        
        Args:
            fine_tune: Whether to allow fine-tuning
        """
        for param in self.model.parameters():
            param.requires_grad = fine_tune