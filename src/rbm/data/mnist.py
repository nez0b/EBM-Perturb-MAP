"""
MNIST data loading and preprocessing utilities.

This module provides functions for loading and preprocessing MNIST data
for RBM training and evaluation.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Optional, List
import numpy as np


def load_mnist_data(
    config: dict,
    train: bool = True
) -> Tuple[DataLoader, int]:
    """
    Load MNIST data according to configuration.
    
    Args:
        config: Configuration dictionary containing data parameters.
        train: Whether to load training or test data.
        
    Returns:
        Tuple of (DataLoader, dataset_size).
    """
    # Extract configuration
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    image_size = data_config.get('image_size', [28, 28])
    data_root = data_config.get('data_root', './data')
    download = data_config.get('download', True)
    digit_filter = data_config.get('digit_filter', None)
    batch_size = training_config.get('batch_size', 64)
    
    # Create transforms
    transform_list = []
    
    # Resize if needed
    if image_size != [28, 28]:
        transform_list.append(transforms.Resize(tuple(image_size)))
    
    transform_list.append(transforms.ToTensor())
    
    transform = transforms.Compose(transform_list)
    
    # Load dataset
    dataset = datasets.MNIST(
        root=data_root,
        train=train,
        download=download,
        transform=transform
    )
    
    # Filter by digit if specified
    if digit_filter is not None:
        indices = (dataset.targets == digit_filter).nonzero().squeeze()
        dataset = Subset(dataset, indices)
        print(f"Filtered dataset to digit {digit_filter}: {len(dataset)} samples")
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        drop_last=train  # Drop last batch for training to ensure consistent batch sizes
    )
    
    return data_loader, len(dataset)


def create_noisy_mnist(
    images: torch.Tensor,
    noise_prob: float = 0.1,
    noise_type: str = 'salt_pepper'
) -> torch.Tensor:
    """
    Add noise to MNIST images for denoising experiments.
    
    Args:
        images: Input images tensor.
        noise_prob: Probability of noise per pixel.
        noise_type: Type of noise ('salt_pepper', 'gaussian', 'dropout').
        
    Returns:
        Noisy images tensor.
    """
    noisy_images = images.clone()
    
    if noise_type == 'salt_pepper':
        # Salt and pepper noise - flip pixel values
        noise_mask = torch.rand_like(images) < noise_prob
        noisy_images[noise_mask] = 1 - noisy_images[noise_mask]
    
    elif noise_type == 'gaussian':
        # Gaussian noise
        noise = torch.randn_like(images) * noise_prob
        noisy_images = torch.clamp(images + noise, 0, 1)
    
    elif noise_type == 'dropout':
        # Dropout noise - set pixels to 0
        noise_mask = torch.rand_like(images) < noise_prob
        noisy_images[noise_mask] = 0
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return noisy_images


def get_mnist_subset(
    dataset: datasets.MNIST,
    indices: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a subset of MNIST data by indices.
    
    Args:
        dataset: MNIST dataset.
        indices: List of indices to extract.
        
    Returns:
        Tuple of (images, labels).
    """
    images = []
    labels = []
    
    for idx in indices:
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)
    
    return torch.stack(images), torch.tensor(labels)


def binarize_images(images: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Binarize images using a threshold.
    
    Args:
        images: Input images tensor.
        threshold: Binarization threshold.
        
    Returns:
        Binarized images tensor.
    """
    return (images > threshold).float()


def create_digit_pairs(
    dataset: datasets.MNIST,
    digit1: int,
    digit2: int,
    num_pairs: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create pairs of images from two different digits for interpolation experiments.
    
    Args:
        dataset: MNIST dataset.
        digit1: First digit class.
        digit2: Second digit class.
        num_pairs: Number of pairs to create.
        
    Returns:
        Tuple of (digit1_images, digit2_images).
    """
    # Find indices for each digit
    indices1 = (dataset.targets == digit1).nonzero().squeeze()
    indices2 = (dataset.targets == digit2).nonzero().squeeze()
    
    # Sample random pairs
    sampled_indices1 = np.random.choice(indices1.numpy(), num_pairs, replace=False)
    sampled_indices2 = np.random.choice(indices2.numpy(), num_pairs, replace=False)
    
    images1, _ = get_mnist_subset(dataset, sampled_indices1.tolist())
    images2, _ = get_mnist_subset(dataset, sampled_indices2.tolist())
    
    return images1, images2


def calculate_data_statistics(data_loader: DataLoader) -> dict:
    """
    Calculate statistics for the dataset.
    
    Args:
        data_loader: DataLoader to analyze.
        
    Returns:
        Dictionary containing dataset statistics.
    """
    total_samples = 0
    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    
    for batch_data, _ in data_loader:
        batch_samples = batch_data.size(0)
        total_samples += batch_samples
        
        # Flatten spatial dimensions
        batch_data = batch_data.view(batch_samples, -1)
        
        pixel_sum += torch.sum(batch_data)
        pixel_squared_sum += torch.sum(batch_data ** 2)
    
    total_pixels = total_samples * batch_data.size(1)
    
    mean = pixel_sum / total_pixels
    var = (pixel_squared_sum / total_pixels) - (mean ** 2)
    std = torch.sqrt(var)
    
    return {
        'total_samples': total_samples,
        'total_pixels': total_pixels.item(),
        'mean': mean.item(),
        'std': std.item(),
        'variance': var.item()
    }