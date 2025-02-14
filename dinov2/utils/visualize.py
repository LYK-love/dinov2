import sys
import os
import torch
import cv2
import torchvision.transforms as tt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import numpy as np


import cv2
import numpy as np
import torch
import torchvision.transforms as tt

def load_preprocess_video(video_path, target_size=448, device='cuda'):
    """
    Loads a video, resizes frames to target_size, and normalizes.

    Args:
    - video_path (str): Path to the input video.
    - target_size (int): Final resize dimension (default 448).
    - device (str): Device to load the tensor onto.

    Returns:
    - torch.Tensor: Video tensor of shape (B, C, target_size, target_size) before preprocess
    - torch.Tensor: Preprocessed video tensor of shape (B, C, target_size, target_size).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Load video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size))  # Resize to 448x448
        frame = frame.astype('float32') / 255.0
        frames.append(frame)

    cap.release()

    # Convert to tensor: (B, H, W, C) -> (B, C, H, W)
    frames = np.stack(frames)
    frames = torch.tensor(frames).permute(0, 3, 1, 2)  # (B, C, H, W)

    frames = frames[:7]
    # Normalize
    transform = tt.Compose([tt.Normalize(mean=0.5, std=0.2)])
    input_tensor = transform(frames).to(device)

    print(f"Preprocessed video tensor shape: {input_tensor.shape}")
    return frames, input_tensor



def get_patch_embeddings(model, input_tensor):
    result = model.forward_features(input_tensor)  # Forward pass

    patch_embeddings = result['x_norm_patchtokens'].detach().cpu().numpy().reshape([input_tensor.shape[0], -1, model.embed_dim])
    return patch_embeddings  # (B, patch_resolution, embedding_dim)

# Step 4: Perform two-stage PCA
def two_stage_pca(patch_embeddings, threshold=0.6):
    """
    Perform two-stage PCA on patch embeddings:
    1. First PCA with 1 component to extract foreground patches.
    2. Second PCA with 3 components to reduce foreground patches to RGB-like embeddings.

    Args:
    - patch_embeddings (np.ndarray): Patch embeddings of shape (B, patch_resolution, embedding_dim)
    - threshold (float): Threshold for selecting foreground patches after the first PCA.

    Returns:
    - reduced_fg_patch_embeddings (np.ndarray): PCA-reduced foreground patch embeddings of shape (total_foreground_patches, 3)
    - nums_of_fg_patches (list): Number of foreground patches for each image.
    - masks
    - reduced_patch_embeddings
    """
    B, patch_resolution, embedding_dim = patch_embeddings.shape

    # First PCA: Extract foreground patches
    fg_pca = PCA(n_components=1)
    all_patch_embeddings = patch_embeddings.reshape(-1, embedding_dim)  # (B*patch_resolution, embedding_dim)
    reduced_patch_embeddings = fg_pca.fit_transform(all_patch_embeddings)  # (B*patch_resolution, 1)
    reduced_patch_embeddings = minmax_scale(reduced_patch_embeddings)  # Scale to (0,1)
    reduced_patch_embeddings = reduced_patch_embeddings.reshape((B, patch_resolution))  # (B, patch_resolution)

    masks = []
    for i in range(B):
        mask = (reduced_patch_embeddings[i] > threshold).ravel()  # Foreground mask
        masks.append(mask)

    nums_of_fg_patches = [np.sum(m) for m in masks]
    fg_patch_embeddings = np.vstack([patch_embeddings[i, m, :] for i, m in enumerate(masks)])  # (total_fg_patches, embedding_dim)

    # Second PCA: Reduce foreground patches to 3 dimensions
    object_pca = PCA(n_components=3)
    reduced_fg_patch_embeddings = object_pca.fit_transform(fg_patch_embeddings)  # (total_foreground_patches, 3)
    reduced_fg_patch_embeddings = minmax_scale(reduced_fg_patch_embeddings)  # Scale to (0,1)

    for i, num_patches in enumerate(nums_of_fg_patches):
        print(f"Num of foreground patches of image {i}: {num_patches}")

    total_foreground_patches = sum(nums_of_fg_patches)
    print(f"Total num of foreground patches: {total_foreground_patches}")
    
    print("Explained variance ratio by PCA components:", object_pca.explained_variance_ratio_)

    return reduced_fg_patch_embeddings, nums_of_fg_patches, masks, reduced_patch_embeddings

def print_video_model_stats(input_tensor, model):
    """
    Prints the video tensor stats and DINO model parameters.

    Args:
    - input_tensor (torch.Tensor): Input video tensor with shape (B, C, H, W).
    - model: The DINO model used for generating patch embeddings.
    """
    B, C, H, W = input_tensor.shape  # Batch size, Channels, Height, Width
    print(f"Input tensor shape: Batch={B}, Channels={C}, Height={H}, Width={W}")

    patch_size = model.patch_size  # Patch size from DINO
    print(f"Patch size: {patch_size}")

    embedding_dim = model.embed_dim  # Embedding dimension from DINO
    print(f"Embedding dimension: {embedding_dim}")

    patch_resolution = (H // patch_size) * (W // patch_size)  # Total number of patches per image
    print(f"Number of patches of each image: {patch_resolution}")
    





def save_tensor_as_video(input_tensor, output_path='tensor_video.mp4', fps=30):
    """
    Saves a PyTorch tensor (B, C, H, W) as an MP4 video.

    Args:
    - input_tensor (torch.Tensor): Video tensor with shape (B, C, H, W).
    - output_path (str): Path to save the output video.
    - fps (int): Frames per second for the output video.
    """
    B, C, H, W = input_tensor.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i in range(B):
        frame = (input_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Video saved to {output_path}")


def save_triple_video(input_tensor, reduced_fg_patch_embeddings, nums_of_fg_patches, masks, reduced_patch_embeddings, patch_resolution, patch_size, output_path='triple_video.mp4', fps=30):
    """
    Saves an MP4 video with three sections:
    - Left: Original frames
    - Middle: Foreground mask from the first PCA component
    - Right: PCA-based foreground patches
    
    Args:
    - input_tensor (torch.Tensor): Original video tensor (B, C, H, W).
    - reduced_fg_patch_embeddings (np.ndarray): PCA-reduced foreground patch embeddings (N, 3).
    - nums_of_fg_patches (list): Number of foreground patches for each frame.
    - masks (list): Boolean masks for foreground patches for each frame.
    - reduced_patch_embeddings (np.ndarray): PCA first component embeddings for foreground mask.
    - patch_resolution (int): Total number of patches in each frame.
    - patch_size (int): Patch size used in the model.
    - output_path (str): Output path for the video.
    - fps (int): Frames per second for the video.
    """
    B, C, H, W = input_tensor.shape
    start_idx = 0

    # Video writer with triple width
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (3 * W, H))  # 3W width for triple view

    for i, mask in enumerate(masks):
        num_patches = nums_of_fg_patches[i]
        
        # ======== PCA Foreground Patches (Right) ========
        patch_image = np.zeros((patch_resolution, 3), dtype='float32')# The background will be value=0 (black in RGB)
        patch_image[mask, :] = reduced_fg_patch_embeddings[start_idx:start_idx + num_patches, :]
        start_idx += num_patches
        color_patches = patch_image.reshape((H // patch_size, W // patch_size, 3))
        pca_frame = cv2.resize((color_patches * 255).astype(np.uint8), (H, W))

        # ======== Foreground Mask (Middle) ========
        mask_image = np.zeros((patch_resolution,), dtype='float32')
        mask_image[mask] = reduced_patch_embeddings[i][mask]  # Use the first PCA component
        mask_image = mask_image.reshape((H // patch_size, W // patch_size))
        mask_frame = cv2.resize((mask_image * 255).astype(np.uint8), (H, W))
        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)

        # ======== Original Frame (Left) ========
        original_frame = (input_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # ======== Combine Frames Horizontally ========
        combined_frame = np.hstack((original_frame, mask_frame, pca_frame))

        # Write the combined frame to the video file
        out.write(cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"Triple video saved to {output_path}")
