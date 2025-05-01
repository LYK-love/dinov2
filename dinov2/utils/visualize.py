import sys
import os
import math
import torch
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import torchvision.transforms as T
from PIL import Image


def load_preprocess_video(video_path, target_size=None, patch_size=14, device='cuda'):
    """
    Loads a video, resizes frames to target_size (if provided), makes them divisible by patch_size, 
    and returns both unnormalized and normalized tensors, along with the video FPS and duration.

    Args:
    - video_path (str): Path to the input video.
    - target_size (int or None): Final resize dimension (e.g., 224 or 448 in the paper). If None, no resizing is applied.
    - patch_size (int): Patch size to make the frames divisible by.
    - device (str): Device to load the tensor onto.

    Returns:
    - torch.Tensor: Unnormalized video tensor (B, C, H, W).
    - torch.Tensor: Normalized video tensor (B, C, H, W).
    - float: Frames per second (FPS) of the video.
    """
    cap = cv2.VideoCapture(video_path)

    # Get FPS and total frame count
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Calculate duration
    duration = total_frames / fps if fps > 0 else 0
    print(f"Video FPS: {fps:.2f}, Total Frames: {int(total_frames)}, Duration: {duration:.2f} seconds")

    frames = []

    # Define transforms dynamically based on target_size
    base_transforms = [T.ToTensor()]
    if target_size is not None:
        base_transforms.append(T.Resize((target_size, target_size)))

    unnormalized_transform = T.Compose(base_transforms)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        unnormalized_frame = unnormalized_transform(image)
        h, w = (
            unnormalized_frame.shape[1] - unnormalized_frame.shape[1] % patch_size,
            unnormalized_frame.shape[2] - unnormalized_frame.shape[2] % patch_size,
        )
        unnormalized_frame = unnormalized_frame[:, :h, :w]
        frames.append(unnormalized_frame)

    cap.release()

    unnormalized_video = torch.stack(frames[:])
    normalized_video = T.Compose([T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])(unnormalized_video.clone())

    # print(f"Unnormalized video shape: {unnormalized_video.shape}")
    # print(f"Normalized video shape: {normalized_video.shape}")

    return unnormalized_video.to(device), normalized_video.to(device), fps





def get_patch_embeddings(model, input_tensor):
    result = model.forward_features(input_tensor)  # Forward pass

    patch_embeddings = result['x_norm_patchtokens'].detach().cpu().numpy().reshape([input_tensor.shape[0], -1, model.embed_dim])
    return patch_embeddings  # (B, patch_num, embedding_dim)

    


def attention_visualize(attentions: np.ndarray, height_in_patches: int, width_in_patches: int, patch_size: int, num_register_tokens: int = 0) -> np.ndarray:
    """
    Visualizes attention maps as colorized frames using the inferno colormap and returns a NumPy array.

    Args:
    - attentions (np.ndarray): Attention array of shape (T, NH, H_p + num_register_tokens +  1, W_p + num_register_tokens + 1).
    - height_in_patches (int): Number of patches in height.
    - width_in_patches (int): Number of patches in width.
    - patch_size (int): Size of each patch in pixels.

    Returns:
    - np.ndarray: Colorized attention maps of shape (T, H, W, 3).
    """
    T, NH = attentions.shape[:2]  # Video length, number of heads

    # Extract the attention of the CLS token to all patch tokens
    attentions = attentions[:, :, 0, num_register_tokens + 1:].reshape(T, NH, height_in_patches, width_in_patches)  # (T, NH, H_p, W_p)

    # Upscale attention maps to original resolution using OpenCV
    upscaled_attentions = np.array([
        np.array([
            cv2.resize(attentions[t, h], (width_in_patches * patch_size, height_in_patches * patch_size), interpolation=cv2.INTER_NEAREST)
            for h in range(NH)
        ])
        for t in range(T)
    ])  # Shape: (T, NH, H, W)

    # Average over all heads
    average_attention = np.mean(upscaled_attentions, axis=1)  # Shape: (T, H, W)

    # It's necessary to normalize the attention maps to [0, 1] for colorization
    min_val, max_val = average_attention.min(), average_attention.max()
    if max_val > min_val:  # Avoid division by zero
        average_attention = (average_attention - min_val) / (max_val - min_val)
        
    # Apply inferno colormap frame by frame
    colorized_frames = np.array([
        plt.cm.inferno(frame)[:, :, :3]  # (H, W, 3)
        for frame in average_attention
    ])  # Shape: (T, H, W, 3)

    return colorized_frames




    
    
def two_stage_pca(patch_embeddings, threshold=0.6):
    """
    Perform two-stage PCA on patch embeddings:
    1. First PCA with 1 component to extract foreground patches.
    2. Second PCA with 3 components to reduce foreground patches to RGB-like embeddings.

    Args:
    - patch_embeddings (np.ndarray): Patch embeddings of shape (B, num_patches_all_images, embedding_dim)
    - threshold (float): Threshold for selecting foreground patches after the first PCA.

    Returns:
    - reduced_patch_embeddings (np.ndarray): PCA-reduced patch embeddings of shape (B, num_patches)
    - reduced_fg_patch_embeddings (np.ndarray): PCA-reduced foreground patch embeddings of shape (num_fg_patches_all_images, 3)
    - num_fg_patches_list (list): List of numbers of foreground patches for each image.
    - masks (list): List of masks for each image
    """
    B, num_patches, embedding_dim = patch_embeddings.shape

    # First PCA: Extract foreground patches
    fg_pca = PCA(n_components=1)
    all_patch_embeddings = patch_embeddings.reshape(-1, embedding_dim)  # (B*num_patches, embedding_dim)
    reduced_patch_embeddings = fg_pca.fit_transform(all_patch_embeddings)  # (B*num_patches, 1)
    reduced_patch_embeddings = minmax_scale(reduced_patch_embeddings)  # Scale to (0,1)
    reduced_patch_embeddings = reduced_patch_embeddings.reshape((B, num_patches))  # (B, num_patches)

    masks = []
    for i in range(B):
        mask = (reduced_patch_embeddings[i] > threshold).ravel()  # Foreground mask
        masks.append(mask)

    num_fg_patches_list = [np.sum(m) for m in masks]
    fg_patch_embeddings = np.vstack([patch_embeddings[i, m, :] for i, m in enumerate(masks)])  # (total_fg_patches, embedding_dim)

    # Second PCA: Reduce foreground patches to 3 dimensions
    object_pca = PCA(n_components=3)
    reduced_fg_patch_embeddings = object_pca.fit_transform(fg_patch_embeddings)  # (total_foreground_patches, 3)
    reduced_fg_patch_embeddings = minmax_scale(reduced_fg_patch_embeddings)  # Scale to (0,1)

    for i, num_patches in enumerate(num_fg_patches_list):
        print(f"Num of foreground patches of image {i}: {num_patches}")

    num_fg_patches_all_images = sum(num_fg_patches_list)
    print(f"Total num of foreground patches: {num_fg_patches_all_images}")
        
    print("Explained variance ratio by PCA components:", object_pca.explained_variance_ratio_)
        

    return reduced_patch_embeddings, reduced_fg_patch_embeddings, num_fg_patches_list, masks

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

    patch_num = (H // patch_size) * (W // patch_size)  # Total number of patches per image
    print(f"Number of patches of each image: {patch_num}")
    
    return B, C, H, W, patch_size, embedding_dim, patch_num
    



def save_np_array_as_video(input_array: np.ndarray, output_path='array_video.mp4', fps: float = 30.0):
    """
    Saves a NumPy array (T, H, W, C) as an MP4 video.

    Args:
    - input_array (np.ndarray): Video array with shape (T, H, W, C) in RGB format.
    - output_path (str): Path to save the output video.
    - fps (float): Frames per second for the output video.
    """
    T, H, W, C = input_array.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))  # ✅ Fix width-height order

    for i in range(T):
        frame = input_array[i]

        # ✅ Scale only if float in range [0,1]
        if np.issubdtype(frame.dtype, np.floating) and frame.min() >= 0 and frame.max() <= 1:
            frame = (frame * 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out.write(frame_bgr)

    out.release()
    print(f"Video saved to {output_path}")




def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize a NumPy array frame to the range [0, 255] if it contains float values."""
    if frame.dtype == np.float32 or frame.dtype == np.float64:
        min_val, max_val = frame.min(), frame.max()
        if max_val > min_val:  # Avoid division by zero
            frame = (frame - min_val) / (max_val - min_val) * 255
        frame = frame.astype(np.uint8)
    return frame

def save_triple_video(
    input_tensor: torch.Tensor,
    reduced_patch_embeddings: np.ndarray,
    reduced_fg_patch_embeddings: np.ndarray,
    nums_of_fg_patches: list,
    masks: list,
    num_patches: int,
    patch_size: int,
    output_path='triple_video.mp4',
    fps: float = 30.0
):
    """
    Saves an MP4 video with three sections:
    - Left: Original frames
    - Middle: Foreground mask from PCA component (B, num_patches)
    - Right: PCA-based foreground patches (B, num_fg_patches, 3)
    """
    B, C, H, W = input_tensor.shape
    start_idx = 0
    frames_list = []

    for i, mask in enumerate(masks):
        num_fg_patches = nums_of_fg_patches[i]

        # ======== PCA Foreground Patches (Right) ========
        patch_image = np.zeros((num_patches, 3), dtype='float32')  # Black background
        patch_image[mask, :] = reduced_fg_patch_embeddings[start_idx:start_idx + num_fg_patches, :]
        start_idx += num_fg_patches
        color_patches = patch_image.reshape((H // patch_size, W // patch_size, 3))
        pca_frame = cv2.resize(color_patches, (W, H))  # Keep float values for normalization
        pca_frame = normalize_frame(pca_frame)  # Normalize separately

        # ======== Foreground Mask (Middle) ========
        mask_image = np.zeros((num_patches,), dtype='float32')
        mask_image[mask] = reduced_patch_embeddings[i][mask]
        mask_image = mask_image.reshape((H // patch_size, W // patch_size))
        mask_frame = cv2.resize(mask_image, (W, H))  # Keep float values for normalization
        mask_frame = normalize_frame(mask_frame)  # Normalize separately
        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel

        # ======== Original Frame (Left) ========
        original_frame = (input_tensor[i].permute(1, 2, 0).cpu().numpy())
        original_frame = normalize_frame(original_frame)
        
        # ======== Combine Frames Horizontally ========
        combined_frame = np.hstack((original_frame, mask_frame, pca_frame))
        frames_list.append(combined_frame)

    frames_array = np.stack(frames_list)  # Shape: (B, H, W, 3)
    
    save_np_array_as_video(frames_array, output_path=output_path, fps=fps)
