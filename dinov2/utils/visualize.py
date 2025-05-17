import sys
import os
import math
from typing import Callable, Optional, Tuple
import torch
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

import torchvision.transforms as T
from PIL import Image


def min_max_normalize(attention_map):
    att_min = attention_map.min()
    att_max = attention_map.max()
    return (attention_map - att_min) / (att_max - att_min + 1e-8)  # Adding epsilon for numerical stability


def load_and_preprocess_video(
    video_path: str,
    target_size: Optional[int] = None,
    patch_size: int = 14,
    device: str = "cuda",
    hook_function: Optional[Callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Loads a video, applies a hook function if provided, and then applies transforms.

    Processing order:
    1. Read raw video frames into a tensor
    2. Apply hook function (if provided)
    3. Apply resizing and other transforms
    4. Make dimensions divisible by patch_size

    Args:
        video_path (str): Path to the input video.
        target_size (int or None): Final resize dimension (e.g., 224 or 448). If None, no resizing is applied.
        patch_size (int): Patch size to make the frames divisible by.
        device (str): Device to load the tensor onto.
        hook_function (Callable, optional): Function to apply to the raw video tensor before transforms.

    Returns:
        torch.Tensor: Unnormalized video tensor (T, C, H, W).
        torch.Tensor: Normalized video tensor (T, C, H, W).
        float: Frames per second (FPS) of the video.
    """
    # Step 1: Load the video frames into a raw tensor
    cap = cv2.VideoCapture(video_path)

    # Get video metadata
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    print(f"Video FPS: {fps:.2f}, Total Frames: {total_frames}, Duration: {duration:.2f} seconds")

    # Read all frames
    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raw_frames.append(frame)
    cap.release()

    # Convert to tensor [T, H, W, C]
    raw_video = torch.tensor(np.array(raw_frames), dtype=torch.float32) / 255.0
    # Permute to [T, C, H, W] format expected by PyTorch
    raw_video = raw_video.permute(0, 3, 1, 2)

    # Step 2: Apply hook function to raw video tensor if provided
    if hook_function is not None:
        raw_video = hook_function(raw_video)

    # Step 3: Apply transforms
    # Create unnormalized tensor by applying resize if needed
    unnormalized_video = raw_video.clone()
    if target_size is not None:
        resize_transform = T.Resize((target_size, target_size))
        # Process each frame
        frames_list = [resize_transform(frame) for frame in unnormalized_video]
        unnormalized_video = torch.stack(frames_list)

    # Step 4: Make dimensions divisible by patch_size
    t, c, h, w = unnormalized_video.shape
    h_new = h - (h % patch_size)
    w_new = w - (w % patch_size)
    if h != h_new or w != w_new:
        unnormalized_video = unnormalized_video[:, :, :h_new, :w_new]

    # Create normalized version
    normalized_video = unnormalized_video.clone()
    # Apply normalization to each frame
    normalize_transform = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    normalized_frames = [normalize_transform(frame) for frame in normalized_video]
    normalized_video = torch.stack(normalized_frames)

    return unnormalized_video.to(device), normalized_video.to(device), fps


def get_model_output(model, input_tensor: torch.Tensor):
    """
    Extracts the class token embedding and patch token embeddings from the model's output.
    Args:
        model: The model object that contains the `forward_features` method.
        input_tensor: A tensor representing the input data to the model.
    Returns:
        tuple: A tuple containing:
            - cls_token_embedding (numpy.ndarray): The class token embedding extracted from the model's output.
            - patch_token_embeddings (numpy.ndarray): The patch token embeddings extracted from the model's output.
    """
    result = model.forward_features(input_tensor)  # Forward pass
    cls_token_embedding = result["x_norm_clstoken"].detach().cpu().numpy()
    patch_token_embeddings = result["x_norm_patchtokens"].detach().cpu().numpy()
    return cls_token_embedding, patch_token_embeddings


def get_cls_token_embeddings(model, input_tensor):
    """
    Extracts CLS token embeddings from the given model and input tensor.
    This function performs a forward pass through the model to obtain
    the normalized CLS token embeddings. The embeddings are then
    detached from the computation graph, moved to the CPU, and reshaped
    into a 2D array with dimensions corresponding to the batch size
    and embedding dimension.
    Args:
        model (torch.nn.Module): The model from which to extract CLS token embeddings.
            The model is expected to have a `forward_features` method that returns
            a dictionary containing the key 'x_norm_cls_token'.
        input_tensor (torch.Tensor): The input tensor to the model, typically
            representing a batch of images. The shape is expected to be
            (B, C, H, W), where B is the batch size, C is the number of channels,
            and H and W are the height and width of the images.
    Returns:
        numpy.ndarray: A 2D array of shape (B, embedding_dim), where B is the batch size
            and embedding_dim is the dimensionality of the CLS token embeddings.
    """
    result = model.forward_features(input_tensor)  # Forward pass

    cls_embeddings = result["x_norm_clstoken"].detach().cpu().numpy()
    return cls_embeddings  # (B, embedding_dim)


def get_patch_embeddings(model, input_tensor):
    """
    Extracts patch embeddings from the given model and input tensor.
    This function performs a forward pass through the model to obtain
    the normalized patch token embeddings. The embeddings are then
    detached from the computation graph, moved to the CPU, and reshaped
    into a 3D array with dimensions corresponding to the batch size,
    number of patches, and embedding dimension.
    Args:
        model (torch.nn.Module): The model from which to extract patch embeddings.
            The model is expected to have a `forward_features` method that returns
            a dictionary containing the key 'x_norm_patchtokens'.
        input_tensor (torch.Tensor): The input tensor to the model, typically
            representing a batch of images. The shape is expected to be
            (B, C, H, W), where B is the batch size, C is the number of channels,
            and H and W are the height and width of the images.
    Returns:
        numpy.ndarray: A 3D array of shape (B, patch_num, embedding_dim), where
            B is the batch size, patch_num is the number of patches, and
            embedding_dim is the dimensionality of the patch embeddings.
    """
    result = model.forward_features(input_tensor)  # Forward pass

    patch_embeddings = (
        result["x_norm_patchtokens"].detach().cpu().numpy().reshape([input_tensor.shape[0], -1, model.embed_dim])
    )
    return patch_embeddings  # (B, patch_num, embedding_dim)


def get_attention_map(
    attn_weights: np.ndarray,
    height_in_patches: int,
    width_in_patches: int,
    num_register_tokens: int = 0,
) -> np.ndarray:
    """
    Extract an attention mapfrom attention weights of shape (T, NH, H_p + num_register_tokens +  1, W_p + num_register_tokens + 1).

    Args:
    - attentions (np.ndarray): Attention array of shape (T, NH, H_p + num_register_tokens +  1, W_p + num_register_tokens + 1).
    - height_in_patches (int): Number of patches in height.
    - width_in_patches (int): Number of patches in width.
    = num_register_tokens (int): Number of register tokens.

    Returns:
    - np.ndarray: normalized attention map of shape (T, H_p, W_p), range [0, 1].
    """
    T, NH = attn_weights.shape[:2]  # Video length, number of heads

    # Extract the attention of the CLS token to all patch tokens, getting H_p * W_p attention weights, then reshape it into a map
    attn_weights = attn_weights[:, :, 0, num_register_tokens + 1 :].reshape(
        T, NH, height_in_patches, width_in_patches
    )  # (T, NH, H_p, W_p)

    # Average over all heads
    average_attention = np.mean(attn_weights, axis=1)  # Shape: (T, H_p, W_p)
    normalized_map = min_max_normalize(average_attention)  # Normalize to [0, 1]
    return normalized_map  # This is the attention map


def colorize_attention_map(
    attention_map: np.ndarray,
    patch_size: int,
) -> np.ndarray:
    """
    Visualizes attention maps as colorized frames using the inferno colormap and returns a NumPy array.

    Args:
    - attentions (np.ndarray): Attention map of shape (T, H_p, W_p), range [0, 1].
    - patch_size (int): Size of each patch in pixels.

    Returns:
    - np.ndarray: Colorized attention maps of shape (T, H, W, 3).
    """
    T, H_p, W_p = attention_map.shape[:3]

    # Upscale attention maps to original resolution using OpenCV
    upscaled_attentions = np.array(
        [
            cv2.resize(
                attention_map[t],
                (H_p * patch_size, W_p * patch_size),
                interpolation=cv2.INTER_NEAREST,
            )
            for t in range(T)
        ]
    )  # Shape: (T, H, W)

    # Apply inferno colormap frame by frame
    colorized_frames = np.array(
        [
            plt.cm.inferno(frame)[:, :, :3] for frame in upscaled_attentions
        ]  # each frame  is of shape (H, W), we colorize it to (H, W, 3)
    )  # Shape: (T, H, W, 3)

    return colorized_frames


def find_percentile_threshold(attn_map, percentile=80):
    """
    Determine threshold based on a percentile of attention values.
    Higher percentile = more focused/selective mask.
    """
    return np.percentile(attn_map, percentile)  # top 20% > threshold >= bottom 80%


def generate_attention_mask(normalized_attn_map: np.ndarray, threshold: float = 0.72) -> np.ndarray:
    """
    Generate a binary mask from a normalized attention map using a threshold.

    Args:
        normalized_attn_map (np.ndarray): Normalized attention map of shape (T, H_p, W_p)
                                         with values in range [0, 1].
        threshold (float): Threshold value between 0 and 1. Attention values greater than
                          or equal to this threshold will be set to 1, otherwise 0.

    Returns:
        np.ndarray: Binary mask of shape (T, H_p, W_p) with values {0, 1}.
    """
    # Ensure the threshold is in valid range
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {threshold}")

    # Apply threshold to create binary mask
    binary_mask = (normalized_attn_map >= threshold).astype(np.uint8)

    return binary_mask


def two_stage_pca(patch_embeddings, threshold=0.6):
    """
    Perform two-stage PCA on patch embeddings:
    1. First PCA with 1 component to extract foreground patches.
    2. Second PCA with 3 components to reduce foreground patches to RGB-like embeddings.

    This computation doesn NOT involve GPU, so all the params are np arrays.

    Args:
    - patch_embeddings (np.ndarray): Patch embeddings of shape (B, num_patches_all_images, embedding_dim)
    - threshold (float): Threshold for selecting foreground patches after the first PCA.

    Returns:
    - reduced_patch_embeddings (np.ndarray): PCA-reduced patch embeddings of shape (B, num_patches). This is the first PCA output.
    - reduced_fg_patch_embeddings (np.ndarray): PCA-reduced foreground patch embeddings of shape (num_fg_patches_all_images, 3). This is the second PCA output.
    - num_fg_patches_list (list): List of numbers of foreground patches for each image. Each element is the num of the first PCA output, whose value is greater than the threshold, of one image in the batch.
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
    fg_patch_embeddings = np.vstack(
        [patch_embeddings[i, m, :] for i, m in enumerate(masks)]
    )  # (total_fg_patches, embedding_dim)

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
    Prints and returns statistics about the input tensor and the model's configuration.
    Args:
        input_tensor (torch.Tensor): The input tensor with shape (B, C, H, W), where
            B is the batch size, C is the number of channels, H is the height, and W is the width.
        model (object): The model object, expected to have attributes `patch_size` (int) and `embed_dim` (int),
            representing the patch size and embedding dimension, respectively.
    Returns:
        tuple: A tuple containing the following:
            - B (int): Batch size of the input tensor.
            - C (int): Number of channels in the input tensor.
            - H (int): Height of the input tensor.
            - W (int): Width of the input tensor.
            - patch_size (int): Patch size used by the model.
            - embedding_dim (int): Embedding dimension of the model.
            - patch_num (int): Total number of patches per image, calculated as (H // patch_size) * (W // patch_size).
    Prints:
        - The shape of the input tensor (Batch, Channels, Height, Width).
        - The patch size of the model.
        - The embedding dimension of the model.
        - The number of patches per image.
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


def save_np_array_as_video(input_array: np.ndarray, output_path="array_video.mp4", fps: float = 30.0):
    """
    Saves a NumPy array (T, H, W, C) as an MP4 video.

    Args:
    - input_array (np.ndarray): Video array with shape (T, H, W, C) in RGB format.
    - output_path (str): Path to save the output video.
    - fps (float): Frames per second for the output video.
    """
    T, H, W, C = input_array.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
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


def assemble_visualizations(
    input_tensor: torch.Tensor,
    reduced_patch_embeddings: np.ndarray,
    reduced_fg_patch_embeddings: np.ndarray,
    nums_of_fg_patches: list,
    masks: list,
    num_patches: int,
    patch_size: int,
):
    B, C, H, W = input_tensor.shape
    start_idx = 0
    frame_list = []

    for i, mask in enumerate(masks):
        num_fg_patches = nums_of_fg_patches[i]

        # ======== PCA Foreground Patches (Right) ========
        patch_image = np.zeros(
            (num_patches, 3), dtype="float32"
        )  # Black background. num_patches == (H // patch_size) * (W // patch_size)
        patch_image[mask, :] = reduced_fg_patch_embeddings[start_idx : start_idx + num_fg_patches, :]
        start_idx += num_fg_patches
        color_patches = patch_image.reshape((H // patch_size, W // patch_size, 3))
        pca_frame = cv2.resize(color_patches, (W, H))  # Keep float values for normalization
        pca_frame = normalize_frame(pca_frame)  # Normalize separately

        # ======== Foreground Mask (Middle) ========
        mask_image = np.zeros((num_patches,), dtype="float32")
        mask_image[mask] = reduced_patch_embeddings[i][mask]
        mask_image = mask_image.reshape((H // patch_size, W // patch_size))
        mask_frame = cv2.resize(mask_image, (W, H))  # Keep float values for normalization
        mask_frame = normalize_frame(mask_frame)  # Normalize separately
        mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel

        # ======== Original Frame (Left) ========
        original_frame = input_tensor[i].permute(1, 2, 0).cpu().numpy()
        original_frame = normalize_frame(original_frame)

        # ======== Combine Frames Horizontally ========
        combined_frame = np.hstack((original_frame, mask_frame, pca_frame))
        frame_list.append(combined_frame)
    return frame_list


def save_triple_image(
    input_tensor: torch.Tensor,
    reduced_patch_embeddings: np.ndarray,
    reduced_fg_patch_embeddings: np.ndarray,
    nums_of_fg_patches: list,
    masks: list,
    num_patches: int,
    patch_size: int,
    output_path="triple_video.png",
):
    """
    Saves a PNG image with three sections:
    - Left: Original frames
    - Middle: Foreground grayscale mask from PCA component (B, num_patches, 3)
    - Right: PCA-based foreground visualization (B, num_fg_patches, 3)
    """
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    # Ensure input_tensor is 4D (B, C, H, W)

    assert input_tensor.shape[0] == 1, "Image must have only one frame."
    assert input_tensor.shape[1] == 3, "Input tensor must have 3 channels (RGB)."

    frame_list = assemble_visualizations(
        input_tensor,
        reduced_patch_embeddings,
        reduced_fg_patch_embeddings,
        nums_of_fg_patches,
        masks,
        num_patches,
        patch_size,
    )
    assert len(frame_list) == 1, "Expected only one frame in the list."
    # saves an image to a specified file.

    cv2.imwrite(output_path, cv2.cvtColor(frame_list[0], cv2.COLOR_RGB2BGR))  # Save the first frame as an image


def save_triple_video(
    input_tensor: torch.Tensor,
    reduced_patch_embeddings: np.ndarray,
    reduced_fg_patch_embeddings: np.ndarray,
    nums_of_fg_patches: list,
    masks: list,
    num_patches: int,
    patch_size: int,
    output_path="triple_video.mp4",
    fps: float = 30.0,
):
    """
    Saves an MP4 video with three sections:
    - Left: Original frames
    - Middle: Foreground mask from PCA component (B, num_patches)
    - Right: PCA-based foreground patches (B, num_fg_patches, 3)
    """
    assert input_tensor.shape[0] > 1, "Video must have more than one frame."
    assert input_tensor.shape[1] == 3, "Input tensor must have 3 channels (RGB)."

    frames_list = assemble_visualizations(
        input_tensor,
        reduced_patch_embeddings,
        reduced_fg_patch_embeddings,
        nums_of_fg_patches,
        masks,
        num_patches,
        patch_size,
    )

    frames_array = np.stack(frames_list)  # Shape: (B, H, W, 3)

    save_np_array_as_video(frames_array, output_path=output_path, fps=fps)


def compute_cosine_similarity(emb1, emb2):
    """
    Compute cosine similarity between two embeddings.

    Args:
        emb1 (np.ndarray): First video embedding of shape (T, feature_dim).
        emb2 (np.ndarray): Second video embedding of shape (T, feature_dim).

    Returns:
        List[float]: Cosine similarity scores for each pair of image embeddings.
    """
    # L2 normalization along the feature dimension
    emb1_norm = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    emb2_norm = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)

    # Compute cosine similarity
    similarity = np.sum(emb1_norm * emb2_norm, axis=1)

    return similarity.tolist()


def get_last_self_attn(model: torch.nn.Module, video: torch.Tensor):
    """
    Get the last self-attention weights from the model for a given video tensor. We collect attention weights for each frame iteratively and stack them.
    This solution saves VRAM but not forward all frames at once. But it should be OKay as DINOv2 doesn't integrate the time dimension processing.

    Parameters:
        model (torch.nn.Module): The model from which to extract the last self-attention weights.
        video (torch.Tensor): Input video tensor with shape (T, C, H, W).

    Returns:
        np.ndarray: Last self-attention weights of shape (T, NH, H_p + num_register_tokens +  1, W_p + num_register_tokens + 1).
    """
    from tqdm import tqdm

    T, C, H, W = video.shape
    last_selfattention_list = []
    with torch.no_grad():
        for i in tqdm(range(T)):
            frame = video[i].unsqueeze(0)  # Add batch dimension for the model

            # Forward pass for the single frame
            last_selfattention = model.get_last_selfattention(frame).detach().cpu().numpy()

            last_selfattention_list.append(last_selfattention)

    return np.vstack(
        last_selfattention_list
    )  # (B, num_heads, num_tokens, num_tokens), where num_tokens = H_p + num_register_tokens + 1


def plot_videos(video1: np.ndarray, video2: np.ndarray, distances: list, output_path=None, fps=10):
    """
    Creates an animation of two videos side by side with their frame-by-frame distances.

    Parameters:
        video1 (np.ndarray): First video with shape (frames, height, width, channels)
        video2 (np.ndarray): Second video with shape (frames, height, width, channels)
        distances (list): List of computed distances between corresponding frames
        output_path (str, optional): Path to save the animation (mp4 or gif)
        fps (int): Frames per second for the animation
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Ensure videos have the same number of frames
    assert len(video1) == len(video2) == len(distances), "Videos and distances must have the same number of frames"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Initialize with first frame
    img1 = ax1.imshow(video1[0])
    ax1.set_title("Original")
    ax1.axis("off")

    img2 = ax2.imshow(video2[0])
    ax2.set_title("Covered")
    ax2.axis("off")

    title = plt.suptitle(f"Frame 0 | Distance: {distances[0]:.6f}", fontsize=16)
    plt.tight_layout()

    def update(frame):
        img1.set_array(video1[frame])
        img2.set_array(video2[frame])
        title.set_text(f"Frame {frame} | Distance: {distances[frame]:.6f}")
        return img1, img2, title

    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(video1), interval=1000 / fps, blit=False)

    if output_path:
        # Determine format based on file extension
        if output_path.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=fps)
            ani.save(output_path, writer=writer)
        elif output_path.endswith(".gif"):
            writer = animation.PillowWriter(fps=fps)
            ani.save(output_path, writer=writer)
        else:
            raise ValueError("Output file must be .mp4 or .gif")
        plt.close()
    else:
        plt.show()

    return ani  # Return animation object to prevent garbage collection


def plot_distance_chart(distances, title="Frame-by-Frame Distance", output_path=None):
    """
    Creates a simple line chart showing distances over frames.

    Parameters:
        distances (list): List of distance values between corresponding frames
        title (str): Title for the chart
        output_path (str, optional): Path to save the chart image
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Create figure
    plt.figure(figsize=(10, 5))

    # Plot distance line with actual values (no percentage conversion)
    plt.plot(distances, "b-", linewidth=2)

    # Add basic stats
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)

    # Add labels and title
    plt.xlabel("Frame Number")
    plt.ylabel("Distance")
    plt.title(f"{title}\nMean: {mean_distance:.4f}, Max: {max_distance:.4f}")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=150)
        plt.close()
    else:
        plt.show()


def colorize_area(video: torch.Tensor, color: list, starting_location: tuple, width: int, height: int):
    """
    Colorize a rectangular area of a video tensor with a specified color.

    Args:
        video (torch.Tensor): Input video tensor with shape [T, C, H, W] or [B, T, C, H, W]
            where T is time/frames, C is channels (typically 3 for RGB), H and W are height and width,
            and B is an optional batch dimension.
        color (list): RGB color values as a list of 3 values in range [0, 1].
        starting_location (tuple): The (y, x) coordinates of the top-left corner of the area to colorize.
        width (int): The width of the area to colorize.
        height (int): The height of the area to colorize.

    Returns:
        torch.Tensor: Video tensor with the specified area colorized.
    """
    # Create a copy of the input video to avoid modifying the original
    colorized_video = video.clone()

    # Check if tensor has batch dimension
    has_batch = len(video.shape) == 5

    # Extract shape information
    if has_batch:
        B, T, C, H, W = video.shape
    else:
        T, C, H, W = video.shape

    # Transform color values to be in the range [0, 1]
    color = [c / 255.0 for c in color]  # Ensure color is in range [0, 1]

    # Ensure color is a tensor of the right shape
    color_tensor = torch.tensor(color, dtype=video.dtype, device=video.device)

    #
    # Ensure color has the right number of channels
    if len(color) != C:
        if len(color) == 3 and C == 1:  # RGB color for grayscale video
            color_tensor = color_tensor.mean()
        elif len(color) == 1 and C == 3:  # Grayscale color for RGB video
            color_tensor = color_tensor.repeat(3)
        else:
            raise ValueError(f"Color channels {len(color)} don't match video channels {C}")

    # Validate coordinates
    # y: the row index, 0<= y <= H-1
    # x: the column index, 0<= x <= W-1
    # (0,0) means the top-left corner of the video
    y, x = starting_location
    if y < 0 or y + height > H or x < 0 or x + width > W:
        raise ValueError(f"The specified area ({x},{y},{width},{height}) is outside video dimensions ({W},{H})")

    # Calculate the alpha blend factor (0.3 means 30% original + 70% color)
    alpha = 0.0

    # Apply colorization to the specified area
    # color_tensor is reshaped to match the dimensions needed for broadcasting
    if has_batch:
        # For batch dimension: [B, T, C, y:y+h, x:x+w]
        original = colorized_video[:, :, :, y : y + height, x : x + width]
        colored = alpha * original + (1 - alpha) * color_tensor.view(1, 1, C, 1, 1)
        colorized_video[:, :, :, y : y + height, x : x + width] = colored
    else:
        # Without batch dimension: [T, C, y:y+h, x:x+w]
        original = colorized_video[:, :, y : y + height, x : x + width]
        colored = alpha * original + (1 - alpha) * color_tensor.view(1, C, 1, 1)
        colorized_video[:, :, y : y + height, x : x + width] = colored

    return colorized_video


def image_tensor_to_np(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        image_tensor (torch.Tensor): Input tensor with shape (T, C, H, W).

    Returns:
        np.ndarray: Converted NumPy array with shape (T, H, W, C).
    """
    T, C, H, W = image_tensor.shape
    return image_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()
