import imageio
import numpy as np
import torch


def save_video(tensor, output_path):
    """
    Saves the video frames to a video file.

    Parameters:
    - tensor (torch.Tensor): The video frames' tensor.
    - output_path (str): The path to save the output video.
    """
    tensor = tensor.to(dtype=torch.float32)
    frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)
    writer = imageio.get_writer(output_path, fps=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
