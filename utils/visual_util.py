import os, cv2
from glob import glob

def images_to_video(
        image_dir: str,
        output_file: str = "output.mp4",
        fps: int = 30,
        img_exts: tuple = (".png", ".jpg", ".jpeg"),
        sort: bool = True,
):
    """
    Converts a directory of images into a video.

    Args:
        image_dir (str): Path to the directory containing image files.
        output_file (str): Name of the output video file (e.g., 'video.mp4').
        fps (int): Frames per second for the output video.
        img_exts (tuple): Allowed image extensions.
        sort (bool): Whether to sort images by filename before writing.
    """
    # Collect image file paths
    image_paths = [f for f in glob(os.path.join(image_dir, "*")) if f.lower().endswith(img_exts)]
    if not image_paths:
        raise ValueError(f"No images with extensions {img_exts} found in {image_dir}.")

    if sort:
        image_paths = sorted(image_paths)

    # Read the first image to get frame size
    first_frame = cv2.imread(image_paths[0])
    if first_frame is None:
        raise ValueError(f"Cannot read image {image_paths[0]}")
    height, width, _ = first_frame.shape

    # Set up the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
    video_writer = cv2.VideoWriter(os.path.join(image_dir, output_file), fourcc, fps, (width, height))

    # Write each image as a frame
    for path in image_paths:
        frame = cv2.imread(path)
        if frame is None:
            print(f"Warning: cannot read image {path}, skipping.")
            continue
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {output_file}")
