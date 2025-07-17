import os
import subprocess

def ffmpegFrameExtractor(input_video_path:os.PathLike,
                         output_frame_path:os.PathLike  =None,
                         fps:int = 1,
                         _Wipe = True,
                         _CUDA = False) -> None:
    """
    Extract frames from a video using FFmpeg, with optional CUDA acceleration.

    Args:
        input_video_path (os.PathLike): Path to the input video file.
        output_frame_path (os.PathLike, optional): Path pattern to save the extracted frames.
            If None, defaults to '<video_dir>/frames/frame_%06d.jpg'.
        fps (int, optional): Number of frames to extract per second of video. Defaults to 1.
        _Wipe (bool, optional): If True, deletes all existing files in the output directory before extraction.
            Defaults to True.
        _CUDA (bool, optional): If True, uses CUDA hardware acceleration for decoding. Defaults to False.

    Returns:
        None: Not much

    Example:
        >>> ffmpegFrameExtractor("video.mp4", fps=2)
        Frames extracted successfully to /path/to/video/frames/frame_%06d.jpg
    """

    if not os.path.isfile(input_video_path):
        raise ValueError("Input video path and output frame path must be provided.")
    
    if output_frame_path is None:
        output_frame_path = os.path.join(os.path.dirname(input_video_path), 'frames', 'frame_%06d.jpg')

    if not os.path.isdir(os.path.split(output_frame_path)[0]):
        os.makedirs(os.path.split(output_frame_path)[0], exist_ok=True)
    else:
        if _Wipe:
            for file in os.listdir(os.path.split(output_frame_path)[0]):
                file_path = os.path.join(os.path.split(output_frame_path)[0], file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

    command = [
        'ffmpeg',]
    if _CUDA:
        command+= [ '-hwaccel', 'cuda',                   # Enable CUDA acceleration
                    # '-hwaccel_output_format', 'cuda',     # Use CUDA pixel format
                    '-c:v', 'h264_cuvid'                 # Use NVIDIA's hardware decoder for H.264
                    ]
    command+= ['-i', input_video_path,]
        
    if _CUDA:
        command += ['-vf', f'hwdownload,format=bgr0,fps={fps}',]       # Move hwdownload+format before fps
    else:
        command += ['-vf', f'fps={fps}',]           # Extract one frame per second
        
    command += [
        '-qscale:v',            '1',                    # Highest quality JPEG (lower = better)
        '-compression_level',   '100',                  # Max compression efficiency (for formats that support it)
        '-f',                   'image2',
        output_frame_path
    ]
    
    try:
        subprocess.run(command, check=True)
        print(f"Frames extracted successfully to {output_frame_path}")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while extracting frames: {e}")
    except FileNotFoundError:
        print("ffmpeg is not installed or not found in the system path. Please install ffmpeg and try again.")


if __name__ == "__main__":
    input_video = r'280\S2-SNr2.1_D\S2-SNr2.1_D_20250621_203528.mp4'  # Replace with your input video file
    ffmpegFrameExtractor(input_video,fps=30,_Wipe=True)