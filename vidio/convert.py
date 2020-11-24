from typing import Union
import os

from .read import VideoReader
from .write import VideoWriter

def convert_video(videofile: Union[str, os.PathLike], movie_format: str, *args, **kwargs) -> None:
    with VideoReader(videofile) as reader:
        with VideoWriter(videofile, movie_format=movie_format, *args, **kwargs) as writer:
            for frame in reader:
                writer.write(frame)