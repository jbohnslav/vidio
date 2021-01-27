from typing import Union
import os

from .read import VideoReader
from .write import VideoWriter

def convert_video(videofile: Union[str, os.PathLike], movie_format: str, *args, **kwargs) -> None:
    with VideoReader(videofile) as reader:
        basename = os.path.splitext(videofile)[0]
        if movie_format == 'ffmpeg':
            out_filename = basename + '.mp4'
        elif movie_format == 'opencv':
            out_filename = basename + '.avi'
        elif movie_format == 'hdf5':
            out_filename = basename + '.h5'
        elif movie_format == 'directory':
            out_filename = basename
        else:
            raise ValueError('unexpected value of movie format: {}'.format(movie_format))
        with VideoWriter(out_filename, movie_format=movie_format, *args, **kwargs) as writer:
            for frame in reader:
                writer.write(frame)