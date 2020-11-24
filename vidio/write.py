import os
import subprocess as sp
import time
import warnings
from queue import Queue
from threading import Thread
from typing import Union

import cv2
import h5py
import numpy as np

from .constants import IMG_EXTENSIONS, VID_EXTENSIONS, COLORSPACES
from .utils import convert_colorspace

class BaseWriter:
    def __init__(self, filename: Union[str, bytes, os.PathLike],
                 height: int = None, width: int = None,
                 fps: int = 30,  in_colorspace: str='RGB', out_colorspace:str='RGB', codec: str = 'MJPG',
                 asynchronous: bool = False, verbose: bool = False) -> None:
        """Initializes a VideoWriter object.

        Args:
            filename: name of movie to be written
            height: height (rows) in frames of movie. None: figure it out when the first frame is written
            width: width (columns) in frames of movie. None: figure it out when the first frame is written
            fps: frames per second. Does nothing for HDF5 encoding
            movie_format: one of 'opencv', 'ffmpeg', or 'hdf5'. See the class docstring for more information
            codec: encoder for OpenCV video writing. I recommend MJPG, 0, DIVX, XVID, or FFV1.
                More info here: http://www.fourcc.org/codecs.php
            filetype: the type of image to save if saving as a directory of images.
                [.bmp, jpg, .png, .tiff]
            colorspace: colorspace of input frames. Necessary because OpenCV expects BGR. Default: RGB
            asynchronous: if True, writes in a background thread. Useful if writing to disk is slower than the image
                generation process.
            verbose: True will generate lots of print statements for debugging

        Returns:
            VideoWriter object
        """
        self.filename = filename

        self.height = height
        self.width = width
        self.fps = fps
        self.codec = codec

        self.in_colorspace = in_colorspace
        self.out_colorspace = out_colorspace
        assert self.in_colorspace in COLORSPACES
        assert self.out_colorspace in COLORSPACES

        self.verbose = verbose

        self.asynchronous = asynchronous
        if self.asynchronous:
            self.save_queue = Queue(maxsize=3000)
            self.save_thread = Thread(target=self.save_worker, args=(self.save_queue,))
            self.save_thread.daemon = True
            self.save_thread.start()
        self.has_stopped = False
        self.writer_obj = None

    def save_worker(self, queue):
        """Worker for asychronously writing video to disk"""
        should_continue = True
        while should_continue:
            try:
                item = queue.get()
                if item is None:
                    if self.verbose:
                        print('Saver stop signal received')
                    should_continue = False
                    break
                self.write_frame(item)
                # print(queue.qsize())
            except Exception as e:
                print(e)
            finally:
                queue.task_done()
        if self.verbose:
            print('out of save queue')

    def write(self, frame: np.ndarray):
        """Writes numpy array to disk"""
        if self.asynchronous:
            self.save_queue.put(frame)
        else:
            self.write_frame(frame)

    def process_frame(self, frame):
        if frame.ndim == 3:
            H, W, C = frame.shape
        # add a gray channel if necessary
        elif frame.ndim == 2:
            H, W = frame.shape
            C = 1
            frame = frame[..., np.newaxis]
        else:
            raise ValueError('Unknown frame dimensions: {}'.format(frame.shape))

        # use the first frame to get height and width
        if self.height is None:
            self.height = H
        else:
            assert self.height == H
        if self.width is None:
            self.width = W
        else:
            assert self.width == W

        if frame.dtype == np.uint8:
            pass
        elif frame.dtype == np.float:
            # make sure that frames are in proper format before writing. We don't want the writer to be implicitly
            # changing pixel values, that should be done outside of this Writer class
            assert (frame.min() >= 0 and frame.max() <= 1)
            frame = (frame * 255).clip(min=0, max=255).astype(np.uint8)

        frame = convert_colorspace(frame, self.in_colorspace, self.out_colorspace)
        return frame

    def write_frame(self, frame: np.ndarray):
       pass

    def __enter__(self):
        # allows use with decorator
        return self

    def __exit__(self, type, value, traceback):
        # allows use with decorator
        self.close()

    def close(self):
        """Stops writing, closes all open file objects"""
        if self.has_stopped:
            return
        if self.asynchronous:
            # wait for save worker to complete, then finish
            self.save_queue.put(None)
            if self.verbose:
                print('joining...')
            self.save_queue.join()
            if self.verbose:
                print('joined.')
            del self.save_queue

    def __del__(self):
        """Destructor"""
        try:
            self.close()
        except BaseException as e:
            if self.verbose:
                print('Error in destructor')
                print(e)
            else:
                pass


class HDF5Writer(BaseWriter):
    def __init__(self, filename: Union[str, bytes, os.PathLike],
                 height: int = None, width: int = None,
                 fps: int = 30, in_colorspace: str='RGB', codec: str = '.png',
                 asynchronous: bool = True, verbose: bool = False, nframes: int=None) -> None:
        out_colorspace = 'RGB'
        super().__init__(filename,  height=height, width=width, fps=fps,
                         in_colorspace=in_colorspace, out_colorspace=out_colorspace,
                         codec=codec, asynchronous=asynchronous, verbose=verbose)

        base, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext != '.h5' and ext != '.hdf5':
            warnings.warn('filename should end in .h5 or .hdf5 for HDF5Writer, not {}'.format(ext))
            self.filename = base + '.h5'
        assert self.codec in IMG_EXTENSIONS

        # we can initialize the writer here because we don't need to know frame size for this file format
        self.writer_obj = None # to get rid of warnings
        self.initialize_writer(nframes)
        self.fnum = 0

    def initialize_writer(self, nframes):
        self.writer_obj = h5py.File(self.filename, 'w')
        datatype = h5py.special_dtype(vlen=np.dtype('uint8'))
        if nframes is None:
            self.writer_obj.create_dataset('frame', (0,), maxshape=(None,), dtype=datatype)
        else:
            self.writer_obj.create_dataset('frame', (nframes,), dtype=datatype)

    def write_frame(self, frame: np.ndarray):
        frame = self.process_frame(frame)
        # PNG by default
        ret, encoded = cv2.imencode(self.codec, frame)
        if not ret:
            raise ValueError('error in encoding frame: {}'.format(frame))
        # resize dataset if necessary
        if self.writer_obj['frame'].shape[0] <= self.fnum:
            self.writer_obj['frame'].resize(self.writer_obj['frame'].shape[0] + 1, axis=0)
        self.writer_obj['frame'][self.fnum] = encoded.squeeze()
        self.fnum += 1

    def close(self):
        # handles closing queues in asynchronous case
        super().close()
        if self.writer_obj is not None:
            self.writer_obj.close()
        self.has_stopped = True


class OpenCVWriter(BaseWriter):
    def __init__(self, filename: Union[str, bytes, os.PathLike],
                 height: int = None, width: int = None,
                 fps: int = 30, in_colorspace: str = 'RGB', codec: str = 'MJPG',
                 asynchronous: bool = True, verbose: bool = False, nframes: int = None) -> None:
        out_colorspace = 'BGR'
        # opencv videowriter expects BGR colorspace
        super().__init__(filename, height=height, width=width, fps=fps,
                         in_colorspace=in_colorspace, out_colorspace=out_colorspace,
                         codec=codec, asynchronous=asynchronous, verbose=verbose)

        if height is not None and width is not None:
            self.writer_obj = self.initialize_writer()
        else:
            self.writer_obj = None

    def initialize_writer(self):
        if self.codec == 0:
            filename = self.filename + '_%06d.bmp'
            fourcc = 0
            fps = 0
        else:
            filename = self.filename
            # filename = filename + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            fps = self.fps
        framesize = (self.width, self.height)
        return cv2.VideoWriter(filename, fourcc, fps, framesize)


    def write_frame(self, frame: np.ndarray):
        # adds width and height if they are currently None
        frame = self.process_frame(frame)
        if self.writer_obj is None:
            self.writer_obj = self.initialize_writer()
        self.writer_obj.write(frame)

    def close(self):
        # handles closing queues in asynchronous case
        super().close()
        if self.writer_obj is not None:
            self.writer_obj.release()
        self.has_stopped = True


class FFMPEGWriter(BaseWriter):
    def __init__(self, filename: Union[str, bytes, os.PathLike], movie_format: str = 'opencv',
                 height: int = None, width: int = None,
                 fps: int = 30, in_colorspace: str = 'RGB', codec: str = 'MJPG', filetype='.png',
                 asynchronous: bool = True, verbose: bool = False, nframes: int = None) -> None:
        out_colorspace = 'YUV420'
        # I use FFMPEG writer for YUV420 colorspace for Nvidia DALI
        super().__init__(filename, height=height, width=width, fps=fps,
                         in_colorspace=in_colorspace, out_colorspace=out_colorspace,
                         codec=codec, asynchronous=asynchronous, verbose=verbose)

        if height is not None and width is not None:
            self.writer_obj = self.initialize_writer()
        else:
            self.writer_obj = None

    def initialize_writer(self):
        """ Initializes a Pipe for streaming video data to a libx264-encoded mp4 file using FFMPEG """
        framesize = (self.width, self.height)
        size_string = '%dx%d' % framesize
        fps = str(self.fps)
        command = ['ffmpeg',
                   '-threads', '1',
                   '-y',  # (optional) overwrite output file if it exists
                   '-f', 'rawvideo',
                   '-s', size_string,  # size of one frame
                   '-pix_fmt', 'yuv420p',
                   '-r', fps,  # frames per second
                   '-i', '-',  # The imput comes from a pipe
                   '-an',  # Tells FFMPEG not to expect any audio
                   '-vcodec', 'libx264',
                   '-crf', '18',
                   self.filename]
        print(command)
        # if you want to print to the command line, change stderr to sp.STDOUT
        pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.DEVNULL)
        return pipe

    def write_frame(self, frame: np.ndarray):
        # adds width and height if they are currently None
        frame = self.process_frame(frame)
        if self.writer_obj is None:
            self.writer_obj = self.initialize_writer()
        try:
            self.writer_obj.stdin.write(frame.tobytes())
        except BaseException as err:
            _, ffmpeg_error = self.writer_obj.communicate()
            error = (str(err) + ("\n\nerror: FFMPEG encountered "
                                 "the following error while writing file:"
                                 "\n\n %s" % (str(ffmpeg_error))))
            raise

    def close(self):
        super().close()
        self.writer_obj.stdin.close()
        if self.writer_obj.stderr is not None:
            self.writer_obj.stderr.close()
        self.writer_obj.wait()
        del self.writer_obj
        self.has_stopped = True


class DirectoryWriter(BaseWriter):
    def __init__(self, filename: Union[str, bytes, os.PathLike],
                 height: int = None, width: int = None,
                 fps: int = 30, in_colorspace: str = 'RGB', codec: str = '.png',
                 asynchronous: bool = True, verbose: bool = False) -> None:
        out_colorspace = 'BGR'
        # opencv expects BGR colorspace
        base, ext = os.path.splitext(filename)
        if ext == '':
            pass
        else:
            warnings.warn('Directory writer called with filename input: {}'.format(filename))
            filename = base
        super().__init__(filename, height=height, width=width, fps=fps,
                         in_colorspace=in_colorspace, out_colorspace=out_colorspace,
                         codec=codec, asynchronous=asynchronous, verbose=verbose)
        assert self.codec in IMG_EXTENSIONS

        if os.path.isdir(filename) or os.path.isfile(filename):
            raise ValueError('Directory already exists: {}'.format(filename))
        os.makedirs(filename)
        self.fnum = 0

    def write_frame(self, frame: np.ndarray):
        frame = self.process_frame(frame)

        filename = os.path.join(self.filename, '{:09d}{}'.format(self.fnum, self.codec))
        cv2.imwrite(filename, frame)
        self.fnum += 1


def VideoWriter(filename, movie_format:str=None, *args, **kwargs):
    """Class for writing videos using OpenCV, FFMPEG libx264, or HDF5 arrays of JPG bytestrings.

    OpenCV: can use encode using MJPG, XVID / DIVX, uncompressed bitmaps, or FFV1 (lossless) encoding
    FFMPEG: can use many codecs, but here only libx264, a common encoder with very high compression rates
    HDF5: Encodes each image as a jpg, and stores as an array of these png encoded bytestrings
        Lossless encoding for larger file sizes, but dramatically faster RANDOM reads!
        Good for if you need often to grab a random frame from anywhere within a video, but slightly slower for
        reading sequential frames.
    directory: encodes each image as a .jpg, .png, .tiff, .bmp, etc. Saves with filename starting at 000000000.jpg

    Useful features:
        - allows for use of a context manager, so you'll never forget to close the writer object
        - Don't need to specify frame size before starting writing
        - Handles OpenCV's bizarre desire to save videos in the BGR colorspace

    Example:
        with VideoWriter('../movie.avi', movie_format = 'opencv') as writer:
            for frame in frames:
                writer.write(frame)
    """
    if movie_format is None:
        base, ext = os.path.splitext(filename)
        ext = ext.lower()
        if ext == '':
            movie_format = 'directory'
        elif ext in ['.avi', '.mp4', '.mov']:
            # default to ffmpeg
            movie_format = 'ffmpeg'
        elif ext in ['.h5', '.hdf5']:
            movie_format = 'hdf5'

    if movie_format == 'opencv':
        return OpenCVWriter(filename, *args, **kwargs)
    elif movie_format == 'directory':
        return DirectoryWriter(filename, *args, **kwargs)
    elif movie_format == 'ffmpeg':
        return FFMPEGWriter(filename, *args, **kwargs)
    elif movie_format == 'hdf5':
        return HDF5Writer(filename, *args, **kwargs)
    else:
        raise ValueError('Unknown movie format: {}'.format(movie_format))

