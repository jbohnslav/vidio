import os
from typing import Union

import cv2
import h5py
import numpy as np


class BaseReader:
    def __init__(self, filename: Union[str, bytes, os.PathLike], nframes) -> None:
        """Initializes a VideoReader object.

        Args:
            filename: name of movie to be read
        Returns:
            VideoReader object
        """
        self.fnum = 0
        self.nframes = nframes

    # Python 2 compatibility:
    def next(self):
        return self.__next__()

    def __next__(self):
        # for use as an iterator
        if self.fnum == self.nframes:
            self.close()
            raise StopIteration

    def read(self, framenum: Union[int, slice]) -> Union[np.ndarray, list]:
        """Read the frame indicated in framenum from disk

        Uses sequential reads where possible if using OpenCV to read
        """
        if type(framenum) == slice:
            return [self.read(i) for i in self.slice_to_list(framenum)]
        if framenum < 0 or framenum > self.nframes:
            raise ValueError('frame number requested outside video bounds: {}'.format(framenum))

    def slice_to_list(self, slice_obj):
        # https://stackoverflow.com/questions/13855288/turn-slice-into-range
        if slice_obj.stop > self.nframes:
            raise ValueError('Slice end {} > nframes {}'.format(slice_obj.stop, self.nframes))
        return list(range(self.nframes)[slice_obj])

    def process_frame(self, frame):
        return frame

    def __len__(self):
        return self.nframes

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __getitem__(self, *args, **kwargs) -> Union[np.ndarray, list]:
        """Wrapper around `read`

        Args:
            framenum (int): frame to be read from disk
        Example:
            frame = reader[10]
        """
        return self.read(*args, **kwargs)

    def close(self):
        """Closes open file objects"""
        pass

    def __del__(self):
        """destructor"""
        self.close()


class OpenCVReader(BaseReader):
    def __init__(self, filename: Union[str, bytes, os.PathLike]):
        self.filename = filename
        self.file_object = cv2.VideoCapture(filename)
        nframes = int(self.file_object.get(cv2.CAP_PROP_FRAME_COUNT))
        super().__init__(filename, nframes)
        self.fps = self.file_object.get(cv2.CAP_PROP_FPS)

    def __next__(self):
        super().__next__()
        ret, frame = self.file_object.read()
        # keep track of current frame to optimize sequential reads
        self.fnum += 1
        return self.process_frame(frame)

    def process_frame(self, frame):
        # opencv reads into BGR colorspace by default
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def read(self, framenum: int) -> np.ndarray:
        """Read the frame indicated in framenum from disk

        Uses sequential reads where possible if using OpenCV to read
        """
        # does checks. if framenum is a slice, calls read recursively. In that case, just return
        output = super().read(framenum)
        if output is not None:
            return output

        if framenum != self.fnum:
            self.file_object.set(int(cv2.CAP_PROP_POS_FRAMES), framenum)
        ret, frame = self.file_object.read()
        if not ret:
            raise ValueError('error decoding frame {} from video {}'.format(framenum, self.filename))
        frame = self.process_frame(frame)
        self.fnum = framenum + 1
        return frame

    def close(self):
        if hasattr(self, 'file_object') and self.file_object is not None:
            self.file_object.release()
            del self.file_object


class FilenameGetter:
    def __init__(self, directory, ending: str = '.png'):
        self.directory = directory
        self.ending = ending

    def __getitem__(self, idx):
        filename = os.path.join(self.directory, '{:09d}'.format(idx) + self.ending)
        return filename


class DirectoryReader(BaseReader):
    def __init__(self, filename: Union[str, bytes, os.PathLike], assume_writer_style=False,
                 filetype='.png') -> None:
        assert os.path.isdir(filename)
        if assume_writer_style:
            self.file_object = FilenameGetter(filename, ending=filetype)
            nframes = 999999999
        else:
            imagefiles = self.find_imagefiles(filename)
            assert len(imagefiles) > 0
            self.file_object = imagefiles
            nframes = len(self.file_object)
        # superclass gets nframes
        super().__init__(filename, nframes)

    def find_imagefiles(self, directory):
        endings = ['.bmp', '.jpg', '.png', '.jpeg', '.tiff', '.tif']
        files = os.listdir(directory)
        files.sort()
        files = [os.path.join(directory, i) for i in files]
        imagefiles = []
        for i in files:
            _, ext = os.path.splitext(i)
            if ext in endings:
                imagefiles.append(i)
        return imagefiles

    def __next__(self):
        super().__next__()
        frame = cv2.imread(self.file_object[self.fnum], 1)
        self.fnum +=1
        return self.process_frame(frame)

    def process_frame(self, frame):
        # opencv reads into BGR colorspace by default
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def read(self, framenum: Union[int, slice]) -> Union[np.ndarray, list]:
        # does checks. if framenum is a slice, calls read recursively. In that case, just return
        output = super().read(framenum)
        if output is not None:
            return output

        frame = cv2.imread(self.file_object[framenum], 1)
        if frame is None:
            raise ValueError('Error reading image file {}'.format(self.file_object[framenum]))
        frame = self.process_frame(frame)
        self.fnum = framenum + 1
        return frame


class HDF5Reader(BaseReader):
    def __init__(self, filename: Union[str, bytes, os.PathLike]) -> None:
        assert os.path.isfile(filename)
        self.filename = filename
        self.file_object = h5py.File(filename, 'r')
        nframes = len(self.file_object['frame'])
        super().__init__(filename, nframes)

    def __next__(self):
        super().__next__()
        # for use as an iterator
        frame = cv2.imdecode(self.file_object['frame'][self.fnum], 1)
        self.fnum += 1
        return self.process_frame(frame)

    def read(self, framenum):
        # does checks. if framenum is a slice, calls read recursively. In that case, just return
        output = super().read(framenum)
        if output is not None:
            return output
        frame = cv2.imdecode(self.file_object['frame'][framenum], 1)
        if frame is None:
            raise ValueError('error decoding frame {} from file {}'.format(framenum, filename))
        self.fnum = framenum + 1
        return self.process_frame(frame)

    def close(self):
        """Closes open file objects"""
        if hasattr(self, 'file_object') and self.file_object is not None:
            try:
                self.file_object.close()
            except TypeError as e:
                print('error in hdf5 destructor')
                # print(e)
                # print(dir(self.file_object))
                # print(self.file_object)
            del self.file_object


def VideoReader(filename, assume_writer_style=False, filetype='.png'):
    """Class for reading videos using OpenCV or JPGs encoded in an HDF5 file.

    Features:
        - can be used as an iterator, or with a decorator
        - Handles OpenCV's bizarre default to read into BGR colorspace
        - Uses sequential reading where possible for speed

    Examples:
        with VideoReader('.../movie.avi') as reader:
            for frame in reader:
                # do something

        with VideoReader('.../movie.h5') as reader:
            # equivalent
            frame = reader[10]
            frame = reader.read(10)

        reader = VideoReader('.../movie.avi')
        for frame in reader:
            # do something
            pass
        reader.close()
    """
    if not os.path.isfile(filename):
        assert os.path.isdir(filename)
        return DirectoryReader(filename, assume_writer_style, filetype)
    else:
        _, ext = os.path.splitext(filename)
        ext = ext[1:].lower()

        if ext == 'h5' or ext == 'hdf5':
            return HDF5Reader(filename)
        elif ext == 'avi' or ext == 'mp4' or ext == 'mov':
            return OpenCVReader(filename)
        else:
            raise ValueError('unknown file extension: {}'.format(ext))

