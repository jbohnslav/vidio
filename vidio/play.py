import os
import time
from typing import Union

import cv2

from .read import VideoReader

class VideoPlayer:
    def __init__(self,
                 filename: Union[str, os.PathLike] = None,
                 frames: list = None,
                 fps: float = 30.0,
                 titles: list = None,
                 write_fnum: bool = True):

        self.filename = filename
        self.frames = frames
        self.fps = fps
        self.waittime = int(1000 / fps)
        if self.filename is not None:
            assert os.path.isfile(filename)
            with VideoReader(self.filename) as reader:
                self.nframes = len(reader)
        else:
            assert self.frames is not None
            self.nframes = len(frames)

        if write_fnum:
            n_digits = len(str(self.nframes))
            fmt = ':0{:d}d'.format(n_digits)
            fmt = '{' + fmt + '}'
            fnum_strings = [fmt.format(i) for i in range(self.nframes)]

        if titles is not None:
            assert (len(titles) == self.nframes)
            if write_fnum:
                self.titles = [fnum_strings[i] + ': ' + '{}'.format(titles[i]) for i in range(self.nframes)]
            else:
                self.titles = ['{}'.format(i) for i in titles]
        else:
            self.titles = None

        # self.repeat = repeat

    def play_from_sequence(self, sequence):
        cv2.namedWindow('VideoPlayer', cv2.WINDOW_AUTOSIZE)
        for i, im in enumerate(sequence):
            t0 = time.perf_counter()
            im = cv2.cvtColor(im.copy(), cv2.COLOR_RGB2BGR)
            if self.titles is not None:
                # x,y = 10, im.shape[1]-10

                x, y = 10, im.shape[0] - 10
                im = cv2.putText(im, self.titles[i], (x, y), cv2.FONT_HERSHEY_COMPLEX,
                                 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('VideoPlayer', im)

            elapsed = int((time.perf_counter() - t0) * 1000)
            key = cv2.waitKey(max([self.waittime, elapsed]))
            if key == 27:
                print('User stopped')
                cv2.destroyAllWindows()
                raise KeyboardInterrupt
        cv2.destroyAllWindows()

    def play(self):
        if self.filename is not None:
            with VideoReader(self.filename) as reader:
                self.play_from_sequence(reader)
        elif self.frames is not None:
            self.play_from_sequence(self.frames)

    def repeat(self):
        should_continue = True
        while should_continue:
            # self.play()
            try:
                self.play()
            except KeyboardInterrupt:
                should_continue = False
            finally:
                cv2.destroyAllWindows()