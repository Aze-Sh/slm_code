from abc import ABC, abstractmethod
from time import sleep
from enum import Enum
from threading import Thread
import numpy as np


class CameraState(Enum):
    Idle: int = 0
    WaitForTrigger: int = 1


class ShutterMode(Enum):
    RollingShutter: int = 0
    GlobalShutter: int = 1
    GlobalStart: int = 2


class AbstractCamera(ABC):
    def __init__(self) -> None:
        self._trigger_counts: int = 0
        self._is_start_wait_for_trigger: bool = False
        self._images: list = list()
        self._camera_state: CameraState = CameraState.Idle
        self._number_of_left_images: int = 0

    def ping(self):
        print("OK")
        return "OK"

    @abstractmethod
    def _prepare_for_imaging(self, **args) -> None:
        """
        Something needs to do before a bunch of triggered imaging.
        """
        pass

    @abstractmethod
    def _get_image(self) -> tuple:
        """
        Get single image from camera.
        """
        pass

    @abstractmethod
    def set_shutter_mode(self, mode: int) -> None:
        """
        mode:
            0 for rolling shutter mode,
            1 for global shutter mode,
            2 for global start mode.
        """
        pass

    @abstractmethod
    def set_frame_rate(self, frame_rate: float) -> None:
        """
        Set frame rate.
        """
        pass

    @abstractmethod
    def set_exposure_time(self, exposure_time: float) -> None:
        """
        Set exposure time.
        """
        pass

    def wait_trigger(self, counts: int, force: bool = True):
        """
        Tell camera to wait for some triggers.
        If force = True, empty all images taken before.
        """
        if counts > 0 and type(counts) == int:
            self._trigger_counts = counts
            if force:
                self._images = list()
            else:
                if len(self._images):
                    raise RuntimeError(
                        'There are still images unread. Read before next imaging, or set force to True.')
            self._is_start_wait_for_trigger = True
        else:
            raise ValueError('Parameter "counts" should be positive integer.')

    def read_images(self, counts: int = 0) -> list:
        """
        Read images from instance images buffer.

        If counts = 0, read all images.
        If counts = other positive integer, read images of counts.
        """
        if counts == 0:
            images = self._images
            self._images = list()
            return images
        elif counts > 0 and type(counts) == int:
            return [self._images.pop(0) for _ in range(counts)]
        else:
            raise ValueError(
                'Parameter "counts" should be 0 or positive integer.')

    def get_number_of_unread_images(self) -> int:
        return len(self._images)

    def get_number_of_untriggered_images(self) -> int:
        """
        Get number of untriggered images.
        """
        if self._camera_state == CameraState.Idle:
            return 0
        elif self._camera_state == CameraState.WaitForTrigger:
            return self._number_of_left_images

    def get_camera_state(self) -> str:
        """
        Get camera state.
        """
        if self._camera_state == CameraState.Idle:
            return "Idle"
        elif self._camera_state == CameraState.WaitForTrigger:
            return "Wait for Trigger"

    def _absorption_imaging_main_logic(self) -> None:
        """
        In this process,\n 
            1. Call wait_trigger() to tell the camera
            the number of triggers in the following experiment sequence.
            2. Trigger camera properly (e.g. a pulse of 2 us).
            3. Call read_images() to read images.
        At any time of the process. get_number_of_left_images() and 
        get_camera_state() could be called for more detailed information.
        """
        while True:
            if self._is_start_wait_for_trigger:
                self._is_start_wait_for_trigger = False
                self._camera_state = CameraState.WaitForTrigger
                self._number_of_left_images = self._trigger_counts
                for i in range(self._trigger_counts):
                    self._prepare_for_imaging()
                    success, image = self._get_image()
                    if not success:
                        print('Fail to get image.')
                    self._images.append(image)
                    self._number_of_left_images -= 1
                self._trigger_counts = 0
                self._number_of_left_images = 0
                self._camera_state = CameraState.Idle
            else:
                sleep(0.5)

    def run(self) -> None:
        Thread(target=self._absorption_imaging_main_logic).start()
