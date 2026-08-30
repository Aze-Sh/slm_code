from rb3_comtools.com_log import *
import threading
from queue import Queue
import numpy as np
import time     

class ServerException(Exception):
    def __init__(self, err):
        super().__init__("Server Error:" + err)

class UndefineException(ServerException):
    def __init__(self, err):
        super().__init__(err + "not defined")
class LockException(ServerException):
    def __init__(self, err):
        super().__init__("Failed to acquire lock while acquire" + err)

class CommonServer:
    def __init__(self,
            monitor_wait_time,
        ):
        self.wait_time = monitor_wait_time

        self._logger = common_logger

        # threading
        self._pipeline = Queue()
        self._thread = None
        self._monitor_stop = threading.Event()
        self._monitor_working = threading.Event()
        self._pipeline_lock = threading.Lock()
        self._monitor_stop.set()
        self._monitor_working.clear()

    # User defined
    def user_defined(func):
        def wrapper(self, *args, **kwargs):
            func(self, *args, **kwargs)
            raise UndefineException(func.__name__)
        return wrapper
    @user_defined
    def _monitor_init(self, *args, **kwargs):
        """setting acquicition mode
        """
    @user_defined
    def _acquire(self):
        """try to acquire data, return None if fail
        """
    @user_defined
    def _monitor_end(self):
        """check whether the device is stop properly
        """
    @user_defined
    def _device_exit(self):
        """correctly close device
        """
    
    # enter & exit
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._device_exit()
        self._logger.info("server exit properly")

    # monitor
    def _monitor(self, *args, **kwargs):
        self._monitor_init(*args, **kwargs)
        self._monitor_working.set()
        while not self._monitor_stop.wait(timeout=self.wait_time):
            res = self._acquire()
            if res is not None:
                if not self._pipeline_lock.acquire(timeout=0.1):
                    raise LockException("pipeline in monitor")
                self._pipeline.put_nowait(res)
                self._pipeline_lock.release()
                print("pipeline input:",self._pipeline.qsize())
                self._logger.info("acquire success")
        self._monitor_end()
        self._monitor_working.clear()
        self._logger.info("monitor stoped")
        # self.__exit__(exc_type=None,exc_value=None,traceback=None)

    def start_monitor(self, *args, **kwargs):
        # check states
        ## program states
        if self._monitor_working.is_set(): # monitor is still working
            warnings.warn(
                "monitor is still working, stopped in order to start new monitor.\n"
            )
            self.stop_monitor()
            assert not self._data_watcher_working.is_set(), "failed to stop data watcher"
        if not self._pipeline.empty(): # data existing in pipeline
            warnings.warn("pipeline is not empty, should check")
            while not self._pipeline.empty():
                self._pipeline.get_nowait()
        ## TODO: check device states

        # init (pipeline), monitor
        self._monitor_stop.clear()
        # start data watcher
        self._logger.info("monitor started")
        self._thread = threading.Thread(target=self._monitor, daemon=True, args=args, kwargs=kwargs)
        self._thread.start()
        while not self._monitor_working.is_set():
            time.sleep(0.1)

    def stop_monitor(self):
        if not self._monitor_stop.is_set():
            self._monitor_stop.set()
            self._thread.join()
            self._monitor_working.clear()
            self._logger.info("monitor stopped")
        else:
            warnings.warn("monitor already stopped")

    # get results
    def _lock(func):
        def wrapper(self, *args, **kw):
            if not self._pipeline_lock.acquire(timeout=0.1):
                raise LockException("read pipeline in " + func.__name__)
            res = func(self, *args, **kw)
            self._pipeline_lock.release()
            return res
        return wrapper

    @_lock
    def get_single(self):
        return None if self._pipeline.empty() else self._pipeline.get_nowait()
    @_lock
    def has_n(self, n):
        return bool(self._pipeline.qsize() >= n)
    @_lock
    def get_num(self):
        return self._pipeline.qsize()
    
    def get_n(self, n):
        return None if not self.has_n(n) else [
            self.get_single() for i in range(n)
        ]

        