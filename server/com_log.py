
import logging
import warnings
import os
# Setup logging and data saving
_currentPath = os.path.dirname(os.path.abspath(__file__))
_logger_path = _currentPath+'/logs'
_data_path = _currentPath+'/data'
if not os.path.exists(_logger_path):
    os.makedirs(_logger_path)
if not os.path.exists(_data_path):
    os.makedirs(_data_path)
logging.basicConfig(
        level=logging.INFO, filename=_logger_path+"/serverlog",
        format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
)
## set up logging to _console
_console = logging.StreamHandler()
_console.setLevel(logging.WARNING)
## set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
_console.setFormatter(formatter)
## add the handler to the root logger
logging.getLogger('').addHandler(_console)
common_logger = logging.getLogger(__name__)

