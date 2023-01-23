import importlib.metadata

from . import com
from . import kin
from . import move
from . import physio
from . import utils
from . import plots
from . import imu
from . import ana

__version__ = importlib.metadata.version(__package__)
del importlib  # clean up the namespace

__all__ = ["com", "kin", "move", "physio", "utils", "plots", "imu", "ana"]
