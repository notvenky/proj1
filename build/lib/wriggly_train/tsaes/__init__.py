

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

from dm_control import suite, composer
from dm_control.utils import containers
from wriggly_train.envs.wriggly.robot import wriggly_from_swimmer

class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__dict__ = self



_DEFAULT_TIME_LIMIT = 20  # (Seconds
_CONTROL_TIMESTEP = .02  # (Seconds)
TARGET_SPEED_MOVE = 10.0



SUITE = containers.TaggedTasks()
suite._DOMAINS["wriggly"] = AttrDict(SUITE=SUITE)

from . import noise
from . import shaping
from . import lookahead
from . import optimizers
from . import algo
from .algo import *