import typing as T
from dm_control import suite, composer
from dm_control.utils import containers
from wriggly_train.envs.wriggly import robots, tasks



# How long the simulation will run [s].
_DEFAULT_TIME_LIMIT = 20
# Target fixed walking speed [m/s].
TARGET_SPEED_MOVE = 0.1
# Target fixed running speed [m/s].
TARGET_SPEED_RUN = 1.5
# Bounds on sampled target speed [m/s].
TARGET_SPEED_MIN = 0.0
TARGET_SPEED_MAX = 1.5
# How long before sampling new target speed [s].
TARGET_DURATION = 5

# Hack `dm_control.suite` because it lacks good registration functionality.
class AttrDict(dict):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__dict__ = self
SUITE = containers.TaggedTasks()
suite._DOMAINS["wriggly"] = AttrDict(SUITE=SUITE)


# @SUITE.add()
# def stand(
#   time_limit=_DEFAULT_TIME_LIMIT,
#   random=None,
#   environment_kwargs={},
#   **task_kwargs,
# ):
#   robot = robots.Wriggly()
#   task = tasks.Stand(robot, **task_kwargs)
#   return composer.Environment(
#     task,
#     time_limit=time_limit,
#     random_state=random,
#     **environment_kwargs,
#   )


@SUITE.add()
def walk(
  time_limit=_DEFAULT_TIME_LIMIT,
  random=None,
  robot_kwargs={},
  environment_kwargs={},
  flatten=True,
  **task_kwargs,
):
  robot = robots.Wriggly(**robot_kwargs)
  task_kwargs.setdefault("target_vx", TARGET_SPEED_MOVE)
  task = tasks.MoveAtFixedVelocity_WalkInThePark(robot, **task_kwargs)
  env = composer.Environment(
    task,
    time_limit=time_limit,
    random_state=random,
    **environment_kwargs,
  )
  env._step_limit = time_limit // task.control_timestep
  return env
@SUITE.add()
def run(
  time_limit=_DEFAULT_TIME_LIMIT,
  random=None,
  robot_kwargs={},
  environment_kwargs={},
  flatten=True,
  **task_kwargs,
):
  robot = robots.Wriggly(**robot_kwargs)
  task_kwargs.setdefault("target_vx", TARGET_SPEED_RUN)
  task = tasks.MoveAtFixedVelocity_WalkInThePark(robot, **task_kwargs)
  env = composer.Environment(
    task,
    time_limit=time_limit,
    random_state=random,
    **environment_kwargs,
  )
  env._step_limit = time_limit // task.control_timestep
  return env
# @SUITE.add()
# def change_speed(
#   time_limit=_DEFAULT_TIME_LIMIT,
#   random=None,
#   environment_kwargs={},
#   **task_kwargs,
# ):
#   robot = robots.Wriggly()
#   task_kwargs.setdefault(‘target_speed_min’, TARGET_SPEED_MIN)
#   task_kwargs.setdefault(‘target_speed_max’, TARGET_SPEED_MAX)
#   task_kwargs.setdefault(‘target_duration’, TARGET_DURATION)
#   task = tasks.MoveAtChangingVelocity(robot, **task_kwargs)
#   return composer.Environment(
#     task,
#     time_limit=time_limit,
#     random_state=random,
#     **environment_kwargs,
#   )
# @SUITE.add()
# def move_to_target(
#   time_limit=_DEFAULT_TIME_LIMIT,
#   random=None,
#   environment_kwargs={},
#   **task_kwargs,
# ):
#   robot = robots.Wriggly()
#   task_kwargs.setdefault(‘reward’, ‘dense’)
#   task = tasks.MoveToTarget(robot, **task_kwargs)
#   return composer.Environment(
#     task,
#     time_limit=time_limit,
#     random_state=random,
#     **environment_kwargs,
#   )

@SUITE.add('benchmarking') # MAX_DISP
def move(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  xml_relative_path = "../../wriggly_mujoco/wriggly_max_disp.xml"
  xml_path = os.path.join(current_dir, xml_relative_path)
  physics = Physics.from_xml_path(xml_path)
  task = Wriggly()
  env = control.Environment(physics, task, time_limit=time_limit,
                            legacy_step=False, n_sub_steps= 2)
  return env

@SUITE.add('benchmarking') # MAX_DISP
def move_no_time(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  xml_relative_path = "../../wriggly_mujoco/wriggly_max_disp.xml"
  xml_path = os.path.join(current_dir, xml_relative_path)
  physics = Physics.from_xml_path(xml_path)
  task = Wriggly(include_time=False)
  env = control.Environment(physics, task, time_limit=time_limit,
                            legacy_step=False, n_sub_steps= 2)
  return env

# APPROACH TARGET
@SUITE.add('benchmarking')  
def approach_target(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  xml_relative_path = "../../wriggly_mujoco/wriggly_apr_target.xml"
  xml_path = os.path.join(current_dir, xml_relative_path)
  physics = Physics.from_xml_path(xml_path)
  task = WrigglyApproachTarget()
  env = control.Environment(physics, task, time_limit=time_limit,
                            legacy_step=False, n_sub_steps= 2)
  
  return env

# CLIMB BENCH
@SUITE.add('benchmarking')
def climb_obstacle(time_limit=_DEFAULT_TIME_LIMIT, random=None,
             environment_kwargs=None):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  xml_relative_path = "../../wriggly_mujoco/wriggly_climb.xml"
  xml_path = os.path.join(current_dir, xml_relative_path)
  physics = Physics.from_xml_path(xml_path)
  task = WrigglyClimbObstacle()
  env = control.Environment(physics, task, time_limit=time_limit,
                            legacy_step=False, n_sub_steps= 2)
  
  return env

# AVOID OBSTACLE
@SUITE.add('benchmarking')
def maxvel(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  xml_relative_path = "../../wriggly_mujoco/wriggly_avoid.xml"
  xml_path = os.path.join(current_dir, xml_relative_path)
  physics = Physics.from_xml_path(xml_path)
  task = WrigglyMaxVel()
  env = control.Environment(physics, task, time_limit=time_limit,
                            legacy_step=False, n_sub_steps= 2)
  
  return env

# ORIENT
@SUITE.add('benchmarking')
def orient(time_limit=_DEFAULT_TIME_LIMIT, random=None,
              environment_kwargs=None):
  current_dir = os.path.dirname(os.path.abspath(__file__))
  xml_relative_path = "../../wriggly_mujoco/wriggly_orient.xml"
  xml_path = os.path.join(current_dir, xml_relative_path)
  physics = Physics.from_xml_path(xml_path)
  task = WrigglyOrient()
  env = control.Environment(physics, task, time_limit=time_limit,
                            legacy_step=False, n_sub_steps= 2)
  
  return env