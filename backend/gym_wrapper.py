import dm_env
from dm_env import specs
# NOTE: using gymnasium, not gym 
import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import torch 

def space2spec(space: gym.Space, name: str = None):
  """Converts an OpenAI Gym space to a dm_env spec or nested structure of specs.

  Box, MultiBinary and MultiDiscrete Gym spaces are converted to BoundedArray
  specs. Discrete OpenAI spaces are converted to DiscreteArray specs. Tuple and
  Dict spaces are recursively converted to tuples and dictionaries of specs.

  Args:
    space: The Gym space to convert.
    name: Optional name to apply to all return spec(s).

  Returns:
    A dm_env spec or nested structure of specs, corresponding to the input
    space.
  """
  print(name, space.dtype)
  if isinstance(space, spaces.Discrete):
    return specs.DiscreteArray(num_values=space.n, dtype=space.dtype, name=name)

  elif isinstance(space, spaces.Box):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                              minimum=space.low, maximum=space.high, name=name)

  elif isinstance(space, spaces.MultiBinary):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype, minimum=0.0,
                              maximum=1.0, name=name)

  elif isinstance(space, spaces.MultiDiscrete):
    return specs.BoundedArray(shape=space.shape, dtype=space.dtype,
                              minimum=np.zeros(space.shape),
                              maximum=space.nvec, name=name)

  elif isinstance(space, spaces.Tuple):
    return tuple(space2spec(s, name) for s in space.spaces)

  elif isinstance(space, spaces.Dict):
    return {key: space2spec(value, name) for key, value in space.spaces.items()}

  else:
    raise ValueError('Unexpected gym space: {}'.format(space))


class DMEnvFromGym(dm_env.Environment):
  """A wrapper to convert an OpenAI Gym environment to a dm_env.Environment."""

  def __init__(self, gym_env: gym.Env):
    self.gym_env = gym_env
    # Convert gym action and observation spaces to dm_env specs.
    self._observation_spec = space2spec(self.gym_env.observation_space,
                                        name='observations')
    self._action_spec = space2spec(self.gym_env.action_space, name='actions')
    self._reset_next_step = True

  def reset(self) -> dm_env.TimeStep:
    self._reset_next_step = False
    observation, info = self.gym_env.reset()
    time_step = dm_env.restart(observation)
    return time_step

  def step(self, action: int) -> dm_env.TimeStep:
    if self._reset_next_step:
      return self.reset()

    # Convert the gym step result to a dm_env TimeStep.
    # NOTE: truncated is added (meedeum) 
    observation, reward, done, truncated, info = self.gym_env.step(action)
    self._reset_next_step = np.logical_or(done, truncated) 
    # self._reset_next_step = done

    if truncated: 
      return dm_env.truncation(reward, observation)
    elif done:
      return dm_env.termination(reward, observation)
    else: 
      return dm_env.transition(reward, observation)

    # if done:
    #   is_truncated = info.get('TimeLimit.truncated', False)
    #   if is_truncated:
    #     return dm_env.truncation(reward, observation)
    #   else:
    #     return dm_env.termination(reward, observation)
    # else:
    #   return dm_env.transition(reward, observation)

  def close(self):
    self.gym_env.close()

  def observation_spec(self):
    return self._observation_spec

  def action_spec(self):
    return self._action_spec
