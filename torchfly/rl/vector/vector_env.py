from ..env import Env


class VectorEnv(Env):
    r"""Base class for vectorized environments.
    Each observation returned from vectorized environment is a batch of observations 
    for each sub-environment. And :meth:`step` is also expected to receive a batch of 
    actions for each sub-environment.
    
    .. note::
    
        All sub-environments should share the identical observation and action spaces.
        In other words, a vector of multiple different environments is not supported. 
    Parameters
    ----------
    num_envs : int
        Number of environments in the vectorized environment.
    observation_space : `gym.spaces.Space` instance
        Observation space of a single environment.
    action_space : `gym.spaces.Space` instance
        Action space of a single environment.
    """
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

        self.closed = False

    def reset_async(self):
        pass

    def reset_wait(self, **kwargs):
        raise NotImplementedError()

    def reset(self):
        r"""Reset all sub-environments and return a batch of initial observations.
        """
        self.reset_async()
        return self.reset_wait()

    def step_async(self, actions):
        pass

    def step_wait(self, **kwargs):
        raise NotImplementedError()

    def step(self, actions):
        r"""Take an action for each sub-environments. 
        Parameters
        ----------
        actions : iterable of samples from `action_space`
            List of actions.
        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        infos : list of dict
            A list of auxiliary information from sub-environments.
        dones : `np.ndarray` instance (dtype `np.bool_`)
            A vector whose entries indicate whether the environment has ended.
        """

        self.step_async(actions=None)
        return self.step_wait()

    def close_extras(self, **kwargs):
        r"""Clean up the extra resources e.g. beyond what's in this base class. """
        raise NotImplementedError()

    def close(self, **kwargs):
        r"""Close all sub-environments and release resources.
        
        It also closes all the existing image viewers, then calls :meth:`close_extras` and set
        :attr:`closed` as ``True``. 
        
        .. warning::
        
            This function itself does not close the environments, it should be handled
            in :meth:`close_extras`. This is generic for both synchronous and asynchronous 
            vectorized environments. 
        
        .. note::
        
            This will be automatically called when garbage collected or program exited. 
            
        """
        if self.closed:
            return
        self.close_extras(**kwargs)
        self.closed = True

    def seed(self, seeds=None):
        """
        Parameters
        ----------
        seeds : list of int, or int, optional
            Random seed for each individual environment. If `seeds` is a list of
            length `num_envs`, then the items of the list are chosen as random
            seeds. If `seeds` is an int, then each environment uses the random
            seed `seeds + n`, where `n` is the index of the environment (between
            `0` and `num_envs - 1`).
        """
        pass

    def __del__(self):
        if not getattr(self, 'closed', True):
            self.close()

    def __repr__(self):
        if self.spec is None:
            return '{}({})'.format(self.__class__.__name__, self.num_envs)
        else:
            return '{}({}, {})'.format(self.__class__.__name__, self.spec.id, self.num_envs)