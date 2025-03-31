import numpy as np

from lasvsim_env.env.env_base import LasvsimEnv
from lasvsim_env.config_base import env_config_default

env_config_mix_cross_multilane_reward = {
    **env_config_default,
}

class ComplexMixRewardLasvsimEnv(LasvsimEnv):
    """
    A simple version of the LasvsimEnv that implements a basic reward structure.
    """
    def __init__(self, **kwargs):
        """
        Initialize the environment with basic parameters.
        """
        print("==================================================")
        print("Using ComplexMixRewardLasvsimEnv")
        print("==================================================")
        super(ComplexMixRewardLasvsimEnv, self).__init__(**kwargs)