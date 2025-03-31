from typing import Dict

from lasvsim_env.config_base import env_config_default, lasvsim_config
from lasvsim_env.env.env_base import LasvsimEnv
from lasvsim_env.env.env_simple_reward import env_config_simple_reward, SimpleRewardLasvsimEnv
from lasvsim_env.env.env_mixed_reward import env_config_mix_cross_multilane_reward, ComplexMixRewardLasvsimEnv

def get_lasvsim_env(lasvsim_config, env_config, impl_id="default") -> LasvsimEnv:
    """
    Get the LasVSim environment based on the implementation ID.
    
    Args:
        impl_id (str): The ID of the environment implementation. 
                       Default is "default".
    
    Returns:
        LasvsimEnv: The LasVSim environment instance.
    """
    if impl_id == "default":
        return LasvsimEnv(**lasvsim_config, env_config=env_config)
    elif impl_id == "simple":
        return SimpleRewardLasvsimEnv(**lasvsim_config, env_config=env_config)
    elif impl_id == "mix":
        return ComplexMixRewardLasvsimEnv(**lasvsim_config, env_config=env_config)
    else:
        raise ValueError(f"Unknown implementation ID: {impl_id}")

def get_lasvsim_config() -> Dict:
    """
    Get the LasVSim configuration.
    
    Returns:
        Dict: The LasVSim configuration dictionary.
    """
    return lasvsim_config


def get_env_config(impl_id="default") -> Dict:
    """
    Get the environment configuration based on the implementation ID.
    
    Args:
        implementation_id (str): The ID of the environment implementation. 
                                 Default is "default".
    
    Returns:
        Dict: The environment configuration dictionary.
    """
    if impl_id == "default":
        return env_config_default
    elif impl_id == "simple":
        return env_config_simple_reward
    elif impl_id == "mix":
        return env_config_mix_cross_multilane_reward
    else:
        raise ValueError(f"Unknown implementation ID: {impl_id}")