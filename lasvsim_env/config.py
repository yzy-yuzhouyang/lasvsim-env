from dataclasses import dataclass, field
from pathlib import Path
import textwrap
from typing import Any, Dict, Optional, Tuple, Union, List

from omegaconf import OmegaConf

FAKE_PATH = Path('/')
Color = Any  # Tuple[int, int, int, int]
PolyOption = Tuple[Color, int, float, bool]  # color, layer, width, filled
ObsNumSur = Any  # Union[int, Dict[str, int]]

@dataclass(frozen=False)
class Config:
    """Configuration for the Lasvsim environment."""

    seed: Optional[int] = None

    # ===== Env =====
    dt: float = 0.1  # Do not change this value.
    max_steps: int = 1000
    penalize_collision: bool = True
    nonimal_acc: bool = False

    # ===== Obs config =====
    obs_dict: Dict[str, int] = field(default_factory=lambda: {
        "ego": 5,
        "single_partner": 11,
        "single_road_point": 16,
        "history_length": 10,
        "partner":1100,
        "max_num_agents_observed": 10,
        "max_num_map_observed": 200,
        "navi": 0,
    })

    # model free reward config
    punish_sur_mode: str = "sum"
    enable_slow_reward: bool = False
    R_step: float = 5.0
    P_lat: float = 5.0
    P_long: float = 2.5
    P_phi: float = 20.0
    P_yaw: float = 10.0
    P_vel: float = 3.0
    P_front: float = 5.0
    P_side: float = 5.0
    P_space: float = 5.0
    P_rear: float = 5.0
    P_steer: float = 50.0
    P_acc: float = 0.2
    P_delta_steer: float = 50.0
    P_jerk: float = 0.1
    P_boundary: float = 0.0
    P_done: float = 2000.0

    safety_lat_margin_front: float = 0.0
    safety_long_margin_front: float = 0.0
    safety_long_margin_side: float = 0.0
    front_dist_thd: float = 50.0
    space_dist_thd: float = 8.0
    rel_v_thd: float = 1.0
    rel_v_rear_thd: float = 0.0
    time_dist: float = 2.0

    # ===== Ego veh dynamics =====
    action_lower_bound: Tuple[float, float] = (-2.0, -0.35)
    action_upper_bound: Tuple[float, float] = ( 1.0,  0.35)
    real_action_lower_bound: Tuple[float, float] = (-3.0, -0.065)
    real_action_upper_bound: Tuple[float, float] = ( 0.8,  0.065)

    # ===== Planning =====
    act_repeat_num: int = 1
    act_seq_len: int = 20
    ref_horizon: int = 20

    # ===== Model config =====
    max_speed: float = 8.0 # default for urban
    v_discount_in_junction_straight: float = 0.75
    v_discount_in_junction_left_turn: float = 0.5
    v_discount_in_junction_right_turn: float = 0.5
    ahead_lane_length_min: float = 6.0
    ahead_lane_length_max: float = 60.0
    dec_before_junction_green: float = 0.8
    dec_before_junction_red: float = 1.3
    min_dist_sur_length: float = 1.8
    min_dist_sur_width: float = 1.8

    
    @staticmethod
    def from_partial_dict(partial_dict) -> "Config":
        base = OmegaConf.structured(Config)
        merged = OmegaConf.merge(base, partial_dict)
        return OmegaConf.to_object(merged)  # type: ignore
