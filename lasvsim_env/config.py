from typing import Dict, Union, Tuple
import os

token_path = os.path.dirname(__file__) + "/lasvsim.token"
print("token_path: ", token_path)
if os.path.exists(token_path):
    with open(token_path, 'r') as file:
        token_str = file.read()
else:
    raise FileNotFoundError("Cannot find token file.")

lasvsim_config = {
    'token': token_str,
    'task_id': 147,
    "server_host" : 'http://172.17.0.191:8290',
}

dt = 0.1

env_config_multilane = {
    "dt": dt,
    "max_steps": 200,
    "eval_max_steps": 400,
    "action_lower_bound": (-2.5 * dt, -0.065 * dt),
    "action_upper_bound": (2.5 * dt, 0.065 * dt),
    "real_action_lower_bound": (-1.5, -0.065),
    "real_action_upper_bound": (0.8, 0.065),
    "max_speed": 12.0,
    "reset_traffic_flow": True,
    
    # obs config
    "obs_dict": {
        "ego": 5,
        "sur_dim": 11,
        "sur_num": 10,
        "sur_history_length": 10,
        "map_vec_dim": 16,
        "map_vec_num": 200,
        "navi": 6,
    },

    #  reward config
    "nonimal_acc": True,
    "penalize_collision": True,
    "punish_sur_mode": "max",
    "enable_slow_reward": True,
    "R_step": 12.0,
    "P_lat": 7.5/3, # lateral penalty
    "P_phi": 2.0/3, # phi penalty
    "P_yaw": 1.5/3, # yaw penalty
    "P_front": 5.0,
    "P_side": 0.0,
    "P_space": 5.0,
    "P_rear": 0.0,
    "P_steer": 0.15,
    "P_acc": 0.2,
    "P_delta_steer": 0.0025,
    "P_jerk": 0.003,
    "P_out_of_map": 200.0,
    "P_collision": 200.0,
    "P_boundary": 0,
    "P_overspeed": 5.0,
    "P_traffic_light_violation": 200.0,
    "safety_lat_margin_front": 0.3,
    "safety_long_margin_front": 0.0,
    "safety_long_margin_side": 2.0,
    "front_dist_thd": 50.0,
    "space_dist_thd": 12.0,
    "rel_v_thd": 1.0,
    "rel_v_rear_thd": 3.0,
    "time_dist": 0.5,
    # ref planning

    # Planning config
    "qx_act_repeat_num": 1,
    "qx_act_seq_len": 20,

    "ref_horizon": 20, #  ref_horizon should be larger than qx_act_repeat_num * qx_act_seq_len
    "ahead_lane_length_min": 6.0,
    "ahead_lane_length_max": 60.0,
    "v_discount_in_junction_straight": 0.75,
    "v_discount_in_junction_left_turn": 0.3,
    "v_discount_in_junction_right_turn": 0.3,
    "dec_before_junction_green": 0.8,
    "dec_before_junction_red": 1.3,
}


def get_env_config() -> Dict:
    return env_config_multilane