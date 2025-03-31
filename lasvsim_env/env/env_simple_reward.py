import numpy as np

from lasvsim_env.env.env_base import LasvsimEnv
from lasvsim_env.config_base import env_config_default

env_config_simple_reward = {
    **env_config_default,
    "P_lat": 1.0,
}

class SimpleRewardLasvsimEnv(LasvsimEnv):
    """
    A simple version of the LasvsimEnv that implements a basic reward structure.
    """
    def __init__(self, **kwargs):
        """
        Initialize the environment with basic parameters.
        """
        print("==================================================")
        print("Using SimpleRewardLasvsimEnv")
        print("==================================================")
        super(SimpleRewardLasvsimEnv, self).__init__(**kwargs)

    def step(self, delta_action: np.ndarray):
        # action: network output, \in [-1, 1]
        self.alive_step += 1

        last_action = self.lasvsim_context.ego.action
        real_action = self.get_real_action(delta_action, last_action)

        step_info = self.simulator.idc_step(self.ego_id,real_action[1],real_action[0],ref_limit=40.0)

        self.update_step_info(step_info)
        self.update_lasvsim_context(real_action)

        terminated, truncated, done_info = self.judge_done(step_info["step_res"])
        reward, rew_info = self.reward_function_safety(terminated)

        obs = self.get_obs_from_context()

        # if terminated or truncated:
        #     print(f"alive step: {self.alive_step:4d}, done info: {[event for event in done_info if done_info[event]]}")
        
        return obs, reward, terminated, truncated, {**rew_info, **done_info, "event_alive_step": self.alive_step, "event_qx_error": 0}

    def reward_function_efficiency(self, ref_list): 
        # 连续奖励：现在只有速度奖励
        # 边界距离惩罚（暂未用到）
        
        ego= self.lasvsim_context.ego
        # 车道跟踪误差（阶跃式）
        # ref_point: List[LineString]
        ref_num = len(ref_list)
        ref_x, ref_y, ref_phi = self.get_closest_ref_point(ref_list).T
        ref_v = np.repeat(self.config["max_speed"], ref_num) # (R, )

        lat_error = -(ego.x - ref_x) * np.sin(ref_phi) + (ego.y - ref_y) * np.cos(ref_phi)
        # phi_error = deal_with_phi_rad(ego.phi - ref_phi)

        # lat_c: 车道跟踪误差的惩罚系数
        te = np.abs(lat_error)
        lat_c = np.zeros_like(te)
        lat_c = np.where((0.0 <= te) & (te <= 0.4), 1.0 - 0.5 * te / 0.4, lat_c)
        lat_c = np.where((0.4 < te) & (te <= 0.8), 0.5, lat_c)
        lat_c = np.where((0.8 < te) & (te <= 1.8), 0.1, lat_c)
        lat_c = np.where(te > 1.8, 0.0, lat_c)

        # 速度奖励（在限速处达到最大，在2倍限速和0处达到0）
        a = 1.0
        ms = self.config["max_speed"]
        if 0.0 <= ego.u <= ms:
            reward_v = a * ego.u / ms
        elif ms < ego.u <= 2 * ms:
            reward_v = a * (2 - ego.u / ms)
        else:
            reward_v = 0.0
        
        reward_v = reward_v * lat_c
        # print(f"ego.u: {ego.u}, lat_c: {lat_c}, lat_error: {te}, reward_v: {reward_v}")

        return reward_v, {
            "reward_v": reward_v,
            "ego_vx": ego.u * np.ones_like(reward_v),
        }
    
    def reward_function_safety(self, terminated):
        # 稀疏奖励
        # done掉 -100
        # 存活 0

        reward_done = -100.0 if terminated else 0.0
        reward_alive = 0.0

        total_reward = 0.0 + reward_done + reward_alive

        return total_reward, {
            "sparse_done": reward_done,
            "sparse_alive": reward_alive
        }
    
    def judge_done(self, res) -> bool:
        # terminated
        park_flag = (self.lasvsim_context.ego.u == 0)
        collision = self.collision_info
        out_of_defined_region = False
        self.out_of_driving_area = self.check_out_of_driving_area()
        out_of_driving_area = self.out_of_driving_area
        self.traffic_light_violation = self.check_traffic_light_violation()
        traffic_light_violation = self.traffic_light_violation
        navigation_violation = self.navigation_violation

        # truncated
        max_step_truncated = (self.alive_step >= self.max_step)
        success = (res["code"] == 1001) and (collision == 0)
        if res["code"] == 1001 and collision == 1:
            raise ValueError("Success and collision at the same time")

        done_info = {
            "category": 0,
            "event_pause": park_flag,
            "event_collision": collision,
            "event_regionout": out_of_defined_region,
            "event_mapout": out_of_driving_area,
            "event_max_step_truncated": max_step_truncated,
            "event_traffic_light_violation": traffic_light_violation,
            "event_navigation_violation": navigation_violation,
            "event_success": success,
        }

        terminated = (
            collision or 
            out_of_defined_region or 
            out_of_driving_area or 
            traffic_light_violation or 
            navigation_violation or
            False
        )
        truncated = max_step_truncated or success # the success sample will be removed from the replay buffer due to plan setting
        
        # if terminated or truncated:
        #     print(f"# DONE: {done_info}")

        return terminated, truncated, done_info
    
    @property
    def sum_keys(self):
        return {
            "sparse_done",
            "sparse_alive",
            "reward_v",
        }
    
    @property
    def avg_keys(self):
        return {
            "ego_vx",
        }
    
    @property
    def max_keys(self):
        return {
            "event_alive_step",
            "event_collision",
            "event_mapout",
            "event_regionout",
            "event_max_step_truncated",
            "event_success",
            "event_qx_error",
            "event_traffic_light_violation",
            "event_navigation_violation",
        }