import os
import random
from time import time
from shapely import segmentize
from collections import deque
from typing import Any, Dict, Tuple, List, Deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Point, LineString, Polygon
    
from lasvsim_openapi.client import Client
from lasvsim_openapi.http_client import HttpConfig
from lasvsim_openapi.simulator_model import SimulatorConfig, Point as QxPoint
from lasvsim_env.lasvsim_dataclasses import EgoVehicle, SurroundingVehicle, LasVSimContext
from lasvsim_env.utils.map_tool.lib.map import Map
from lasvsim_env.utils.lib import \
    point_project_to_line, compute_waypoints_by_intervals, compute_waypoint, create_box_polygon
from lasvsim_env.utils.math_utils import \
    deal_with_phi_rad, convert_ref_to_ego_coord, \
    inverse_normalize_action, cal_dist, \
    get_indices_of_k_smallest, convert_ground_coord_to_ego_coord, \
    calculate_perpendicular_points
from lasvsim_env.dataclass import EgoVehicle, SurroundingVehicle, LasVSimContext
from lasvsim_env.traj_processor import compute_intervals, compute_intervals_in_junction, compute_intervals_initsegment_green, compute_intervals_initsegment_red

def add_map_objs(line_string, map_objs, max_speed, obj_type):
    """
    Add segmentized line_string to map_objs.
    Args:
        line_string: shaple.LineString.
        map_objs: list.
        obj_type: one-hot list, e.g. [0, 0, 1, 0, 0] for center lanes.
    """
    default_light_status = [0, 0, 1] # 默认交通灯（无灯）

    xs = np.array(line_string.xy[0]).astype(np.float32)
    ys = np.array(line_string.xy[1]).astype(np.float32)
    xys = np.array([xs, ys]).T
    map_obj_xs = (xs[:-1] + xs[1:]) / 2
    map_obj_ys = (ys[:-1] + ys[1:]) / 2
    lengths = np.linalg.norm(xys[1:] - xys[:-1], axis=1) / 2
    orientations = np.arctan2(ys[1:] - ys[:-1], xs[1:] - xs[:-1])
    # 将每条linestring的所有vector添加到map_objs
    count = 0
    for x, y, l, o in zip(map_obj_xs, map_obj_ys, lengths, orientations):
        map_objs.append([
            x, y, l, 0, 
            np.cos(o), np.sin(o), max_speed,
            *obj_type,
            *default_light_status
        ])
        count += 1
    return count

class LasvsimEnv():
    def __init__(
        self,
        token: str,
        env_config: Dict = {},
        task_id=None,
        record_id=None,
        is_testing: bool = False,
        **kwargs: Any,
    ):
        self.metadata = [('authorization', 'Bearer ' + token)]
        assert task_id is not None, "None task id"

        # ================== 1. Build a connection ==================
        endpoint = kwargs['server_host']
        assert endpoint == "http://localhost:8290", "endpoint should be localhost:8290, please check."
        self.qx_client = Client(HttpConfig(
            endpoint=endpoint,  # 接口地址
            token=token,  # 授权token
        ))
        if not is_testing: # 训练环境
            scene_list = self.qx_client.train_task.get_scene_id_list(task_id)
            self.scenario_list = scene_list.scene_id_list
            self.version_list = scene_list.scene_version_list
            self.scenario_id = self.scenario_list[0]
            self.simulator = self.init_remote_lasvsim(
                scenario_id=scene_list.scene_id_list[0],
                scenario_version=scene_list.scene_version_list[0]
            )
        else: # 测试环境
            print("initializing test environment...")
            new_record = self.qx_client.process_task.copy_record(task_id, record_id)
            self.scenario_list = [new_record.scen_id]
            self.version_list = [new_record.scen_ver]
            self.scenario_id = new_record.scen_id
            self.simulator = self.qx_client.init_simulator_from_config(
                                SimulatorConfig(
                                    scen_id=new_record.scen_id,
                                    scen_ver=new_record.scen_ver,
                                    sim_record_id=new_record.sim_record_id,
                                )
            )
            
      

        # ================== 2. Init simulator ==================
        # init variables
        self.config = env_config
        self.scenario_cnt = 0
        self.alive_step = 0
        self.max_step = env_config["max_steps"]
        self.action_lower_bound = np.array(self.config["action_lower_bound"])
        self.action_upper_bound = np.array(self.config["action_upper_bound"])
        self.action_center = (self.action_upper_bound +
                              self.action_lower_bound) / 2
        self.action_half_range = (
            self.action_upper_bound - self.action_lower_bound) / 2
        self.real_action_upper = np.array(
            self.config["real_action_upper_bound"])
        self.real_action_lower = np.array(
            self.config["real_action_lower_bound"])

        self.max_num_map_observed = self.config['obs_dict']['max_num_map_observed']
        self.surr_veh_num = self.config['obs_dict']['max_num_agents_observed']
        
        # init ego vehicle
        test_vehicle = self.get_remote_lasvsim_test_veh_list()
        self.ego_id = random.choice(test_vehicle.list)
        self.lasvsim_context = LasVSimContext(
            ego=EgoVehicle(),
            ref_list=[],
            sur_list=[]
        )
        self.history_sur_veh: Deque = deque([[] for _ in range(10)], maxlen=10)
        self.can_not_get_lane_id = False
        self.step_remote_lasvsim()
        self.update_lasvsim_context()
        # self._render_init(render_info=render_info)

        # ================== 3. Process static map, surroundings and render ==================
        self.connections = {}
        self.junctions = {}
        self.lanes = {}
        self.segments = {}
        self.links = {}
        self.map_dict = {} # scenaio_id -> Qxmap
        for i in range(len(self.scenario_list)):
            cur_map = self.get_remote_hdmap(self.scenario_list[i], self.version_list[i])
            self.map_dict[self.scenario_list[i]] = cur_map

        self.convert_map(self.scenario_list[0])
        print("len(self.lanes): ", len(self.lanes))
        # 根据self.lanes，将车道中心线转化为self.map_objs
        # 首先将每条车道构造成linestring对象，利用segmentize函数切成小段
        
        ROAD_EDGE   = [1, 0, 0, 0, 0, 0]
        LINE_EDGE   = [0, 1, 0, 0, 0, 0]
        CENTER_LINE = [0, 0, 1, 0, 0, 0]
        STOP_LINE   = [0, 0, 0, 1, 0, 0]
        ZEBRA       = [0, 0, 0, 0, 1, 0]
        VIRTUAL     = [0, 0, 0, 0, 0, 1]

        map_objs = []
        for key in self.map_dict: # 每张地图
            for seg in self.map_dict[key].data.segments: # 每个segment
                for link in seg.ordered_links: # 每个link
                    # 左右道路边界
                    linestring = LineString([(p.x, p.y) for p in link.left_boundary.points])
                    segmentized_linestring = segmentize(linestring, max_segment_length=5.0)
                    count = add_map_objs(segmentized_linestring, map_objs, max_speed=0.0, obj_type=ROAD_EDGE) # 道路边界线
                    print(f"finish adding left boundary of {link.id} with {count} vectors.")
                    linestring = LineString([(p.x, p.y) for p in link.right_boundary.points])
                    segmentized_linestring = segmentize(linestring, max_segment_length=5.0)
                    count = add_map_objs(segmentized_linestring, map_objs, max_speed=0.0, obj_type=ROAD_EDGE) # 道路边界线
                    print(f"finish adding right boundary of {link.id} with {count} vectors.")
                    
                    # 对每个车道
                    for i, lane in enumerate(link.ordered_lanes):
                        print(f"processsing lane {i} "+ lane.id)
                        if lane.type == 0:
                            print(f"lane {lane.id} is type 0. continue.")
                            continue
                        elif lane.type == 1:
                            # 添加车道中心线
                            lane_linestring = LineString([(p.point.x, p.point.y) for p in lane.center_line])
                            segmentized_linestring = segmentize(lane_linestring, max_segment_length=5.0)
                            count = add_map_objs(segmentized_linestring, map_objs, max_speed=16.67, obj_type=CENTER_LINE)
                        elif lane.type == 2:
                            print(f"lane {lane.id} is bicycle lane.")
                            continue
                        elif lane.type == 3:
                            print(f"lane {lane.id} is pedestrian lane.")
                            continue

                        # 添加车道线
                        if i > 0 and link.ordered_lanes[i-1].type == 0: # 如果是第一条机动车道，则加入左侧车道线
                            # linestringlinestring = LineString([(p.x, p.y) for p in lane.left_boundary.points])
                            linestring = LineString([(p.point.x - p.left_width * np.sin(p.heading), \
                                                      p.point.y + p.left_width * np.cos(p.heading)) for p in lane.center_line])
                            segmentized_linestring = segmentize(linestring, max_segment_length=5.0)
                            count += add_map_objs(segmentized_linestring, map_objs, max_speed=0.0, obj_type=LINE_EDGE)
                        
                        # 其他情况，加入右侧车道线
                        # linestring = LineString([(p.x, p.y) for p in lane.right_boundary.points])
                        linestring = LineString([(p.point.x + p.right_width * np.sin(p.heading), \
                                                  p.point.y - p.right_width * np.cos(p.heading)) for p in lane.center_line])
                        segmentized_linestring = segmentize(linestring, max_segment_length=5.0)
                        count += add_map_objs(segmentized_linestring, map_objs, max_speed=0.0, obj_type=LINE_EDGE)

                        # 停止线
                        if lane.stopline:
                            linestring = LineString([(p.x, p.y) for p in lane.stopline.shape.points])
                            segmentized_linestring = segmentize(linestring, max_segment_length=5.0)
                            count += add_map_objs(segmentized_linestring, map_objs, max_speed=0.0, obj_type=STOP_LINE)
                        
                        print(f"finish adding lane {lane.id} with {count} vectors.")
            
            for junc in self.map_dict[key].data.junctions: # 每个junction
                pass #TODO: 添加人行道、连接线


        # 每个小段是一个vector，将这些信息按照观测形式进行保存，得到self.map_objs
        # self.map_objs是长度为(S, 16)的向量，S表示切出来的vector数量（大约几千）
        # 14维观测的设置按照RL Planner文档，这里的x,y,phi取绝对坐标
        self.map_objs = np.array(map_objs)
        print(f"finish initializing self.map_objs with shape: {self.map_objs.shape}.")
        
        self.surrounding_deque=deque([[] for _ in range(10)], maxlen=10)

    def init_remote_lasvsim(self, scenario_id: str, scenario_version: str):
        print(f"[LasvsimEnv] init_remote_lasvim with scenario_id={scenario_id} and version={scenario_version}...")
        return self.qx_client.init_simulator_from_config(SimulatorConfig(
            scen_id=scenario_id,
            scen_ver=scenario_version,
        ))

    def reset_remote_lasvsim(self):
        return self.simulator.reset()

    def step_remote_lasvsim(self):
        return self.simulator.step()

    def stop_remote_lasvsim(self, simulation_id: str = None):
        return self.simulator.stop()

    def update_lasvsim_context(self, real_action: np.ndarray = None):
        time_1 = time()
        ego_context = self.get_ego_context(real_action)
        time_2 = time()
        # print(f"----get_ego_context: {(time_2 - time_1) * 1000} ms.")
        ref_contex = self.get_ref_context()
        time_3 = time()
        # print(f"----get_ref_context: {(time_3 - time_2) * 1000} ms.")
        sur_context = self.get_sur_context()
        time_4 = time()
        # print(f"----get_sur_context: {(time_4 - time_3) * 1000} ms.")
        
        # Update history_sur_veh
        self.history_sur_veh.append(sur_context)
        self.history_sur_veh[-1].sort(key=SurroundingVehicle.distance_key)
        
        self.lasvsim_context = LasVSimContext(
            ego=ego_context,
            ref_list=ref_contex,
            sur_list=sur_context
        )
        time_5 = time()
        # print(f"----set lasvsim_context: {(time_5 - time_4) * 1000} ms.")


    def get_all_ref_param(self) -> np.ndarray:
        # return: ref_param [VARIABLE_NUM, ref_horizon, per_point_dim]
        ref_horizon = self.config["ref_horizon"]
        ref_list = self.lasvsim_context.ref_list
        traffic_light = 0 # TODO: get traffic light from qianxing self.lasvsim_context.xxxx
        max_speed = self.config["max_speed"]
        dt = self.config["dt"]

        path_planning_mode = "green"
        if traffic_light == 0:
            am = self.config["dec_before_junction_green"]
            path_planning_mode = "green"
        else:
            am = self.config["dec_before_junction_red"]
            path_planning_mode = "red"
            min_ahead_lane_length = self.config["ahead_lane_length_min"]
        
        driving_task = "s"
        if driving_task == "s":
            ref_v_junction = max_speed * self.config["v_discount_in_junction_straight"]
        elif driving_task == "l":
            ref_v_junction = max_speed * self.config["v_discount_in_junction_left_turn"]
        elif driving_task == "r":
            ref_v_junction = max_speed * self.config["v_discount_in_junction_right_turn"]
        else:
            raise ValueError("Error driving task: {}".format(driving_task))
        
        cur_v = max_speed

        ref_param = []
        for ref_line in ref_list:
            
            # ref_info = ref_info_list[ref_list.index(ref_line)]
            ref_info = [
                {'destination': True}
            ]
            current_part = ref_info[0]
            ego = self.lasvsim_context.ego
            position_on_ref = point_project_to_line(
                ref_line, *ego.ground_position)
            
            if current_part['destination'] == True:
                position_on_ref = point_project_to_line(ref_line, *ego.ground_position)
                intervals, ref_v = compute_intervals(ref_info, ref_horizon, cur_v, max_speed, dt, 0)
            elif current_part['destination'] == False and current_part["in_junction"] == True:
                intervals, ref_v = compute_intervals_in_junction(
                    ref_horizon, ref_v_junction, dt)
            elif current_part["in_junction"] == False and current_part['destination'] == False:
                if path_planning_mode == "green":
                    intervals, ref_v = compute_intervals_initsegment_green(
                        position_on_ref, current_part, ref_horizon, max_speed, ref_v_junction, dt, am)
                elif path_planning_mode == "red":
                    intervals, ref_v = compute_intervals_initsegment_red(
                        position_on_ref, current_part, ref_horizon, max_speed, dt, am, min_ahead_lane_length)
                else:
                    raise ValueError("Error path_planning_mode")
            else:
                raise ValueError("Error ref_line")
            # repeat the last v
            ref_v = np.append(ref_v, ref_v[-1])
            ref_v = np.expand_dims(ref_v, axis=1)
            
            ref_array = compute_waypoints_by_intervals(ref_line, position_on_ref, intervals)
            ref_array = np.concatenate((ref_array, ref_v), axis=-1)
            ref_param.append(ref_array)
        return np.array(ref_param)
        
    def get_obs_from_context(self):
        # get obs from self.lasvisim_context
        # Return: np.ndarray([4305])
        ego_dim = 5
        sur_num = 10 # max number of surrounding vehicles
        sur_horizon = 10 # history length
        sur_dim = 11 # x, y, cosphi, sinphi, speed, length, width, type(4)
        vec_num = 200
        vec_dim = 16 # x, y, length, width, cosphi, sinphi, max_speed, type(6), light(3)
        
        obs = np.zeros(
            ego_dim +
            sur_num * sur_horizon * sur_dim +
            vec_num * vec_dim
        )
        
        # -------------- 自车状态更新 -------------- 
        obs[0:5] = [self.lasvsim_context.ego.u,
                    self.lasvsim_context.ego.v,
                    self.lasvsim_context.ego.w,
                    self.lasvsim_context.ego.action[0],
                    self.lasvsim_context.ego.action[1]]
        
        # 地图状态更新
        # 根据self.lasvsim_context中的自车位置，从self.map_objs中选取出200个距离自车最近的vector
        # 构成观测的第1105-4304维（共3200维）
        # 其中，各vector到自车的距离应从小到大排列，即第1个vector距离自车应为最近
        ego_x, ego_y, ego_phi = (self.lasvsim_context.ego.x, 
                                 self.lasvsim_context.ego.y, 
                                 self.lasvsim_context.ego.phi)
        ego_center = np.array([ego_x, ego_y])
        obj_centers = self.map_objs[:, :2]
        # Calculate distances to ego center
        distances = np.sqrt(np.sum((obj_centers - ego_center) ** 2, axis=1))

        # Use partition to find indices of self.max_num_map_observed nearest objects 
        selected_indices = np.argpartition(distances, self.max_num_map_observed)[:self.max_num_map_observed]
        sorted_indices = selected_indices[np.argsort(distances[selected_indices])]
        selected_objs = self.map_objs[sorted_indices]
        
        cos_tf = np.cos(-ego_phi)
        sin_tf = np.sin(-ego_phi)
        
        # Transform x,y coordinates
        x_ego = (selected_objs[:, 0] - ego_x) * cos_tf - (selected_objs[:, 1] - ego_y) * sin_tf
        y_ego = (selected_objs[:, 0] - ego_x) * sin_tf + (selected_objs[:, 1] - ego_y) * cos_tf
        
        # Transform orientation angles (columns 4,5 contain cos/sin of orientation)
        phi_abs = np.arctan2(selected_objs[:, 5], selected_objs[:, 4])
        phi_ego = deal_with_phi_rad(phi_abs - ego_phi)
        cos_phi_ego = np.cos(phi_ego)
        sin_phi_ego = np.sin(phi_ego)
        
        # Construct transformed objects array
        transformed_objs = selected_objs.copy()
        transformed_objs[:, 0] = x_ego
        transformed_objs[:, 1] = y_ego
        transformed_objs[:, 4] = cos_phi_ego
        transformed_objs[:, 5] = sin_phi_ego

        obs[ego_dim + sur_num * sur_horizon * sur_dim:] = transformed_objs.ravel()
        
        # -------------- 周车状态更新 -------------- 
        all_sur_veh_obs = np.zeros((sur_num, sur_horizon, sur_dim), dtype=np.float32)

        # latest surrounding vehicles
        veh_id_list = []
        for i, sur_veh in enumerate(self.lasvsim_context.sur_list):
            if i >= sur_num: 
                break
            veh_id_list.append(sur_veh.veh_id)
        
        # assert there is no duplicated element in veh_id_list
        assert len(veh_id_list) == len(set(veh_id_list))
        
        # padding in whole horizon L
        all_sur_veh_obs[len(veh_id_list):, :, 10] = 1

        # history surrounding vehicles
        for j in range(sur_horizon):
            for sur_veh in self.history_sur_veh[-1-j]:
                if sur_veh.veh_id in veh_id_list:
                    idx = veh_id_list.index(sur_veh.veh_id)
                    all_sur_veh_obs[idx, j, :7] = [
                        sur_veh.rel_x, sur_veh.rel_y, 
                        np.cos(sur_veh.rel_phi), np.sin(sur_veh.rel_phi),
                        sur_veh.u, sur_veh.length, sur_veh.width
                    ]
                    all_sur_veh_obs[idx, j, 7] = 1
                    # TODO: 增加行人、自行车
            # padding
            all_sur_veh_obs[
                all_sur_veh_obs[:, j, 7] == 0, # 虚拟周车
                j,
                10
            ] = 1
        
        obs[ego_dim : ego_dim + sur_num * sur_horizon * sur_dim] = all_sur_veh_obs.ravel()
        
        return obs

    def step(self, action: np.ndarray):
        # action: network output, \in [-1, 1]
        self.alive_step += 1

        time_1 = time()
        action = inverse_normalize_action(action, self.action_half_range, self.action_center)
        real_action = action + self.lasvsim_context.ego.last_action
        real_action = np.clip(
            real_action, self.real_action_lower, self.real_action_upper)
        
        time_2 = time()
        # print(f"--before set remote control: {(time_2 - time_1) * 1000} ms.")
        self.set_remote_lasvsim_veh_control(real_action)
        time_3 = time()
        # print(f"--set remote control: {(time_3 - time_2) * 1000} ms.")
        self.step_remote_lasvsim()
        time_4 = time()
        # print(f"--step_remote_lasvsim: {(time_4 - time_3) * 1000} ms.")
        self.update_lasvsim_context(real_action)
        time_5 = time()
        # print(f"--update_lasvsim_context: {(time_5 - time_4) * 1000} ms.")

        reward, rew_info = self.reward_function_multilane()
        time_6 = time()
        # print(f"--reward_function_multilane: {(time_6 - time_5) * 1000} ms.")

        obs = self.get_obs_from_context()
        truncated = self.alive_step >= self.max_step
        time_7 = time()
        # print(f"--get_obs_from_context: {(time_7 - time_6) * 1000} ms.")
        return obs, reward, self.judge_done(), truncated, rew_info

    def reset(self):
        test_vehicle_list = []
        self.alive_step = 0
        if self.scenario_cnt < 10:
            while len(test_vehicle_list) == 0:
                self.reset_remote_lasvsim()
                self.step_remote_lasvsim()
                test_vehicle = self.get_remote_lasvsim_test_veh_list()
                if test_vehicle is not None:
                    test_vehicle_list = test_vehicle.list
            self.scenario_cnt += 1
        else:
            while len(test_vehicle_list) == 0:
                self.stop_remote_lasvsim()
                idx = random.randint(0, len(self.scenario_list) - 1)
                self.scenario_id = self.scenario_list[idx]
                self.simulator = self.init_remote_lasvsim(
                    scenario_id=self.scenario_id,
                    scenario_version=self.version_list[idx]
                )
                self.step_remote_lasvsim()
                test_vehicle = self.get_remote_lasvsim_test_veh_list()
                if (test_vehicle is not None):
                    test_vehicle_list = test_vehicle.list
                self.convert_map(self.scenario_list[idx])
            self.scenario_cnt = 0

        self.ego_id = test_vehicle_list[0]
        # TODO: 速度和位置的随机初始化
        random_init_v = np.random.uniform(0, 10)
        self.simulator.set_vehicle_moving_info(self.ego_id, random_init_v)
        # 获取自车位置
        vehicles_position = self.get_remote_lasvsim_veh_position()
        x = vehicles_position.position_dict.get(self.ego_id).point.x
        y = vehicles_position.position_dict.get(self.ego_id).point.y
        phi = vehicles_position.position_dict.get(self.ego_id).phi
        random_offset_x = np.random.normal(0.0, 1.0)
        random_offset_y = np.random.normal(0.0, 1.0)
        random_offset_phi = np.random.normal(0.0, 0.1)
        self.simulator.set_vehicle_position(self.ego_id, QxPoint(
            x + random_offset_x,
            y + random_offset_y,
            phi + random_offset_phi
        ))

        self.update_lasvsim_context()
        obs = self.get_obs_from_context()
        info = {}
        return obs, info

    # from rlplanner
    def model_free_reward_multilane_batch(self,
                                          t: np.ndarray,  # time step
                                          ref_param, # [R, 2N+1, 4]
                                          ) -> Tuple[List[np.ndarray], List[dict]]:
        # all inputs are batched
        ego= self.lasvsim_context.ego

        ego_state = (ego.x, ego.y, ego.u, ego.v, ego.phi, ego.w)
        ego_x, ego_y, ego_vx, ego_vy, ego_phi, ego_r = ego_state

        last_acc, last_steer = ego.action[0], ego.action[1] * 180 / np.pi
        last_last_acc, last_last_steer = ego.last_action[0], ego.last_action[1] * 180 / np.pi
        delta_steer = (last_steer - last_last_steer) / self.config["dt"]
        jerk = (last_acc - last_last_acc) / self.config["dt"]

        ref_states = ref_param[:, t, :]  # [R, 4]
        next_ref_states = ref_param[:, t + 1, :]  # [R, 4]
        ref_x, ref_y, ref_phi, ref_v = ref_states.T
        next_ref_v = next_ref_states[:, 3]

        # live reward
        rew_step = 0.5 * np.clip(ego_vx, 0, 2.0) * \
            np.ones(ref_param.shape[0])  # 0~1

        # tracking_error
        tracking_error = -(ego_x - ref_x) * np.sin(ref_phi) + \
            (ego_y - ref_y) * np.cos(ref_phi)
        delta_phi = deal_with_phi_rad(
            ego_phi - ref_phi) * 180 / np.pi  # degree
        ego_r = ego_r * 180 / np.pi  # degree
        speed_error = ego_vx - ref_v

        # tracking_error
        punish_dist_lat = 5 * np.where(
            np.abs(tracking_error) < 0.3,
            np.square(tracking_error),
            0.02 * np.abs(tracking_error) + 0.084,
        )  # 0~1 0~6m 50% 0~0.3m

        punish_vel_long = 0.5*np.where(
            np.abs(speed_error) < 1,
            np.square(speed_error),
            0.2*np.abs(speed_error)+0.8,
        )  # 0~1 0~11.5m/s 50% 0~1m/s

        punish_head_ang = 0.05 * np.where(
            np.abs(delta_phi) < 3,
            np.square(delta_phi),
            np.abs(delta_phi) + 8,
        )  # 0~1  0~12 degree 50% 0~3 degree

        ego_r = ego_r * np.ones(ref_param.shape[0])
        punish_yaw_rate = 0.1 * np.where(
            np.abs(ego_r) < 2,
            np.square(ego_r),
            np.abs(ego_r) + 2,
        )  # 0~1  0~8 degree/s 50% 0~2 degree/s

        punish_overspeed = np.clip(
            np.where(
                ego_vx > 1.05 * ref_v,
                1 + np.abs(ego_vx - 1.05 * ref_v),
                0, ),
            0, 2)

        # reward related to action
        nominal_steer = self._get_nominal_steer_by_state_batch(
            ego_state, ref_param) * 180 / np.pi

        abs_steer = np.abs(last_steer - nominal_steer)
        reward_steering = -np.where(abs_steer < 4,
                                    np.square(abs_steer), 2 * abs_steer + 8)

        self.out_of_action_range = abs_steer > 20

        if ego_vx < 0.1 and self.config["enable_slow_reward"]:
            reward_steering = reward_steering * 5

        abs_ax = np.abs(last_acc) * np.ones(ref_param.shape[0])
        reward_acc_long = -np.where(abs_ax < 2, np.square(abs_ax), 2 * abs_ax)

        delta_steer = delta_steer * np.ones(ref_param.shape[0])
        reward_delta_steer = - \
            np.where(np.abs(delta_steer) < 4, np.square(
                delta_steer), 2 * np.abs(delta_steer) + 8)
        jerk = jerk * np.ones(ref_param.shape[0])
        reward_jerk = -np.where(np.abs(jerk) < 2,
                                np.square(jerk), 2 * np.abs(jerk) + 8)

        # if self.in_multilane:  # consider more comfortable reward
        #     reward_acc_long = reward_acc_long * 2
        #     reward_jerk = reward_jerk * 2
        #     reward_steering = reward_steering * 2
        #     reward_delta_steer = reward_delta_steer * 2
        #     punish_yaw_rate = punish_yaw_rate * 2

        # if self.turning_direction != 0:  # left is positive =1
        #     punish_dist_lat = punish_dist_lat * 0.5
        #     punish_head_ang = punish_head_ang * 0.5
        #     punish_yaw_rate = punish_yaw_rate * 0.2
        #     reward_steering = reward_steering * 0.2
        #     tracking_bias_direrction = np.sign(
        #         tracking_error)  # left is positive
        #     phi_direrction = np.sign(delta_phi)  # left is positive
        #     condition = (self.turning_direction != tracking_bias_direrction) & (
        #         self.turning_direction != phi_direrction) & (np.abs(tracking_error) > 0.05) & (np.abs(delta_phi) > 2)
        #     punish_dist_lat = np.where(
        #         condition, punish_dist_lat + 4, punish_dist_lat)
        #     punish_head_ang = np.where(
        #         condition, punish_head_ang + 4, punish_head_ang)

        break_condition = (ref_v < 1.5) & (
            (next_ref_v - ref_v) < -0.1) | (ref_v < 1.0)
        if break_condition.any() and self.config["nonimal_acc"]:
            nominal_acc = np.where(break_condition, -1.5, 0)
            # remove the effect of tracking error
            punish_dist_lat = np.where(break_condition, 0, punish_dist_lat)
            punish_head_ang = np.where(break_condition, 0, punish_head_ang)
            reward_acc_long = np.where(break_condition, 0, reward_acc_long)
        else:
            nominal_acc = np.zeros(ref_param.shape[0])
            punish_nominal_acc = np.zeros(ref_param.shape[0])

        if self.braking_mode and self.config["nonimal_acc"]:
            nominal_acc = -1.5 * np.ones(ref_param.shape[0])
            punish_vel_long = np.zeros(ref_param.shape[0])

        if break_condition.any() or self.braking_mode:
            rew_step = np.where(break_condition, 1.0, rew_step)

        delta_acc = np.abs(nominal_acc - last_acc)
        punish_nominal_acc = (nominal_acc != 0) * np.where(delta_acc <
                                                           0.5, np.square(delta_acc), delta_acc - 0.25)

        # tracking related reward
        scaled_punish_dist_lat = punish_dist_lat * self.config["P_lat"]
        scaled_punish_vel_long = punish_vel_long * self.config["P_long"]
        scaled_punish_head_ang = punish_head_ang * self.config["P_phi"]
        scaled_punish_yaw_rate = punish_yaw_rate * self.config["P_yaw"]
        scaled_punish_overspeed = punish_overspeed * 3  # TODO: hard coded value

        # action related reward
        scaled_reward_steering = reward_steering * self.config["P_steer"]
        scaled_reward_acc_long = reward_acc_long * self.config["P_acc"]
        scaled_reward_delta_steer = reward_delta_steer * self.config["P_delta_steer"]
        scaled_reward_jerk = reward_jerk * self.config["P_jerk"]
        scaled_punish_nominal_acc = punish_nominal_acc * 8  # TODO: hard coded value

        # live reward
        scaled_rew_step = rew_step * self.config["R_step"]

        reward_ego_state = scaled_rew_step - \
            (scaled_punish_dist_lat +
             scaled_punish_vel_long +
             scaled_punish_head_ang +
             scaled_punish_yaw_rate +
             scaled_punish_nominal_acc +
             scaled_punish_overspeed) + \
            (scaled_reward_steering +
             scaled_reward_acc_long +
             scaled_reward_delta_steer +
             scaled_reward_jerk)

        reward_ego_state = np.clip(reward_ego_state, -5, 30)

        rewards = reward_ego_state.tolist()
        infos = [{
            "env_tracking_error": np.abs(tracking_error[i]),
            "env_speed_error": np.abs(speed_error[i]),
            "env_delta_phi": np.abs(delta_phi[i]),
            "state_nominal_steer": nominal_steer[i],
            "state_nominal_acc": nominal_acc[i],

            "env_reward_step": rew_step[i],

            "env_reward_steering": reward_steering[i],
            "env_reward_acc_long": reward_acc_long[i],
            "env_reward_delta_steer": reward_delta_steer[i],
            "env_reward_jerk": reward_jerk[i],

            "env_reward_dist_lat": -punish_dist_lat[i],
            "env_reward_vel_long": -punish_vel_long[i],
            "env_reward_head_ang": -punish_head_ang[i],
            "env_reward_yaw_rate": -punish_yaw_rate[i],

            "env_scaled_reward_part2": reward_ego_state[i],
            "env_scaled_reward_step": scaled_rew_step[i],
            "env_scaled_reward_dist_lat": -scaled_punish_dist_lat[i],
            "env_scaled_reward_vel_long": -scaled_punish_vel_long[i],
            "env_scaled_reward_head_ang": -scaled_punish_head_ang[i],
            "env_scaled_reward_yaw_rate": -scaled_punish_yaw_rate[i],
            "env_scaled_reward_nominal_acc": -scaled_punish_nominal_acc[i],
            "env_scaled_reward_overspeed": -scaled_punish_overspeed[i],
            "env_scaled_reward_steering": scaled_reward_steering[i],
            "env_scaled_reward_acc_long": scaled_reward_acc_long[i],
            "env_scaled_reward_delta_steer": scaled_reward_delta_steer[i],
            "env_scaled_reward_jerk": scaled_reward_jerk[i],
        } for i in range(ref_param.shape[0])]

        return rewards, infos

    def _get_nominal_steer_by_state_batch(self,
                                          ego_state,
                                          ref_param):
        # ref_param: [R, 2N+1, 4]
        # use ref_state_index to determine the start, from 2N+1 to 3
        # ref_line: [R, 3, 4]
        def cal_curvature(x1, y1, x2, y2, x3, y3):
            # cal curvature by three points in batch format
            # dim of x1 is [R]
            a = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            b = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
            c = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
            k = np.zeros_like(a)
            i = (a * b * c) != 0
            area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
            k[i] = 2 * area[i] / (a[i] * b[i] * c[i])
            return k

        ref_line = np.stack([ref_param[:, i, :]
                            for i in [0, 5, 10]], axis=1)  # [R, 3, 4]
        ref_x_ego_coord, ref_y_ego_coord, ref_phi_ego_coord = \
            convert_ref_to_ego_coord(ref_line[:, :, :3], ego_state)  # [R, 3]

        # nominal action
        x1, y1 = ref_x_ego_coord[:, 0], ref_y_ego_coord[:, 0]  # [R,]
        x2, y2 = ref_x_ego_coord[:, 1], ref_y_ego_coord[:, 1]
        x3, y3 = ref_x_ego_coord[:, 2], ref_y_ego_coord[:, 2]
        nominal_curvature = cal_curvature(x1, y1, x2, y2, x3, y3)
        nominal_steer = nominal_curvature * 2.65  # FIXME: hard-coded: wheel base
        nominal_steer = np.clip(
            nominal_steer, self.real_action_lower[1], self.real_action_upper[1])

        return nominal_steer

    # from rlplanner
    def reward_function_multilane(self):
        ego = self.lasvsim_context.ego
        # cal reference_closest
        ref_list = self.lasvsim_context.ref_list
        # try:
        closest_idx = np.argmin([ref_line.distance(ego.polygon)
                                    for ref_line in ref_list])
        # except Exception as e:
            # breakpoint()
        reference_closest = ref_list[closest_idx]
        # tracking_error cost
        position_on_ref = point_project_to_line(
            reference_closest, ego.x, ego.y)
        current_first_ref_x, current_first_ref_y, \
            current_first_ref_phi = compute_waypoint(
                reference_closest, position_on_ref)

        tracking_error = np.sqrt((ego.x - current_first_ref_x) ** 2 +
                                 (ego.y - current_first_ref_y) ** 2)
        delta_phi = deal_with_phi_rad(ego.phi - current_first_ref_phi)

        self.out_of_range = tracking_error > 4 or np.abs(delta_phi) > np.pi/4
        self.in_junction = ego.in_junction
        # self.in_multilane = self.engine.context.scenario_id in self.config["multilane_scenarios"]  # FIXME: hardcoded scenario_id
        # direction = vehicle.direction
        # if self.in_junction:
        #     self.turning_direction = 1 if direction == "l" else -1 if direction == "r" else 0
        # else:
        #     self.turning_direction = 0

        # TODO: hard coded value

        # ax = vehicle.ax

        # collision risk cost
        # ego_vx = vehicle.vx
        # ego_W = vehicle.width
        # ego_L = vehicle.length

        safety_lat_margin_front = self.config["safety_lat_margin_front"]
        safety_lat_margin_rear = safety_lat_margin_front  # TODO: safety_lat_margin_rear
        safety_long_margin_front = self.config["safety_long_margin_front"]
        safety_long_margin_side = self.config["safety_long_margin_side"]
        front_dist_thd = self.config["front_dist_thd"]
        space_dist_thd = self.config["space_dist_thd"]
        rel_v_thd = self.config["rel_v_thd"]
        rel_v_rear_thd = self.config["rel_v_rear_thd"]
        time_dist = self.config["time_dist"]

        punish_done = self.config["P_done"]

        pun2front = 0.
        pun2side = 0.
        pun2space = 0.
        pun2rear = 0.

        pun2front_sum = 0.
        pun2side_sum = 0.
        pun2space_sum = 0.
        pun2rear_sum = 0.

        min_front_dist = np.inf

        sur_list: List[SurroundingVehicle] = self.lasvsim_context.sur_list
        sur_list = [sur for sur in sur_list if sur.mask==1]
        # sur_info = self.engine.context.vehicle.surrounding_veh_info
        # ego_edge = self.engine.context.vehicle.edge
        # ego_lane = self.engine.context.vehicle.lane
        # if self.config["ignore_opposite_direction"] and self.engine.context.scenario_id in self.config["multilane_scenarios"]:  # FIXME: hardcoded scenario_id
        #     sur_info = [s for s in sur_info if s.road_id == ego_edge]

        ego_W = ego.width
        ego_L = ego.length
        ego_vx = ego.u
        for sur_vehicle in sur_list:
            rel_x = sur_vehicle.rel_x
            rel_y = sur_vehicle.rel_y
            sur_vx = sur_vehicle.u
            sur_lane = sur_vehicle.lane_id
            sur_W = sur_vehicle.width
            sur_L = sur_vehicle.length
            # [1 - tanh(x)]: 0.25-> 75%  0.5->54%, 1->24%, 1.5->9.5% 2->3.6%, 3->0.5%
            if ((np.abs(rel_y) < (ego_W + sur_W) / 2 - 1)) \
                    and (rel_x > (ego_L + sur_L) / 2):
                min_front_dist = min(
                    min_front_dist, rel_x - (ego_L + sur_L) / 2)

            pun2front_cur = np.where(
                (np.abs(rel_y) < (ego_W + sur_W) / 2 + safety_lat_margin_front) and
                (rel_x >= 0) and (rel_x < front_dist_thd) and (ego_vx > sur_vx),
                np.clip(1. - np.tanh((rel_x-(ego_L + sur_L) / 2
                        - safety_long_margin_front) / (time_dist*(np.max(ego_vx, 0) + 0.1))),
                        0., 1.),
                0,
            )
            pun2front = np.maximum(pun2front, pun2front_cur)
            pun2front_sum += pun2front_cur

            pun2side_cur = np.where(
                np.abs(rel_x) < (ego_L + sur_L) / 2 + safety_long_margin_side and rel_y *
                delta_phi > 0 and rel_y > (ego_W + sur_W) / 2,
                np.clip(1. - np.tanh((np.abs(rel_y) - (ego_W + sur_W) / 2) /
                        (np.abs(ego_vx*np.sin(delta_phi))+0.01)), 0., 1.),
                0,
            )
            pun2side = np.maximum(pun2side, pun2side_cur)
            pun2side_sum += pun2side_cur

            pun2space_cur = np.where(
                np.abs(rel_y) < (ego_W + sur_W) /
                2 and rel_x >= 0 and rel_x < space_dist_thd and ego_vx > sur_vx + rel_v_thd,
                np.clip(1. - (rel_x - (ego_L + sur_L) / 2) /
                        (space_dist_thd - (ego_L + sur_L) / 2), 0., 1.),
                0,) + np.where(
                np.abs(rel_x) < (ego_L + sur_L) / 2 +
                safety_long_margin_side and np.abs(
                    rel_y) > (ego_W + sur_W) / 2,
                np.clip(
                    1. - np.tanh(3.0*(np.abs(rel_y) - (ego_W + sur_W) / 2)), 0., 1.),
                0,)
            pun2space = np.maximum(pun2space, pun2space_cur)
            pun2space_sum += pun2space_cur

            pun2rear_cur = np.where(
                (np.abs(rel_y) < (ego_W + sur_W) / 2 + safety_lat_margin_rear) and rel_x < 0 and rel_x > -
                space_dist_thd and ego_vx < sur_vx - rel_v_rear_thd,
                np.clip(1. - (-1)*(rel_x + (ego_L + sur_L) / 2) /
                        (space_dist_thd - (ego_L + sur_L) / 2), 0., 1.),
                0,)
            pun2rear = np.maximum(pun2rear, pun2rear_cur)
            pun2rear_sum += pun2rear_cur

        if self.config["punish_sur_mode"] == "sum":
            pun2front = pun2front_sum
            pun2side = pun2side_sum
            pun2space = pun2space_sum
            pun2rear = pun2rear_sum
        elif self.config["punish_sur_mode"] == "max":
            pass
        else:
            print(self.config["punish_sur_mode"])
            raise ValueError(
                f"Invalid punish_sur_mode")
        scaled_pun2front = pun2front * self.config["P_front"]
        scaled_pun2side = pun2side * self.config["P_side"]
        scaled_pun2space = pun2space * self.config["P_space"]
        scaled_pun2rear = pun2rear * self.config["P_rear"]

        # self.braking_mode = (
        #     min_front_dist < 4) and not self.in_junction and not self.in_multilane  # trick
        self.braking_mode = False

        punish_collision_risk = scaled_pun2front + \
            scaled_pun2side + scaled_pun2space + scaled_pun2rear

        if ego_vx <= 0.01:
            punish_collision_risk = 0

        # exclude scenarios without surrounding vehicles
        self.active_collision = self.check_collision() and ego_vx > 0.01

        # out of driving area cost
        # TODO: boundary cost = 0  when boundary info is not available
        if self.in_junction or self.config["P_boundary"] == 0:
            punish_boundary = 0.
        else:
            rel_angle = np.abs(delta_phi)
            left_distance = np.abs(ego.left_boundary_distance)
            right_distance = np.abs(ego.right_boundary_distance)
            min_left_distance = left_distance - \
                (ego_L / 2)*np.sin(rel_angle) - (ego_W / 2)*np.cos(rel_angle)
            min_right_distance = right_distance - \
                (ego_L / 2)*np.sin(rel_angle) - (ego_W / 2)*np.cos(rel_angle)
            boundary_safe_margin = 0.15
            boundary_distance = np.clip(np.minimum(
                min_left_distance, min_right_distance), 0., None)

            punish_boundary = np.where(
                boundary_distance < boundary_safe_margin,
                np.clip((1. - boundary_distance/boundary_safe_margin), 0., 1.),
                0.0,
            )
        scaled_punish_boundary = punish_boundary * self.config["P_boundary"]

        # action related reward

        reward = - scaled_punish_boundary

        punish_collision_risk = punish_collision_risk if (
            self.config["penalize_collision"]) else 0.
        reward -= punish_collision_risk

        event_flag = 0  # nomal driving (on lane, stop)
        reward_done = 0
        reward_collision = 0
        # Event reward: target reached, collision, out of driving area
        if self.check_out_of_driving_area() or self.out_of_range:  # out of driving area
            reward_done = - punish_done
            event_flag = 1
        elif self.active_collision:  # collision by ego vehicle
            reward_collision = -20 if self.config["penalize_collision"] else 0.
            event_flag = 2
        elif self.braking_mode:  # start to brake
            event_flag = 3

        reward += (reward_done + reward_collision)

        # if vehicle.arrive_success:
        #     reward += 200.
        return reward, {
            "category": event_flag,
            "env_pun2front": pun2front,
            "env_pun2side": pun2side,
            "env_pun2space": pun2space,
            "env_pun2rear": pun2rear,
            "env_scaled_reward_part1": reward,
            "env_scaled_reward_done": reward_done,
            "env_scaled_reward_collision": reward_collision,
            "env_scaled_reward_collision_risk": - punish_collision_risk,
            "env_scaled_pun2front": scaled_pun2front,
            "env_scaled_pun2side": scaled_pun2side,
            "env_scaled_pun2space": scaled_pun2space,
            "env_scaled_pun2rear": scaled_pun2rear,
            "env_scaled_reward_boundary": - scaled_punish_boundary,
        }
    
    def check_collision(self) -> bool:
        return self.simulator.get_vehicle_collision_status(self.ego_id).collision_status

    def check_out_of_driving_area(self) -> bool:
        vehicles_position = self.get_remote_lasvsim_veh_position()
        ego_pos = vehicles_position.position_dict.get(self.ego_id).type
        out_of_driving_area_flag = (ego_pos == 3)
        return out_of_driving_area_flag

    def judge_done(self) -> bool:
        time_1 = time()
        collision = self.check_collision()
        time_2 = time()
        # print(f"--check_collision:{(time_2 - time_1) * 1000} ms.")
        out_of_driving_area = self.check_out_of_driving_area()
        time_3 = time()
        # print(f"--check_out_of_driving_area:{(time_3 - time_2) * 1000} ms.")
        park_flag = (self.lasvsim_context.ego.u == 0)
        out_of_defined_region = self.out_of_range
        self._render_done_info = {
            "Pause": park_flag,
            "RegionOut": out_of_defined_region,
            "Collision": collision,
            "MapOut": out_of_driving_area,
            "alive_step": self.alive_step
        }
        done = collision or out_of_defined_region or out_of_driving_area
        time_4 = time()
        # print(f"--before judge_done:{(time_4 - time_3) * 1000} ms.")
        if done:
            print('# DONE')
            print(self._render_done_info)
        return done

    def get_ego_context(self, real_actiton: np.ndarray = None):
        time_1 = time()
        vehicles_position = self.get_remote_lasvsim_veh_position()
        time_2 = time()
        # print(f"------get_remote_lasvsim_veh_position: {(time_2 - time_1) * 1000} ms.")
        vehicles_baseInfo = self.get_remote_lasvsim_veh_base_info()
        time_3 = time()
        # print(f"------get_remote_lasvsim_veh_base_info: {(time_3 - time_2) * 1000} ms.")
        vehicles_movingInfo = self.get_remote_lasvsim_veh_moving_info()
        time_4 = time()
        # print(f"------get_remote_lasvsim_veh_moving_info: {(time_4 - time_3) * 1000} ms.")

        x = vehicles_position.position_dict.get(self.ego_id).point.x
        y = vehicles_position.position_dict.get(self.ego_id).point.y
        phi = vehicles_position.position_dict.get(self.ego_id).phi
        junction_id = vehicles_position.position_dict.get(
            self.ego_id).junction_id
        lane_id = vehicles_position.position_dict.get(self.ego_id).lane_id
        link_id = vehicles_position.position_dict.get(self.ego_id).link_id
        segment_id = vehicles_position.position_dict.get(
            self.ego_id).segment_id
        ego_pos = vehicles_position.position_dict.get(self.ego_id).position_type
        in_junction = (ego_pos == 2)

        length = vehicles_baseInfo.info_dict.get(self.ego_id).base_info.length
        width = vehicles_baseInfo.info_dict.get(self.ego_id).base_info.width

        u = vehicles_movingInfo.moving_info_dict.get(self.ego_id).u
        v = vehicles_movingInfo.moving_info_dict.get(self.ego_id).v
        w = vehicles_movingInfo.moving_info_dict.get(self.ego_id).w

        # assert not in_junction
        self.can_not_get_lane_id = False
        try:
            target_lane = self.lanes[lane_id]
        except Exception as e:
            # print('X'*50)
            # print('can_not_get_lane_id')
            # breakpoint()
            self.can_not_get_lane_id = True

        if self.can_not_get_lane_id:
            right_boundary_distance = 1.75
            left_boundary_distance = 1.75
        else:
            # print("Normal lane id")
            left1, right1 = calculate_perpendicular_points(
                target_lane.center_line[0].point.x,
                target_lane.center_line[0].point.y,
                target_lane.center_line[0].heading,
                target_lane.center_line[0].left_width)
            left2, right2 = calculate_perpendicular_points(
                target_lane.center_line[1].point.x,
                target_lane.center_line[1].point.y,
                target_lane.center_line[1].heading,
                target_lane.center_line[1].left_width)
            # print("center1: ", target_lane.center_line[0])
            # print("center2: ", target_lane.center_line[1])
            # print("boundary points: ", left1, right1)
            # print("boundary points: ", left2, right2)
            # breakpoint()

            left_boundary_lane = LineString([left1, left2])
            right_boundary_lane = LineString([right1, right2])
            Rb_position = point_project_to_line(right_boundary_lane, x, y)
            Rb_x, Rb_y, _ = compute_waypoint(right_boundary_lane, Rb_position)
            right_boundary_distance = cal_dist(Rb_x, Rb_y, x, y)

            Lb_position = point_project_to_line(left_boundary_lane, x, y)
            Lb_x, Lb_y, _ = compute_waypoint(left_boundary_lane, Lb_position)
            left_boundary_distance = cal_dist(Lb_x, Lb_y, x, y)

        polygon = create_box_polygon(x, y, phi, length, width)

        if real_actiton is not None:
            action = real_actiton
        else:
            action = np.array([0.0]*2)
        state = np.array([x, y, u, v, phi, w])
        last_action = self.lasvsim_context.ego.action

        time_5 = time()
        # print(f"------other in get_ego_context: {(time_5 - time_4) * 1000} ms.")

        return EgoVehicle(
            x=x, y=y, phi=phi, u=u, v=v, w=w,
            length=length, width=width,
            action=action,
            state=state,
            last_action=last_action,
            junction_id=junction_id, lane_id=lane_id,
            link_id=link_id, segment_id=segment_id,
            in_junction=in_junction,
            left_boundary_distance=left_boundary_distance,
            right_boundary_distance=right_boundary_distance,
            polygon=polygon
        )

    def get_ref_context(self):
        time_1 = time()
        ref_points = self.get_remote_lasvsim_ref_line()
        time_2 = time()
        # print(f"------get_remote_lasvsim_ref_line: {(time_2 - time_1) * 1000} ms.")
        # breakpoint()
        ref_lines = ref_points.reference_lines
        if len(ref_lines)==0:
            # print('X'*50)
            # print("Zero ref!!!")
            if not self.can_not_get_lane_id:
                lane_id = self.lasvsim_context.ego.lane_id
                target_lane = self.lanes[lane_id]
                ref_line_xy = np.array([[p.point.x, p.point.y] for p in target_lane.center_line])
                ref_line_string = LineString(ref_line_xy)
                return [ref_line_string] * len(self.lasvsim_context.ref_list)
            else:
                return self.lasvsim_context.ref_list
        # print("Normal ref")
        ref_context = []
        for ref_line in ref_lines:
            ref_line_xy = np.array([[point.x, point.y]
                                for point in ref_line.points])
            ref_line_string = LineString(ref_line_xy)
            ref_context.append(ref_line_string)
        time_3 = time()
        # print(f"------other: {(time_3 - time_2) * 1000} ms.")
        return ref_context

    def get_sur_context(self):
        perception_info = self.get_remote_lasvsim_perception_info()
        around_moving_objs = perception_info.list

        ego_x, ego_y, ego_phi = self.lasvsim_context.ego.x, \
            self.lasvsim_context.ego.y, \
            self.lasvsim_context.ego.phi

        # filter neighbor vehicles for better efficiency
        distances = [
            cal_dist(
                obj.position.point.x,
                obj.position.point.y,
                ego_x,
                ego_y
            )
            for obj in around_moving_objs
        ]

        # sort out the smallest k distance vehicles
        if (len(distances) > self.surr_veh_num):
            indices = get_indices_of_k_smallest(distances, self.surr_veh_num)
        else:
            indices = range(len(distances))

        # append info of the smallest k distance vehicles
        sur_context = []
        for i in indices:
            sur_x, sur_y, sur_phi = \
                around_moving_objs[i].position.point.x, \
                around_moving_objs[i].position.point.y, \
                around_moving_objs[i].position.phi
            rel_x, rel_y, rel_phi = convert_ground_coord_to_ego_coord(
                sur_x, sur_y, sur_phi,
                ego_x, ego_y, ego_phi
            )
            distance = distances[i]
            u = around_moving_objs[i].moving_info.u
            length = around_moving_objs[i].base_info.length
            width = around_moving_objs[i].base_info.width
            veh_id = around_moving_objs[i].obj_id
            lane_id = around_moving_objs[i].position.lane_id
            sur_vehicle = SurroundingVehicle(
                x=sur_x, y=sur_y, phi=sur_phi,
                rel_x=rel_x, rel_y=rel_y, rel_phi=rel_phi,
                u=u, distance=distance,
                length=length, width=width,
                veh_id=veh_id, lane_id=lane_id,
                mask=1
            )
            # print("veh_id:", veh_id)
            # if rel_phi < np.pi/2 and distance > 0.01: # TODO: 可以去掉？
            sur_context.append(sur_vehicle)

        # sur_context.extend(SurroundingVehicle()
        #                    for _ in range(self.surr_veh_num - len(sur_context)))
        return sur_context

    def convert_map(self, scenario_id: str):
        link_nav = self.simulator.get_vehicle_navigation_info(self.ego_id).navigation_info.link_nav
        print(f"link_nav: {link_nav}")
        traffic_map = self.map_dict[scenario_id]
        for segment in traffic_map.data.segments:
            for link in segment.ordered_links:
                print(f"processing {link.id}.")
                if not link.id in link_nav:
                    print(f"link {link.id} is not in {link_nav}, not adding lanes. continue.")
                    continue
                for lane in link.ordered_lanes:
                    # print("lane: ", lane)
                    self.lanes[lane.id] = lane

    def get_remote_hdmap(self, scenario_id: str, version: str):
        return self.qx_client.resources.get_hd_map(scenario_id, version)

    def get_remote_lasvsim_test_veh_list(self):
        return self.simulator.get_test_vehicle_id_list()

    def set_remote_lasvsim_veh_control(self, real_action: np.ndarray):
        lon_acc, ste_wheel = real_action
        return self.simulator.set_vehicle_control_info(self.ego_id, ste_wheel, lon_acc)

    def get_remote_lasvsim_veh_position(self):
        return self.simulator.get_vehicle_position([self.ego_id])

    def get_remote_lasvsim_veh_base_info(self):
        return self.simulator.get_vehicle_base_info([self.ego_id])

    def get_remote_lasvsim_veh_moving_info(self):
        return self.simulator.get_vehicle_moving_info([self.ego_id])

    def get_remote_lasvsim_ref_line(self):
        return self.simulator.get_vehicle_reference_lines(self.ego_id)

    def get_remote_lasvsim_perception_info(self):
        return self.simulator.get_vehicle_perception_info(self.ego_id)