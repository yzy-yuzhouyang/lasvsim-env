import os
import random
from shapely import segmentize, simplify
from collections import deque
from typing import Any, Dict, Tuple, List, Deque
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from gops.utils.python_timer import Timeit, timeit
    
from lasvsim_openapi.client import Client
from lasvsim_openapi.http_client import HttpConfig
from lasvsim_openapi.simulator_model import SimulatorConfig, Point as QxPoint
from lasvsim_env.utils.lib import \
    point_project_to_line, compute_waypoints_by_intervals, compute_waypoint, create_box_polygon
from lasvsim_env.utils.math_utils import \
    deal_with_phi_rad, convert_ref_to_ego_coord, \
    inverse_normalize_action, cal_dist, \
    get_indices_of_k_smallest, convert_ground_coord_to_ego_coord
from lasvsim_env.dataclass import EgoVehicle, SurroundingVehicle, LasVSimContext
from lasvsim_env.traj_processor import compute_intervals, compute_intervals_in_junction, compute_intervals_initsegment_green, compute_intervals_initsegment_red

def add_map_objs(line_string, map_objs, max_speed, obj_type, simplify_tol=0.2):
    """
    Add segmentized line_string to map_objs.
    Args:
        line_string: shapely.LineString.
        map_objs: list.
        obj_type: one-hot list, e.g. [0, 0, 1, 0, 0, 0] for center lanes.
        simplify_tol: float, tolerance for simplifying the line_string.
    """

    if simplify_tol is not None:
        line_string = simplify(line_string, tolerance=simplify_tol)
    line_string = segmentize(line_string, max_segment_length=5.0)
    default_light_status = [0, 0, 1]

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

NAVI_UNKNOWN  = np.array([1, 0, 0, 0, 0])
NAVI_STRAIGHT = np.array([0, 1, 0, 0, 0])
NAVI_LEFT     = np.array([0, 0, 1, 0, 0])
NAVI_RIGHT    = np.array([0, 0, 0, 1, 0])
NAVI_UTURN    = np.array([0, 0, 0, 0, 1])

def get_nav_obs_by_flow_direction(flow_direction):
    if flow_direction == 0:
        return NAVI_UNKNOWN
    elif flow_direction == 1:
        return NAVI_STRAIGHT
    elif flow_direction == 2:
        return NAVI_LEFT
    elif flow_direction == 3:
        return NAVI_RIGHT
    elif flow_direction == 4:
        return NAVI_UTURN
    else:
        print(f"Error: unknown flow direction: {flow_direction}")
        return NAVI_UNKNOWN

class LasvsimEnv():
    def __init__(
        self,
        token: str,
        env_config: Dict = {},
        task_id=None,
        env_idx=0,
        is_testing: bool = False,
        server_host: str = "",
        **kwargs: Any,
    ):
        assert task_id is not None, "None task id"

        # ================== 1. Build a connection ==================
        self.qx_client = Client(HttpConfig(
            endpoint=server_host,  # 接口地址
            token=token,  # 授权token
        ))

        # ================== 2. Create simulator ==================
        if not is_testing: # 训练环境
            scene_list = self.qx_client.train_task.get_scene_id_list(task_id)
            scenario_list = scene_list["scene_id_list"]
            version_list = scene_list["scene_version_list"]
            assert len(scenario_list) == len(version_list), "Error: scenario_list and version_list have different lengths."

            scenario_id = scenario_list[env_idx % len(scenario_list)]
            scenario_version = version_list[env_idx % len(scenario_list)]

            self.simulator = self.init_remote_lasvsim(
                scenario_id=scenario_id,
                scenario_version=scenario_version
            )
        else: # 测试环境
            print("initializing test environment...")
            record_ids = self.qx_client.process_task.get_task_record_ids(task_id)["record_ids"]
            scen_ids = [self.qx_client.process_task.get_record_scenario(task_id, record_id)["scen_id"] for record_id in record_ids]
            # remove the duplicated scen_id, and get the index of the non-duplicated scenarios in the original list
            unique_scen_ids, unique_scen_indices = np.unique(scen_ids, return_index=True)
            # randomly select one scenario
            random_index = random.randint(0, len(unique_scen_indices) - 1)
            unique_scen_indice = unique_scen_indices[random_index]
            record_id = record_ids[unique_scen_indice]
            print(f"randomly select scenario: {unique_scen_ids[random_index]}, record_id: {record_id}")

            new_record = self.qx_client.process_task.copy_record(task_id, record_id)
            
            scenario_id = new_record["scen_id"]
            scenario_version = new_record["scen_ver"]

            self.record_id = new_record["new_record_id"]      # useful in other functions
            self.sim_record_id = new_record["sim_record_id"]  # useful in other functions

            self.simulator = self.init_remote_lasvsim(
                scenario_id=scenario_id,
                scenario_version=scenario_version,
                sim_record_id=self.sim_record_id,
            )
        print(f"env_idx: {env_idx}, scenario_id: {scenario_id}", flush=True)
            
        # ================== 3. Init variables ==================
        self.config = env_config
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

        self.ego_dim = self.config['obs_dict']['ego']
        self.sur_num = self.config['obs_dict']['sur_num']
        self.sur_horizon = self.config['obs_dict']['sur_history_length']
        self.sur_dim = self.config['obs_dict']['sur_dim'] # 11: x, y, cosphi, sinphi, speed, length, width, type(4)
        self.map_vec_num = self.config['obs_dict']['map_vec_num']
        self.vec_dim = self.config['obs_dict']['map_vec_dim'] # 16: x, y, length, width, cosphi, sinphi, max_speed, type(6), light(3)
        self.nav_dim = self.config['obs_dict']['navi']
        self.obs_dim = (self.ego_dim + 
                        self.sur_num * self.sur_horizon * self.sur_dim + 
                        self.map_vec_num * self.vec_dim + 
                        self.nav_dim)
        
        # init ego vehicle
        test_vehicle_list = self.get_remote_lasvsim_test_veh_list()["list"]
        assert len(test_vehicle_list) == 1, "Error: Only one test vehicle is allowed currently."
        self.ego_id = test_vehicle_list[0]
        print(f"ego_id: {self.ego_id}")

        self.lasvsim_context = LasVSimContext(
            ego=EgoVehicle(),
            ref_list=[],
            sur_list=[]
        )
        self.history_sur_veh: Deque = deque([[] for _ in range(self.sur_num)], maxlen=self.sur_num)
        self.can_not_get_lane_id = False
        self.global_link_nav = self.get_ego_navigation_info()

        # ================== 4. Process static map and surroundings ==================
        self.movement_id_to_direction = {}
        self.qx_map = self.get_remote_hdmap(scenario_id, scenario_version)

        self.laneid2lane = {}
        self.linkid2link = {}
        self.convert_map(self.qx_map)
        # print("len(self.laneid2lane): ", len(self.laneid2lane))
        
        ROAD_EDGE   = [1, 0, 0, 0, 0, 0]
        LINE_EDGE   = [0, 1, 0, 0, 0, 0]
        CENTER_LINE = [0, 0, 1, 0, 0, 0]
        STOP_LINE   = [0, 0, 0, 1, 0, 0]
        ZEBRA       = [0, 0, 0, 0, 1, 0]
        VIRTUAL     = [0, 0, 0, 0, 0, 1]

        total_count = 0
        map_objs = []
        linkid2map_obj = {}
        movementid2map_obj = {}
        map_objs_in_junction = [] # index of map_objs indicating whether a map_obj belongs to a junction
        map_objs_is_center_line = [] # index of map_objs indicating whether a map_obj is a center line
        for seg in self.qx_map["data"]["segments"]:
            for link in seg["ordered_links"]:
                # 左右道路边界
                linestring = LineString([(p.get("x", 0), p.get("y", 0)) for p in link["left_boundary"]["points"]]) # TODO: 这里不应该默认给0，等待接口修复
                count = add_map_objs(linestring, map_objs, max_speed=0.0, obj_type=ROAD_EDGE) # 道路边界线
                linestring = LineString([(p.get("x", 0), p.get("y", 0)) for p in link["right_boundary"]["points"]]) # TODO: 这里不应该默认给0，等待接口修复
                count += add_map_objs(linestring, map_objs, max_speed=0.0, obj_type=ROAD_EDGE) # 道路边界线
                
                # 对每个车道
                for i, lane in enumerate(link["ordered_lanes"]):
                    # print(f"processsing lane {i} "+ lane["id"])
                    lane_type = lane.get("type", 0) # 0: unknown, 1: 机动车道， 2: 非机动车道， 3: 人行道 # TODO: 这里不该默认给0，等待接口修复
                    if lane_type == 0:
                        continue
                    elif lane_type == 1:
                        # 添加车道中心线
                        linestring = LineString([(p["point"]["x"], p["point"]["y"]) for p in lane["center_line"]])
                        _count = add_map_objs(linestring, map_objs, max_speed=self.config["max_speed"], obj_type=CENTER_LINE)
                        map_objs_is_center_line.extend(list(range(len(map_objs) - _count, len(map_objs))))
                        count += _count
                    elif lane_type == 2:
                        continue
                    elif lane_type == 3:
                        continue

                    # 添加车道线
                    if i > 0 and link["ordered_lanes"][i-1].get("type", 0) == 0: # 如果是第一条机动车道，则加入左侧车道线 # TODO: 这里不该默认给0，等待接口修复
                        linestring = LineString([(p["point"]["x"] - p["left_width"] * np.sin(p.get("heading", 0)), # TODO: 这里不该默认给0，等待接口修复
                                                  p["point"]["y"] + p["left_width"] * np.cos(p.get("heading", 0))) # TODO: 这里不该默认给0，等待接口修复
                                                  for p in lane["center_line"]])
                        count += add_map_objs(linestring, map_objs, max_speed=0.0, obj_type=LINE_EDGE)
                    
                    # 其他情况，加入右侧车道线
                    linestring = LineString([(p["point"]["x"] - p["right_width"] * np.sin(p.get("heading", 0)), # TODO: 这里不该默认给0，等待接口修复
                                              p["point"]["y"] + p["right_width"] * np.cos(p.get("heading", 0))) # TODO: 这里不该默认给0，等待接口修复
                                              for p in lane["center_line"]])
                    count += add_map_objs(linestring, map_objs, max_speed=0.0, obj_type=LINE_EDGE)

                    # 停止线
                    if "stopline" in lane.keys():
                        linestring = LineString([(p["x"], p["y"]) for p in lane["stopline"]["shape"]["points"]])
                        count += add_map_objs(linestring, map_objs, max_speed=0.0, obj_type=STOP_LINE)
                    
                    # print(f"finish adding lane {lane.id} with {count} vectors.")
                if link["id"] in linkid2map_obj.keys():
                    linkid2map_obj[link["id"]].extend(list(range(len(map_objs) - count, len(map_objs))))
                else:
                    linkid2map_obj[link["id"]] = list(range(len(map_objs) - count, len(map_objs)))
                total_count += count

        for junc in self.qx_map["data"]["junctions"]:
            if junc["type"] == 1:
                continue
            elif junc["type"] == 2:
                # 所有movements
                for movement in junc.get("movements", {}):
                    if movement["id"] in self.movement_id_to_direction.keys():
                        # print(f"Error: duplicated movement id: {movement['id']}")
                        pass
                    self.movement_id_to_direction[movement["id"]] = movement["flow_direction"]

                # 路口连接线
                for connection in junc.get("connections", {}):
                    linestring = LineString([(p['x'], p['y']) for p in connection["path"]["points"]])
                    count = add_map_objs(linestring, map_objs, max_speed=self.config["max_speed"], obj_type=CENTER_LINE, simplify_tol=0.2)
                    if connection["movement_id"] in movementid2map_obj.keys():
                        movementid2map_obj[connection["movement_id"]].extend(list(range(len(map_objs) - count, len(map_objs))))
                    else:
                        movementid2map_obj[connection["movement_id"]] = list(range(len(map_objs) - count, len(map_objs)))
                    map_objs_in_junction.extend(list(range(len(map_objs) - count, len(map_objs))))
                    total_count += count
                    # count += add_map_objs(linestring, map_objs, max_speed=6.0, obj_type=CENTER_LINE)
                
                # 人行道
                for crosswalk in junc.get("crosswalks", {}):
                    xs = np.array([p['x'] for p in crosswalk['shape']['points']])
                    ys = np.array([p['y'] for p in crosswalk['shape']['points']])
                    xys = np.array([(p['x'], p['y']) for p in crosswalk['shape']['points']])
                    a = np.linalg.norm(xys[1] - xys[0])
                    b = np.linalg.norm(xys[2] - xys[1])
                    if a > b:
                        l = a / 2  # always use the longer side as length
                        w = b / 2
                        orientation = np.arctan2(ys[1] - ys[0], xs[1] - xs[0])  # use the direction of the longer side
                    else:
                        l = b / 2
                        w = a / 2
                        orientation = np.arctan2(ys[2] - ys[1], xs[2] - xs[1])
                    orientation += np.pi if orientation < 0 else 0  # make sure the orientation is in [0, pi)
                    map_objs.append([
                        xs.mean(), ys.mean(), l, w, 
                        np.cos(orientation), np.sin(orientation), 0.0,
                        *ZEBRA,
                        0, 0, 1  # default light status
                    ])
                    total_count += 1

        assert total_count == len(map_objs), f"Error: total_count={total_count}, len(map_objs)={len(map_objs)}"
        assert len(map_objs_in_junction) == sum([len(idx) for idx in movementid2map_obj.values()]), f"Error: len(map_objs_in_junction)={len(map_objs_in_junction)}, sum([len(movementid2map_obj[m]) for m in movementid2map_obj.keys()])={sum([len(movementid2map_obj[m]) for m in movementid2map_obj.keys()])}"

        self.map_objs = np.array(map_objs)
        map_len = len(self.map_objs)

        self.movementid2map_obj = {}
        is_movement = np.zeros((len(movementid2map_obj), map_len), dtype=bool)
        for idx, value in enumerate(movementid2map_obj.values()):
            is_movement[idx, value] = True
        self.movementid2map_obj = {key: is_movement[idx] for idx, key in enumerate(movementid2map_obj.keys())}

        self.linkid2map_obj = {}
        is_link = np.zeros((len(linkid2map_obj), map_len), dtype=bool)
        for idx, value in enumerate(linkid2map_obj.values()):
            is_link[idx, value] = True
        self.linkid2map_obj = {key: is_link[idx] for idx, key in enumerate(linkid2map_obj.keys())}
        
        self.map_objs_in_junction = np.zeros(map_len, dtype=bool)
        self.map_objs_in_junction[map_objs_in_junction] = True

        self.map_objs_is_center_line = np.zeros(map_len, dtype=bool)
        self.map_objs_is_center_line[map_objs_is_center_line] = True
        # print(f"finish initializing self.map_objs with shape: {self.map_objs.shape}.")
        
        self.surrounding_deque=deque([[] for _ in range(10)], maxlen=10)
        
        # 设置一些全局变量，每步更新，在obs和reward计算中都会用到
        veh_base = self.simulator.get_vehicle_base_info([self.ego_id])["info_dict"][self.ego_id]["base_info"]
        self.ego_length , self.ego_width = veh_base["length"],veh_base["width"]
        self.pos_info = None
        self.moving_info = None
        self.bound_info = None
        self.perception_info = None
        self.reference_info = None
        self.nav_info = None
        self.step_info = None
        self.collision_info = None

    def init_remote_lasvsim(self, scenario_id: str, scenario_version: str, sim_record_id: str = None):
        # print(f"[LasvsimEnv] init_remote_lasvim with scenario_id={scenario_id} and version={scenario_version}...")
        return self.qx_client.init_simulator_from_config(SimulatorConfig(
            scen_id=scenario_id,
            scen_ver=scenario_version,
            sim_record_id=sim_record_id,
        ))

    def update_lasvsim_context(self, real_action: np.ndarray = None):
        ego_context = self.get_ego_context(real_action)
        ref_context = self.get_ref_context(ego_context)
        sur_context = self.get_sur_context(ego_context, ref_context)
        
        # Update history_sur_veh
        self.history_sur_veh.append(sur_context)
        self.history_sur_veh[-1].sort(key=SurroundingVehicle.distance_key)
        
        self.lasvsim_context = LasVSimContext(
            ego=ego_context,
            ref_list=ref_context,
            sur_list=sur_context
        )
    
    def update_step_info(self,step_info):
        self.pos_info = step_info["position"]
        self.moving_info = step_info["moving_info"]
        self.bound_info = step_info["dis_to_link_boundary"]
        self.perception_info = step_info["perception_infos"]
        self.reference_info = step_info["reference_lines"]
        self.nav_info = step_info["navigation_info"]
        self.step_info = step_info["step_res"]
        self.collision_info = step_info["collision_status"]

    def get_all_ref_param(self, ref_linestring: List[LineString]) -> np.ndarray:
        # return: ref_param [VARIABLE_NUM, ref_horizon, per_point_dim]
        ref_horizon = self.config["ref_horizon"]
        ref_list = ref_linestring
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
        
        obs = np.zeros(
            self.ego_dim +
            self.sur_num * self.sur_horizon * self.sur_dim +
            self.map_vec_num * self.vec_dim +
            self.nav_dim,
        )
        
        # -------------- 1.自车观测更新 -------------- 
        obs[0:5] = [self.lasvsim_context.ego.u,
                    self.lasvsim_context.ego.v,
                    self.lasvsim_context.ego.w,
                    self.lasvsim_context.ego.action[0],
                    self.lasvsim_context.ego.action[1]]
        
        # -------------- 2.周车观测更新 -------------- 
        all_sur_veh_obs = np.zeros((self.sur_num, self.sur_horizon, self.sur_dim), dtype=np.float32)

        # latest surrounding vehicles
        veh_id_list = []
        for i, sur_veh in enumerate(self.lasvsim_context.sur_list):
            if i >= self.sur_num: 
                break
            veh_id_list.append(sur_veh.veh_id)
        
        # assert there is no duplicated element in veh_id_list
        assert len(veh_id_list) == len(set(veh_id_list))
        
        # padding in whole horizon L
        all_sur_veh_obs[len(veh_id_list):, :, 10] = 1

        # history surrounding vehicles
        for j in range(self.sur_horizon):
            for sur_veh in self.history_sur_veh[-1-j]:
                if sur_veh.veh_id in veh_id_list:
                    idx = veh_id_list.index(sur_veh.veh_id)
                    # print(f"in j={j}, idx={idx}, add sur {sur_veh.veh_id}, rel_x: {sur_veh.rel_x:10.2f}, rel_y: {sur_veh.rel_y:10.2f}, rel_phi: {sur_veh.rel_phi:10.2f}")
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
        
        obs[self.ego_dim : self.ego_dim + self.sur_num * self.sur_horizon * self.sur_dim] = all_sur_veh_obs.ravel()


        # -------------- 3.地图观测更新 -------------- 
        ego_x, ego_y, ego_phi = (self.lasvsim_context.ego.x, 
                                 self.lasvsim_context.ego.y, 
                                 self.lasvsim_context.ego.phi)
        ego_center = np.array([ego_x, ego_y])

        # 将ego_center按照自车速度向前递推2s
        ego_u = self.lasvsim_context.ego.u
        ego_center = ego_center + ego_u * np.array([np.cos(ego_phi), np.sin(ego_phi)]) * 2.0

        map_objs = self.map_objs.copy()
        # Update light status for map objs
        map_objs = self.update_light_status(map_objs)

        # Ignore objects in the junction that do not correspond to the current movement_id
        movement_id = self.lasvsim_context.ego.movement_id
        if movement_id is not None and movement_id != "":
            movement_idx = self.movementid2map_obj[movement_id]
        else:
            movement_idx = np.zeros_like(self.map_objs_in_junction, dtype=bool)
        ignore_indices = np.logical_and(self.map_objs_in_junction, ~movement_idx)
        navi_map_objs = map_objs[~ignore_indices]

        if len(navi_map_objs) < self.map_vec_num:
            navi_map_objs = np.concatenate([navi_map_objs, np.zeros((self.map_vec_num - len(navi_map_objs), self.vec_dim))], axis=0)
            # Set mask to 1 for padded objects
            navi_map_objs[navi_map_objs.shape[0]:, -4] = 1 # FIXME: hard code
            selected_objs = navi_map_objs
        else:
            # Calculate distances to ego center
            obj_centers = navi_map_objs[:, :2]
            distances_square = np.sum((obj_centers - ego_center) ** 2, axis=1)
            # Use partition to find indices of self.map_vec_num nearest objects 
            selected_indices = np.argpartition(distances_square, self.map_vec_num)[:self.map_vec_num]
            sorted_indices = selected_indices[np.argsort(distances_square[selected_indices])]
            # Select the updated objects           
            selected_objs = navi_map_objs[sorted_indices]
        
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

        obs[self.ego_dim + self.sur_num * self.sur_horizon * self.sur_dim:
            self.obs_dim - self.nav_dim] = transformed_objs.ravel()

        # -------------- 4.导航观测更新 --------------
        if self.nav_dim > 0:
            obs[self.obs_dim - self.nav_dim] = np.clip(self.lasvsim_context.ego.dis_to_next_junction, 0, 200) / 200.0
            flow_direction = self.lasvsim_context.ego.flow_direction
            obs[self.obs_dim - self.nav_dim + 1:] = get_nav_obs_by_flow_direction(flow_direction)
            
        return obs

    def update_light_status(self, map_objs):
        """
        更新 self.map_objs 中的信号灯状态。
        1. 路口中的所有对象的信号灯状态根据实际信号灯状态更新；
        2. 导航车道中心线的信号灯状态更新为绿灯。
        """
        # 偏离路口就没有movement_id
        movement_id = self.lasvsim_context.ego.movement_id

        if movement_id is not None and movement_id != "" and movement_id != "default":
            # 获取信号灯状态
            light_status = self.lasvsim_context.ego.traffic_light

            # 根据 light_status 设置对应的 one-hot 向量
            if light_status == "green":
                one_hot_vector = np.array([1, 0, 0]) # 绿灯
            elif light_status == "red" or light_status == "yellow":
                one_hot_vector = np.array([0, 1, 0]) # 红灯或黄灯
            elif light_status == "unknown":
                one_hot_vector = np.array([0, 0, 1])
            else:
                raise ValueError(f"Unknown light status: {light_status}")
            
            # 查找 movementid2map_obj 中的对应索引并更新信号灯状态
            if movement_id in self.movementid2map_obj:
                movement_idx = self.movementid2map_obj[movement_id]  # 获取所有与 movement_id 相关的索引
                map_objs[movement_idx, 13:16] = one_hot_vector[np.newaxis, :]  # 14 到 16 维存储信号灯状态

        for link in self.nav_info["link_nav"]:
            if link in self.linkid2map_obj:
                link_idx = self.linkid2map_obj[link]
                indices = np.logical_and(link_idx, self.map_objs_is_center_line)
                map_objs[indices, 13:16] = np.array([[1, 0, 0]])
            else:
                raise ValueError(f"Error: link {link} not in linkid2map_obj.")
            
        return map_objs
            
    def step(self, delta_action: np.ndarray):
        # action: network output, \in [-1, 1]
        self.alive_step += 1

        last_action = self.lasvsim_context.ego.action
        real_action = self.get_real_action(delta_action, last_action)

        step_info = self.simulator.idc_step(self.ego_id,real_action[1],real_action[0],ref_limit=40.0)

        self.update_step_info(step_info)
        self.update_lasvsim_context(real_action)

        reward, rew_info = self.reward_function_multilane()

        obs = self.get_obs_from_context()

        terminated, truncated, done_info = self.judge_done(step_info["step_res"])

        # if terminated or truncated:
        #     print(f"alive step: {self.alive_step:4d}, done info: {[event for event in done_info if done_info[event]]}")
        
        return obs, reward, terminated, truncated, {**rew_info, **done_info, "event_alive_step": self.alive_step, "event_qx_error": 0}

    def reset(self, reset_traffic_flow: bool = False):
        self.alive_step = 0
        random_link_nav = self.global_link_nav[np.random.choice(len(self.global_link_nav)):]
        reset_vehicle = [{"link_path": random_link_nav, "vehicle_id": self.ego_id}] if reset_traffic_flow else []
        self.reset_remote_lasvsim(reset_traffic_flow, reset_vehicle)
        res = self.step_remote_lasvsim(0.0, 0.0)
        self.update_step_info(res)

        # 速度和位置的随机初始化
        random_init_v = np.random.uniform(self.config["reset_v_min"], self.config["reset_v_max"])
        self.set_ego_speed(random_init_v)

        random_offset_x = np.random.normal(0.0, 1.0)
        random_offset_y = np.random.normal(0.0, 1.0)
        random_offset_phi = np.random.normal(0.0, 0.1)

        self.pos_info["point"]["x"] += random_offset_x
        self.pos_info["point"]["y"] += random_offset_y
        self.pos_info["phi"] += random_offset_phi

        self.set_ego_position(
            self.pos_info["point"]["x"],
            self.pos_info["point"]["y"],
            self.pos_info["phi"]
        )

        self.update_lasvsim_context()
        obs = self.get_obs_from_context()
        info = {}
        return obs, info

    def get_closest_ref_point(self, ref_param: List[LineString]) -> np.ndarray:
        ego= self.lasvsim_context.ego
        position_on_ref_list = [point_project_to_line(ref_line, ego.x, ego.y) for ref_line in ref_param]
        ref_state_list = [compute_waypoint(ref_line, position_on_ref) for ref_line, position_on_ref in zip(ref_param, position_on_ref_list)] # [R, 3]
        return np.array(ref_state_list) # [R, 3]

    # from rlplanner
    def get_reward(self, ref_param: List[LineString]) -> Tuple[List[np.ndarray], List[dict]]:
        # all inputs are batched
        ego= self.lasvsim_context.ego

        ego_state = (ego.x, ego.y, ego.u, ego.v, ego.phi, ego.w)
        ego_x, ego_y, ego_vx, ego_vy, ego_phi, ego_r = ego_state

        last_acc, last_steer = ego.action[0], ego.action[1] * 180 / np.pi
        last_last_acc, last_last_steer = ego.last_action[0], ego.last_action[1] * 180 / np.pi
        delta_steer = (last_steer - last_last_steer) / self.config["dt"]
        jerk = (last_acc - last_last_acc) / self.config["dt"]

        # Note: ref_param is fixed during the planning process, but self.lasvsim_context.ref_list is updated every step
        ref_num = len(ref_param)
        ref_x, ref_y, ref_phi = self.get_closest_ref_point(ref_param).T
        ref_v = np.repeat(self.config["max_speed"], ref_num) # (R, )
        next_ref_v = ref_v

        # live reward
        rew_step = np.ones(ref_num)  # 0~1

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

        ego_r = ego_r * np.ones(ref_num)
        punish_yaw_rate = 0.1 * np.where(
            np.abs(ego_r) < 2,
            np.square(ego_r),
            np.abs(ego_r) + 2,
        )  # 0~1  0~8 degree/s 50% 0~2 degree/s

        punish_overspeed = np.zeros(ref_num)
        index_lowspeed = ego_vx < ref_v
        punish_overspeed[index_lowspeed] = 2 * (1 - ego_vx / ref_v[index_lowspeed])
        index_overspeed = ego_vx > 1.1 * ref_v
        punish_overspeed[index_overspeed] = (1 + ego_vx - ref_v[index_overspeed])
        punish_overspeed = np.clip(punish_overspeed, 0, 2)

        # # reward related to action
        # nominal_steer = self._get_nominal_steer_by_state_batch(
        #     ego_state, ref_param) * 180 / np.pi

        # abs_steer = np.abs(last_steer - nominal_steer)
        # reward_steering = -np.where(abs_steer < 4,
        #                             np.square(abs_steer), 2 * abs_steer + 8)

        # self.out_of_action_range = abs_steer > 20

        # if ego_vx < 0.1 and self.config["enable_slow_reward"]:
        #     reward_steering = reward_steering * 5

        reward_steering = np.zeros(ref_num) 
        
        abs_ax = np.abs(last_acc) * np.ones(ref_num)
        reward_acc_long = -np.where(abs_ax < 2, np.square(abs_ax), 2 * abs_ax)

        delta_steer = delta_steer * np.ones(ref_num)
        reward_delta_steer = - \
            np.where(np.abs(delta_steer) < 4, np.square(
                delta_steer), 2 * np.abs(delta_steer) + 8)
        jerk = jerk * np.ones(ref_num)
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
            nominal_acc = np.zeros(ref_num)
            punish_nominal_acc = np.zeros(ref_num)

        if self.braking_mode and self.config["nonimal_acc"]:
            nominal_acc = -1.5 * np.ones(ref_num)
            punish_vel_long = np.zeros(ref_num)

        if break_condition.any() or self.braking_mode:
            rew_step = np.where(break_condition, 1.0, rew_step)

        delta_acc = np.abs(nominal_acc - last_acc)
        punish_nominal_acc = (nominal_acc != 0) * np.where(delta_acc <
                                                           0.5, np.square(delta_acc), delta_acc - 0.25)

        # tracking related reward
        scaled_punish_dist_lat = punish_dist_lat * self.config["P_lat"]
        scaled_punish_head_ang = punish_head_ang * self.config["P_phi"]
        scaled_punish_yaw_rate = punish_yaw_rate * self.config["P_yaw"]
        scaled_punish_overspeed = punish_overspeed * self.config["P_overspeed"]  # TODO: hard coded value

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
             scaled_punish_head_ang +
             scaled_punish_yaw_rate +
             scaled_punish_nominal_acc +
             scaled_punish_overspeed) + \
            (scaled_reward_steering +
             scaled_reward_acc_long +
             scaled_reward_delta_steer +
             scaled_reward_jerk)

        rewards = reward_ego_state
        infos = {
            "reward_part2": reward_ego_state,
            "reward_step": scaled_rew_step,
            "reward_dist_lat": -scaled_punish_dist_lat,
            "reward_head_ang": -scaled_punish_head_ang,
            "reward_nominal_acc": -scaled_punish_nominal_acc,
            "reward_overspeed": -scaled_punish_overspeed,
            "reward_yaw_rate": -scaled_punish_yaw_rate,
            "reward_steering": scaled_reward_steering,
            "reward_acc_long": scaled_reward_acc_long,
            "reward_delta_steer": scaled_reward_delta_steer,
            "reward_jerk": scaled_reward_jerk,

            "ego_vx": np.repeat(ego_vx, ref_num),
            "ego_speed2limit": speed_error,
            "ego_abs_phi_error": np.abs(delta_phi),
            "ego_abs_tracking_error": np.abs(tracking_error),
            "ego_abs_yaw_rate": np.abs(ego_r),
            
            "action_abs_steer": np.repeat(np.abs(last_steer), ref_num),
            "action_abs_acc": np.repeat(np.abs(last_acc), ref_num),

            "action_abs_delta_steer": np.abs(delta_steer) * self.config["dt"],
            "action_abs_delta_acc": np.abs(jerk) * self.config["dt"],
        }

        return rewards, infos

    def _get_nominal_steer_by_state_batch(
            self,
            ego_state,
            ref_param: List[LineString]
        ):
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
        
        ref_param_array = self.get_all_ref_param(ref_param)

        ref_line = np.stack([ref_param_array[:, i, :]
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
        closest_idx = np.argmin([ref_line.distance(ego.polygon) for ref_line in ref_list])
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

        punish_out_of_map = self.config["P_out_of_map"]
        punish_collision = self.config["P_collision"]

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
            raise ValueError(f"Invalid punish_sur_mode: {self.config['punish_sur_mode']}")
        
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
        self.active_collision = self.collision_info and ego_vx > 0.01

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
        reward_traffic_light_violation = 0
        reward_navigation_violation = 0

        # Event reward: target reached, collision, out of driving area
        self.out_of_driving_area = self.check_out_of_driving_area()
        self.traffic_light_violation = self.check_traffic_light_violation()
        flow_direction_near_junction = self.get_direction_near_junction()

        if self.out_of_driving_area or self.out_of_range:  # out of driving area
            reward_done = - punish_out_of_map
            event_flag = 1
        elif self.active_collision:  # collision by ego vehicle
            reward_collision = - punish_collision if self.config["penalize_collision"] else 0.
            event_flag = 2
        elif self.braking_mode:  # start to brake
            event_flag = 3
        elif self.traffic_light_violation:  # traffic light violation
            reward_traffic_light_violation = - self.config["P_traffic_light_violation"]
            event_flag = 4
        elif self.navigation_violation:
            reward_navigation_violation = - punish_out_of_map # use the same reward as out of map
            event_flag = 5
        elif flow_direction_near_junction == 2:
            event_flag = 6 # NAVI_LEFT
        elif flow_direction_near_junction == 3:
            event_flag = 7 # NAVI_RIGHT
        elif flow_direction_near_junction == 4:
            event_flag = 8 # NAVI_UTURN

        reward += (reward_done + reward_collision + reward_traffic_light_violation + reward_navigation_violation)

        return reward, {
            "category": event_flag,

            "reward_part1": reward,
            "reward_done": reward_done,
            "reward_boundary": - scaled_punish_boundary,
            "reward_collision": reward_collision,
            "reward_collision_risk": - punish_collision_risk,
            "reward_traffic_light_violation": reward_traffic_light_violation,
            "rewardcomp_pun2front": scaled_pun2front,
            "rewardcomp_pun2side": scaled_pun2side,
            "rewardcomp_pun2space": scaled_pun2space,
            "rewardcomp_pun2rear": scaled_pun2rear,
        }
    
    def check_out_of_driving_area(self) -> bool:
        out_of_driving_area_flag = (self.pos_info["type"] == 3)
        return out_of_driving_area_flag

    def check_traffic_light_violation(self) -> bool:
        return (self.lasvsim_context.ego.traffic_light == "red" or \
                self.lasvsim_context.ego.traffic_light == "yellow") \
            and self.lasvsim_context.ego.dis_to_next_junction < 10

    def get_direction_near_junction(self)-> str:
        if self.lasvsim_context.ego.dis_to_next_junction > 20:
            return 0
        else:
            return self.lasvsim_context.ego.flow_direction

    def judge_done(self, res) -> bool:
        # terminated
        park_flag = (self.lasvsim_context.ego.u == 0)
        collision = self.collision_info
        out_of_defined_region = self.out_of_range
        out_of_driving_area = self.out_of_driving_area
        traffic_light_violation = self.traffic_light_violation
        navigation_violation = self.navigation_violation

        # truncated
        max_step_truncated = (self.alive_step >= self.max_step)
        success = (res["code"] == 1001) and (collision == 0)
        if res["code"] == 1001 and collision == 1:
            raise ValueError("Success and collision at the same time")

        done_info = {
            "event_pause": park_flag,
            "event_collision": collision,
            "event_regionout": out_of_defined_region,
            "event_mapout": out_of_driving_area,
            "event_max_step_truncated": max_step_truncated,
            "event_traffic_light_violation": traffic_light_violation,
            "event_navigation_violation": navigation_violation,
            "event_success": success,
        }

        terminated = collision or out_of_defined_region or out_of_driving_area or traffic_light_violation or navigation_violation
        truncated = max_step_truncated or success # the success sample will be removed from the replay buffer due to plan setting
        
        # if terminated or truncated:
        #     print(f"# DONE: {done_info}")

        return terminated, truncated, done_info

    def get_ego_context(self, real_actiton: np.ndarray = None):
        # ego_position = self.get_remote_lasvsim_veh_position()["position_dict"].get(self.ego_id)
        # ego_base_info = self.get_remote_lasvsim_veh_base_info()["info_dict"].get(self.ego_id)
        # ego_moving_info = self.get_remote_lasvsim_veh_moving_info()["moving_info_dict"].get(self.ego_id)

        x = self.pos_info["point"]["x"]
        y = self.pos_info["point"]["y"]
        phi = self.pos_info["phi"]
        junction_id = self.pos_info["junction_id"]
        lane_id = self.pos_info["lane_id"]
        link_id = self.pos_info["link_id"]
        segment_id = self.pos_info["segment_id"]
        ego_pos = self.pos_info["type"]
        in_junction = (ego_pos == 2)

        # length = ego_base_info["base_info"]["length"]
        # width = ego_base_info["base_info"]["width"]

        u = self.moving_info["u"]
        v = self.moving_info["v"]
        w = self.moving_info["w"]

        # get navigation_violation in lane
        self.can_not_get_lane_id = False
        self.navigation_violation = False
        if ego_pos == 1: # on lane
            link_nav_id = self.nav_info["link_nav"]
            curent_lane_id = [lane['id'] for lane in self.linkid2link[link_nav_id[0]]['ordered_lanes']] if len(link_nav_id) > 0 else []
            if lane_id not in curent_lane_id:
                self.navigation_violation = True
                self.can_not_get_lane_id = True
                # print(f"lane_id: {lane_id} not in lane_nav")
        else:
            self.can_not_get_lane_id = True
            # print('X'*50)
            # print('can_not_get_lane_id')
            # breakpoint()
        
        left_boundary_distance = self.bound_info["left"]
        right_boundary_distance = self.bound_info["right"]

        if left_boundary_distance < -1 or left_boundary_distance > 100.0:
            print("[Warning] left_boundary_distance: ", left_boundary_distance)
            # left_boundary_distance = 20.0
            # raise ValueError(f"left_boundary_distance: {left_boundary_distance}")
        if right_boundary_distance < -1 or right_boundary_distance > 100.0:
            print("[Warning] right_boundary_distance: ", right_boundary_distance)
            # right_boundary_distance = 20.0
            # raise ValueError(f"right_boundary_distance: {right_boundary_distance}")

        polygon = create_box_polygon(x, y, phi, self.ego_length, self.ego_width)

        if real_actiton is not None:
            action = real_actiton
        else:
            action = np.array([0.0]*2)
        state = np.array([x, y, u, v, phi, w])
        last_action = self.lasvsim_context.ego.action

        # update traffic light, movement_id, dis_to_next_junction and flow_direction
        vehicle_navigation = self.simulator.get_idc_vehicle_nav(self.ego_id)
        movement_id = vehicle_navigation["next_movement_id"]
        dis_to_next_junction = vehicle_navigation["dis_to_next_junction"]

        # 偏离路口就没有movement_id
        traffic_light = "unknown"
        flow_direction = 0
        if dis_to_next_junction is None:
            dis_to_next_junction = 200
        if movement_id is not None and movement_id != "" and movement_id != "default":
            # 0:无信号灯或信号灯损坏 | 1:红灯 | 2:绿灯 | 3:黄灯
            light_status = self.simulator.get_movement_signal(movement_id)["current_signal"]
            if light_status == 0:
                traffic_light = "unknown"
            elif light_status == 1:
                traffic_light = "red"
            elif light_status == 2:
                traffic_light = "green"
            elif light_status == 3:
                traffic_light = "yellow"
            else:
                raise ValueError(f"Invalid light status: {light_status}")

            flow_direction = self.movement_id_to_direction[movement_id]
            
        return EgoVehicle(
            x=x, y=y, phi=phi, u=u, v=v, w=w,
            length=self.ego_length, width=self.ego_width,
            action=action,
            state=state,
            last_action=last_action,
            junction_id=junction_id, lane_id=lane_id, movement_id=movement_id,
            link_id=link_id, segment_id=segment_id,
            in_junction=in_junction,
            left_boundary_distance=left_boundary_distance,
            right_boundary_distance=right_boundary_distance,
            polygon=polygon,
            traffic_light=traffic_light,
            dis_to_next_junction=dis_to_next_junction,
            flow_direction=flow_direction
        )

    def get_ref_context(self, ego_context):
        ref_lines = self.reference_info

        if len(ref_lines)==0:
            if self.navigation_violation == 0 and self.pos_info["type"] != 3:
                # raise ValueError("ref_lines is empty, but navigation_violation is False")
                print("ref_lines is empty, but navigation_violation is False")
            if not self.can_not_get_lane_id:
                lane_id = self.lasvsim_context.ego.lane_id
                target_lane = self.laneid2lane[lane_id]
                ref_line_xy = [[p["point"]["x"], p["point"]["y"]] for p in target_lane["center_line"]]
                ref_line_string = LineString(ref_line_xy)
                return [ref_line_string] * len(self.lasvsim_context.ref_list)
            else:
                return self.lasvsim_context.ref_list

        # remove the unnecessary ref lines near junction
        # Note: the index of the leftmost one is 0, and that of the rightmost one is -1
        if ego_context.dis_to_next_junction < 20 and ego_context.in_junction == 0:
            if ego_context.flow_direction == 1: # straight
                if len(ref_lines) > 3:
                    ref_lines = ref_lines[1:] # remove the leftmost line
            elif ego_context.flow_direction == 2 or ego_context.flow_direction == 4: # left or uturn
                ref_lines = [ref_lines[0]] # only keep the leftmost line
            elif ego_context.flow_direction == 3: # right
                ref_lines = [ref_lines[-1]] # only keep the rightmost line
            
        ref_context = [
            LineString([[point["x"], point["y"]] for point in ref_line["points"]]) 
            for ref_line in ref_lines
        ]
                    
        return ref_context

    def get_sur_context(self, ego_context, ref_context):
        # perception_info = self.get_remote_lasvsim_perception_info()
        around_moving_objs = self.perception_info

        ego_x, ego_y, ego_phi = ego_context.x, \
            ego_context.y, \
            ego_context.phi

        # filter neighbor vehicles for better efficiency
        distances = [
            cal_dist(
                obj["position"]["point"]["x"],
                obj["position"]["point"]["y"],
                ego_x,
                ego_y
            )
            for obj in around_moving_objs
        ]

        # sort out the smallest k distance vehicles
        if (len(distances) > self.sur_num):
            indices = get_indices_of_k_smallest(distances, self.sur_num)
        else:
            indices = range(len(distances))

        # append info of the smallest k distance vehicles
        sur_context = []
        for i in indices:
            sur_x, sur_y, sur_phi = \
                around_moving_objs[i]["position"]["point"]["x"], \
                around_moving_objs[i]["position"]["point"]["y"], \
                around_moving_objs[i]["position"]["phi"]
            rel_x, rel_y, rel_phi = convert_ground_coord_to_ego_coord(
                sur_x, sur_y, sur_phi,
                ego_x, ego_y, ego_phi
            )
            distance = distances[i]
            u = around_moving_objs[i]["moving_info"]["u"]
            length = around_moving_objs[i]["base_info"]["length"]
            width = around_moving_objs[i]["base_info"]["width"]
            veh_id = around_moving_objs[i]["obj_id"]
            lane_id = around_moving_objs[i]["position"]["lane_id"]
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
        #                    for _ in range(self.sur_num - len(sur_context)))
        return sur_context

    def convert_map(self, qx_map):
        # print(f"link_nav: {link_nav}")
        for segment in qx_map["data"]["segments"]:
            for link in segment["ordered_links"]:
                self.linkid2link[link["id"]] = link
                for lane in link["ordered_lanes"]:
                    self.laneid2lane[lane["id"]] = lane

    def get_real_action(self, delta_action: np.ndarray, last_action: np.ndarray):
        # input normalized increment action, output clipped real action
        delta_action = inverse_normalize_action(delta_action, self.action_half_range, self.action_center)
        real_action = delta_action + last_action
        real_action = np.clip(real_action, self.real_action_lower, self.real_action_upper)
        return real_action

    def get_ego_navigation_info(self):
        return self.simulator.get_vehicle_navigation_info(self.ego_id)["navigation_info"]["link_nav"]

    def reset_remote_lasvsim(self, reset_traffic_flow: bool = False, reset_vehicle: List = []):
        # print("reset_traffic_flow: ", reset_traffic_flow)
        return self.simulator.reset(reset_traffic_flow, reset_vehicle)

    def step_remote_lasvsim(self, steer, acc):
        return self.simulator.idc_step(self.ego_id, steer, acc, ref_limit=40.0)

    def stop_remote_lasvsim(self):
        return self.simulator.stop()
    
    def get_remote_hdmap(self, scenario_id: str, version: str):
        return self.qx_client.resources.get_hd_map(scenario_id, version)
    
    def set_ego_position(self, x, y, phi):
        return self.simulator.set_vehicle_position(self.ego_id, QxPoint(x, y, phi))

    def set_ego_speed(self, v):
        return self.simulator.set_vehicle_moving_info(self.ego_id, v)

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
    

if __name__ == "__main__":
    from lasvsim_env.config import get_env_config
    env = LasvsimEnv(
        token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjIwNCwib2lkIjoxMDEsIm5hbWUiOiLlvKDlt7fohb4iLCJpZGVudGl0eSI6Im5vcm1hbCIsInBlcm1pc3Npb25zIjpbXSwiaXNzIjoidXNlciIsInN1YiI6Ikxhc1ZTaW0iLCJleHAiOjE3NDE4NTk3MTMsIm5iZiI6MTc0MTI1NDkxMywiaWF0IjoxNzQxMjU0OTEzLCJqdGkiOiIyMDQifQ.MsHgmYVMBk3KJvBuVwQlmUe4CppXQeeyPtt9L99dbg0",
        env_config=get_env_config(),
        task_id=147,
        is_testing=False,
        server_host="http://172.17.0.191:8290"
    )
    env.stop_remote_lasvsim()
    
