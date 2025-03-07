import numpy as np
import matplotlib.pyplot as plt
import time

from dataclasses import dataclass
from typing import List
from numpy.typing import NDArray

@dataclass
class RefPath:
    """
    A class to represent a reference path in the map.

    Attributes:
    ----------
    path : np.ndarray
        A numpy array with shape (4, N) where 4 represents [x, y, phi, speed] and N is the number of points.
    lane_id_list : List[str]
        A list of lane IDs associated with the path.
    link_id_list : List[str]
        A list of link IDs associated with the path.
    road_id_list : List[str]
        A list of road IDs associated with the path.
    """
    path: np.array
    lane_id_list: List[str]
    link_id_list: List[str]
    road_id_list: List[str]

# 计算道路的航向角
def xy2phi(x: NDArray[np.float32], y: NDArray[np.float32]) -> NDArray[np.float32]:
    dx = np.gradient(x)
    dy = np.gradient(y)
    phi = np.arctan2(dy, dx)
    return phi

# 拼接多个路径
def concat_ref_paths(ref_paths: List[RefPath]) -> RefPath:
    path = np.concatenate([ref_path.path for ref_path in ref_paths], axis=1)
    lane_id_list = [lane_id for ref_path in ref_paths for lane_id in ref_path.lane_id_list]
    
    # 去重处理 link_id 和 road_id
    link_id_set = set()
    link_id_list = []
    for ref_path in ref_paths:
        for link_id in ref_path.link_id_list:
            if link_id not in link_id_set:
                link_id_set.add(link_id)
                link_id_list.append(link_id)

    road_id_set = set()
    road_id_list = []
    for ref_path in ref_paths:
        for road_id in ref_path.road_id_list:
            if road_id not in road_id_set:
                road_id_set.add(road_id)
                road_id_list.append(road_id)

    return RefPath(path, lane_id_list, link_id_list, road_id_list)

# 等时间间隔离散化，ref_path是待离散化的参考路径，t是时间步长，max_step是最大离散化点数，不包括起点
def path_discrete_t(ref_path: RefPath, time_step: float, max_step = 50) -> RefPath:
    path = ref_path.path
    x = path[0, :]
    y = path[1, :]
    speed = path[3, :]
    
    # Initialize new path with the first point
    new_path = [path[:, 0]]
    
    path_length = 0.0
    path_time = 0.0
    calculated_step_num = 0
    prev_x = x[0]
    prev_y = y[0]

    for i in range(1, len(x)):
        curr_x = x[i]
        curr_y = y[i]

        path_length_step = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
        path_length += path_length_step

        speed_avg = (speed[i - 1] + speed[i]) / 2
        path_time_step = path_length_step / (speed_avg + 1e-6)
        path_time += path_time_step
        # print("path_time: ", path_time)
        while path_time >= (calculated_step_num + 1) * time_step:
            calculated_step_num += 1
            last_path_time = path_time - path_time_step
            ratio = (calculated_step_num * time_step - last_path_time) / path_time_step
            
            if ratio < -0.1 or ratio > 1.1:
                raise ValueError("Ratio is not in [0, 1]")

            x_new = (1 - ratio) * prev_x + ratio * curr_x
            y_new = (1 - ratio) * prev_y + ratio * curr_y
            speed_new = (1 - ratio) * speed[i - 1] + ratio * speed[i]
            phi_new = xy2phi(np.array([x_new, curr_x]), np.array([y_new, curr_y]))[0]

            new_point = np.array([x_new, y_new, phi_new, speed_new])
            new_path.append(new_point)

            if calculated_step_num >= max_step:
                break
        
        if calculated_step_num >= max_step:
            break

        prev_x = curr_x
        prev_y = curr_y
    
    # 如果calculated_step_num < max_step，进行匀速地推
    while calculated_step_num < max_step:
        calculated_step_num += 1
        speed = new_path[-1][3]
        x_new = new_path[-1][0] + speed * time_step * np.cos(new_path[-1][2])
        y_new = new_path[-1][1] + speed * time_step * np.sin(new_path[-1][2])
        new_point = np.array([x_new, y_new, new_path[-1][2], speed])
        new_path.append(new_point)

    # 将新路径转换为 numpy 数组
    new_path = np.array(new_path).T
    new_ref_path = RefPath(
        path=new_path,
        lane_id_list=ref_path.lane_id_list,
        link_id_list=ref_path.link_id_list,
        road_id_list=ref_path.road_id_list
    )
    return new_ref_path


def nearest_point_on_segment(A, B, P):
    AP = P - A
    AB = B - A
    AB_squared = np.dot(AB, AB)
    
    if AB_squared == 0:
        # A and B are the same point
        return A
    
    t = np.dot(AP, AB) / AB_squared
    t = np.clip(t, 0, 1)
    return A + t * AB

def find_closest_foot_point(x: float, y: float, ref_path: RefPath):
    path = ref_path.path
    coarse_search_interval = 4
    fine_search_range = 5

    min_distance = float('inf')
    approximate_closest_idx = -1

    for i in range(0, path.shape[1], coarse_search_interval):
        distance = (x - path[0, i]) ** 2 + (y - path[1, i]) ** 2
        if distance < min_distance:
            min_distance = distance
            approximate_closest_idx = i

    start_idx = max(0, approximate_closest_idx - fine_search_range)
    end_idx = min(path.shape[1] - 1, approximate_closest_idx + fine_search_range)

    closest_point = None
    closest_idx = -1
    min_distance = float('inf')

    for i in range(start_idx, end_idx):
        A = path[:2, i]
        B = path[:2, i + 1]
        P = np.array([x, y])
        point_on_segment = nearest_point_on_segment(A, B, P)
        distance = np.sum((P - point_on_segment) ** 2)
        if distance < min_distance:
            min_distance = distance
            closest_point = point_on_segment
            closest_idx = i

    return closest_point, closest_idx

def path_cut(ref_path: RefPath, start_x: float, start_y: float) -> RefPath:
    path = ref_path.path
    speed = path[3, :]

    closest_point, start_index = find_closest_foot_point(start_x, start_y, ref_path)

    # Convert closest_point to the same dimension as path
    closest_phi = xy2phi(np.array([closest_point[0], path[0, start_index + 1]]), 
                         np.array([closest_point[1], path[1, start_index + 1]]))[0]
    closest_speed = speed[start_index]
    closest_point_full = np.array([closest_point[0], closest_point[1], closest_phi, closest_speed]).reshape(4, 1)

    # Concatenate the closest point with the rest of the path
    cut_path = np.concatenate((closest_point_full, path[:, start_index + 1:]), axis=1)

    new_ref_path = RefPath(
        path=cut_path,
        lane_id_list=ref_path.lane_id_list,
        link_id_list=ref_path.link_id_list,
        road_id_list=ref_path.road_id_list
    )
    return new_ref_path

if __name__ == "__main__":
    x1 = np.array([0, 5, 10])
    y1 = np.array([0, 2, 0])
    speed1 = np.array([5, 5, 5])
    path1 = np.vstack((x1, y1, xy2phi(x1, y1), speed1))
    ref_path_1 = RefPath(path1, ["lane1"], ["link1"], ["road1"])

    x2 = np.array([10, 15, 20])
    y2 = np.array([0, 2, 0])
    speed2 = np.array([5, 5, 5])
    path2 = np.vstack((x2, y2, xy2phi(x2, y2), speed2))
    ref_path_2 = RefPath(path2, ["lane2"], ["link2"], ["road1"])

    # 测试 concat_ref_paths
    ref_path = concat_ref_paths([ref_path_1, ref_path_2])
    print("ref_path: ", ref_path)

    # 测试 path_discrete_t
    time_step = 0.1
    new_ref_path = path_discrete_t(ref_path, time_step)
    print("new_ref_path size: ", new_ref_path.path.shape)
    print("path x: ", new_ref_path.path[0])
    print("path y: ", new_ref_path.path[1])
    plt.scatter(ref_path.path[0], ref_path.path[1], c='r')
    plt.scatter(new_ref_path.path[0], new_ref_path.path[1], c='b', s=10)
    plt.savefig("path_discrete_t.png", dpi=300)
