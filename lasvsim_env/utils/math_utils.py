import torch
from torch import Tensor
from typing import Tuple
import math
import numpy as np

def angle_normalize(x):
    return ((x + math.pi) % (2 * math.pi)) - math.pi

def deal_with_phi_rad(phi: float):
    return (phi + math.pi) % (2 * math.pi) - math.pi

def convert_ref_to_ego_coord(ref_obs_absolute, ego_state): # rewrite this function in numpy
    ref_x, ref_y, ref_phi = ref_obs_absolute[:,0], ref_obs_absolute[:,1], ref_obs_absolute[:,2]
    x, y, phi = ego_state[0], ego_state[1], ego_state[4]
    ref_x_ego_coord, ref_y_ego_coord, ref_phi_ego_coord = convert_ground_coord_to_ego_coord(
        ref_x, ref_y, ref_phi, x, y, phi)
    return ref_x_ego_coord, ref_y_ego_coord, ref_phi_ego_coord

def inverse_normalize_action(action: np.array, 
                             action_half_range: np.array,
                             action_center: np.array) -> np.array:
    action = action * action_half_range + action_center
    return action

def cal_dist(x1, y1, x2, y2):
    return ((x1-x2)**2+(y1-y2)**2)**0.5

# get the indices of the smallest k elements
def get_indices_of_k_smallest(arr, k):
    idx = np.argpartition(arr, k)
    return idx[:k]

def calculate_perpendicular_points(x0, y0, direction_radians, distance):
    dx = -math.sin(direction_radians)
    dy = math.cos(direction_radians)

    x1 = x0 + distance * dx
    y1 = y0 + distance * dy
    x2 = x0 - distance * dx
    y2 = y0 - distance * dy

    return (x1, y1), (x2, y2)

def convert_ground_coord_to_ego_coord(x, y, phi, ego_x, ego_y, ego_phi):
    shift_x, shift_y = shift(x, y, ego_x, ego_y)
    x_ego_coord, y_ego_coord, phi_ego_coord \
        = rotate(shift_x, shift_y, phi, ego_phi)
    return x_ego_coord, y_ego_coord, phi_ego_coord

def convert_ego_coord_to_ground_coord(x, y, phi, ego_x, ego_y, ego_phi):
    shift_x, shift_y, phi_ground_coord = rotate(x, y, phi, -ego_phi)
    x_ground_coord, y_ground_coord = shift(shift_x, shift_y, -ego_x, -ego_y)
    return x_ground_coord, y_ground_coord, phi_ground_coord


def shift(orig_x, orig_y, shift_x, shift_y):
    shifted_x = orig_x - shift_x
    shifted_y = orig_y - shift_y
    return shifted_x, shifted_y


def rotate(orig_x, orig_y, orig_phi, rotate_phi):
    rotated_x = orig_x * math.cos(rotate_phi) + orig_y * math.sin(rotate_phi)
    rotated_y = -orig_x * math.sin(rotate_phi) + \
                orig_y * math.cos(rotate_phi)
    rotated_phi = deal_with_phi_rad(orig_phi - rotate_phi)
    return rotated_x, rotated_y, rotated_phi


def cal_curvature(x1, y1, x2, y2, x3, y3):
    # cal curvature by three points in batch format
    # dim of x1 is [batch, 1]
    a = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    b = torch.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    c = torch.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    k = torch.zeros_like(x1)
    i = (a * b * c) != 0
    # area > 0 for left turn, area < 0 for right turn
    area = x1[i] * (y2[i] - y3[i]) + x2[i] * \
           (y3[i] - y1[i]) + x3[i] * (y1[i] - y2[i])
    k[i] = 2 * area / (a[i] * b[i] * c[i])
    return k

def ego_predict_model(ego_state: np.ndarray,
                      action: np.ndarray,
                      Ts: float,
                      vehicle_spec: tuple) -> np.ndarray:
    # parameters
    m, Iz, lf, lr, Cf, Cr, vx_max, vx_min = vehicle_spec
    x, y, vx, vy, phi, omega = ego_state
    ax, steer = action

    return np.array([
        x + Ts * (vx * np.cos(phi) - vy * np.sin(phi)),
        y + Ts * (vy * np.cos(phi) + vx * np.sin(phi)),
        np.clip(vx + Ts * ax, vx_min, vx_max),
        (-(lf * Cf - lr * Cr) * omega + Cf * steer * vx + m * omega * vx * vx - m * vx * vy / Ts) / (Cf + Cr - m * vx / Ts),
        phi + Ts * omega,
        (-Iz * omega * vx / Ts - (lf * Cf - lr * Cr) * vy + lf * Cf * steer * vx) / (
            (lf * lf * Cf + lr * lr * Cr) - Iz * vx / Ts)
    ])

def convert_ego_coord_to_ground_coord(rel_x, rel_y, rel_phi, ego_x, ego_y, ego_phi):
    delta_x, delta_y, phi = rotate(rel_x, rel_y, rel_phi, -ego_phi)
    x, y = shift(delta_x, delta_y, -ego_x, -ego_y)
    return x, y, phi