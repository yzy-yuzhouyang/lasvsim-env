import numpy as np
from functools import partial
from typing import Tuple, Dict, List, Optional, Any

def compute_intervals_in_junction(total_num: int,
                      ref_v_junction: float,
                      dt: float,) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for green mode
    intervals = np.zeros(total_num)
    ref_v = np.zeros(total_num)
    # ref_v from ref_v_junction to ref_v_lane
    # ahead_length = current_part['length'] - position_on_ref
    # n1 = int(ahead_length / ref_v_junction)
    # intervals[:n1] = ref_v_junction * dt
    # intervals[n1:] = ref_v_lane * dt
    # ref_v[:n1] = ref_v_junction
    # ref_v[n1:] = ref_v_lane

    # ref_v keeps ref_v_junction for all points if ego is in the junction
    intervals[:] = ref_v_junction * dt
    ref_v[:] = ref_v_junction
    return intervals, ref_v


def compute_intervals_initsegment_green(position_on_ref: float,
                      current_part: Dict[str, float],
                      total_num: int,
                      ref_v_lane: float,
                      ref_v_junction: float,
                      dt: float,
                      am: float) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for green mode
    intervals = np.zeros(total_num)
    ref_v = np.zeros(total_num)
    position_on_ref = np.clip(position_on_ref, a_min=0,
                              a_max=current_part['length'])
    # ref_v from ref_v_lane to ref_v_junction
    ahead_length = current_part['length'] - position_on_ref
    v0 = np.sqrt(2 * am * ahead_length + ref_v_junction ** 2)
    ref_v = [v0 - am * dt * i for i in range(total_num)]
    # when v < v_junction and v > v_lane, a = am
    a_list = [am if v > ref_v_junction and v <
                    ref_v_lane else 0 for v in ref_v]
    ref_v = np.clip(ref_v, a_min=ref_v_junction, a_max=ref_v_lane)
    intervals = [v * dt - 0.5 * a * dt * dt for v, a in zip(ref_v, a_list)]
    intervals = np.clip(
        intervals, a_min=ref_v_junction * dt, a_max=ref_v_lane * dt)
    return intervals, ref_v


def compute_intervals_initsegment_red(position_on_ref: float,
                      current_part: Dict[str, float],
                      total_num: int,
                      ref_v_lane: float,
                      dt: float,
                      am: float,
                      min_ahead_lane_length: float) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for red mode
    intervals = np.zeros(total_num)
    ref_v = np.zeros(total_num)
    position_on_ref = np.clip(position_on_ref, a_min=0,
                              a_max=current_part['length'])
    # ref_v from ref_v_lane to 0
    ahead_length = current_part['length'] - position_on_ref - (min_ahead_lane_length - 4) # FIXME: 4 is 0.8*ego_length, which has been considered in current_part['length'] for red light
    if ahead_length < 0.01:
        intervals = np.zeros((total_num, ))
        ref_v = np.zeros((total_num, ))
    else:
        v0 = np.sqrt(2 * am * ahead_length)
        ref_v = [v0 - am * dt * i for i in range(total_num)]
        # when v < v_junction and v > v_lane, a = am
        a_list = [am if v > 0 and v <
                        ref_v_lane else 0 for v in ref_v]
        ref_v = np.clip(ref_v, a_min=0, a_max=ref_v_lane)
        intervals = [v * dt for v, a in zip(ref_v, a_list)]
        intervals = np.clip(
            intervals, a_min=0 * dt, a_max=ref_v_lane * dt)
    return intervals, ref_v


def compute_intervals(ref_info: List[Dict[str, float]],
                      total_num: int,
                      cur_v: float,
                      ref_v_lane: float,
                      dt: float,
                      acc: float,
                      ) -> Tuple[np.ndarray, np.ndarray]:
    # velocity planning for green mode
    # total_num = total_num - 1
    ref_v = np.ones(total_num) * ref_v_lane
    current_part = ref_info[0]

    assert current_part['destination'] == True, "Error ref_line"
    if acc is not None: # calculate ref_v using acc
        ref_v = [cur_v + acc * dt * i for i in range(total_num)]
        ref_v = np.clip(ref_v, a_min=0, a_max=ref_v_lane+10)

    intervals = ref_v * dt

    return intervals, ref_v