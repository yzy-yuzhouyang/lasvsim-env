import numpy as np
from shapely import segmentize, simplify

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
    

def add_map_objs(line_string, map_objs, max_speed, obj_type, simplify_tol=0.2):
    """
    Add segmentized line_string to map_objs.
    Args:
        line_string: shapely.LineString.
        map_objs: existing list. Newly added map objects will be appended to this list.
        max_speed: float, max speed of the line.
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