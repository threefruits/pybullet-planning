from .fetch_never_collisions import NEVER_COLLISIONS
import os
import re
import math
import random
import numpy as np
from itertools import combinations
from collections import namedtuple

from .utils import multiply, get_link_pose, set_joint_position, set_joint_positions, get_joint_positions, get_min_limit, get_max_limit, quat_from_euler, read_pickle, set_pose, \
    get_pose, euler_from_quat, link_from_name, point_from_pose, invert, Pose, \
    unit_pose, joints_from_names, PoseSaver, get_aabb, get_joint_limits, ConfSaver, get_bodies, create_mesh, remove_body, \
    unit_from_theta, violates_limit, \
    violates_limits, add_line, get_body_name, get_num_joints, approximate_as_cylinder, \
    approximate_as_prism, unit_quat, unit_point, angle_between, quat_from_pose, compute_jacobian, \
    movable_from_joints, quat_from_axis_angle, LockRenderer, Euler, get_links, get_link_name, \
    get_extend_fn, get_moving_links, link_pairs_collision, get_link_subtree, \
    clone_body, get_all_links, pairwise_collision, tform_point, get_camera_matrix, ray_from_pixel, pixel_from_ray, dimensions_from_camera_matrix, \
    wrap_angle, TRANSPARENT, PI, OOBB, pixel_from_point, set_all_color, wait_if_gui

ARM = 'main_arm'
ARM_NAMES = (ARM)

FETCH_GROUPS = {
    'base': ['x', 'y', 'theta'],
    'torso': ['torso_lift_joint'],
    'head': ['head_pan_joint', 'head_tilt_joint'],
    'main_arm': ['shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint'],
    'main_gripper': ['l_gripper_finger_joint', 'r_gripper_finger_joint'],
}

FETCH_TOOL_FRAMES = {ARM: 'tool_link'}
FETCH_GRIPPER_ROOTS = {ARM: 'gripper_link'}
FETCH_BASE_LINK = 'base_link'
HEAD_LINK_NAME = 'head_camera_link'


def get_disabled_collisions(fetch):
    # disabled_names = PR2_ADJACENT_LINKS
    # disabled_names = PR2_DISABLED_COLLISIONS
    disabled_names = NEVER_COLLISIONS
    # disabled_names = PR2_DISABLED_COLLISIONS + NEVER_COLLISIONS
    link_mapping = {get_link_name(fetch, link): link for link in get_links(fetch)}
    return {
        (link_mapping[name1], link_mapping[name2])
        for name1, name2 in disabled_names
        if (name1 in link_mapping) and (name2 in link_mapping)
    }

FETCH_URDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models/fetch_description/robots/fetch.urdf')



def get_base_pose(hsr):
    return get_link_pose(hsr, link_from_name(hsr, FETCH_BASE_LINK))

def arm_conf(arm, arm_config):
    if arm == ARM:
        return arm_config

def base_conf(arm, base_config):
    if arm == ARM:
        return base_config

def base_arm_conf(arm, base_config, arm_config):
    base_arm_conf = []
    if arm == ARM:
        for base in base_config:
            base_arm_conf.append(base)
        for arm in arm_config:
            base_arm_conf.append(arm)
        return base_arm_conf

# def get_carry_conf(arm, grasp_type):
#     return arm_conf(arm, HSR_CARRY_CONFS[grasp_type])

def get_other_arm(arm):
    for other_arm in ARM_NAMES:
        if other_arm != arm:
            return other_arm
    raise ValueError(arm)


#####################################

def get_groups():
    return sorted(FETCH_GROUPS)

def get_group_joints(robot, group):
    return joints_from_names(robot, FETCH_GROUPS[group])

def get_group_conf(robot, group):
    return get_joint_positions(robot, get_group_joints(robot, group))

def set_group_conf(robot, group, positions):
    set_joint_positions(robot, get_group_joints(robot, group), positions)

def set_group_positions(robot, group_positions):
    for group, positions in group_positions.items():
        set_group_conf(robot, group, positions)

def get_group_positions(robot):
    return {group: get_group_conf(robot, group) for group in get_groups()}

#####################################


# End-effectors

def get_arm_joints(robot, arm):
    return get_group_joints(robot, arm)

def get_base_joints(robot, arm):
    return joints_from_names(robot, FETCH_GROUPS['base'])

def get_torso_joints(robot, arm):
    return joints_from_names(robot, FETCH_GROUPS['torso'])

def get_torso_arm_joints(robot, arm):
    return joints_from_names(robot, FETCH_GROUPS['torso'] + FETCH_GROUPS[arm])

def get_base_arm_joints(robot, arm):
    return joints_from_names(robot, FETCH_GROUPS['base'] + FETCH_GROUPS[arm])

def get_base_torso_joints(robot):
    return joints_from_names(robot, FETCH_GROUPS['base'] + FETCH_GROUPS['torso'])

def get_base_torso_arm_joints(robot):
    return joints_from_names(robot, FETCH_GROUPS['base'] + FETCH_GROUPS['torso'] + FETCH_GROUPS['main_arm'])

def set_arm_conf(robot, arm, conf):
    set_joint_positions(robot, get_arm_joints(robot, arm), conf)

def get_gripper_link(robot, arm):
    return link_from_name(robot, FETCH_TOOL_FRAMES[arm])

def get_gripper_joints(robot, arm):
    return get_group_joints(robot, 'main_gripper')

def set_gripper_position(robot, arm, position):
    gripper_joints = get_gripper_joints(robot, arm)
    set_joint_positions(robot, gripper_joints, [position] * len(gripper_joints))

def open_arm(robot, arm):
    for joint in get_gripper_joints(robot, arm):
        set_joint_position(robot, joint, get_max_limit(robot, joint))

def close_arm(robot, arm):
    for joint in get_gripper_joints(robot, arm):
        set_joint_position(robot, joint, get_min_limit(robot, joint))

open_gripper = open_arm
close_gripper = close_arm