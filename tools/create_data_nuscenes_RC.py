# -*- coding: utf-8 -*-
# Copyright (c) OpenMMLab. All rights reserved.

import pickle
import numpy as np
from nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import mmcv

from data_converter import nuscenes_converter_RC as nuscenes_converter

map_name_from_general_to_detection = {
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.wheelchair': 'ignore',
    'human.pedestrian.stroller': 'ignore',
    'human.pedestrian.personal_mobility': 'ignore',
    'human.pedestrian.police_officer': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'animal': 'ignore',
    'vehicle.car': 'car',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.truck': 'truck',
    'vehicle.construction': 'construction_vehicle',
    'vehicle.emergency.ambulance': 'ignore',
    'vehicle.emergency.police': 'ignore',
    'vehicle.trailer': 'trailer',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
    'movable_object.pushable_pullable': 'ignore',
    'movable_object.debris': 'ignore',
    'static_object.bicycle_rack': 'ignore',
}
classes = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]


def get_gt(info):
    """Generate gt labels from info."""
    ego2global_rotation = info['cams']['CAM_FRONT']['ego2global_rotation']
    ego2global_translation = info['cams']['CAM_FRONT']['ego2global_translation']
    trans = -np.array(ego2global_translation)
    rot = Quaternion(ego2global_rotation).inverse
    gt_boxes, gt_labels = [], []
    for ann_info in info['ann_infos']:
        if (map_name_from_general_to_detection[ann_info['category_name']]
                not in classes
                or ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] <= 0):
            continue
        box = Box(
            ann_info['translation'],
            ann_info['size'],
            Quaternion(ann_info['rotation']),
            velocity=ann_info['velocity'],
        )
        box.translate(trans)
        box.rotate(rot)
        box_xyz = np.array(box.center)
        box_dxdydz = np.array(box.wlh)[[1, 0, 2]]
        box_yaw = np.array([box.orientation.yaw_pitch_roll[0]])
        box_velo = np.array(box.velocity[:2])
        gt_box = np.concatenate([box_xyz, box_dxdydz, box_yaw, box_velo])
        gt_boxes.append(gt_box)
        gt_labels.append(
            classes.index(
                map_name_from_general_to_detection[ann_info['category_name']]))
    return gt_boxes, gt_labels


def nuscenes_data_prep(root_path, info_prefix, version, max_sweeps=10):
    """Prepare data related to nuScenes dataset."""
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix, version=version, max_sweeps=max_sweeps)


def add_ann_adj_info(extra_tag, version='v1.0-mini'):
    dataroot = './data/nuscenes/'
    nuscenes = NuScenes(version, dataroot)
    dataset = pickle.load(
        open(f'./data/nuscenes/{extra_tag}_infos_train.pkl', 'rb'))
    for id in mmcv.track_iter_progress(range(len(dataset['infos']))):
        info = dataset['infos'][id]
        sample = nuscenes.get('sample', info['token'])
        ann_infos = []
        for ann in sample['anns']:
            ann_info = nuscenes.get('sample_annotation', ann)
            velocity = nuscenes.box_velocity(ann_info['token'])
            if np.any(np.isnan(velocity)):
                velocity = np.zeros(3)
            ann_info['velocity'] = velocity
            ann_infos.append(ann_info)
        dataset['infos'][id]['ann_infos'] = ann_infos
        dataset['infos'][id]['ann_infos'] = get_gt(dataset['infos'][id])
        dataset['infos'][id]['scene_token'] = sample['scene_token']

    with open(f'./data/nuscenes/{extra_tag}_infos_train.pkl', 'wb') as fid:
        pickle.dump(dataset, fid)


if __name__ == '__main__':
    root_path = './data/nuscenes'
    extra_tag = 'nuscenes_RC_mini'
    version = 'v1.0-mini'

    nuscenes_data_prep(
        root_path=root_path,
        info_prefix=extra_tag,
        version=version,
        max_sweeps=0)

    print('add_ann_infos')
    add_ann_adj_info(extra_tag, version)
