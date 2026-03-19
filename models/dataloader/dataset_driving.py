"""
Code that loads the dataset for training.
partially taken from https://github.com/autonomousvision/carla_garage/blob/main/team_code/data.py
(MIT licence)
"""

import os
import ujson
import json
import numpy as np
import random
import cv2
import re
import gzip

import torch
from models.utils.custom_types import DatasetOutput
from models.dataloader.dataset_base import BaseDataset
import models.utils.transfuser_utils as t_u

VIZ_DATA = False

class Data_Driving(BaseDataset):  # pylint: disable=locally-disabled, invalid-name
    """
    Custom dataset that dynamically loads a CARLA dataset from disk.
    """

    def __init__(self,
            **cfg,
        ):
        super().__init__(**cfg)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        # Disable threading because the data loader will already split in threads.
        cv2.setNumThreads(0)

        data = {}
        images = self.images[index]
        measurements = self.measurements[index]
        sample_start = self.sample_start[index]
        augment_exists = self.augment_exists[index]
        boxes = self.boxes[index]

        loaded_measurements, current_measurement, measurement_file_current = self.load_current_and_future_measurements(
            measurements,
            sample_start
            )
        
        data['measurement_path'] = measurement_file_current

        route_config_path = measurement_file_current.split(self.data_path)[-1].lstrip('/')
        route_id = route_config_path.split("Rep0_")[1].split("_")[0]

        path_parts = route_config_path.split('/')[:-3]
        relative_dir = '/'.join(path_parts)
        route_config_path = os.path.join(self.repo_path, relative_dir, f"{route_id}.xml")

        scenario_type = extract_scenario_type(route_config_path)[0]['type']

        loaded_boxes, current_boxes = self.load_current_and_future_boxes(boxes, sample_start)
        nearby_actors = [box for box in current_boxes if box['class'] == 'car']
        nearby_walkers = [box for box in current_boxes if box['class'] == 'walker']
        nearby_actors_by_id = {box['id']: box for box in nearby_actors}
        nearby_walkers_by_id = {box['id']: box for box in nearby_walkers}
        ids = [box['id'] for box in nearby_actors]
        ids_walkers = [box['id'] for box in nearby_walkers]
        ids_all = ids + ids_walkers
        
        # load data for future frames, repeating the last available frame if not found
        future_nearby_actors = []
        future_nearby_walkers = []
        future_nearby_actors_by_id = {id: [] for id in ids}
        future_nearby_actors_used_time_stamps_by_id = {id: [] for id in ids}
        future_nearby_walkers_by_id = {id: [] for id in ids_walkers}
        future_nearby_walkers_used_time_stamps_by_id = {id: [] for id in ids_walkers}

        for i, future in enumerate(loaded_boxes[self.hist_len:]):
            tmp_actors = []
            tmp_walkers = []
            for id in ids_all:
                tmp_box = [box for box in future if 'id' in box and box['id'] == id]
                if tmp_box:
                    if tmp_box[0]['class'] == 'car':
                        tmp_actors.append(tmp_box[0])
                    elif tmp_box[0]['class'] == 'walker':
                        tmp_walkers.append(tmp_box[0])
                    else:
                        raise ValueError('Unknown class')
                else:
                    if id in future_nearby_actors_by_id:
                        if len(future_nearby_actors_by_id[id]) > 0:
                            tmp_actors.append(future_nearby_actors_by_id[id][-1])
                            future_nearby_actors_used_time_stamps_by_id[id].append(future_nearby_actors_used_time_stamps_by_id[id][-1])
                        else:
                            tmp_actors.append(nearby_actors_by_id[id])
                            future_nearby_actors_used_time_stamps_by_id[id].append(0)
                if id in future_nearby_actors_by_id:
                    future_nearby_actors_by_id[id].append(tmp_actors[-1])
                    if tmp_box:
                        future_nearby_actors_used_time_stamps_by_id[id].append(i)
                elif id in future_nearby_walkers_by_id and len(tmp_walkers) > 0:
                    future_nearby_walkers_by_id[id].append(tmp_walkers[-1])
                    future_nearby_walkers_used_time_stamps_by_id[id].append(i)

            if len(tmp_actors) > 0:
                future_nearby_actors.append(tmp_actors)
            if len(tmp_walkers) > 0:
                future_nearby_walkers.append(tmp_walkers)

        speed_limit = current_measurement['speed_limit']
        target_speed = current_measurement['target_speed']

        ego_actor = [box for box in current_boxes if box['class'] == 'ego_car'][0]
        ego_actor["steer"] = current_measurement['steer']
        ego_actor["throttle"] = current_measurement['throttle']
        ego_actor["brake"] = current_measurement['brake']
        ego_actor["id"] = 0
        ego_position = current_measurement['pos_global']
        ego_yaw = current_measurement['theta']
        route_global = np.asarray([t_u.conversion_2d(rout, ego_position, -ego_yaw) for rout in current_measurement['route']])

        next_egos = [box for all_future_actors in loaded_boxes[self.hist_len:] for box in all_future_actors if box['class'] == 'ego_car']
        for i, next_ego in enumerate(next_egos):
            next_ego["id"] = 0
            next_ego["steer"] = loaded_measurements[self.hist_len + i]['steer']
            next_ego["throttle"] = loaded_measurements[self.hist_len + i]['throttle']
            next_ego["brake"] = loaded_measurements[self.hist_len + i]['brake']

        next_ego_by_id = {0: next_egos}
        all_ego_positions = [ego_position] + [loaded_measurements[self.hist_len + i]['pos_global'] for i in range(len(next_egos))]
        all_ego_yaws = [ego_yaw] + [loaded_measurements[self.hist_len + i]['theta'] for i in range(len(next_egos))]

        bbs_walkers = self.get_bbs(nearby_walkers, future_nearby_walkers_by_id, future_nearby_walkers_used_time_stamps_by_id, all_ego_positions, all_ego_yaws)
        bbs_other_actors = self.get_bbs(nearby_actors, future_nearby_actors_by_id, future_nearby_actors_used_time_stamps_by_id, all_ego_positions, all_ego_yaws)
        bbs_other_actors = {**bbs_other_actors, **bbs_walkers}

        forecasts_ego_adjusted = []
        tmp_forecasts, final_speed = self.forecast_vehicles(ego_actor, next_ego_by_id, ego_position=ego_position, ego_yaw=ego_yaw, route=route_global, target_speed=10, return_final_speed=True)
        forecasts_ego_adjusted.append(tmp_forecasts)

        intersection_bb = []
        intersection_timesteps = []
        if bbs_other_actors is not None:
            for j, ego_bounding_boxes_route_new_single in enumerate(forecasts_ego_adjusted):
                if len(intersection_bb) < j+1:
                    intersection_bb.append([])
                    intersection_timesteps.append([])
                for i, ego_bounding_box in enumerate(ego_bounding_boxes_route_new_single[0]):
                    intersects_with_ego = False
                    for vehicle_id, bounding_boxes in bbs_other_actors.items():
                        if i >= len(bounding_boxes)-1:
                            continue
                        ego_bounding_box.location.z = 0
                        bounding_boxes[i+1].location.z = 0
                        intersects_with_ego = self.check_obb_intersection(ego_bounding_box, bounding_boxes[i+1])
                        if intersects_with_ego:
                            intersection_bb[j].append(bounding_boxes[i+1])
                            intersection_timesteps[j].append(i+1)
                            intersects_with_ego = False

        forecasts_ego_forecasted_wps = []
        for forecast_adjusted in forecasts_ego_adjusted:
            forecasts_ego_forecasted_wps.append([[f.location.x, f.location.y] for f in forecast_adjusted[0]])
        intersection_bb_bool = [len(intersection_bb[i])>0 for i in range(len(intersection_bb))]
        
        # Determine whether the augmented camera or the normal camera is used.
        if augment_exists and random.random() <= self.img_shift_augmentation_prob and self.img_shift_augmentation:
            augment_sample = True
            aug_rotation = current_measurement['augmentation_rotation']
            aug_translation = current_measurement['augmentation_translation']
        else:
            augment_sample = False
            aug_rotation = 0.0
            aug_translation = 0.0

        data = self.load_waypoints(data, loaded_measurements, aug_translation, aug_rotation)
       
        speed_rounded = round(current_measurement['speed'], 1)
        data['speed'] = current_measurement['speed']

        data = self.load_route(data, current_measurement, aug_translation, aug_rotation)

        target_point = np.array(current_measurement['target_point'])
        next_target_point = np.array(current_measurement['target_point_next'])

        commentary_exists = False
        commentary = ''
        if self.use_commentary:
            commentary_file_path = measurement_file_current.replace('measurements', 'commentary').replace('data/', 'commentary/')
            if 'validation_' in commentary_file_path:
                commentary_exists = False
            else:
                try:
                    with gzip.open(commentary_file_path, 'rt') as f:
                        commentary_file = ujson.load(f)
                        commentary_exists = True
                except (FileNotFoundError, ujson.JSONDecodeError):
                    commentary_exists = False
                    commentary_file = None

                if commentary_file is not None:

                    commentary = commentary_file['commentary']
                    # we only augment in 60% of the cases and use the default commentary in 40% of the cases
                    # augmentation is used to increase generalization to a broader set of sentences
                    # but we do not want to overfit to the augmented sentences
                    if self.commentary_augmentation and random.random() < 0.6:
                        if commentary_file['commentary_template'] in self.templates_commentary:
                            commentary = random.choice(self.templates_commentary[commentary_file['commentary_template']])
                            for key, value in commentary_file['placeholder'].items():
                                if key in commentary:
                                    commentary = commentary.replace(key, value)
                            if re.search(r'<.*?>', commentary):
                                print(f"WARNING: {commentary} contains placeholders that are not replaced. Using default commentary.")
                                commentary = commentary_file['commentary']

                    commentary = commentary.replace('..', '.')
                    commentary = commentary.replace('in in', 'in')
        
        target_options, placeholder_values = self.get_navigational_conditioning( data, current_measurement, target_point, next_target_point)
            
        answer = ''

        prompt_random = random.random()
        
        if self.use_commentary and commentary_exists and prompt_random < self.prompt_probabilities['commentary']:
            if random.random() < 0.2:
                if random.random() < 0.5:
                    prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} {commentary} Predict the waypoints."
                else:
                    prompt = f"Current speed: {speed_rounded} m/s. Command: {commentary} Predict the waypoints."
                answer = f"Waypoints:"
            else:
                prompt = f"Current speed: {speed_rounded} m/s. {random.choice(target_options)} What should the ego do next?"
                answer = f"{commentary} Waypoints:"
        else:
            if scenario_type not in self.scenario_prompts:
                prompt = f"Current speed: {speed_rounded} m/s. {target_options[0]} Predict the waypoints."
                style = "Normal"
            else:
                selected_style = self.prompt_style
                style_prompts = self.scenario_prompts[scenario_type].get(selected_style, [])
                selected_prompt = style_prompts[index % len(style_prompts)]
                prompt = f"Current speed: {speed_rounded} m/s. {target_options[0]} {selected_prompt} Predict the waypoints."
                style = selected_style
            answer = f"Waypoints:"

        answer = answer.replace('..', '.')
        prompt = prompt.replace('..', '.')

        data = self.load_images(data, images, augment_sample=augment_sample)
        

        conversation_answer = [
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{answer}"},
                ],
            },
        ]
        conversation_all = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image"},
                ],
            },
            {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{answer}"},
                ],
            }
        ]
        
        images = [data['rgb']]

        data_new = DatasetOutput(
            conversation = conversation_all,
            answer = conversation_answer,
            image_ff = data['rgb'],
            image_ff_org_size=data['rgb_org_size'],
            waypoints = data["waypoints"],
            waypoints_1d = data["waypoints_1d"],
            path = data['route_adjusted'],
            target_points = data['target_points'],
            speed = data['speed'],
            placeholder_values = placeholder_values,
            measurement_path = data['measurement_path'],
            dataset = 'driving',
            bbs_other_actors = bbs_other_actors,
            route_global = route_global,
            current_ego_action = [ego_actor['steer'], ego_actor['throttle'], ego_actor['brake']],
            current_ego_position = ego_actor['position'],
            current_ego_yaw = ego_actor['yaw'],
            current_ego_speed = ego_actor['speed'],
            ego_extent = ego_actor['extent'],
            speed_limit = speed_limit,
            target_speed = target_speed,
            prompt = prompt,
            scenario_type = scenario_type,
            style = style,
            driver_id = self.driver_id,
        )
        
        if VIZ_DATA:
            self.visualise_cameras(data_new, commentary, data['route_adjusted'], data['waypoints'], options=None, prompt=prompt, answer=answer, name="img")
        return data_new

import xml.etree.ElementTree as ET
def extract_scenario_type(xml_file_path):
    """Extract scenario type from route XML file"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        scenarios = root.findall('.//scenario')
        
        scenario_types = []
        for scenario in scenarios:
            scenario_type = scenario.get('type')
            scenario_name = scenario.get('name')
            if scenario_type:
                scenario_types.append({
                    'name': scenario_name,
                    'type': scenario_type
                })
        
        return scenario_types
        
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return []
    except FileNotFoundError:
        print(f"File not found: {xml_file_path}")
        return []

if __name__ == "__main__":
    from hydra import compose, initialize
    from models.config import TrainConfig
    
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    

    initialize(config_path="../config")
    cfg = compose(config_name="config")
    
    cfg.data_module.base_dataset.use_commentary = True
    cfg.data_module.base_dataset.use_qa = True
    cfg.data_module.base_dataset.img_shift_augmentation = False

    print('Test Dataset')
    dataset = Data_Driving(                        
                        split="train",
                        bucket_name='all',
                        **cfg.data_module,
                        **cfg.data_module.base_dataset,
    )

    for i in range(len(dataset)):
        data = dataset[i]
        break