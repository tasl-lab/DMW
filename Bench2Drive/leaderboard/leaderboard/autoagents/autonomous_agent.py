#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the base class for all autonomous agents
"""

from __future__ import print_function

from enum import Enum

import carla
from srunner.scenariomanager.timer import GameTime

from leaderboard.utils.route_manipulation import downsample_route
from leaderboard.envs.sensor_interface import SensorInterface
import math
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider


class Track(Enum):

    """
    This enum represents the different tracks of the CARLA AD leaderboard.
    """
    SENSORS = 'SENSORS'
    MAP = 'MAP'
    SENSORS_QUALIFIER = 'SENSORS_QUALIFIER'
    MAP_QUALIFIER = 'MAP_QUALIFIER'


class AutonomousAgent(object):

    """
    Autonomous agent base class. All user agents have to be derived from this class
    """

    def __init__(self, carla_host, carla_port, debug=False):
        self.track = Track.SENSORS
        #  current global plans to reach a destination
        self._global_plan = None
        self._global_plan_world_coord = None

        # this data structure will contain all sensor data
        self.sensor_interface = SensorInterface()

        self.wallclock_t0 = None

        self.get_hero()

    def setup(self, path_to_conf_file):
        """
        Initialize everything needed by your agent and set the track attribute to the right type:
            Track.SENSORS : CAMERAS, LIDAR, RADAR, GPS and IMU sensors are allowed
            Track.MAP : OpenDRIVE map is also allowed
        """
        pass

    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]

        """
        sensors = []

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        :return: control
        """
        control = carla.VehicleControl()
        control.steer = 0.0
        control.throttle = 0.0
        control.brake = 0.0
        control.hand_brake = False

        return control

    def destroy(self):
        """
        Destroy (clean-up) the agent
        :return:
        """
        pass

    def __call__(self):
        """
        Execute the agent call, e.g. agent()
        Returns the next vehicle controls
        """
        input_data = self.sensor_interface.get_data(GameTime.get_frame())

        timestamp = GameTime.get_time()

        if not self.wallclock_t0:
            self.wallclock_t0 = GameTime.get_wallclocktime()
        wallclock = GameTime.get_wallclocktime()
        wallclock_diff = (wallclock - self.wallclock_t0).total_seconds()
        sim_ratio = 0 if wallclock_diff == 0 else timestamp/wallclock_diff

        print('=== [Agent] -- Wallclock = {} -- System time = {} -- Game time = {} -- Ratio = {}x'.format(
            str(wallclock)[:-3], format(wallclock_diff, '.3f'), format(timestamp, '.3f'), format(sim_ratio, '.3f')), flush=True)

        control = self.run_step(input_data, timestamp)
        control.manual_gear_shift = False

        return control

    @staticmethod
    def get_ros_version():
        return -1

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        """
        Set the plan (route) for the agent
        """
        self.org_dense_route_world_coord = global_plan_world_coord
        ds_ids = downsample_route(global_plan_world_coord, 50)
        self._global_plan_world_coord = [(global_plan_world_coord[x][0], global_plan_world_coord[x][1]) for x in ds_ids]
        self._global_plan = [global_plan_gps[x] for x in ds_ids]
        self._plan_gps_HACK = global_plan_gps
    
    def get_hero(self):
        hero_actor = None
        from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
        for actor in CarlaDataProvider.get_world().get_actors():
            if 'role_name' in actor.attributes and actor.attributes['role_name'] == 'hero':
                hero_actor = actor
                break
        self.hero_actor = hero_actor
    
    def get_metric_info(self):
        
        def vector2list(vector, rotation=False):
            if rotation:
                return [vector.roll, vector.pitch, vector.yaw]
            else:
                return [vector.x, vector.y, vector.z]

        output = {}
        output['acceleration'] = vector2list(self.hero_actor.get_acceleration())
        output['angular_velocity'] = vector2list(self.hero_actor.get_angular_velocity())
        output['forward_vector'] = vector2list(self.hero_actor.get_transform().get_forward_vector())
        output['right_vector'] = vector2list(self.hero_actor.get_transform().get_right_vector())
        output['location'] = vector2list(self.hero_actor.get_transform().location)
        output['rotation'] = vector2list(self.hero_actor.get_transform().rotation, rotation=True)
        output['bounding_box_extent']=vector2list(self.hero_actor.bounding_box.extent)
        output['speed'] = CarlaDataProvider.get_velocity(self.hero_actor)

        # --- 新增的指标 ---
        world = self.hero_actor.get_world()
        carla_map = world.get_map()
        ego_location = self.hero_actor.get_location()
        ego_transform = self.hero_actor.get_transform()
        
        # 1. 获取当前限速 (单位: km/h)
        output['speed_limit'] = self.hero_actor.get_speed_limit()

        # 2. 获取当前车道信息
        waypoint = carla_map.get_waypoint(ego_location)
        output['lane_info'] = {
            'lane_id': waypoint.lane_id,
            'lane_type': str(waypoint.lane_type),
            'lane_width': waypoint.lane_width,
            'is_junction': waypoint.is_junction
        }

        # 3. 更新并获取变道次数
        current_lane_id = waypoint.lane_id
        if self.previous_lane_id is not None and self.previous_lane_id != current_lane_id and not waypoint.is_junction:
            self.lane_change_count += 1
        self.previous_lane_id = current_lane_id
        
        output['lane_change_count'] = self.lane_change_count

        # 4. 计算与前方车辆的距离并获取其信息
        min_distance_to_front_vehicle = float('inf')
        front_vehicle_actor = None # 用来临时存储前方的车辆对象
        
        for vehicle in world.get_actors().filter('vehicle.*'):
            if vehicle.id == self.hero_actor.id:
                continue
            
            other_location = vehicle.get_location()
            other_waypoint = carla_map.get_waypoint(other_location, project_to_road=True, lane_type=carla.LaneType.Driving)

            if other_waypoint is not None and other_waypoint.lane_id == current_lane_id:
                vec_to_other = other_location - ego_location
                
                if vec_to_other.dot(ego_transform.get_forward_vector()) > 0:
                    distance = ego_location.distance(other_location)
                    if distance < min_distance_to_front_vehicle:
                        min_distance_to_front_vehicle = distance
                        front_vehicle_actor = vehicle # 存储找到的最近的前方车辆对象
        
        # 将距离信息存入 output
        output['distance_to_front_vehicle'] = min_distance_to_front_vehicle if min_distance_to_front_vehicle != float('inf') else -1.0
        
        # # 如果找到了前方车辆，就提取它的信息存入一个字典
        # if front_vehicle_actor:
        #     front_velocity = front_vehicle_actor.get_velocity()
        #     # 计算前车的速度 (km/h)
        #     front_speed_kmh = 3.6 * math.sqrt(front_velocity.x**2 + front_velocity.y**2 + front_velocity.z**2)

        #     output['front_vehicle_info'] = {
        #         "id": front_vehicle_actor.id,
        #         "type": front_vehicle_actor.type_id,
        #         "location": vector2list(front_vehicle_actor.get_location()),
        #         "velocity": vector2list(front_velocity),
        #         "speed_kmh": front_speed_kmh,
        #         "color": front_vehicle_actor.attributes.get('color', 'N/A')
        #     }
        # else:
        #     # 如果没找到前方车辆，则存入 None
        #     output['front_vehicle_info'] = None
            
        return output