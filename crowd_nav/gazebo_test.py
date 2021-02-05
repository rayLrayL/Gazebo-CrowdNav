import logging
import argparse
import sys
if sys.version_info[0] ==2:
    import ConfigParser as configparser
else:
    import configparser
import os
import torch
import numpy as np
import gym
import math
import matplotlib.pyplot as plt

from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.policy.orca import ORCA
from crowd_sim.envs.utils.state import ObservableState
from crowd_sim.envs.utils.human import Human

import rospy

import time
import pc2obs

import easyGo

print("init pc2obs")
pc2obs.pc2obs_init()
voxel_size = 0.3

#ROBOT MOVE
SPEED = 15# 14
ROTATE_SPEED = 25 # 15
ANGULAR_SPEED = 0.5

# Set goal position
GOAL_X = 0
GOAL_Y = 3

def GoEasy(direc):
	if direc == 4: # Backward
		easyGo.mvStraight(- SPEED, -1)
	elif direc == 0 or direc == 1: # Go straight
		easyGo.mvStraight(SPEED, -1)
	elif direc == 2: # turn left
		easyGo.mvRotate(ROTATE_SPEED, -1, False)
	elif direc == 3: # turn right
		easyGo.mvRotate(ROTATE_SPEED, -1, True)
	elif direc == 5: # stop
		easyGo.stop()

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs/env.config')
    parser.add_argument('--policy_config', type=str, default='configs/policy.config')
    parser.add_argument('--policy', type=str, default='orca')
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--il', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    parser.add_argument('--visualize', default=True, action='store_true')
    parser.add_argument('--phase', type=str, default='test')
    parser.add_argument('--test_case', type=int, default=0)
    parser.add_argument('--square', default=False, action='store_true')
    parser.add_argument('--circle', default=False, action='store_true')
    parser.add_argument('--video_file', type=str, default=None)
    parser.add_argument('--traj', default=False, action='store_true')
    parser.add_argument('--plot', default=False, action='store_true')
    args = parser.parse_args()

    if args.model_dir is not None:
        env_config_file = os.path.join(args.model_dir, os.path.basename(args.env_config))
        policy_config_file = os.path.join(args.model_dir, os.path.basename(args.policy_config))
        if args.il:
            model_weights = os.path.join(args.model_dir, 'il_model.pth')
        else:
            if os.path.exists(os.path.join(args.model_dir, 'resumed_rl_model.pth')):
                model_weights = os.path.join(args.model_dir, 'resumed_rl_model.pth')
            else:
                model_weights = os.path.join(args.model_dir, 'rl_model.pth')
    else:
        env_config_file = args.env_config
        policy_config_file = args.env_config

    # configure logging and device
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    if policy.trainable:
        if args.model_dir is None:
            parser.error('Trainable policy must be specified with a model weights directory')
        policy.get_model().load_state_dict(torch.load(model_weights, map_location=torch.device('cpu')))

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    if args.square:
        env.test_sim = 'square_crossing'
    if args.circle:
        env.test_sim = 'circle_crossing'
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    policy.set_phase(args.phase)
    policy.set_device(device)
    # set safety space for ORCA in non-cooperative simulation
    if isinstance(robot.policy, ORCA):
        if robot.visible:
            robot.policy.safety_space = 0
        else:
            # because invisible case breaks the reciprocal assumption
            # adding some safety space improves ORCA performance. Tune this value based on your need.
            robot.policy.safety_space = 0
        logging.info('ORCA agent buffer: %f', robot.policy.safety_space)

    policy.set_env(env)
    robot.print_info()

    if args.visualize:

        obs = env.reset(args.phase, args.test_case)
        done = False
        last_pos = np.array(robot.get_position())
        policy_time = 0.0
        numFrame = 0

        dist = 20.0
        obs_flg = 0

        sim_time = False
        while sim_time == False:
            samples, robot_state, sim_time = pc2obs.pc2obs(voxel_size = voxel_size)
        t1 = float(sim_time)
        while (dist > 0.8):
            #t1 = time.time()
            env.humans = []
            samples, robot_state, sim_time = pc2obs.pc2obs(voxel_size = voxel_size)
            if type(samples) == type(False):
                print("Not Connect")
                continue
            dist = math.sqrt((GOAL_X - robot_state[0])**2 + (GOAL_Y - robot_state[1])**2)
            if obs_flg == 0 and dist < 10:
                os.system('sh ./init.sh')
                obs_flg = 1
            print("dist: {}".format(dist))
            # rotate and shift obs position
            t2 = time.time()
            yaw = robot_state[2]
            rot_matrix = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
            #samples = np.array([sample + robot_state[0:2] for sample in samples[:,0:2]])

            if len(samples) == 1:
                samples = np.array([np.dot(rot_matrix, samples[0][0:2]) + robot_state[0:2]])
                print(samples)
            elif len(samples) > 1:
                samples = np.array([np.dot(rot_matrix, sample) + robot_state[0:2] for sample in samples[:,0:2]])

            obs_position_list = samples
            obs = []

            for ob in obs_position_list:
                human = Human(env.config, 'humans')
                # px, py, gx, gy, vx, vy, theta
                human.set(ob[0], ob[1], ob[0], ob[1], 0, 0, 0)
                env.humans.append(human)
                # px, py, vx, vy, radius
                obs.append(ObservableState(ob[0], ob[1], 0, 0, voxel_size/2))
            if len(obs_position_list) == 0:
                human = Human(env.config, 'humans')
                # SARL, CADRL
                # human.set(0, -10, 0, -10 , 0, 0, 0)
                # LSTM
                human.set(robot_state[0]+10, robot_state[1]+10, robot_state[0]+10, robot_state[1]+10 , 0, 0, 0)
                env.humans.append(human)
                # SARL, CADRL
                # human.set(robot_state[0]+10, robot_state[1]+10, robot_state[0]+10, robot_state[1]+10, 0, 0, voxel_size/2)
                # LSTM
                obs.append(ObservableState(robot_state[0]+10, robot_state[1]+10, 0, 0, voxel_size/2))
            robot.set_position(robot_state)
            robot.set_velocity([math.sin(yaw), math.cos(yaw)])
            #print(obs)
            action = robot.act(obs)
            obs, _, done, info = env.step(action)
            current_pos = np.array(robot.get_position())
            current_vel = np.array(robot.get_velocity())
            #print("Velocity: {}, {}".format(current_vel[0], current_vel[1]))
            logging.debug('Speed: %.2f', np.linalg.norm(current_pos - last_pos) / robot.time_step)
            last_pos = current_pos
            policy_time += time.time()-t1
            numFrame += 1
            #print(t2-t1, time.time() - t2)

            diff_angle = (-yaw + math.atan2(current_vel[0], current_vel[1]))
            if diff_angle > math.pi:
                diff_angle = diff_angle - 2*math.pi
            elif diff_angle < - math.pi:
                diff_angle = diff_angle + 2*math.pi
            #print("diff_angle: {}, {}, {}".format(diff_angle, yaw ,-math.atan2(current_vel[0], current_vel[1])))
            if diff_angle < -0.7:
                direc = 2 # turn left
            elif diff_angle > 0.7:
                direc = 3 # turn right
            else:
                direc = 1 # go straight
            # GoEasy(direc)
            vx = SPEED * math.sqrt(current_vel[0]**2+current_vel[1]**2)
            if diff_angle > 0:
                v_ang = ANGULAR_SPEED * min(diff_angle/(math.pi/2), 1)
            else:
                v_ang = ANGULAR_SPEED * max(diff_angle/(math.pi/2), -1)
            print(vx, -v_ang)
            easyGo.mvCurve(vx, -v_ang)
            if args.plot:
                plt.scatter(current_pos[0], current_pos[1], label='robot')
                plt.arrow(current_pos[0], current_pos[1], current_vel[0], current_vel[1], width=0.05, fc='g', ec='g')
                plt.arrow(current_pos[0], current_pos[1], math.sin(yaw), math.cos(yaw), width=0.05, fc='r', ec='r')
                if len(samples) == 1:
                    plt.scatter(samples[0][0], samples[0][1], label='obstacles')
                elif len(samples) > 1:
                    plt.scatter(samples[:,0], samples[:,1], label='obstacles')
                plt.xlim(-6.5,6.5)
                plt.ylim(-9,4)
                plt.legend()
                plt.title("gazebo test")
                plt.xlabel("x (m)")
                plt.ylabel("y (m)")
                plt.pause(0.001)
                plt.cla()
                plt.clf()
            print("NAV TIME {}".format(float(sim_time)-t1))
            time.sleep(0.7)
        print("NAV TIME {}".format(float(sim_time)-t1))
        easyGo.stop()
        plt.close()
        print("Average took {} sec per iteration, {} Frame".format(policy_time/numFrame, numFrame))
    else:
        explorer.run_k_episodes(env.case_size[args.phase], args.phase, print_failure=True)


if __name__ == '__main__':
    main()
