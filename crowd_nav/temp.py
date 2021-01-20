
import numpy as np
import rospy
import time
import threading
import math
from sensor_msgs.msg import PointCloud2
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
import sensor_msgs.point_cloud2 as pc2
import matplotlib.pyplot as plt

rospy.init_node('temp', anonymous=False)
points_raw = 0
robot_state = 0

def points_callback(data):
	global points_raw
	points_raw = data

def state_callback(data):
	global robot_state
	## robot state -> x, y, yaw (differenct x,y axis)
	robot_state = np.array([data.linear.y, data.linear.x, data.angular.z])

def listener():
	rospy.Subscriber("/obs_center", PointCloud2, points_callback)
	rospy.Subscriber("/robot_state", Twist, state_callback)
	rospy.spin()

obs_listener = threading.Thread(target=listener)
obs_listener.setDaemon(True)
obs_listener.start()

def talker():
	global  points_raw, robot_state
	if type(points_raw) == type(0) or type(robot_state) == type(0):
		print("NOT CONNECTED")
		return False
	points = pc2.read_points(points_raw, skip_nans=True)
	samples = np.array(list(points), dtype=np.float32)
	yaw = robot_state[2]
	print(robot_state)
	if len(samples):
		samples = samples[:, 0:2]
		rot_matrix = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
		samples = np.array([np.dot(rot_matrix, sample) + robot_state[0:2] for sample in samples])
		plt.scatter(samples[:,0], samples[:,1])
	plt.scatter(robot_state[0], robot_state[1])
	plt.arrow(robot_state[0], robot_state[1], math.sin(yaw), math.cos(yaw), width=0.05)
	plt.xlim(-2.5,2.5)
	plt.ylim(0,5)
	plt.legend()
	plt.title("Top view points after filter processing")
	plt.xlabel("x (m)")
	plt.ylabel("y (m)")
	plt.pause(0.05)
	plt.cla()
	plt.clf()


while True:
	talker()
	time.sleep(0.1)
rospy.signal_shutdown("esc")
exit()