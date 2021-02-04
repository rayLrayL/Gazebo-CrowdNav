import rospy
import threading
import time
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

global obs_pos1, obs_pos2, obs_pos3, obs_pos4, obs_pos5, self_pos, sim_time
obs_pos1 = [0,0]
obs_pos2 = [0,0]
obs_pos3 = [0,0]
obs_pos4 = [0,0]
obs_pos5 = [0,0]
self_pos = [0,0]
sim_time = 0.0

def ob1_callback(data):
    global obs_pos1
    _x = data.pose.pose.position.x
    _y = data.pose.pose.position.y
    obs_pos1 = [_x, _y]

def ob2_callback(data):
    global obs_pos2
    _x = data.pose.pose.position.x
    _y = data.pose.pose.position.y
    obs_pos2 = [_x, _y]

def ob3_callback(data):
    global obs_pos3
    _x = data.pose.pose.position.x
    _y = data.pose.pose.position.y
    obs_pos3 = [_x, _y]

def ob4_callback(data):
    global obs_pos4
    _x = data.pose.pose.position.x
    _y = data.pose.pose.position.y
    obs_pos4 = [_x, _y]

def ob5_callback(data):
    global obs_pos5
    _x = data.pose.pose.position.x
    _y = data.pose.pose.position.y
    obs_pos5 = [_x, _y]

def self_callback(data):
    global self_pos
    _x = data.pose.pose.position.x
    _y = data.pose.pose.position.y
    self_pos = [_x, _y]

def time_callback(data):
    global sim_time
    _sec = data.clock.secs
    _nsec = data.clock.nsecs
    sim_time = _sec + _nsec * 0.000000001

def listener():
    rospy.init_node('logger', anonymous=True)
    print('listener ready')
    rospy.Subscriber("/odom", Odometry, self_callback)
    rospy.Subscriber("/tb3_0/odom", Odometry, ob1_callback)
    rospy.Subscriber("/tb3_1/odom", Odometry, ob2_callback)
    rospy.Subscriber("/tb3_2/odom", Odometry, ob3_callback)
    rospy.Subscriber("/tb3_3/odom", Odometry, ob4_callback)
    rospy.Subscriber("/tb3_4/odom", Odometry, ob5_callback)
    rospy.Subscriber("/clock", Clock, time_callback)
    rospy.spin()


def printer():
    global obs_pos1, obs_pos2, obs_pos3, obs_pos4, obs_pos5, self_pos, sim_time
    while True:
        #print(sim_time)
        print(sim_time, self_pos[0], self_pos[1], obs_pos1[0], obs_pos1[1], obs_pos2[0], obs_pos2[1], obs_pos3[0], obs_pos3[1], obs_pos4[0], obs_pos4[1], obs_pos5[0], obs_pos5[1])
        time.sleep(0.5)


def load_printer():
    print_thread = threading.Thread(target=printer)
    print_thread.start()

if __name__ == "__main__":
	load_printer()
	listener()
else:
    print("ERROR!")
