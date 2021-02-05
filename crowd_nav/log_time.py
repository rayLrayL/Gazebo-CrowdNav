import rospy
import threading
import time
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock

global sim_time, flg, t1, t2

sim_time = 0.0
flg = 0

def time_callback(data):
    global sim_time
    _sec = data.clock.secs
    _nsec = data.clock.nsecs
    sim_time = _sec + _nsec * 0.000000001

def listener():
    rospy.init_node('logger', anonymous=True)
    rospy.Subscriber("/clock", Clock, time_callback)
    rospy.spin()


def printer():
    global sim_time, t1, t2, flg
    while not rospy.is_shutdown():
        if sim_time != 0.0 and flg == 0:
            t1 = float(sim_time)
            flg = 1
        time.sleep(0.01)
    t2 = float(sim_time)
    print()
    print("Total time: {} secs".format(t2 - t1))


def load_printer():
    print_thread = threading.Thread(target=printer)
    print_thread.start()

if __name__ == "__main__":
	load_printer()
	listener()
else:
    print("ERROR!")
