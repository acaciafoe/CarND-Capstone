#!/usr/bin/env python

from __future__ import print_function

import rospy
import math
import numpy as np

from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from styx_msgs.msg import Lane
from geometry_msgs.msg import PoseStamped, TwistStamped
from twist_controller import Controller

# https://stackoverflow.com/a/43015816
import matplotlib
gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
for gui in gui_env:
    try:
        print("testing", gui)
        matplotlib.use(gui,warn=False, force=True)
        from matplotlib import pyplot as plt
        break
    except:
        continue

from Queue import Queue, Empty

CURRENT_VELOCITY_UNIT = 2.23694 # mps to Mph
WAYPOINT_VELOCITY_UNIT = 2.23694 # mps to Mph
BUFFER_SIZE = 128

class Observer(object):
    def __init__(self, *args, **kwargs):
        rospy.init_node('observer')

        self.current_velocity = None
        self.waypoint_velocity = None
        self.current_buffer = list()
        self.waypoint_buffer = list()

        self.process_q = Queue()

        self.setup_plot()

        rospy.Subscriber('/final_waypoints', Lane, self.final_waypoints_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)

        self.loop()

    def setup_plot(self):
        # https://stackoverflow.com/a/15724978
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.set_title("Speed over time")
        ax.set_ylabel("Mph")
        #ax.set_aspect('equal')
        ax.set_xlim(0, 160)
        ax.set_ylim(0, 50)
        ax.hold(True)

        plt.show(False)
        plt.draw()

        self.velocity_points = ax.plot([], [], '.')[0]
        self.waypoint_points = ax.plot([], [], '.')[0]

        self.plot_fig = fig
        self.plot_ax = ax
        self.plot_background = fig.canvas.copy_from_bbox(ax.bbox)

    def plot(self):

        velocity_y = self.current_buffer[:]
        velocity_x = list(range(len(velocity_y)))
        waypoint_y = self.waypoint_buffer[:]
        waypoint_x = list(range(len(waypoint_y)))

        # update data
        self.velocity_points.set_data(velocity_x, velocity_y)
        self.waypoint_points.set_data(waypoint_x, waypoint_y)

        # restore background
        self.plot_fig.canvas.restore_region(self.plot_background)

        # redraw just the points
        self.plot_ax.draw_artist(self.velocity_points)
        self.plot_ax.draw_artist(self.waypoint_points)

        # fill in the axes rectangle
        self.plot_fig.canvas.blit(self.plot_ax.bbox)

    def loop(self):
        while True:
            try:
                process = self.process_q.get(True, 0.1)
                self.plot()
            except Empty as ex:
                if rospy.is_shutdown():
                    break

    def process_and_plot(self):
        if self.current_velocity is None or self.waypoint_velocity is None:
            return
        self.current_buffer.append(self.current_velocity)
        self.current_buffer = self.current_buffer[-BUFFER_SIZE:]
        self.waypoint_buffer.append(self.waypoint_velocity)
        self.waypoint_buffer = self.waypoint_buffer[-BUFFER_SIZE:]
        self.process_q.put(1)
        return True

    def current_velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x * CURRENT_VELOCITY_UNIT

    def final_waypoints_cb(self, msg):
        self.waypoint_velocity = msg.waypoints[0].twist.twist.linear.x * WAYPOINT_VELOCITY_UNIT
        self.process_and_plot()

if __name__ == '__main__':
    try:
        Observer()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start observer node.')
