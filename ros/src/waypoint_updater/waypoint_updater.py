#!/usr/bin/env python

import math
import rospy
import tf
import waypoint_helper as Helper
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
'''

LOOKAHEAD_WPS = 100 # Number of waypoints we will publish. You can change this number
DISTANCE_STOP_AT_TRAFFIC = 28 # stop distance before traffic light
MAX_DECEL = 1.0 # max deceleration in ms-2

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # Subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        self.waypoints_sub = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        # Publishers
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=2)
        self.closest_waypoint_pub = rospy.Publisher('closest_waypoint', Int32, queue_size=1)

        self.base_waypoints = None
        self.current_pose = None
        self.num_waypoints = 0
        self.closest_waypoint = None
        self.traffic = -1

        self.publish()
        rospy.spin()

    def publish(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            final_waypoints = self.get_final_waypoints()
            if final_waypoints:
                lane = Lane()
                lane.header.frame_id = '/world'
                lane.header.stamp = rospy.Time.now()
                lane.waypoints = final_waypoints
                self.final_waypoints_pub.publish(lane)
            rate.sleep()

    def pose_cb(self, msg):
        """ Callback for current vehicle pose """
        self.current_pose = msg.pose

    def waypoints_cb(self, waypoints):
        """ Callback for base waypoints """
        self.base_waypoints = waypoints.waypoints
        self.num_waypoints = len(self.base_waypoints)
        self.waypoints_sub.unregister()

    def traffic_cb(self, msg):
        """ Callback for traffic lights """
        self.traffic = int(msg.data)

    def obstacle_cb(self, msg):
        """ Callback for obstacles """
        pass

    def get_final_waypoints(self):
        if not self.current_pose or not self.base_waypoints:
            return None

        final_waypoints, self.closest_waypoint = Helper.look_ahead_waypoints(self.current_pose,
                                                                             self.base_waypoints,
                                                                             self.closest_waypoint,
                                                                             LOOKAHEAD_WPS)

        self.closest_waypoint_pub.publish(Int32(self.closest_waypoint))

        # If we have a traffic light ahead
        if self.traffic != -1:
            
            stop_point_lookup = {318:283, 784:741, 2095:2029, 2625:2567, 6322:6281, 7036:6995, 8565:8528, 9773:9715}
            
            how_far_before_traffic_light_is_the_stop_point = 35
            how_far_before_stop_point_to_begin_decel = 25
            
            total_waypoints = len(self.base_waypoints)

            if self.traffic in stop_point_lookup:
                stop_point = stop_point_lookup[self.traffic]
            else:
                stop_point = (self.traffic - how_far_before_traffic_light_is_the_stop_point + total_waypoints) % total_waypoints

            distance_to_stop_point = Helper.distance(self.base_waypoints, self.closest_waypoint, stop_point)

            # Check if we are close enough to start decelerating
            # We want to start decelerating 30 meters before
            if distance_to_stop_point <= how_far_before_stop_point_to_begin_decel:

                # If yes, adjust waypoint speeds so that we stop at the traffic light
                final_waypoints = Helper.smooth_decel_till_stop_waypoints(self.base_waypoints,
                                                                          final_waypoints,
                                                                          self.closest_waypoint,
                                                                          stop_point,
                                                                          distance_to_stop_point,
                                                                          how_far_before_stop_point_to_begin_decel,
                                                                          total_waypoints)

        # rospy.logout("closest %d, traffic %d", self.closest_waypoint, self.traffic)
        # info = "speed for waypoint: " + ", ".join("%05.2f" % wp.twist.twist.linear.x for wp in final_waypoints[:10])
        # rospy.logout(info)

        return final_waypoints

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
