#!/usr/bin/env python3

import sys
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
import numpy as np
import tf2_ros as tf
import tf_conversions
from tf2_geometry_msgs import PoseStamped
from moveit_commander.conversions import pose_to_list
from moveit_commander import MoveGroupCommander
from math import pi, dist, fabs, cos
from moveit_commander.exception import MoveItCommanderException
import transforms3d.quaternions as quaternions
import transforms3d.euler as euler
from geometry_msgs.msg import Pose, Point, Quaternion
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
import requests
import json
import ast
from actionlib import SimpleActionClient
from pal_interaction_msgs.msg import TtsAction, TtsGoal




def all_close(goal, actual, tolerance):
    """
    Convenience method for testing if the values in two lists are within a tolerance of each other.
    For Pose and PoseStamped inputs, the angle between the two quaternions is compared (the angle
    between the identical orientations q and -q is calculated correctly).
    @param: goal       A list of floats, a Pose or a PoseStamped
    @param: actual     A list of floats, a Pose or a PoseStamped
    @param: tolerance  A float
    @returns: bool
    """
    if type(goal) is list:
        for index in range(len(goal)):
            if abs(actual[index] - goal[index]) > tolerance:
                return False

    elif type(goal) is geometry_msgs.msg.PoseStamped:
        return all_close(goal.pose, actual.pose, tolerance)

    elif type(goal) is geometry_msgs.msg.Pose:
        x0, y0, z0, qx0, qy0, qz0, qw0 = pose_to_list(actual)
        x1, y1, z1, qx1, qy1, qz1, qw1 = pose_to_list(goal)
        # Euclidean distance
        d = dist((x1, y1, z1), (x0, y0, z0))
        # phi = angle between orientations
        cos_phi_half = fabs(qx0 * qx1 + qy0 * qy1 + qz0 * qz1 + qw0 * qw1)
        return d <= tolerance and cos_phi_half >= cos(tolerance / 2.0)

    return True

class MoveGroupPythonInterface(object):
    def __init__(self):
        super(MoveGroupPythonInterface, self).__init__()

        # Initialize the moveit_commander and a ROS node
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node("arm_python_interface", anonymous=True)

        # Setup the MoveGroupCommander for controlling the robot's arm
        self.arm_group = "arm"
        self.arm = moveit_commander.MoveGroupCommander(self.arm_group)
        
        self.arm_torso_group = "arm_torso"
        self.arm_torso = moveit_commander.MoveGroupCommander(self.arm_torso_group)

        # Create a RobotCommander object to access robot's kinematic model and joint states
        self.robot = moveit_commander.RobotCommander()

        self.gripper_group = "gripper"
        self.gripper = moveit_commander.MoveGroupCommander(self.gripper_group)

        # Create a PlanningSceneInterface object to interact with the environment
        self.scene = moveit_commander.PlanningSceneInterface()

        # Create a tf buffer and listener
        self.tf_buffer = tf.Buffer()
        self.tf_listener = tf.TransformListener(self.tf_buffer)
        # Give time for TF data to populate the buffer
        rospy.sleep(1.0)

        self.arm.stop()  # This stops the movement and cancels the goal


    def go_to_joint_state(self, joints, group="arm"):
            ## Planning to a Joint Goal
            if group == "arm":
                group = self.arm
            if group == "arm_torso":
                group = self.arm_torso

            # We get the joint values from the group and change some of the values:
            joint_goal = group.get_current_joint_values()
            print(joint_goal)
            joint_goal[0] = joints[0]
            joint_goal[1] = joints[1]
            joint_goal[2] = joints[2]
            joint_goal[3] = joints[3]
            joint_goal[4] = joints[4]
            joint_goal[5] = joints[5]
            joint_goal[6] = joints[6]
            
            if len(joint_goal) > 6: 
                joint_goal[7] = joints[7]
            

            # The go command can be called with joint values, poses, or without any
            # parameters if you have already set the pose or joint target for the group
            success = group.go(joint_goal, wait=True)

            # Calling ``stop()`` ensures that there is no residual movement
            group.stop()


            # For testing:
            current_joints = group.get_current_joint_values()
            return all_close(joint_goal, current_joints, 0.01), success
    

    def get_position_relative_frame(self, target_frame):
        listener = tf.TransformListener(self.tf_buffer)
        # Wait for the transform to be available
        try:
            # Get the transform from the target frame to base_link
            transform = self.tf_buffer.lookup_transform("base_link", target_frame, rospy.Time(0))
            return transform
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            rospy.logerr("Error looking up transform: {}".format(e))
            return None
    

    def move_gripper(self, position):
        # Get the current joint values of the gripper
        gripper_current_state = self.gripper.get_current_joint_values()
        
        # Modify the joint values for the gripper to close it
        # Assuming the gripper has a single joint, you can set it to a value that represents closing
        # Joint limits seems to be 0.001 for close and 0.04 for open. 
        gripper_current_state[0] = position[0]
        gripper_current_state[1] = position[1]

        # Set the target joint values for the gripper
        self.gripper.set_joint_value_target(gripper_current_state)

        # Command the gripper to move
        self.gripper.go()


    def plan_cartesian_path(self, frame_name, pos, avoidCollision = True):
        ## Cartesian Paths
        ## ^^^^^^^^^^^^^^^
        ## You can plan a Cartesian path directly by specifying a list of waypoints
        ## for the end-effector to go through. If executing  interactively in a
        ## Python shell, set scale = 1.0.
        
        waypoints = []
    
        # Create a PoseStamped message to hold the waypoints
        wpose = PoseStamped()
        wpose.header.frame_id = frame_name
        wpose.pose.position.x = pos[0]
        wpose.pose.position.y = pos[1]
        wpose.pose.position.z = pos[2]
       
        pos_quaternion = tf_conversions.transformations.quaternion_from_euler(pos[3], pos[4], pos[5])

        wpose.pose.orientation.x = pos_quaternion[0]
        wpose.pose.orientation.y = pos_quaternion[1]
        wpose.pose.orientation.z = pos_quaternion[2]
        wpose.pose.orientation.w = pos_quaternion[3]
        
        
        transformed_pose_stamped = self.tf_buffer.transform(wpose, "base_footprint")
        waypoints.append(transformed_pose_stamped.pose)

       
        (plan, fraction) = self.arm.compute_cartesian_path(
            waypoints, 0.01, 0.0, avoid_collisions = avoidCollision  # waypoints to follow  # eef_step
        )  # jump_threshold

        # Note: We are just planning, not asking arm to actually move the robot yet:
        self.arm.execute(plan, wait=True)


    def go_home(self, gripper_position = [0.04, 0.04]):
        self.move_gripper(gripper_position)
        homeJoints = [0.20001658562027336, -1.3399542821641688, -0.1999877279909187, 
                    1.9400579944469158, -1.5699072941908805, 1.3699524984623137, -0.00011856858080827593]    
        joint_goal = self.arm.get_current_joint_values()
    
        joint_goal[0] = homeJoints[0]
        joint_goal[1] = homeJoints[1]
        joint_goal[2] = homeJoints[2]
        joint_goal[3] = homeJoints[3]
        joint_goal[4] = homeJoints[4]
        joint_goal[5] = homeJoints[5]
        joint_goal[6] = homeJoints[6]

        # The go command can be called with joint values, poses, or without any
        # parameters if you have already set the pose or joint target for the group
        success = self.arm.go(joint_goal, wait=True)

        # Calling ``stop()`` ensures that there is no residual movement
        self.arm.stop()

        # For testing:
        current_joints = self.arm.get_current_joint_values()
        return all_close(joint_goal, current_joints, 0.01), success


    def hand_wave(self, num_waves=3):
        """
        Perform a waving gesture by moving the arm back and forth.
        """
        # Define positions for a wave
        # You will need to know your robot's joint angles and limits
        # for the shoulder, elbow, and wrist for this to work.
        
        # First, get the current arm state (joint positions)
        current_state = self.arm.get_current_joint_values()

        # Assume the following joint positions (adjust for your robot):
        # Raise arm position (initial wave start position)
        wave_up = [0.07000998941539205, -0.8258644817377165, -3.2050192517142464, 2.244894700026078, -0.05274449211889464, -0.05541413782115412, 0.19723327627220383]
        wave_down = [0.07000998941539205, -1.008644817377165, -3.2050192517142464, 2.04894700026078, -0.05274449211889464, -0.05541413782115412, 0.19723327627220383] 
        # Define the wave movement: back and forth
        # Perform the waving motion a specified number of times
        
        for i in range(num_waves):
            # Raise the arm to start the wave
            self.arm.set_joint_value_target(wave_up)
            self.arm.go(wait=True)
            # Move the arm back down (simulating a wave motion)
            self.arm.set_joint_value_target(wave_down)
            self.arm.go(wait=True)
            

        # Return the arm to the initial position (resting pose)
        self.arm.set_joint_value_target(current_state)
        self.arm.go(wait=True)
        

    def hand_shake(self):
        """
        Perform a waving gesture by moving the arm back and forth.
        """
        # Define positions for a wave
        # You will need to know your robot's joint angles and limits
        # for the shoulder, elbow, and wrist for this to work.
        
        # First, get the current arm state (joint positions)
        current_state = self.arm.get_current_joint_values()

        # Assume the following joint positions (adjust for your robot):
        # Raise arm position (initial wave start position)
        shake_up = [1.522739744435411, -0.7213990044332632, -3.093896209483195, 1.182844354108822, 0.5097782026485257, 0.037647377040676444, -1.9811939115033754]
        shake_down = [1.522739744435411, -0.7213990044332632, -3.093896209483195, 1.1002844354108822, 0.5097782026485257, 0.037647377040676444, -1.9811939115033754] 
        self.move_gripper(position=[0.04, 0.04])
        # Define the wave movement: back and forth
        # Perform the waving motion a specified number of times
        for i in range(3):
            # Raise the arm to start the wave
            self.arm.set_joint_value_target(shake_up)
            self.arm.go(wait=True)
            # Move the arm back down (simulating a wave motion)
            self.arm.set_joint_value_target(shake_down)
            self.arm.go(wait=True)
            

        # Return the arm to the initial position (resting pose)
        self.arm.set_joint_value_target(current_state)
        self.arm.go(wait=True)


    def head_move_position_call(self, position=[0, 0]):
        trajectory = JointTrajectory()
        trajectory.joint_names = ["head_1_joint", "head_2_joint"]

        head_pub = rospy.Publisher("/head_controller/command", JointTrajectory, queue_size=10)
        point = JointTrajectoryPoint()
        point.positions = [position[0], position[1]]
        point.time_from_start = rospy.Duration(2.0)

        trajectory.points.append(point)

        rate = rospy.Rate(1)

        for _ in range(5):
            head_pub.publish(trajectory)
            rate.sleep()

    def head_move(self):
        trajectory = JointTrajectory()
        trajectory.joint_names = ["head_1_joint", "head_2_joint"]

        trajectory_up = JointTrajectory()
        trajectory_up.joint_names = ["head_1_joint", "head_2_joint"]

        head_pub = rospy.Publisher("/head_controller/command", JointTrajectory, queue_size=10)
        point = JointTrajectoryPoint()
        point.positions = [0, -0.5]
        point.time_from_start = rospy.Duration(0.5)

        point_up = JointTrajectoryPoint()
        point_up.positions = [0, 0.5]
        point_up.time_from_start = rospy.Duration(0.5)

        trajectory_up.points.append(point_up)

        trajectory.points.append(point)

        rate = rospy.Rate(3)

        for _ in range(5):
            head_pub.publish(trajectory)
            rate.sleep()
            head_pub.publish(trajectory_up)
            rate.sleep()

    def grasp(self, object="cup", shelf_position = "left"):

        if shelf_position == "left":

            if object == "measurement tape" or object == "screwdriver" or object == "toolbox":
                start_position = [
                    0.20001563843928377, 1.0495157342496433, 0.06526745769991871,
                    -2.5513353479881573, 1.8065538756538682, 0.19662746492963695,
                    -0.9942679501715539, 0.16379137858206794
                ]
            else:
                start_position = [
                    0.03596841035264681, 0.8191854638052432, 0.5229520363865069,
                    -1.5328352942981178, 2.067011928392827, 0.2407423875265853,
                    0.255152175362931, 0.2361737918973196]

        elif shelf_position == "center":
            if object == "measurement tape" or object == "screwdriver" or object == "toolbox":
                start_position = [
                    0.20001563843928377, 1.0495157342496433, 0.06526745769991871,
                    -2.5513353479881573, 1.8065538756538682, 0.19662746492963695,
                    -0.9942679501715539, 0.16379137858206794
                ]
                
            else:
                start_position = [
                    0.03596841035264681, 0.8191854638052432, 0.5229520363865069,
                    -1.5328352942981178, 2.067011928392827, 0.2407423875265853,
                    0.255152175362931, 0.2361737918973196]

        else:
            if object == "measurement tape" or object == "screwdriver" or object == "toolbox":
                start_position = [
                    0.20001563843928377, 1.0495157342496433, 0.06526745769991871,
                    -2.5513353479881573, 1.8065538756538682, 0.19662746492963695,
                    -0.9942679501715539, 0.16379137858206794
                ]
            else:
                start_position = [
                    0.03596841035264681, 0.8191854638052432, 0.5229520363865069,
                    -1.5328352942981178, 2.067011928392827, 0.2407423875265853,
                    0.255152175362931, 0.2361737918973196]

        
        self.go_to_joint_state(start_position, group="arm_torso")
        rospy.sleep(1.0)
        
        end_effector_frame = "gripper_grasping_frame"
        movement_sequence = {
            "measurement tape": [
                ([-0.13 + 0.3, 0.0, 0, -np.pi/2, 0, 0], [0.04, 0.04]),
                ([-0.13, 0, 0.04, -np.pi/2, 0, 0], None),
                ([-0.13 - 0.3, 0, 0, -np.pi/2, 0, 0], [0.02, 0.02])
            ],
            "toolbox": [
                ([-0.13 + 0.3, 0.12, 0, -np.pi/2, 0, 0], [0.04, 0.04]),
                ([-0.13, 0, 0.04, -np.pi/2, 0, 0], None),
                ([-0.13 - 0.3, 0, 0, -np.pi/2, 0, 0], [0.02, 0.02])
            ],
            "screwdriver": [
                ([-0.13 + 0.3, -0.12, 0, -np.pi/2, 0, 0], [0.04, 0.04]),
                ([-0.13, 0, 0.05, -np.pi/2, 0, 0], None),
                ([-0.13 - 0.3, 0, 0, -np.pi/2, 0, 0], [0.001, 0.001])
            ],
            "water bottle": [
                ([-0.13 + 0.25, 0.0, 0, -np.pi/2, 0, 0], [0.04, 0.04]),
                ([-0.13, 0, 0.05, -np.pi/2, 0, 0], None),
                ([-0.13 - 0.3, 0, 0, -np.pi/2, 0, 0], [0.02, 0.02])
            ],
            "lego brick": [
                ([-0.13 + 0.25, -0.15, 0, -np.pi/2, 0, 0], [0.04, 0.04]),
                ([-0.13, 0, 0.07, -np.pi/2, 0, 0], None),
                ([-0.13 - 0.3, 0, 0, -np.pi/2, 0, 0], [0.001, 0.001])
            ]
        }
        
        if object in movement_sequence:
            for pos, gripper_pos in movement_sequence[object]:
                if gripper_pos:
                    self.move_gripper(gripper_pos)
                self.plan_cartesian_path(end_effector_frame, pos=pos)
        
        self.go_to_joint_state(start_position, group="arm_torso")
        rospy.sleep(1)
        end_position = [
            0.03596841035264681, 0.8192928438847045, 0.9626734617803616,
             -1.3650002301001702, 0.7747080120879289, 0.5567832344026737,
             -0.5265056357128566, 0.24185211533758702]

        self.go_to_joint_state(end_position, group="arm_torso")

        self.move_gripper([0.04, 0.04])


    def frame_exists(self, frame_id):
        # checks if a frame exists
        try: 
            self.tf_buffer.lookup_transform('base_link', frame_id, rospy.Time(0))
            return True 
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return False

def call_server(input_string):
    url = 'http://172.27.15.38:5015/process'
    headers = {'Content-Type': 'application/json'}
    data = {'input_string': input_string}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        result = response.json()
        return result['result']
    
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None
    
# extract the json string from the response
def extract_between_braces(input_string):
    start = input_string.find('{')
    end = input_string.find('}', start) + 1
    if start > 0 and end > start:
        return input_string[start:end]
    else:
        return None
    
def main():
    try:
        # Initialize the node
        move_node = MoveGroupPythonInterface()

        current_joints = move_node.arm.get_current_joint_values()
        current_joints = move_node.arm_torso.get_current_joint_values()
        print(current_joints)

        
        #transform = move_node.tf_buffer.lookup_transform("base_link", "gripper_grasping_frame", rospy.Time(0))
        #result = move_node.frame_exists("gripper_grasping_frame")
        #print(result)

        move_node.grasp(object="lego brick")
        #move_node.grasp(object="toolbox")
        #move_node.head_move()
        # Example usage
        """while True:
            user_input = input("User: ")
            
            result = call_server(user_input)
            
            if result:
                print(result)
                result = extract_between_braces(result)
                result = ast.literal_eval(result)
                print(result["function"])
                print(result["response"])
                response = result["response"]

            client = SimpleActionClient('/tts', TtsAction)
            client.wait_for_server()
            # Create a goal to say our sentence
            goal = TtsGoal()
            goal.rawtext.text = response
            goal.rawtext.lang_id = "en_GB"
            # Send the goal and wait
            client.send_goal(goal)
            #client.send_goal(goal)

            # Ensure 'function' key exists and contains a callable function in the list
            function_call = result["function"]
            print("This is the output of the function call  : ", function_call)
            if function_call:
                # Get the method name from the response (e.g., 'hand_wave()')

                method_name = function_call[0][0:function_call[0].find('(')]

                print("THIS IS THE METHOD NAME : ", method_name)

                sep = "'"
                temp = function_call[0]
                
                if method_name == "grasp": 
                    arg = temp[temp.find(sep)+len(sep):temp.rfind(sep)]
                    print("THIS IS ARG : ", arg)

                # Use hasattr to check if the method exists in the move_node object
                if hasattr(move_node, method_name):  # Check if the method exists
                    method = getattr(move_node, method_name)  # Get the method dynamically
                    if method_name == "grasp": 
                        method(arg)  # Call the method
                    else: 
                        method()

                else:
                    print(f"Method '{method_name}' not found in MoveNode class.")
            else:
                print("No function to execute.")
        """
        #move_node.grasp_on_table()
        #position = [0.04, 0.04]
        #move_node.move_gripper(position)

        #move_node.move_head()
        #move_node.gohome()
        #move_node.wave()
        #move_node.handshake()


        print("Program is done")
    except rospy.ROSInterruptException:
        return
    except KeyboardInterrupt:
        exit()


if __name__ == "__main__":
    main()
