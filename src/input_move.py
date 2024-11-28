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


    def continuous_cartesian_motion(self, waves=1):
        """
        Make the robot arm perform continuous Cartesian movement through multiple poses.
        """
        wave_up = [0.07000998941539205, -0.8258644817377165, -3.2050192517142464, 2.244894700026078, -0.05274449211889464, -0.05541413782115412, 0.19723327627220383]
        self.go_to_joint_state(wave_up, "arm_torso")
        # Get the current pose of the end effector
        start_pose = self.arm.get_current_pose().pose
        print(start_pose)
        # Define a series of poses to move through

        euler = tf_conversions.transformations.euler_from_quaternion([start_pose.orientation.x, start_pose.orientation.y, start_pose.orientation.z, start_pose.orientation.w])
        quat = tf_conversions.transformations.quaternion_from_euler(euler[0], euler[1], euler[2])

        poses = [
            Pose(position=start_pose.position, orientation=start_pose.orientation),  # Current position
            Pose(position=Point(start_pose.position.x + 0.04, start_pose.position.y       , start_pose.position.z - 0.06), orientation=start_pose.orientation),
            Pose(position=Point(start_pose.position.x + 0.04, start_pose.position.y - 0.07, start_pose.position.z - 0.06), orientation=start_pose.orientation),
            Pose(position=Point(start_pose.position.x + 0.04, start_pose.position.y - 0.17, start_pose.position.z - 0.07), orientation=start_pose.orientation),
            Pose(position=Point(start_pose.position.x + 0.04, start_pose.position.y - 0.05, start_pose.position.z - 0.06), orientation=start_pose.orientation),
            Pose(position=Point(start_pose.position.x + 0.04, start_pose.position.y - 0.17, start_pose.position.z - 0.07), orientation=start_pose.orientation),
            Pose(position=Point(start_pose.position.x + 0.04, start_pose.position.y - 0.05, start_pose.position.z - 0.06), orientation=start_pose.orientation),
        ]
        for i in range(waves):
            # Plan and execute the Cartesian path through the poses
            (plan, fraction) = self.arm.compute_cartesian_path(poses, 0.05, 0.0)  # 0.01 is the step size
            if fraction == 1.0:
                self.arm.execute(plan, wait=True)
            else:
                rospy.logwarn("Cartesian path planning failed to complete.")


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
        

    def hand_shake(self, shakes=3):
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
        for i in range(shakes):
            # Raise the arm to start the wave
            self.arm.set_joint_value_target(shake_up)
            self.arm.go(wait=True)
            # Move the arm back down (simulating a wave motion)
            self.arm.set_joint_value_target(shake_down)
            self.arm.go(wait=True)
            

        # Return the arm to the initial position (resting pose)
        self.arm.set_joint_value_target(current_state)
        self.arm.go(wait=True)


    def head_move(self, position=[0, 0]):
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

    def grasp(self, object="cup"):

        start_position = [0.349675502384862, 1.3545978800104312, 1.0208427848256607, 0.1870407584101612, 1.8024274126002853, -1.416253529267913, 0.8398546052777839, 0.23353193570618702]
        self.go_to_joint_state(start_position, group="arm_torso")
        
        rospy.sleep(1.0)

        start_position = [0.3495973101884432, 0.6028146036908067, 1.0207660847689026, 0.038411388424427435, 1.769645808341898, -1.3717273219062864, 0.632098367458853, -0.39945199084057714]
        self.go_to_joint_state(start_position, group="arm_torso")
        
        rospy.sleep(1.0)
        
        end_effector = self.arm.get_current_pose().pose
        print(end_effector)
        end_effector_frame = "gripper_grasping_frame"
        if end_effector is not None:
            # Access the translation values
            x = end_effector.position.x
            y = end_effector.position.y
            z = end_effector.position.z
            # Access the quaternion values for orientation
            qx = end_effector.orientation.x
            qy = end_effector.orientation.y
            qz = end_effector.orientation.z
            qw = end_effector.orientation.w
            euler = tf_conversions.transformations.euler_from_quaternion([qx, qy, qz, qw])

            rospy.loginfo("Euler values based on transform: {}".format(euler))

        if object == "cup":

            position = [0.04, 0.04]
            self.move_gripper(position)

            start_pos = [-0.13, 0.0, 0.13, np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0)]

            initial_pos = start_pos

            self.plan_cartesian_path(end_effector_frame, pos=initial_pos)

            position = [0.02, 0.02]
            self.move_gripper(position)

            start_pos = [-0.17, 0.0, -0.12, np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0)]

            initial_pos = start_pos

            self.plan_cartesian_path(end_effector_frame, pos=initial_pos)


        elif object == "measurement tape":
            position = [0.04, 0.04]
            self.move_gripper(position)

            start_pos = [-0.13, 0.14, 0, np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0)]

            initial_pos = start_pos

            self.plan_cartesian_path(end_effector_frame, pos=initial_pos)

            start_pos = [-0.05, 0, 0.09, np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0)]

            initial_pos = start_pos

            self.plan_cartesian_path(end_effector_frame, pos=initial_pos)

            position = [0.01, 0.01]
            self.move_gripper(position)

            start_pos = [-0.17, 0, -0.1, np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0)]

            initial_pos = start_pos

            self.plan_cartesian_path(end_effector_frame, pos=initial_pos)
        
        elif object == "tool box":
            position = [0.04, 0.04]
            self.move_gripper(position)

            start_pos = [-0.13, -0.15, 0, np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0)]

            initial_pos = start_pos

            self.plan_cartesian_path(end_effector_frame, pos=initial_pos)

            start_pos = [-0.06, 0, 0.11, np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0)]

            initial_pos = start_pos

            self.plan_cartesian_path(end_effector_frame, pos=initial_pos)

            position = [0.01, 0.01]
            self.move_gripper(position)

            start_pos = [-0.17, 0, -0.1, np.deg2rad(-90), np.deg2rad(0), np.deg2rad(0)]

            initial_pos = start_pos

            self.plan_cartesian_path(end_effector_frame, pos=initial_pos)
        

        
        if object != "cup":
            start_position = [0.3495973101884432, 0.6028146036908067, 1.0207660847689026, 0.038411388424427435, 1.769645808341898, -1.3717273219062864, 0.632098367458853, -0.39945199084057714]
            self.go_to_joint_state(start_position, group="arm_torso")
        
            rospy.sleep(1.0)
            self.go_home(gripper_position=[0.02,0.02])

            start_position = [0.149480021893815, 1.2404068355090583, -0.39097515991961834, -2.0907054871220603, 1.5351430548098606, -0.15168997280333177, 0.3028130395796744, -0.3437858947852056]
            self.go_to_joint_state(start_position, group="arm_torso")

            rospy.sleep(1)
        else:
            start_position = [0.3495973101884432, 1.8059010139747502, 1.0209194848824188, 0.0673733298562641, 1.1372691803831936, -1.5231505154027998, 1.3938144253499627, -0.02399531654105813]
            self.go_to_joint_state(start_position, group="arm_torso")
        
            rospy.sleep(1.0)

        position = [0.04, 0.04]
        self.move_gripper(position)

        rospy.sleep(1)

        self.go_home()


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
        move_node.grasp(object="tool box")
        move_node.grasp(object="measurement tape")
        move_node.grasp(object="cup")
        #move_node.go_home()
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
            client.send_goal_and_wait(goal)
            #client.send_goal(goal)

            # Ensure 'function' key exists and contains a callable function in the list
            function_call = result["function"]
            print(function_call)
            if function_call:
                # Get the method name from the response (e.g., 'hand_wave()')
                method_name = function_call[0].strip('()')  # Remove the parentheses

                # Use hasattr to check if the method exists in the move_node object
                if hasattr(move_node, method_name):  # Check if the method exists
                    method = getattr(move_node, method_name)  # Get the method dynamically
                    method()  # Call the method
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
