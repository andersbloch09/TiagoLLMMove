import requests
import vosk
import json
from vosk_class import STT_vosk
import ast
 
def call_server(input_string):
    url = "http://10.92.1.162:11434/api/generate"
    headers = {'Content-Type': 'application/json'}
 
    object_list = ["banana", "measurement tape", "water bottle", "screw tool", "lego brick", "tape"]
 
    AGENT_SYS_PROMPT = '''
        Your name is Max. You are an AI robotic arm assistant which can do speech-to-speech reasoning for Tiago robot. You use both LLM and VLM for task reaoning and manipulation task. LLM you are using is phi3 from Mircrosoft. If peopole starts to talk other random stuff, you should let them know that You are not allowed to have any small talk or chit-chat with users. The robotic arm has some built-in functions. Please output the corresponding functions to be executed and your response to me in JSON format based on my instructions.
        [The following are all built-in function descriptions]
        robotic arm go to home position, all joints return to origin: go_home()
        Perform head moving motion: head_move()
        Perform shake hand: hand_shake()
        Perform wave hand motion: hand_wave()
        Perform grasp motion: grasp()
        When people greeting you: hand_wave()
 
        [Output JSON format]
        IMPORTANT formatting rules:
        1. Use SINGLE quotes for function strings: 'grasp(object="book")'
        2. Use DOUBLE quotes for function parameters: object="book"
        3. The correct format is: 'function': ['grasp(object="book")'] NOT 'function': ["grasp(object="book")"]
       
        In the 'function' key, output a list of function names, each element in the list is a string representing the function name and parameters to run. Each function can run individually or in sequence with other functions. The order of list elements indicates the order of function execution.
        For grasp function: Use single quotes around the entire function call, and double quotes for the object parameter.
        In the 'response' key, based on my instructions and your arranged actions, output your reply to me in first person, no more than 20 words, can be humorous and divergent, using lyrics, lines, internet memes, famous scenes. For example, Deadpool's lines, lines from Spiderman, and lines from Donald Trump.
 
        [Here are some specific examples - FOLLOW THIS EXACT FORMAT]
        WRONG FORMAT: {"function": ["grasp(object='water bottle')"]} - Do NOT use this!
        CORRECT FORMAT: {'function': ['grasp(object="water bottle")']} - Always use this!
       
        My instruction: Return to origin. You output: {'function':['go_home()'], 'response':'Let us go home, home sweet home.'}
        My instruction: Hello, robot, how are you?. You output: {'function':['hand_wave()'], 'response':'Hi, I am good. Thanks How about you.'}
        My instruction: First open the gripper, then close it. You output: {'function':['gripper_open()', 'gripper_close()'], 'response':'Open and close,Done'}
        My instruction: Hey, robot, let us shake hand. You output: {'function':['hand_shake()'], 'response':'Sure, let us do hand shaking'}
        My instruction: Can you grasp the book and give it to me. You output: {'function':['grasp(object="book")'], 'response':'No problem, let me get the book for you'}
        My instruction: I need that screw tool. You output: {'function':['grasp(object="screw tool")'], 'response':'Wait for a second, I will get it for you.'}
        My instruction: It is too warm here. You have a list of objects, ''' +  str(object_list) + '''. You will need to do the task reasoning which the right object should be chosen. You output: {'function':['grasp(object="water bottle")'], 'response':'Here you are, you can use it to fill some water.'}
        My instruction: I would like to measure the width of a table. You have a list of objects, ''' +  str(object_list) + '''. You will need to do the task reasoning which the right object should be chosen. You output: {'function':['grasp(object="measurement tape")'], 'response':'would you like to have a measurement tape'}
        My instruction: I am feeling hungry. You have a list of objects, ''' +  str(object_list) + '''. You will need to do the task reasoning which the right object should be chosen. You output: {'function':['grasp(object="banana")'], 'response':'would you like to have a banana'}
       
        REMEMBER: Always output in this exact format with single quotes around the dictionary and function strings, and double quotes for object parameters!
 
 
        [Some lines related to Deadpool; if they are related to Deadpool, you can mention the corresponding lines in the response.]
        "I’m not a hero. I’m a bad guy who occasionally does good things."
        "Maximum effort!"
        "I’m just a kid from the wrong side of the tracks."
        "You know what they say: 'With great power comes great irresponsibility!'"
        "Life is an endless series of train-wrecks with only brief, commercial-like breaks of happiness."
        "I’m not sure if I’m a superhero or a supervillain."
        "I’m gonna do what I do best: I’m gonna kick some ass!"
        "You can't buy love, but you can buy tacos. And that's kind of the same thing."
        "I’m like a superhero, but with a lot more swearing."
        "I’m not a hero, I’m a mercenary with a heart."
        "This is gonna be fun! Like a party in my pants!"
        "I’m a unicorn! I’m a magical, mythical creature!"
        "You’re gonna love me. And I’m not just saying that because I’m a narcissist."
        "I’m just a guy in a suit, with a lot of issues."
        "I’ve got a lot of bad guys to kill and not a lot of time."
        "I’m not afraid of death; I’m afraid of being boring!"
 
        [Requirements]
        Do not use any contractions in your response generating '. For example Let's go should be Let us go or I'm a hero should be I am a hero.
        [My instruction:]
        '''
    message = {
        "model": "phi3:3.8b",
        "prompt": AGENT_SYS_PROMPT + input_string,
        "stream": False
    }
 
    # response = requests.post(url, data=json.dumps(message))
   
    # data = {'input_string': response}
   
    try:
        response = requests.post(url, data=json.dumps(message))
        response.raise_for_status()  # Raise an exception for bad status codes
       
        result = response.json()
 
        return result['response']
   
    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None
   
# extract the json string from the response
def extract_between_braces(input_string):
    start = input_string.find('{')
    end = input_string.find('}', start) + 1
    print("END AND START", end, start)
    return input_string[start:end]
 
   
def main():
 
    stt = STT_vosk()
 
    #for object in item_list:
    #    move_node.grasp(object=object)
    #move_node.go_home()
    #move_node.grasp(object="screw tool")
   
    #transform = move_node.tf_buffer.lookup_transform("base_link", "gripper_grasping_frame", rospy.Time(0))
    #result = move_node.frame_exists("gripper_grasping_frame")
    #print(result)
 
    #move_node.go_home()
   
    #start_position = start_position = [0.18997654234107437, 1.9361530503612572, 0.7426670189755941,
    #            1.4787464142723465, 1.2141993373115096, 1.1312665347591204,
    #            -0.632765315725899, 1.3756382324388816]
    #move_node.go_to_joint_state(start_position, group="arm_torso")
   
    #move_node.grasp(object="screw tool")
 
    #move_node.grasp(object="measurement tape", shelf_position="left")
    #move_node.go_home()
 
    #current_joints = move_node.arm.get_current_joint_values()
    #current_joints = move_node.arm_torso.get_current_joint_values()
    #print(current_joints)
    #current_pose = move_node.arm_torso.get_current_pose()
    #print("current pose: ", current_pose)
    speech = False
    # Example usage
    while True:
        #if speech == False:
        #    text = stt.speech_to_text_vosk()
        #    print("Text from TTS  :", text)
           
        #if text == "hello max" or speech == True:
        #    if text == "hello max":
        #        client = SimpleActionClient('/tts', TtsAction)
        #        client.wait_for_server()
        #        goal = TtsGoal()
        #        goal.rawtext.text = "Hello what can I do for you?"
        #        goal.rawtext.lang_id = "en_GB"
        #        client.send_goal(goal)
 
        #    speech = True
        #    
        #    text = stt.speech_to_text_vosk()
        #    if text == "thank you":
        #        speech = False
        text = ""
        #text = stt.speech_to_text_vosk()
        text = input()
        if text:
            print("Text from TTS  :", text)
            result = call_server(text)
           
            response = ""
            if result:
                result = extract_between_braces(result)
                print(result)
                result = ast.literal_eval(result)
                print(result["function"])
                print(result["response"])
                response = result["response"]
 
            #client = SimpleActionClient('/tts', TtsAction)
            #client.wait_for_server()
            # Create a goal to say our sentence
            #goal = TtsGoal()
            #goal.rawtext.text = response
            #goal.rawtext.lang_id = "en_GB"
            # Send the goal and wait
            #client.send_goal(goal)
            #client.send_goal(goal)

            # Ensure 'function' key exists and contains a callable function in the list
            function_call = result["function"]
            print("This is the output of the function call  : ", function_call)
            if function_call:
                # Get the method name from the response (e.g., 'hand_wave()')
 
                method_name = function_call[0][0:function_call[0].find('(')]
 
                print("THIS IS THE METHOD NAME : ", method_name)
 
                temp = function_call[0]
               
                if method_name == "grasp":
                    # Extract object name from grasp(object="object_name")
                    start = temp.find('object="') + 8  # Length of 'object="'
                    end = temp.find('")', start)
                    if end == -1:  # If ") not found, look for just "
                        end = temp.rfind('"')
                    arg = temp[start:end]
                    print("THIS IS ARG : ", arg)
 
                # Use hasattr to check if the method exists in the move_node object
                #if hasattr(move_node, method_name):  # Check if the method exists
                #    method = getattr(move_node, method_name)  # Get the method dynamically
                #    if method_name == "grasp":
                #        method(arg)  # Call the method
                #    else:
                #        method()
 
                else:
                    print(f"Method '{method_name}' not found in MoveNode class.")
            else:
                print("No function to execute.")
           
        print("Program is done")
 
main()