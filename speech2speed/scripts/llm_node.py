#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
#from langchain.agents import initialize_agent, AgentType
from langchain.agents import create_agent # newer version of langchain
from langchain.tools import tool

# Import your new message type alongside the standard ones
from speech2speed_interface.msg import TrajContext 
from speech2speed_interface.srv import String
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float32

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

import os 
import time
from dataLogger import export_string

# ===================== LangChain Tool =====================
def make_publish_context_tool(node):
    @tool
    def publish_context(goal_x: float, goal_y: float, goal_z: float, v_const: float, a: float) -> str:
        """ 
        Publishes high-level trapezoidal trajectory parameters to the trajectory generation node.
        Use this tool to specify the TRAPEZOIDAL trajectory parameters, in order to command the robot.
        Assume the robot starts at (0,0,0) with 0 velocity, and points towards the goal.
        The tobot front is positive x direction, left is positive y direction, and up is positive z direction.
        
        Args:
            goal_x: The X coordinate of the goal position (meters).
            goal_y: The Y coordinate of the goal position (meters).
            goal_z: The Z coordinate of the goal position (meters).
            v_const: The constant phase velocity of the trajectory (m/s).
            a: The acceleration/deceleration rate in the acceleration phase (m/s^2).
            
        The robot has the size of about 0.40m x 0.50m.
        The robot has the following kinematic constraints:
            Min velocity: 0.1 m/s
            Max velocity: 0.2 m/s
            Min acceleration: 0.02 m/s^2
            Max acceleration: 0.04 m/s^2

        Returns: success or error message
        """
        try:
            msg = TrajContext()
            
            # 1. Set the goal vector
            msg.s_goal = Vector3(x=float(goal_x), y=float(goal_y), z=float(goal_z))
            
            # 2. Set the kinematics
            msg.v_const = float(v_const)
            msg.a = float(a)
            
            # 3. Set defaults for FM to overwrite/use
            # Assuming q_init is 0,0,0 initially, or the FM ignores this until it reads current state
            msg.q_init = Vector3(x=0.0, y=0.0, z=0.0)
            msg.part = msg.CONSTANT 

            # Publish the message to the topic
            node.context_publisher.publish(msg)

            return f"Success: Published trajectory context -> Goal:({goal_x},{goal_y},{goal_z}), Vel:{v_const}, Accel:{a}"

        except Exception as e:
            return f"Error publishing context topic: {str(e)}"
            
    return publish_context


class LlmNode(Node):
    def __init__(self):
        super().__init__('llm_node')

        # Updated System Instructions for the new paradigm
        self.systemInstructions = [
            {"role": "system", "content": "You are the high-level planning brain of a mobile robot, attached with a LangChain tool function. Your job is to interpret the prompt and plan the trapezoidal trajectory, by specifying the kinematic parameters."},
            {"role": "system", "content": "When given a prompt, you must infer and  determine the appropriate values of spatial goal (x, y, z), a constant phase velocity (v_const), and an acceleration rate (a)."},
            {"role": "system", "content": "Call the 'publish_context' tool with these parameters to command the robot trajectory. The publish_context tool will handle publishing the context message to the ROS2 trajectory generation node."},
            {"role": "system", "content": "You must call a tool with the correct format of trapezoidal trajectory parameters in the tool arguments."},
            {"role": "system", "content": "Call the tool once for the entire trajectory."}
        ]
        self.history = self.systemInstructions.copy()

        # Create ROS2 Interfaces =======================================
        self.create_service(String, 'LlmPrompt', self.prompt_callback)
        
        # CHANGED: We now use a Publisher instead of a Service Client
        self.context_publisher = self.create_publisher(TrajContext, 'traj_context', 10)

        # Initialize LLM
        '''
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            self.get_logger().warning("OPENAI_API_KEY not set!")
            
        self.llm = ChatOpenAI(model="gpt-5-nano", api_key=api_key)
        '''

        # initialize Google Gemini LLM instead
        # Initialize LLM
        api_key = os.environ.get("GOOGLE_API_KEY", None)
        if api_key is None:
            self.get_logger().warning("GOOGLE_API_KEY not set!")
            
        # Using Gemini Flash for fast, tool-calling compatible inference
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key,
            temperature=0.0 # Recommended 0 for deterministic robotics planning
        )


        # Register the new tool
        self.publish_context_tool = make_publish_context_tool(self)
        tools = [self.publish_context_tool]

        # Create Agent
        self.agent = create_react_agent(self.llm, tools)
        self.get_logger().info("LlmNode initialized. Listening on /LlmPrompt, publishing to /traj_context.")

    def prompt_callback(self, req, res):
        self.log_info(f"Received message: {req.prompt}")
        self.start_time = self.get_clock().now()
        res = String.Response()
        user_text = req.prompt.strip()

        if user_text.lower() == "reset history":
            system_prompts = [m for m in self.history if m.get("role") == "system"]
            self.history = system_prompts[:] if system_prompts else self.systemInstructions.copy()
            res.response = "Conversation history cleared."
            return res

        self.history.append({"role": "user", "content": user_text})

        try:
            result = self.agent.invoke({"messages": self.history})
            assistant_text = None

            # Update for Gemini multimodal response compatibility
            if isinstance(result, dict) and 'messages' in result:
                try:
                    msgs = result['messages']
                    if msgs:
                        last = msgs[-1]
                        raw_content = getattr(last, "content", None) or last.get("content", None)
                        
                        # --- THE FIX: Handle the Gemini multimodal list ---
                        if isinstance(raw_content, list):
                            # Extract text from the list block (e.g. [{'type': 'text', 'text': '...'}])
                            assistant_text = str(raw_content[0].get('text', ''))
                        else:
                            # It's already a standard string
                            assistant_text = str(raw_content) if raw_content is not None else None
                        # --------------------------------------------------
                except Exception:
                    assistant_text = None

            if assistant_text is None:
                if isinstance(result, dict) and result.get("output"):
                    assistant_text = result.get("output")
                elif isinstance(result, str):
                    assistant_text = result

            if assistant_text is None:
                assistant_text = str(result)

            self.log_info("\n================================================\n")
            self.log_info(f"Agent reply: {assistant_text}")
            self.history.append({"role": "assistant", "content": assistant_text})

            MAX_HISTORY = 200 
            if len(self.history) > MAX_HISTORY:
                system_prompts = [m for m in self.history if m.get("role") == "system"]
                non_system = [m for m in self.history if m.get("role") != "system"]
                kept_non_system = non_system[-(MAX_HISTORY - len(system_prompts)):]
                self.history = system_prompts + kept_non_system

            self.log_info(f"Thinking time: {(self.get_clock().now() - self.start_time).nanoseconds / 1e6} ms")
            res.response = str(assistant_text)
            
        except Exception as e:
            self.get_logger().error(f"Agent error: {e}")
            res.response = f"Error: {str(e)}"

        export_string(text = f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        export_string(text = f'----------------------------------------------------------------')
        return res

    def log_info(self, text: str, file_name = "saved_log.txt"):
        export_string(text, file_name)
        self.get_logger().info(text)

def main(args=None):
    rclpy.init(args=args)
    node = LlmNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()