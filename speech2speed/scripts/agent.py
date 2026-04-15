#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool

from speech2speed_interface.srv import TwistTraj, String
from speech2speed_interface.msg import TwistSimpleStamped
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3

from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from speech2speed.llm import HFChatWrapper

from langchain_core.output_parsers import StrOutputParser
import os 

import time
from dataLogger import export_string

# run this before running this script:
# chmod +x ~/physical_ai/ws/install/speech2speed/lib/speech2speed/agent.py

# ================================================================
# IMPORTANT: Do NOT hardcode API keys in source. Set OPENAI_API_KEY
# in your environment (export OPENAI_API_KEY="sk-...").
# ================================================================

# ===================== LangChain Tool =====================
def make_call_traj_service(node):
    @tool
    def call_traj_service(vectors: str) -> str:
        """ Calls the ROS2 service name "/Traj" to send a trajectory. 
            This function will return success or error message according to the service response.

            Input: string of vectors in format 'time1:v_x1,v_y1,v_z1,w_x1,w_y1,w_z1 time2:v_x2,v_y2,v_z2,w_x2,w_y2,w_z2; ...' 
            Example Input: '0.0:1.0,2.0,3.0,0.1,0.2,0.3; 1.0:7.0,8.0,9.0,0.4,0.5,0.6'

            Output: success or error message"""
        try:
            vector_list = []
            for item in vectors.split(';'):
                item = item.strip()
                if not item:
                    continue
                time_str, coords = item.split(':')
                vx_str, vy_str, vz_str, wx_str, wy_str, wz_str = coords.split(',')
                vector_list.append((float(time_str), float(vx_str), float(vy_str), float(vz_str), float(wx_str), float(wy_str), float(wz_str)))

            req = TwistTraj.Request()
            for t, vx, vy, vz, wx, wy, wz in vector_list:
                msg = TwistSimpleStamped()
                msg.time = Float32(data=t)
                msg.linear = Vector3(x=vx, y=vy, z=vz)
                msg.angular = Vector3(x=wx, y=wy, z=wz)
                req.twist_traj.append(msg)

            # Use the node's client directly
            future = node.scheduler_client.call_async(req)

            return f"Service call success."

        #     if future.done():
        #         return f"Service call successful: {future.result()}"
        #     else:
        #         return f"Service call failed: {future.exception()}"
        except Exception as e:
            return f"Error calling /Traj service: {str(e)}"
    return call_traj_service


class AgentNode(Node):
    instance = None  # keep reference for tools
    scheduler_client = None

    def __init__(self):
        super().__init__('agent_node')
        AgentNode.instance = self

        # Conversation history stored as list of {"role": "user"/"assistant"/"system", "content": "..."}
        # Prepopulate with an optional system prompt to guide agent behavior.
        self.systemInstructions = [
            {"role": "system", "content": "You are a agentic brain of a robot that can call ROS2 tools. Your role is to plan the trajectory and send it as formatted data to the robot."},
            {"role": "system", "content": "When given a prompt to generate a trajectory, call a tool with the correct format of trajectory data."},
            {"role": "system", "content": "You must call the tool function in order to make the robot work. Ensure to response the full data in the correct format."},
            {"role": "system", "content": "Response the full trajectory data in one single tool call, even if it is long. Dont hesitate to generate long trajectories."}
        ]
        self.history = self.systemInstructions.copy()

        # Create services/clients =======================================
        self.create_service(String, 'Prompt', self.prompt_callback)
        AgentNode.scheduler_client = self.create_client(TwistTraj, 'Traj')

        # Initialize LLM (use env var for API key)
        api_key = os.environ.get("OPENAI_API_KEY", None)
        if api_key is None:
            self.get_logger().warning("OPENAI_API_KEY not set. Make sure to set it in the environment.")
        # Use ChatOpenAI wrapper from your environment (adjust args if your wrapper differs)
        self.llm = ChatOpenAI(model="gpt-5-nano", api_key=api_key)

        # Register tools
        self.call_traj_tool = make_call_traj_service(self)
        tools = [self.call_traj_tool]

        # Create the agent using your existing helper
        # create_react_agent accepts an llm + tools; it will use messages passed in invoke()
        self.agent = create_react_agent(self.llm, tools)

        self.get_logger().info("AgentNode initialized and ready.")

    def prompt_callback(self, req, res):
        """
        ROS2 service callback. The incoming req.prompt is treated as the latest user utterance.
        The agent will be invoked with the full conversation history so it can remember previous turns.
        Special commands:
          - "reset history" (case-insensitive): clears conversation history (keeps system prompt)
        """
        self.log_info(f"Received message: {req.prompt}")
        self.start_time = self.get_clock().now()
        res = String.Response()

        user_text = req.prompt.strip()

        # If user wants to reset conversation history, allow it
        if user_text.lower() == "reset history":
            # Preserve system prompt if present
            system_prompts = [m for m in self.history if m.get("role") == "system"]
            self.history = system_prompts[:] if system_prompts else self.systemInstructions.copy()
            res.response = "Conversation history cleared."
            return res

        # Append user's message to history
        self.history.append({"role": "user", "content": user_text})

        try:
            # Invoke the agent with the full history so it has context.
            # Many chat/agent APIs accept a `messages` parameter as a list of {role, content}.
            # create_react_agent's .invoke is expected to accept that format in your original code.
            result = self.agent.invoke({"messages": self.history})

            # The structure returned by your agent implementation previously had result['messages'].
            # We'll attempt to extract assistant content robustly.
            assistant_text = None

            # Case 1: result provides a messages list (each with content)
            if isinstance(result, dict) and 'messages' in result:
                try:
                    # If messages is a list of objects with .content or ['content']
                    msgs = result['messages']
                    if msgs:
                        last = msgs[-1]
                        assistant_text = getattr(last, "content", None) or last.get("content", None)
                except Exception:
                    assistant_text = None

            # Case 2: result directly returns a text string
            if assistant_text is None:
                if isinstance(result, dict) and result.get("output"):
                    assistant_text = result.get("output")
                elif isinstance(result, str):
                    assistant_text = result

            # Fallback
            if assistant_text is None:
                assistant_text = str(result)

            # Log and append assistant reply to history
            self.log_info("\n================================================\n")
            self.log_info(f"Agent reply: {assistant_text}")
            self.history.append({"role": "assistant", "content": assistant_text})

            # Optionally limit history length to avoid unbounded growth
            MAX_HISTORY = 200  # total messages (tweak as needed)
            if len(self.history) > MAX_HISTORY:
                # Keep system prompt(s) + last (MAX_HISTORY - system_count) messages
                system_prompts = [m for m in self.history if m.get("role") == "system"]
                non_system = [m for m in self.history if m.get("role") != "system"]
                kept_non_system = non_system[-(MAX_HISTORY - len(system_prompts)):]
                self.history = system_prompts + kept_non_system

            self.log_info(f"Thinking time: {(self.get_clock().now() - self.start_time).nanoseconds / 1e6} ms")
            res.response = assistant_text
        except Exception as e:
            self.get_logger().error(f"Agent error: {e}")
            res.response = f"Error: {str(e)}"

        # export_string time stamp now at the end
        export_string(text = f'Time: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
        export_string(text = f'----------------------------------------------------------------')
        #export_string(text='\n')
        return res

    def log_info(self, text: str, file_name = "saved_log.txt"):
        export_string(text, file_name)
        self.get_logger().info(text)

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
