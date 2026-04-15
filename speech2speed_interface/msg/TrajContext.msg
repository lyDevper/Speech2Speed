# TrajectPart Enum Constants
uint8 ACCELERATION = 0
uint8 CONSTANT = 1
uint8 DECELERATION = 2

# Goal position (3D vector)
geometry_msgs/Vector3 s_goal

# Kinematic parameters
float32 v_const
float32 a

# Current pose at window start (Optional/Filled by FM if needed)
geometry_msgs/Vector3 q_init

# Current state machine part (Default: CONSTANT)
uint8 part