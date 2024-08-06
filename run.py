
# Install minimal dependencies (`torch`, `transformers`, `timm`, `tokenizers`, ...)
# > pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt
from transformers import AutoModelForVision2Seq, AutoProcessor

from robotic_arm_package.robotic_arm import *
from PIL import Image
from camera import realsense
import torch
import cv2
import numpy as np

# variables >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# initial position of the end of arm
initial_position = {"pos": None,
                    "joint": (170.355, -28.3, -64, 6.78, -82.19, 191.235)}

# initialize robot position with pos when True, else initialize with joint
initialize_with_pos = False

# 0 for close, 255 for open
gripper_open = 0

# the maximum z coordinate the robot can reach
z_max = 0.182

# instr = "flip the bottle"
instr = "lift the cube near the doll"

# motion inverse
inverse_x = True
inverse_y = True
inverse_z = False
# rotation inverse
inverse_x_rot = True
inverse_y_rot = True
inverse_z_rot = False


# list of inverse-or-not variables
is_inverse_list = [inverse_x, inverse_y, inverse_z, inverse_x_rot, inverse_y_rot, inverse_z_rot]

# functions >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


def refresh_cam(color_image, delay=1, name="color_img"):
    cv2.imshow(name, color_image)
    cv2.waitKey(delay)


# main >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# config robot
robot = Arm(RM65, "192.168.1.18")
robot.Set_Collision_Stage(4, True)
robot.Pos_Step_Cmd(0, 0.01, 1, True)
# move end of arm to initial position
if initialize_with_pos:
    robot.Movej_P_Cmd(initial_position["pos"], 20, 0, 0, True)
else:
    robot.Movej_Cmd(initial_position["joint"], 20, 0, 0, True)

# set gripper
robot.Set_Tool_Voltage(3)
robot.Set_Modbus_Mode(1, 115200, 1, True)
robot.Write_Registers(1, 1000, 1, [0, 0], 9, True)
robot.Write_Registers(1, 1000, 1, [0, 1], 9, True)
time.sleep(2)

# Load Processor & VLA
# 智能 易用 可靠 通用
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to("cuda:0")

# initialize camera
camera = realsense.RealSenseCamera()
camera.start_camera()
print("Camera started")
while True:
    # Grab image input & format prompt
    color_img, depth_img, _, point_cloud, depth_frame = camera.read_align_frame()

    refresh_cam(color_img, 1)
    # 调整大小为 224x224
    image_resized = cv2.resize(color_img, (224, 224))

    # 转置为 (3, 224, 224)
    image_transposed = np.transpose(image_resized, (2, 0, 1))

    # image: Image.Image = get_from_camera(...)
    # prompt = "In: What action should the robot take to {<pick>}?\nOut:"
    prompt = "In: What action should the robot take to "+instr+"?\nOut:"

    img = Image.fromarray(image_resized).convert('RGB')

    # Predict Action (7-DoF; un-normalize for BridgeData V2)
    inputs = processor(prompt, img).to("cuda:0", dtype=torch.bfloat16)
    action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    print(action)
    # Execute...

    current_pose = robot.Get_Current_Arm_State()[2]

    target_pose = [current_pose[i] + (-action[i] if is_inverse_list[i] else action[i]) for i in range(len(action[0:6]))]

    if target_pose[2] < z_max:
        target_pose[2] = z_max
    
    robot.Movej_P_Cmd(target_pose, 20, 0, 0, True)

    if action[6] > 0.5:
        gripper_open = 255
    if action[6] <= 0.5:
        gripper_open = 0

    robot.Write_Registers(1, 1000, 3, [0, 9, gripper_open, 0, 255, 255], 9)



