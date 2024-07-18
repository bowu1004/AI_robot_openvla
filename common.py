import time

import numpy as np

from config import config
from .robotic_arm import Arm

from loguru import logger


def get_catch_object_weight(whole_arm):
    _, _, zero_force, _, _ = whole_arm.Get_Force_Data()
    obj_kg = np.sqrt(zero_force[0] ** 2 + zero_force[1] ** 2 + zero_force[2] ** 2) / 9.81 - 0.03692589891730488
    print(f"zero_force is {zero_force}.")
    print(f"obj_kg is {obj_kg}.")
    return obj_kg


def close_gripper(arm_type: int, whole_arm: Arm):
    if arm_type == 10:
        whole_arm.Write_Registers(1, 1000, 3, [0, 9, 0, 0, config.GRIPPER_FORCE, config.GRIPPER_V], 9)
        time.sleep(1)
    else:
        whole_arm.Set_Gripper_Pick_On(500, 500)


def open_gripper(arm_type: int, whole_arm: Arm):
    if arm_type == 10:
        whole_arm.Write_Registers(1, 1000, 3, [0, 9, 255, 0, config.GRIPPER_FORCE, config.GRIPPER_V], 9)
    else:
        whole_arm.Set_Gripper_Release(500)


def check_catch_result(arm_type: int, whole_arm: Arm):
    if config.GRIPPER_CHECK:
        if arm_type == 10:
            register_data = whole_arm.Read_Multiple_Holding_Registers(1, 2000, 6, 9)
            gripper_status = int(bin(register_data[1][2])[2:].zfill(8)[:2], 2)

            hex_status = format(gripper_status, '08x')[-1]
            logger.debug("hex_status is %s", hex_status)

            if int(hex_status) in (2, 1):
                logger.info(f"gripper catch successfully!")
                return True
            return False

        _, state = whole_arm.Get_Gripper_State(retry=1)
        logger.info(f"获取夹爪状态为【{_}】")
        if state.actpos > 5:
            logger.info(f"gripper catch successfully!")
            return True
        return False
    else:
        return True
