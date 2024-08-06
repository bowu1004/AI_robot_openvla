from abc import ABCMeta, abstractmethod


class BaseCamera(metaclass=ABCMeta):
    """摄像头基类"""

    def __init__(self):
        pass

    @abstractmethod
    def start_camera(self):
        """启动相机"""
        pass

    @abstractmethod
    def stop_camera(self):
        """停止相机"""
        pass

    @abstractmethod
    def set_resolution(self, resolution_width, resolution_height):
        """设置相机彩色图像分辨率"""
        pass

    @abstractmethod
    def set_frame_rate(self, fps):
        """设置相机彩色图像帧率"""
        pass

    @abstractmethod
    def read_frame(self):
        """读取一帧彩色图像和深度图像"""
        pass

    @abstractmethod
    def get_camera_intrinsics(self):
        """获取彩色图像和深度图像的内参"""
        pass