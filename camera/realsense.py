import time
import numpy as np
import pyrealsense2 as rs
from camera import base

# TODO: 加载配置文件


class RealSenseCamera(base.BaseCamera):
    """Intel RealSense相机类"""

    def __init__(self):
        """
        初始化相机对象
        :param _config: 相机配置参数，默认为空字典
        """
        self._D435_IMG_WIDTH = 640
        self._D435_IMG_HEIGHT = 480
        self._D435_FRAME_RATE = 30
        self._colorizer = rs.colorizer()
        self._config = rs.config()

    def _set__config(self, color_width, color_height, frames_rate):
        """
        设置相机机配置信息
        :param color_width: 彩色图像宽度
        :param color_height: 彩色图像高度
        :param depth_format: 深度图像格式
        """
        self._config.enable_stream(
            rs.stream.color,
            color_width,
            color_height,
            rs.format.bgr8,
            frames_rate,
        )
        self._config.enable_stream(
            rs.stream.depth,
            color_width,
            color_height,
            rs.format.z16,
            frames_rate,
        )

    # TODO: 调节白平衡消除色差
    def start_camera(self):
        """
        启动相机并获取内参信息
        """

        color_width = self._D435_IMG_WIDTH
        color_height = self._D435_IMG_HEIGHT
        frames_rate = self._D435_FRAME_RATE

        self._pipeline = rs.pipeline()
        self.point_cloud = rs.pointcloud()
        self._align = rs.align(rs.stream.color)
        self._set__config(color_width, color_height, frames_rate)
        profile = self._pipeline.start(self._config)

        self._depth_intrinsics = (
            profile.get_stream(rs.stream.depth)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self._color_intrinsics = (
            profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        time.sleep(1)

    def stop_camera(self):
        """
        停止相机
        """
        self._pipeline.stop()

    def set_resolution(self, width, height):
        """
        设置相机图像分辨率与帧率，需要在相机启动前
        :param width: 分辨率宽度
        :param height: 分辨率高度
        """
        self._D435_IMG_WIDTH = width
        self._D435_IMG_HEIGHT = height

    def set_frame_rate(self, fps):
        """
        设置相机彩色图像帧率，需要在相机启动前
        :param fps: 帧率
        """
        self._D435_FRAME_RATE = fps

    # # 调节相机白平衡等补偿
    # def set_exposure(self, exposure):


    def read_frame(self):
        """
        读取一帧彩色图像和深度图像
        :return: 彩色图像和深度图像的NumPy数组
        """
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        colorized_depth = np.asanyarray(
            self._colorizer.colorize(depth_frame).get_data()
        )
        points = self.point_cloud.calculate(depth_frame)
        point_cloud = np.asanyarray(points.get_vertices())
        return color_image, depth_image, colorized_depth, point_cloud

    def read_align_frame(self, if_colorized_depth=False, if_point_cloud=False):
        """
        读取一帧对齐的彩色图像和深度图像
        :return: 彩色图像和深度图像的NumPy数组
        """
        frames = self._pipeline.wait_for_frames()
        aligned_frames = self._align.process(frames)
        aligned_color_frame = aligned_frames.get_color_frame()
        self._aligned_depth_frame = aligned_frames.get_depth_frame()

        color_image = np.asanyarray(aligned_color_frame.get_data())
        depth_image = np.asanyarray(self._aligned_depth_frame.get_data())
        if if_colorized_depth:
            colorized_depth = np.asanyarray(
                self._colorizer.colorize(self._aligned_depth_frame).get_data()
            )
        else:
            colorized_depth = None
        if if_point_cloud:
            points = self.point_cloud.calculate(self._aligned_depth_frame)
            point_cloud = np.asanyarray(points.get_vertices())
        else:
            point_cloud = None

        return color_image, depth_image, colorized_depth, point_cloud, self._aligned_depth_frame

    def get_camera_intrinsics(self):
        """
        获取彩色图像和深度图像的内参信息
        :return: 彩色图像和深度图像的内参信息
        """
        # 宽高：.width, .height; 焦距：.fx, .fy; 像素坐标：.ppx, .ppy; 畸变系数：.coeffs
        return self._color_intrinsics, self._depth_intrinsics

    def get_3d_camera_coordinate(self, depth_pixel):
        """
        获取相机坐标系下的三维坐标
        :param depth_pixel: 深度图像中的像素坐标

        :return: 深度值和相机坐标
        """
        x = depth_pixel[0]
        y = depth_pixel[1]
        distance = self._aligned_depth_frame.get_distance(x, y)  # 获取该像素点对应的深度
        camera_coordinate = rs.rs2_deproject_pixel_to_point(
            # self._depth_intrinsics, depth_pixel, distance
            self._color_intrinsics, depth_pixel, distance
        )
        return distance, camera_coordinate
