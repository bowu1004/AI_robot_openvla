from base_camera import BaseCamera
import cv2

class OpenCVCamera(BaseCamera):
    """基于OpenCV的摄像头类"""
    
    def __init__(self, device_id=0):
        """初始化视频捕获
        
        参数:
            device_id: 摄像头设备ID
        """
        self.cap = cv2.VideoCapture(device_id)

    def get_frame(self):
        """获取当前帧
        
        返回:
            frame: 当前帧的图像数据,取不到时返回None
        """
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def get_frame_info(self):
        """获取当前帧信息
        
        返回:
            dict: 帧信息字典
        """
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        channels = int(self.cap.get(cv2.CAP_PROP_FRAME_CHANNELS))
        
        return {
            'width': width,
            'height': height,
            'channels': channels
        }