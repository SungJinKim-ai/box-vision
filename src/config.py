import os
import cv2

class Config:
    """애플리케이션 설정 클래스"""
    
    def __init__(self):
        # 서버 설정
        self.HOST = '0.0.0.0'
        self.PORT = 5000
        self.DEBUG = True  # 프로덕션 환경에서는 False로 설정
        
        # 카메라 설정
        self.CAMERA_INDEX = 1  # 기본 웹캠 (필요에 따라 변경)
        
        # 모델 설정
        self.MODEL_PATH = os.path.join('models', 'yolov8n.pt')  # 기본 YOLOv8 모델
        self.CONFIDENCE_THRESHOLD = 0.6  # 객체 감지 신뢰도 임계값 (높은 값으로 설정)
        
        # 병(Bottle) 클래스 ID (COCO 데이터셋 기준)
        self.BOTTLE_CLASS_ID = 39  # bottle 클래스 ID
        
        # 병(Bottle)만 감지하도록 설정
        # COCO 데이터셋의 모든 클래스 ID를 제외하고 병(Bottle) 클래스만 포함
        self.EXCLUDE_CLASSES = list(range(0, 80))  # 모든 클래스 제외
        self.EXCLUDE_CLASSES.remove(self.BOTTLE_CLASS_ID)  # 병(Bottle) 클래스는 제외 목록에서 제거
        
        # 최대 감지 객체 수 (낮게 설정하여 오탐율 감소)
        self.MAX_DETECTIONS = 20
        
        # 스냅샷 설정
        self.SNAPSHOT_DIR = os.path.join('static', 'snapshots')
        if not os.path.exists(self.SNAPSHOT_DIR):
            os.makedirs(self.SNAPSHOT_DIR)

        # DirectShow 백엔드만 사용하도록 수정
        self.camera = cv2.VideoCapture(self.CAMERA_INDEX, cv2.CAP_DSHOW) 