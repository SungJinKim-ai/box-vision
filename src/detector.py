import cv2
import time
import logging
import numpy as np
from ultralytics import YOLO
import torch
from src.bottle_classifier import BottleClassifier

# PyTorch 2.6 호환성을 위한 패치
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

class ObjectDetector:
    """YOLOv8 기반 객체 감지 클래스"""
    
    def __init__(self, model_path, confidence_threshold=0.5, exclude_classes=None, max_det=50):
        """
        객체 감지기 초기화
        
        Args:
            model_path (str): YOLOv8 모델 파일 경로
            confidence_threshold (float): 객체 감지 신뢰도 임계값
            exclude_classes (list): 제외할 클래스 ID 목록
            max_det (int): 최대 감지 객체 수
        """
        self.confidence_threshold = confidence_threshold
        self.exclude_classes = exclude_classes or []
        self.max_det = max_det
        
        # 통계 정보
        self.stats = {
            "total_detections": 0,
            "detection_counts": {},
            "start_time": time.time(),
            "frames_processed": 0,
            "normal_bottles": 0,
            "abnormal_bottles": 0,
            "bottle_types": {},
            "contamination_levels": {}
        }
        
        # 병 분류기 초기화
        self.bottle_classifier = None
        
        # COCO 데이터셋 클래스 이름
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # 모델 로드
        try:
            logging.info(f"YOLOv8 모델 로드 중: {model_path}")
            import torch.serialization
            # 필요한 클래스들을 안전한 글로벌로 추가
            from ultralytics.nn.tasks import DetectionModel
            from torch.nn.modules.container import Sequential
            torch.serialization.add_safe_globals([DetectionModel, Sequential])
            
            # 또는 weights_only=False 옵션을 사용하여 모델 로드 (신뢰할 수 있는 소스인 경우)
            self.model = YOLO(model_path)
            logging.info("YOLOv8 모델 로드 완료")
        except Exception as e:
            logging.error(f"모델 로드 중 오류 발생: {str(e)}")
            raise
    
    def detect(self, frame):
        """
        이미지에서 객체 감지 수행
        
        Args:
            frame (numpy.ndarray): 입력 이미지
            
        Returns:
            tuple: (처리된 이미지, 감지 결과 목록)
        """
        # 통계 업데이트
        self.stats["frames_processed"] += 1
        
        # 이미지 전처리 - 노이즈 제거 및 선명도 향상
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 모델 추론
        try:
            logging.info("객체 감지 수행 중...")
            results = self.model(frame, conf=self.confidence_threshold, max_det=self.max_det, iou=0.45)[0]
            logging.info(f"감지 결과: {len(results.boxes)} 개체")
            
            # 결과 처리
            detections = []
            
            for r in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, cls = r
                
                # 클래스 ID가 제외 목록에 있으면 건너뜀
                if int(cls) in self.exclude_classes:
                    logging.info(f"제외된 클래스: {int(cls)}")
                    continue
                
                # 너무 작은 객체는 제외 (오탐율 감소)
                box_width = x2 - x1
                box_height = y2 - y1
                min_size = min(frame.shape[0], frame.shape[1]) * 0.02  # 이미지 크기의 2% 미만인 객체 제외
                
                if box_width < min_size or box_height < min_size:
                    logging.info(f"너무 작은 객체 제외: 크기 {box_width}x{box_height}")
                    continue
                
                # 클래스 이름 가져오기
                class_id = int(cls)
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Class {class_id}"
                
                logging.info(f"감지된 객체: {class_name}, 신뢰도: {conf:.2f}")
                
                # 객체 영역 추출
                obj_roi = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # 정상/비정상 판단 (병에 대한 규칙 기반 방법)
                is_abnormal = self._check_abnormal(obj_roi, class_name, box_width, box_height)
                
                # 병 유형 및 오염도 분류 (병 클래스인 경우)
                bottle_type = None
                contamination_level = None
                bottle_type_confidence = 0.0
                contamination_confidence = 0.0
                
                if class_name == 'bottle' and obj_roi.size > 0:
                    bottle_type, contamination_level, bottle_type_confidence, contamination_confidence = self._classify_bottle(obj_roi)
                
                # 감지 정보 저장
                detection = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": float(conf),
                    "class_id": class_id,
                    "class": class_name,
                    "is_abnormal": is_abnormal
                }
                
                # 병 분류 정보 추가
                if bottle_type and contamination_level:
                    detection["bottle_type"] = bottle_type
                    detection["contamination_level"] = contamination_level
                    detection["bottle_type_confidence"] = bottle_type_confidence
                    detection["contamination_confidence"] = contamination_confidence
                
                detections.append(detection)
                
                # 통계 업데이트
                self.stats["total_detections"] += 1
                if class_name in self.stats["detection_counts"]:
                    self.stats["detection_counts"][class_name] += 1
                else:
                    self.stats["detection_counts"][class_name] = 1
                
                # 정상/비정상 병 카운트 업데이트
                if class_name == 'bottle':
                    if is_abnormal:
                        self.stats["abnormal_bottles"] += 1
                    else:
                        self.stats["normal_bottles"] += 1
                    
                    # 병 유형 및 오염도 통계 업데이트
                    if bottle_type:
                        if bottle_type in self.stats["bottle_types"]:
                            self.stats["bottle_types"][bottle_type] += 1
                        else:
                            self.stats["bottle_types"][bottle_type] = 1
                    
                    if contamination_level:
                        if contamination_level in self.stats["contamination_levels"]:
                            self.stats["contamination_levels"][contamination_level] += 1
                        else:
                            self.stats["contamination_levels"][contamination_level] = 1
                
                # 바운딩 박스 그리기 (정상: 녹색, 비정상: 빨간색)
                color = (0, 0, 255) if is_abnormal else (0, 255, 0)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # 클래스 이름과 신뢰도 표시
                status = "비정상" if is_abnormal else "정상"
                label = f"{class_name}: {conf:.2f} ({status})"
                
                # 병 유형 및 오염도 정보 추가
                if bottle_type and contamination_level:
                    label += f" | {bottle_type}, {contamination_level}"
                
                cv2.putText(frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            return frame, detections
        except Exception as e:
            logging.error(f"객체 감지 중 오류 발생: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return frame, []
    
    def _classify_bottle(self, bottle_image):
        """
        병 이미지를 분류하여 유형과 오염도 반환
        
        Args:
            bottle_image (numpy.ndarray): 병 이미지
            
        Returns:
            tuple: (병 유형, 오염도, 유형 신뢰도, 오염도 신뢰도)
        """
        try:
            # 병 분류기가 초기화되지 않은 경우 초기화
            if self.bottle_classifier is None:
                self.bottle_classifier = BottleClassifier()
                # 모델 로드 시도
                self.bottle_classifier.load_model()
            
            # 병 분류기가 학습되지 않은 경우
            if not self.bottle_classifier.is_trained:
                return None, None, 0.0, 0.0
            
            # 병 분류 수행
            bottle_type, contamination_level, bottle_type_confidence, contamination_confidence = self.bottle_classifier.predict(bottle_image)
            
            return bottle_type, contamination_level, bottle_type_confidence, contamination_confidence
        
        except Exception as e:
            logging.error(f"병 분류 중 오류 발생: {str(e)}")
            return None, None, 0.0, 0.0
    
    def _check_abnormal(self, obj_roi, class_name, width, height):
        """
        객체의 정상/비정상 여부를 판단
        
        Args:
            obj_roi (numpy.ndarray): 객체 영역 이미지
            class_name (str): 객체 클래스 이름
            width (float): 객체 너비
            height (float): 객체 높이
            
        Returns:
            bool: 비정상 여부 (True: 비정상, False: 정상)
        """
        try:
            # 객체 영역이 비어 있으면 비정상으로 판단
            if obj_roi.size == 0:
                return True
            
            # 병(bottle) 클래스에 대한 정상/비정상 판단
            if class_name == 'bottle':
                # 1. 종횡비 확인 (병의 일반적인 종횡비는 0.3~0.5 사이)
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < 0.25 or aspect_ratio > 0.6:
                    logging.info(f"비정상 병 감지: 종횡비 {aspect_ratio:.2f}")
                    return True
                
                # 2. 색상 분포 확인 (HSV 색상 공간 사용)
                hsv = cv2.cvtColor(obj_roi, cv2.COLOR_BGR2HSV)
                
                # 색상 히스토그램 계산
                h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
                s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
                v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
                
                # 색상 분포의 표준편차 계산
                h_std = np.std(h_hist)
                s_std = np.std(s_hist)
                v_std = np.std(v_hist)
                
                # 색상 분포가 비정상적으로 균일하거나 불균일하면 비정상
                if h_std < 3 or h_std > 40 or s_std < 5 or s_std > 50:
                    logging.info(f"비정상 병 감지: 색상 분포 H:{h_std:.2f}, S:{s_std:.2f}")
                    return True
                
                # 3. 텍스처 분석 (엣지 검출)
                gray = cv2.cvtColor(obj_roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                edge_ratio = np.count_nonzero(edges) / (width * height) if width * height > 0 else 0
                
                # 엣지 비율이 비정상적으로 높거나 낮으면 비정상
                if edge_ratio < 0.05 or edge_ratio > 0.3:
                    logging.info(f"비정상 병 감지: 엣지 비율 {edge_ratio:.2f}")
                    return True
                
                # 4. 대칭성 확인 (병은 일반적으로 좌우 대칭)
                if obj_roi.shape[1] > 10:  # 너비가 충분히 큰 경우에만 대칭성 확인
                    left_half = obj_roi[:, :obj_roi.shape[1]//2]
                    right_half = obj_roi[:, obj_roi.shape[1]//2:]
                    right_half_flipped = cv2.flip(right_half, 1)  # 우측 절반을 좌우 반전
                    
                    # 좌우 절반의 크기가 다른 경우 조정
                    if left_half.shape[1] != right_half_flipped.shape[1]:
                        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
                        left_half = left_half[:, :min_width]
                        right_half_flipped = right_half_flipped[:, :min_width]
                    
                    # 좌우 절반의 차이 계산
                    if left_half.size > 0 and right_half_flipped.size > 0:
                        diff = cv2.absdiff(left_half, right_half_flipped)
                        asymmetry = np.mean(diff) / 255.0  # 0~1 사이 값으로 정규화
                        
                        if asymmetry > 0.3:  # 비대칭성이 30% 이상이면 비정상
                            logging.info(f"비정상 병 감지: 비대칭성 {asymmetry:.2f}")
                            return True
                
                # 5. 학습된 분류기를 사용하여 오염도 확인
                if self.bottle_classifier and self.bottle_classifier.is_trained:
                    _, contamination_level, _, _ = self.bottle_classifier.predict(obj_roi)
                    if contamination_level and contamination_level != "깨끗함":
                        logging.info(f"비정상 병 감지: 오염도 {contamination_level}")
                        return True
            
            # 기본적으로 정상으로 판단
            return False
        
        except Exception as e:
            logging.error(f"정상/비정상 판단 중 오류 발생: {str(e)}")
            return False
    
    def get_stats(self):
        """
        감지 통계 정보 반환
        
        Returns:
            dict: 통계 정보
        """
        # 실행 시간 계산
        elapsed_time = time.time() - self.stats["start_time"]
        
        # 평균 FPS 계산
        avg_fps = self.stats["frames_processed"] / elapsed_time if elapsed_time > 0 else 0
        
        # 통계 정보 업데이트
        stats = {
            "total_detections": self.stats["total_detections"],
            "detection_counts": self.stats["detection_counts"],
            "frames_processed": self.stats["frames_processed"],
            "elapsed_time": int(elapsed_time),
            "avg_fps": round(avg_fps, 2),
            "normal_bottles": self.stats["normal_bottles"],
            "abnormal_bottles": self.stats["abnormal_bottles"],
            "bottle_types": self.stats["bottle_types"],
            "contamination_levels": self.stats["contamination_levels"]
        }
        
        return stats 