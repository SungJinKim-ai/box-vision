import cv2
import numpy as np
import os
import json
import logging
from datetime import datetime
import pickle

class BottleClassifier:
    """병 유형 및 오염도 분류기"""
    
    def __init__(self):
        """분류기 초기화"""
        self.is_trained = False
        self.model = None
        
        # 병 유형 및 오염도 정보
        self.bottle_types = ["소형", "중형", "대형"]
        self.contamination_levels = ["깨끗함", "약간 오염", "심한 오염"]
        
        # 학습 데이터 저장 경로
        self.data_dir = "data/training/bottles"
        self.model_dir = "models"
        
        # 학습 데이터
        self.samples = []
        self.features = []
        self.labels = []
        
        # 디렉토리 생성
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 기존 학습 데이터 로드
        self._load_samples()
    
    def _load_samples(self):
        """저장된 학습 샘플 로드"""
        try:
            # 샘플 정보 파일 확인
            sample_info_path = os.path.join(self.data_dir, "samples_info.json")
            if os.path.exists(sample_info_path):
                with open(sample_info_path, "r", encoding="utf-8") as f:
                    self.samples = json.load(f)
                logging.info(f"학습 샘플 정보 로드 완료: {len(self.samples)}개")
        except Exception as e:
            logging.error(f"학습 샘플 로드 중 오류: {str(e)}")
            self.samples = []
    
    def save_sample(self, image, bottle_type, contamination_level):
        """새로운 학습 샘플 저장"""
        try:
            # 이미지가 비어있는 경우
            if image is None or image.size == 0:
                logging.warning("저장할 이미지가 비어 있습니다.")
                return False
            
            # 타임스탬프 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # 이미지 파일 저장
            image_filename = f"bottle_{timestamp}.jpg"
            image_path = os.path.join(self.data_dir, image_filename)
            cv2.imwrite(image_path, image)
            
            # 특징 추출
            features = self._extract_features(image)
            
            # 샘플 정보 저장
            sample_info = {
                "image_path": image_path,
                "bottle_type": bottle_type,
                "contamination_level": contamination_level,
                "timestamp": timestamp,
                "features": features.tolist() if features is not None else None
            }
            
            # 샘플 목록에 추가
            self.samples.append(sample_info)
            
            # 샘플 정보 파일 업데이트
            self._save_samples_info()
            
            logging.info(f"학습 샘플 저장 완료: {bottle_type}, {contamination_level}")
            return True
        except Exception as e:
            logging.error(f"학습 샘플 저장 중 오류: {str(e)}")
            return False
    
    def _save_samples_info(self):
        """샘플 정보 파일 저장"""
        try:
            sample_info_path = os.path.join(self.data_dir, "samples_info.json")
            with open(sample_info_path, "w", encoding="utf-8") as f:
                json.dump(self.samples, f, ensure_ascii=False, indent=2)
            logging.info(f"샘플 정보 파일 저장 완료: {len(self.samples)}개")
            return True
        except Exception as e:
            logging.error(f"샘플 정보 파일 저장 중 오류: {str(e)}")
            return False
    
    def _extract_features(self, image):
        """이미지에서 특징 추출"""
        try:
            # 이미지 크기 조정
            image = cv2.resize(image, (128, 128))
            
            # 색상 특징
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
            s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
            v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
            
            # 히스토그램 정규화
            h_hist = h_hist.flatten() / np.sum(h_hist) if np.sum(h_hist) > 0 else h_hist.flatten()
            s_hist = s_hist.flatten() / np.sum(s_hist) if np.sum(s_hist) > 0 else s_hist.flatten()
            v_hist = v_hist.flatten() / np.sum(v_hist) if np.sum(v_hist) > 0 else v_hist.flatten()
            
            # 텍스처 특징
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_ratio = np.count_nonzero(edges) / (image.shape[0] * image.shape[1])
            
            # 형태 특징
            aspect_ratio = image.shape[1] / image.shape[0] if image.shape[0] > 0 else 0
            
            # 특징 벡터 생성
            features = np.concatenate([
                h_hist, s_hist, v_hist,
                np.array([edge_ratio, aspect_ratio])
            ])
            
            return features
        except Exception as e:
            logging.error(f"특징 추출 중 오류: {str(e)}")
            return None
    
    def train(self):
        """분류기 학습"""
        try:
            # 충분한 샘플이 있는지 확인
            if len(self.samples) < 6:  # 각 클래스당 최소 1개 이상의 샘플 필요
                logging.warning(f"학습을 위한 샘플이 부족합니다. 현재 {len(self.samples)}개")
                return False
            
            # 특징 벡터와 레이블 준비
            features = []
            bottle_type_labels = []
            contamination_labels = []
            
            for sample in self.samples:
                if sample["features"] is not None:
                    features.append(np.array(sample["features"]))
                    bottle_type_labels.append(self.bottle_types.index(sample["bottle_type"]))
                    contamination_labels.append(self.contamination_levels.index(sample["contamination_level"]))
            
            if len(features) < 6:
                logging.warning(f"유효한 특징 벡터가 부족합니다. 현재 {len(features)}개")
                return False
            
            # 특징 벡터 배열로 변환
            X = np.array(features)
            y_bottle_type = np.array(bottle_type_labels)
            y_contamination = np.array(contamination_labels)
            
            # 간단한 k-NN 분류기 사용
            from sklearn.neighbors import KNeighborsClassifier
            
            # 병 유형 분류기
            bottle_type_model = KNeighborsClassifier(n_neighbors=3)
            bottle_type_model.fit(X, y_bottle_type)
            
            # 오염도 분류기
            contamination_model = KNeighborsClassifier(n_neighbors=3)
            contamination_model.fit(X, y_contamination)
            
            # 모델 저장
            self.model = {
                "bottle_type_model": bottle_type_model,
                "contamination_model": contamination_model,
                "features_mean": np.mean(X, axis=0),
                "features_std": np.std(X, axis=0) + 1e-10  # 0으로 나누기 방지
            }
            
            # 모델 파일 저장
            model_path = os.path.join(self.model_dir, "bottle_classifier.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            
            self.is_trained = True
            logging.info("모델 학습 및 저장 완료")
            return True
        except Exception as e:
            logging.error(f"모델 학습 중 오류: {str(e)}")
            return False
    
    def load_model(self):
        """저장된 모델 로드"""
        try:
            model_path = os.path.join(self.model_dir, "bottle_classifier.pkl")
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                self.is_trained = True
                logging.info("모델 로드 완료")
                return True
            else:
                logging.warning("저장된 모델 파일이 없습니다.")
                return False
        except Exception as e:
            logging.error(f"모델 로드 중 오류: {str(e)}")
            return False
    
    def predict(self, image):
        """이미지 분류"""
        try:
            # 모델이 없으면 로드 시도
            if not self.is_trained or self.model is None:
                if not self.load_model():
                    logging.warning("학습된 모델이 없습니다.")
                    return None, None, 0.0, 0.0
            
            # 특징 추출
            features = self._extract_features(image)
            if features is None:
                return None, None, 0.0, 0.0
            
            # 특징 정규화
            features_normalized = (features - self.model["features_mean"]) / self.model["features_std"]
            
            # 병 유형 예측
            bottle_type_idx = self.model["bottle_type_model"].predict([features_normalized])[0]
            bottle_type = self.bottle_types[bottle_type_idx]
            
            # 오염도 예측
            contamination_idx = self.model["contamination_model"].predict([features_normalized])[0]
            contamination_level = self.contamination_levels[contamination_idx]
            
            # 신뢰도 계산 (거리 기반)
            bottle_type_neighbors = self.model["bottle_type_model"].kneighbors([features_normalized], return_distance=True)
            contamination_neighbors = self.model["contamination_model"].kneighbors([features_normalized], return_distance=True)
            
            bottle_type_confidence = 1.0 / (1.0 + np.mean(bottle_type_neighbors[0][0]))
            contamination_confidence = 1.0 / (1.0 + np.mean(contamination_neighbors[0][0]))
            
            return bottle_type, contamination_level, bottle_type_confidence, contamination_confidence
        except Exception as e:
            logging.error(f"예측 중 오류: {str(e)}")
            return None, None, 0.0, 0.0
    
    def get_sample_counts(self):
        """병 유형 및 오염도별 샘플 개수 반환"""
        counts = {
            "total": len(self.samples),
            "bottle_types": {bottle_type: 0 for bottle_type in self.bottle_types},
            "contamination_levels": {level: 0 for level in self.contamination_levels},
            "combinations": {}
        }
        
        for sample in self.samples:
            bottle_type = sample["bottle_type"]
            contamination_level = sample["contamination_level"]
            
            counts["bottle_types"][bottle_type] += 1
            counts["contamination_levels"][contamination_level] += 1
            
            combo_key = f"{bottle_type}_{contamination_level}"
            if combo_key not in counts["combinations"]:
                counts["combinations"][combo_key] = 0
            counts["combinations"][combo_key] += 1
        
        return counts 