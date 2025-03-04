from flask import Flask, render_template, Response, jsonify, request
import cv2
import time
import os
import logging
import base64
from datetime import datetime
from src.detector import ObjectDetector
from src.config import Config
from src.bottle_classifier import BottleClassifier
import numpy as np

# 설정 로드
config = Config()

# 로깅 설정
if not os.path.exists('logs'):
    os.makedirs('logs')

log_filename = f"logs/detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    encoding='utf-8',
    datefmt='%m/%d %H:%M'
)

# Flask 앱 초기화
app = Flask(__name__)

# Flask 내장 로깅 비활성화
import logging as flask_logging
flask_logging.getLogger('werkzeug').disabled = True
app.logger.disabled = True

# 객체 감지기 초기화
detector = None
bottle_classifier = None

class Camera:
    def __init__(self):
        self.camera = None
        self.last_access = time.time()
        self.is_running = False
        self.initialize_camera()

    def initialize_camera(self):
        """여러 방법으로 카메라 연결 시도"""
        # 카메라가 이미 열려있다면 닫기
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        # 사용 가능한 카메라 목록 확인
        available_cameras = []
        for i in range(10):  # 0부터 9까지 인덱스 확인
            try:
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available_cameras.append(i)
                    cap.release()
            except:
                pass
        
        logging.info(f"사용 가능한 카메라 인덱스: {available_cameras}")
        
        # 연결 방법 목록
        connection_methods = []
        
        # 사용 가능한 카메라가 있으면 해당 인덱스로 시도
        for idx in available_cameras:
            connection_methods.append((f"카메라 인덱스 {idx}", lambda i=idx: cv2.VideoCapture(i)))
            connection_methods.append((f"DirectShow 카메라 {idx}", lambda i=idx: cv2.VideoCapture(i, cv2.CAP_DSHOW)))
            connection_methods.append((f"MSMF 카메라 {idx}", lambda i=idx: cv2.VideoCapture(i, cv2.CAP_MSMF)))
        
        # 사용 가능한 카메라가 없으면 기본 방법 시도
        if not available_cameras:
            connection_methods = [
                ("기본 카메라 0", lambda: cv2.VideoCapture(0)),
                ("DirectShow 카메라 0", lambda: cv2.VideoCapture(0, cv2.CAP_DSHOW)),
                ("MSMF 카메라 0", lambda: cv2.VideoCapture(0, cv2.CAP_MSMF)),
                ("기본 카메라 1", lambda: cv2.VideoCapture(1))
            ]
        
        # 각 방법 시도
        for method_name, method_func in connection_methods:
            try:
                logging.info(f"카메라 연결 방법 '{method_name}' 시도 중...")
                self.camera = method_func()
                
                # 카메라 설정
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
                
                # 연결 확인 (실제 프레임 읽기 시도)
                success, test_frame = self.camera.read()
                if success and test_frame is not None and test_frame.size > 0:
                    logging.info(f"카메라 연결 성공 (방법: {method_name})")
                    self.is_running = True
                    return True
                else:
                    logging.warning(f"카메라 연결 방법 '{method_name}' 실패: 프레임을 읽을 수 없음")
                    self.camera.release()
                    self.camera = None
            except Exception as e:
                logging.error(f"카메라 연결 방법 '{method_name}' 오류: {str(e)}")
                if self.camera is not None:
                    self.camera.release()
                    self.camera = None
        
        # 모든 방법 실패 - 가상 카메라 생성
        logging.error("모든 카메라 연결 방법 실패, 가상 카메라 생성")
        self.create_virtual_camera()
        return False
        
    def create_virtual_camera(self):
        """웹캠이 없을 때 가상 카메라 생성"""
        self.is_running = True
        # 검은색 배경에 텍스트 표시
        self.virtual_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self.virtual_frame, "Camera Not Available", (120, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(self.virtual_frame, "Please check your webcam connection", (80, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.virtual_frame, "Press 'Stream Start' to retry", (120, 320),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    def get_frame(self):
        """카메라에서 프레임 가져오기"""
        # 마지막 접근 시간 업데이트
        self.last_access = time.time()
        
        # 카메라가 없거나 실행 중이 아니면 재연결 시도
        if self.camera is None and not hasattr(self, 'virtual_frame'):
            if not self.initialize_camera():
                # 재연결 실패시 에러 이미지 반환
                error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img, "Camera Error", (180, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(error_img, "Check your webcam connection", (120, 280),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return error_img
        
        # 가상 카메라 모드인 경우
        if hasattr(self, 'virtual_frame') and self.camera is None:
            return self.virtual_frame.copy()
        
        try:
            # 프레임 읽기 시도
            success, frame = self.camera.read()
            
            if not success or frame is None or frame.size == 0:
                logging.warning("프레임 읽기 실패, 카메라 재연결 시도")
                self.is_running = False
                if self.initialize_camera():
                    return self.get_frame()  # 재귀적으로 다시 시도
                else:
                    # 재연결 실패시 가상 카메라 프레임 반환
                    if hasattr(self, 'virtual_frame'):
                        return self.virtual_frame.copy()
                    else:
                        # 가상 카메라도 없는 경우 에러 이미지 반환
                        error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(error_img, "Frame Error", (180, 240),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        return error_img
            
            return frame
            
        except Exception as e:
            logging.error(f"프레임 읽기 오류: {str(e)}")
            self.is_running = False
            
            # 가상 카메라 프레임 반환
            if hasattr(self, 'virtual_frame'):
                return self.virtual_frame.copy()
            else:
                # 가상 카메라도 없는 경우 에러 이미지 반환
                error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_img, f"Camera Error: {str(e)}", (100, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return error_img

    def cleanup(self):
        """카메라 자원 해제"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.is_running = False

# 전역 카메라 객체
camera = Camera()

@app.route('/')
def index():
    """메인 페이지 렌더링"""
    return render_template('index.html')

@app.route('/training')
def training():
    """학습 인터페이스 페이지 렌더링"""
    global bottle_classifier
    if bottle_classifier is None:
        bottle_classifier = BottleClassifier()
    return render_template('training.html')

def generate_frames():
    """비디오 프레임 생성 함수"""
    global detector
    
    # 이전 감지 결과 초기화
    generate_frames.prev_detections = []
    
    if detector is None:
        logging.info("객체 감지기 초기화 중...")
        detector = ObjectDetector(model_path=config.MODEL_PATH, 
                                confidence_threshold=config.CONFIDENCE_THRESHOLD,
                                exclude_classes=config.EXCLUDE_CLASSES)
        logging.info("객체 감지기 초기화 완료")
    
    # FPS 계산을 위한 변수
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    
    while True:
        try:
            # 프레임 가져오기
            frame = camera.get_frame()
            
            # FPS 계산
            fps_frame_count += 1
            elapsed_time = time.time() - fps_start_time
            if elapsed_time > 1.0:
                fps = fps_frame_count / elapsed_time
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # 프레임 크기 조정 (처리 속도 향상 및 오탐율 감소)
            frame_resized = cv2.resize(frame, (640, 480))
            
            # 객체 감지 수행
            frame_processed, detections = detector.detect(frame_resized)
            
            # 연속 프레임에서 감지 결과 안정화 (오탐율 감소)
            if hasattr(generate_frames, 'prev_detections') and len(generate_frames.prev_detections) > 0:
                # 이전 프레임과 현재 프레임의 감지 결과를 비교하여 안정적인 결과만 유지
                stable_detections = []
                for d in detections:
                    # 높은 신뢰도를 가진 감지 결과만 유지
                    if d['confidence'] > config.CONFIDENCE_THRESHOLD + 0.1:
                        stable_detections.append(d)
                    # 또는 이전 프레임에서도 감지된 객체인 경우 유지
                    elif any(prev_d['class'] == d['class'] for prev_d in generate_frames.prev_detections):
                        stable_detections.append(d)
                
                detections = stable_detections
            
            # 현재 감지 결과 저장
            generate_frames.prev_detections = detections
            
            # FPS 표시
            cv2.putText(frame_processed, f"FPS: {fps:.2f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 감지된 객체 수 표시
            cv2.putText(frame_processed, f"Objects: {len(detections)}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 로그 기록
            if len(detections) > 0:
                logging.info(f"감지된 객체: {len(detections)}, 클래스: {[d['class'] for d in detections]}")
            
            # JPEG로 인코딩
            ret, buffer = cv2.imencode('.jpg', frame_processed)
            frame = buffer.tobytes()
            
            # 프레임 전송
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                   
        except Exception as e:
            logging.error(f"프레임 처리 중 오류 발생: {str(e)}")
            time.sleep(1)  # 오류 발생시 잠시 대기
            continue

@app.route('/video_feed')
def video_feed():
    """비디오 피드 엔드포인트"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot')
def take_snapshot():
    """현재 프레임의 스냅샷 저장"""
    global detector
    
    if detector is None:
        return jsonify({"error": "객체 감지기가 초기화되지 않았습니다."}), 500
    
    # 웹캠 초기화
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        return jsonify({"error": "웹캠을 열 수 없습니다."}), 500
    
    # 프레임 캡처
    success, frame = cap.read()
    cap.release()
    
    if not success:
        return jsonify({"error": "프레임을 캡처할 수 없습니다."}), 500
    
    # 객체 감지 수행
    try:
        frame, detections = detector.detect(frame)
        
        # 스냅샷 저장 디렉토리 확인
        if not os.path.exists('static/snapshots'):
            os.makedirs('static/snapshots')
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"static/snapshots/snapshot_{timestamp}.jpg"
        
        # 이미지 저장
        cv2.imwrite(filename, frame)
        
        # 정상/비정상 객체 수 계산
        normal_count = sum(1 for d in detections if not d.get('is_abnormal', False))
        abnormal_count = sum(1 for d in detections if d.get('is_abnormal', False))
        
        # 로그 기록
        logging.info(f"스냅샷 저장: {filename}, 감지된 객체: {len(detections)}, 정상: {normal_count}, 비정상: {abnormal_count}")
        
        return jsonify({
            "success": True,
            "filename": filename,
            "detections": len(detections),
            "objects": [d['class'] for d in detections],
            "normal_count": normal_count,
            "abnormal_count": abnormal_count
        })
    
    except Exception as e:
        logging.error(f"스냅샷 처리 중 오류 발생: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/stats')
def get_stats():
    """현재 통계 정보 반환"""
    global detector
    
    if detector is None:
        return jsonify({"error": "객체 감지기가 초기화되지 않았습니다."}), 500
    
    stats = detector.get_stats()
    
    # 정상/비정상 통계는 이미 detector.get_stats()에서 제공됨
    # 병(Bottle) 클래스만 감지하므로 추가 처리 필요 없음
    
    return jsonify(stats)

# 학습 인터페이스 API 엔드포인트
@app.route('/api/capture', methods=['POST'])
def api_capture():
    """이미지 캡처 API"""
    try:
        logging.info("이미지 캡처 요청 수신")
        
        # 카메라에서 프레임 가져오기
        frame = camera.get_frame()
        
        if frame is None:
            logging.error("프레임을 가져올 수 없습니다.")
            return jsonify({"success": False, "message": "프레임을 가져올 수 없습니다."}), 500
        
        # 이미지 저장 경로
        timestamp = int(time.time())
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(config.CAPTURE_DIR, filename)
        
        # 디렉토리가 없으면 생성
        os.makedirs(config.CAPTURE_DIR, exist_ok=True)
        
        # 이미지 저장
        cv2.imwrite(filepath, frame)
        logging.info(f"이미지 저장 완료: {filepath}")
        
        # 이미지를 base64로 인코딩
        _, buffer = cv2.imencode('.jpg', frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 이미지 URL 생성
        image_url = f"/static/captures/{filename}"
        
        return jsonify({
            "success": True,
            "image_url": image_url,
            "image_data": img_base64
        })
        
    except Exception as e:
        logging.error(f"이미지 캡처 중 오류 발생: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/save_sample', methods=['POST'])
def api_save_sample():
    """학습 샘플 저장 API"""
    global bottle_classifier
    
    try:
        if bottle_classifier is None:
            bottle_classifier = BottleClassifier()
        
        # 요청 데이터 파싱
        data = request.json
        if not data:
            return jsonify({"success": False, "message": "요청 데이터가 없습니다."}), 400
        
        image_data = data.get('image_data')
        bottle_type = data.get('bottle_type')
        contamination_level = data.get('contamination_level')
        
        if not image_data or not bottle_type or not contamination_level:
            return jsonify({"success": False, "message": "필수 데이터가 누락되었습니다."}), 400
        
        # Base64 이미지 디코딩
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"success": False, "message": "이미지 디코딩에 실패했습니다."}), 400
        
        # 샘플 저장
        success = bottle_classifier.save_sample(image, bottle_type, contamination_level)
        
        if not success:
            return jsonify({"success": False, "message": "샘플 저장에 실패했습니다."}), 500
        
        return jsonify({"success": True})
    
    except Exception as e:
        logging.error(f"샘플 저장 중 오류 발생: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/train_model', methods=['POST'])
def api_train_model():
    """모델 학습 API"""
    global bottle_classifier
    
    try:
        if bottle_classifier is None:
            bottle_classifier = BottleClassifier()
        
        # 모델 학습
        success = bottle_classifier.train()
        
        if not success:
            return jsonify({"success": False, "message": "모델 학습에 실패했습니다."}), 500
        
        return jsonify({"success": True})
    
    except Exception as e:
        logging.error(f"모델 학습 중 오류 발생: {str(e)}")
        return jsonify({"success": False, "message": str(e)}), 500

@app.route('/api/sample_stats')
def api_sample_stats():
    """학습 샘플 통계 API"""
    global bottle_classifier
    
    try:
        if bottle_classifier is None:
            bottle_classifier = BottleClassifier()
        
        # 샘플 통계 가져오기
        stats = bottle_classifier.get_sample_counts()
        
        return jsonify(stats)
    
    except Exception as e:
        logging.error(f"샘플 통계 조회 중 오류 발생: {str(e)}")
        return jsonify({"error": str(e)}), 500

# 애플리케이션 종료시 카메라 정리
@app.teardown_appcontext
def cleanup(error):
    camera.cleanup()

if __name__ == '__main__':
    logging.info("서버 시작")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG) 