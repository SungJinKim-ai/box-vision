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
import threading

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

# 콘솔 로깅 추가
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%m/%d %H:%M')
console_handler.setFormatter(console_formatter)
logging.getLogger().addHandler(console_handler)

# Flask 앱 초기화
app = Flask(__name__)
logging.info("Flask 앱 초기화 완료")

# Flask 내장 로깅 비활성화
import logging as flask_logging
flask_logging.getLogger('werkzeug').disabled = True
app.logger.disabled = True

# 전역 객체 (지연 초기화를 위해 None으로 설정)
detector = None
bottle_classifier = None
camera_instance = None  # 카메라 객체를 None으로 초기화
detector_loading = False  # 감지기 로딩 중 플래그
detector_ready = threading.Event()  # 감지기 준비 완료 이벤트

class Camera:
    def __init__(self):
        logging.info("카메라 객체 생성 중...")
        self.camera = None
        self.last_access = time.time()
        self.is_running = False
        self.camera_lock = threading.Lock()
        self.initialization_thread = None
        self.available_cameras = []  # 사용 가능한 카메라 목록 저장
        self.create_virtual_camera()  # 초기에 가상 카메라 생성
        logging.info("카메라 객체 생성 완료 (가상 카메라 모드)")
        # 초기화는 첫 프레임 요청 시 시작
        
    def check_available_cameras(self):
        """Windows 10에서 사용 가능한 카메라 장치 확인 (숨겨진 장치 포함)"""
        available_cameras = []
        
        # 방법 1: OpenCV를 통한 확인
        for i in range(10):  # 0부터 9까지 인덱스 확인
            try:
                # 다른 프로그램이 카메라를 사용 중인 경우에도 접근 시도
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow 백엔드 사용
                
                # 카메라 연결 시도
                is_opened = cap.isOpened()
                
                # 프레임 읽기 시도 (실제 작동 여부 확인)
                frame_readable = False
                if is_opened:
                    for attempt in range(3):  # 여러 번 시도
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            frame_readable = True
                            break
                        time.sleep(0.1)  # 약간의 지연
                
                if is_opened:
                    # 카메라 정보 가져오기
                    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    
                    status = 'available' if frame_readable else 'in_use'
                    status_msg = '사용 가능' if frame_readable else '다른 프로그램에서 사용 중'
                    
                    available_cameras.append({
                        'index': i,
                        'name': f'Camera {i}',
                        'resolution': f'{int(width)}x{int(height)}',
                        'status': status,
                        'status_msg': status_msg
                    })
                    
                    logging.info(f"카메라 {i} 상태: {status_msg}")
                    
                    # 다른 프로그램이 사용 중인 경우에도 목록에 추가
                    if not frame_readable:
                        logging.warning(f"카메라 {i}가 다른 프로그램에서 사용 중입니다. 해당 프로그램을 종료하고 다시 시도하세요.")
                
                # 항상 자원 해제
                cap.release()
                
            except Exception as e:
                logging.error(f"카메라 {i} 확인 중 오류: {str(e)}")
                
        # 방법 2: Windows 특정 방법 (PowerShell 명령 실행)
        try:
            import subprocess
            
            # 1. 모든 카메라 장치 확인 (숨겨진 장치 포함)
            result = subprocess.run(
                ['powershell', '-Command', "Get-PnpDevice -Class Camera,Image | Select-Object Status,FriendlyName"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # PowerShell 출력 파싱
                lines = result.stdout.strip().split('\n')
                device_info = []
                
                current_device = {}
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('Status') or line.startswith('--'):
                        continue
                        
                    if line.startswith('OK') or line.startswith('Error') or line.startswith('Unknown'):
                        # 새 장치 시작
                        if current_device and 'name' in current_device:
                            device_info.append(current_device)
                        
                        status = line.split(' ')[0]
                        name = line[len(status):].strip()
                        current_device = {'status': status, 'name': name}
                    else:
                        # 이전 장치의 이름 계속
                        if current_device:
                            current_device['name'] = (current_device.get('name', '') + ' ' + line).strip()
                
                # 마지막 장치 추가
                if current_device and 'name' in current_device:
                    device_info.append(current_device)
                
                logging.info(f"PowerShell로 감지된 모든 카메라 장치: {device_info}")
                
                # 2. 문제가 있는 장치 확인
                result_error = subprocess.run(
                    ['powershell', '-Command', "Get-PnpDevice -Class Camera,Image -Status Error | Select-Object FriendlyName"],
                    capture_output=True,
                    text=True
                )
                
                error_devices = []
                if result_error.returncode == 0:
                    lines = result_error.stdout.strip().split('\n')
                    error_devices = [line.strip() for line in lines if line.strip() and not line.startswith('FriendlyName')]
                    
                    if error_devices:
                        logging.warning(f"오류 상태의 카메라 장치: {error_devices}")
                        
                        # 오류 상태의 장치도 목록에 추가
                        for device in error_devices:
                            # 이미 OpenCV로 감지된 장치와 중복 확인
                            if not any(d['name'] == device for d in available_cameras):
                                available_cameras.append({
                                    'index': -1,  # 인덱스 알 수 없음
                                    'name': device,
                                    'resolution': 'unknown',
                                    'status': 'error',
                                    'status_msg': '오류 상태'
                                })
                
                # OpenCV로 감지된 카메라와 PowerShell로 감지된 카메라 이름 매핑
                for device in device_info:
                    if device['status'] == 'OK':
                        for i, cam in enumerate(available_cameras):
                            if cam['status'] in ['available', 'in_use'] and 'Camera' in cam['name']:
                                # 기본 이름을 실제 장치 이름으로 업데이트
                                available_cameras[i]['name'] = device['name']
                                break
        except Exception as e:
            logging.error(f"PowerShell 카메라 감지 중 오류: {str(e)}")
            
        # 방법 3: 카메라를 사용 중인 프로세스 확인
        try:
            # 카메라를 사용 중인 프로세스 확인 (Windows 전용)
            result = subprocess.run(
                ['powershell', '-Command', "Get-Process | Where-Object {$_.Modules.ModuleName -like '*video*' -or $_.Modules.ModuleName -like '*camera*'} | Select-Object ProcessName"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                camera_processes = [line.strip() for line in lines if line.strip() and not line.startswith('ProcessName')]
                
                if camera_processes:
                    logging.warning(f"카메라를 사용 중인 것으로 의심되는 프로세스: {camera_processes}")
                    logging.warning("위 프로세스를 종료하고 다시 시도하면 카메라 접근이 가능할 수 있습니다.")
        except Exception as e:
            logging.error(f"카메라 사용 프로세스 확인 중 오류: {str(e)}")
            
        self.available_cameras = available_cameras
        logging.info(f"최종 감지된 카메라 장치: {self.available_cameras}")
        
        # 카메라가 감지되지 않은 경우 사용자에게 도움이 될 수 있는 정보 로깅
        if not available_cameras:
            logging.warning("카메라가 감지되지 않았습니다. 다음 사항을 확인하세요:")
            logging.warning("1. 카메라가 물리적으로 연결되어 있는지 확인하세요.")
            logging.warning("2. 장치 관리자에서 카메라 드라이버 상태를 확인하세요.")
            logging.warning("3. 다른 프로그램이 카메라를 사용 중인지 확인하세요.")
            logging.warning("4. USB 포트를 변경해 보세요.")
            logging.warning("5. 컴퓨터를 재부팅한 후 다시 시도해 보세요.")
        elif not any(cam['status'] == 'available' for cam in available_cameras):
            logging.warning("모든 카메라가 다른 프로그램에서 사용 중이거나 오류 상태입니다.")
            logging.warning("다음 프로그램을 종료하고 다시 시도해 보세요: Webex, Zoom, Teams, Skype 등")
        
        return available_cameras

    def start_camera_initialization(self):
        """백그라운드 스레드에서 카메라 초기화 시작"""
        if self.initialization_thread is None or not self.initialization_thread.is_alive():
            self.initialization_thread = threading.Thread(target=self.initialize_camera)
            self.initialization_thread.daemon = True
            self.initialization_thread.start()
            logging.info("카메라 초기화 백그라운드 스레드 시작")

    def initialize_camera(self):
        """여러 방법으로 카메라 연결 시도 (백그라운드 스레드에서 실행)"""
        with self.camera_lock:
            # 카메라가 이미 열려있다면 닫기
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            
            # 사용 가능한 카메라 목록 확인
            self.check_available_cameras()
            
            # 사용 가능한 카메라가 없는 경우
            if not self.available_cameras:
                logging.warning("사용 가능한 카메라가 없습니다.")
                return False
                
            # 1. 먼저 사용 가능한 상태의 카메라 시도
            available_indices = [cam['index'] for cam in self.available_cameras if cam['status'] == 'available']
            
            # 2. 사용 가능한 카메라가 없으면 사용 중인 카메라도 시도
            if not available_indices:
                logging.warning("사용 가능한 카메라가 없어 사용 중인 카메라에 접근을 시도합니다.")
                available_indices = [cam['index'] for cam in self.available_cameras if cam['status'] == 'in_use' and cam['index'] >= 0]
            
            # 3. 그래도 없으면 종료
            if not available_indices:
                logging.error("접근 가능한 카메라가 없습니다.")
                return False
            
            # 설정된 카메라 인덱스 먼저 시도 (USB 카메라 포함)
            if config.CAMERA_INDEX in available_indices:
                connection_methods = [
                    # DirectShow는 Windows에서 USB 카메라에 가장 적합한 방법
                    (f"DirectShow USB 카메라 {config.CAMERA_INDEX}", lambda: cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_DSHOW)),
                    # Media Foundation도 Windows에서 USB 카메라 지원
                    (f"MSMF USB 카메라 {config.CAMERA_INDEX}", lambda: cv2.VideoCapture(config.CAMERA_INDEX, cv2.CAP_MSMF)),
                    # 일반 인덱스 방식 (마지막 시도)
                    (f"카메라 인덱스 {config.CAMERA_INDEX}", lambda: cv2.VideoCapture(config.CAMERA_INDEX))
                ]
                
                # 설정된 인덱스로 연결 시도
                if any(self._try_connect_camera(method_name, method_func) for method_name, method_func in connection_methods):
                    return True
            
            # 설정된 인덱스로 연결 실패시 다른 카메라 시도
            for idx in available_indices:
                if idx == config.CAMERA_INDEX:  # 이미 시도한 인덱스는 건너뛰기
                    continue
                    
                methods = [
                    # DirectShow는 Windows에서 USB 카메라에 가장 적합한 방법
                    (f"DirectShow USB 카메라 {idx}", lambda i=idx: cv2.VideoCapture(i, cv2.CAP_DSHOW)),
                    # Media Foundation도 Windows에서 USB 카메라 지원
                    (f"MSMF USB 카메라 {idx}", lambda i=idx: cv2.VideoCapture(i, cv2.CAP_MSMF)),
                    # 일반 인덱스 방식 (마지막 시도)
                    (f"카메라 인덱스 {idx}", lambda i=idx: cv2.VideoCapture(i))
                ]
                if any(self._try_connect_camera(method_name, method_func) for method_name, method_func in methods):
                    return True
            
            # 모든 방법 실패 - 가상 카메라 유지
            logging.error("모든 카메라 연결 방법 실패, 가상 카메라 유지")
            return False

    def _try_connect_camera(self, method_name, method_func):
        """카메라 연결 시도 헬퍼 함수"""
        try:
            logging.info(f"카메라 연결 방법 '{method_name}' 시도 중...")
            temp_camera = method_func()
            
            # 카메라 설정
            temp_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            temp_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            temp_camera.set(cv2.CAP_PROP_FPS, 30)
            temp_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 버퍼 크기 최소화
            
            # 연결 확인 (실제 프레임 읽기 시도)
            success, test_frame = temp_camera.read()
            if success and test_frame is not None and test_frame.size > 0:
                logging.info(f"카메라 연결 성공 (방법: {method_name})")
                with self.camera_lock:
                    if self.camera is not None:
                        self.camera.release()
                    self.camera = temp_camera
                    self.is_running = True
                return True
            else:
                logging.warning(f"카메라 연결 방법 '{method_name}' 실패: 프레임을 읽을 수 없음")
                temp_camera.release()
                return False
        except Exception as e:
            logging.error(f"카메라 연결 방법 '{method_name}' 오류: {str(e)}")
            return False
            
    def create_virtual_camera(self):
        """웹캠이 없을 때 가상 카메라 생성"""
        self.is_running = True
        # 검은색 배경에 텍스트 표시
        self.virtual_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(self.virtual_frame, "Camera Connecting...", (120, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(self.virtual_frame, "Please wait while we connect to your camera", (80, 280),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(self.virtual_frame, "Press 'Stream Start' to retry", (120, 320),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
    def get_frame(self):
        """카메라에서 프레임 가져오기"""
        # 마지막 접근 시간 업데이트
        self.last_access = time.time()
        
        # 첫 프레임 요청 시 카메라 초기화 시작
        if self.initialization_thread is None:
            self.start_camera_initialization()
        
        # 카메라가 초기화 중이거나 연결되지 않은 경우 가상 프레임 반환
        with self.camera_lock:
            if self.camera is None:
                return self.virtual_frame.copy()
            
            try:
                # 프레임 읽기 시도
                success, frame = self.camera.read()
                
                if not success or frame is None or frame.size == 0:
                    logging.warning("프레임 읽기 실패, 카메라 재연결 시도")
                    self.is_running = False
                    self.start_camera_initialization()  # 백그라운드에서 재연결 시도
                    return self.virtual_frame.copy()
                
                return frame
                
            except Exception as e:
                logging.error(f"프레임 읽기 오류: {str(e)}")
                self.is_running = False
                self.start_camera_initialization()  # 백그라운드에서 재연결 시도
                return self.virtual_frame.copy()

    def cleanup(self):
        """카메라 자원 해제"""
        with self.camera_lock:
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            self.is_running = False

# 카메라 객체 지연 초기화 함수
def get_camera():
    global camera_instance
    if camera_instance is None:
        logging.info("카메라 객체 초기화 중...")
        camera_instance = Camera()
        logging.info("카메라 객체 초기화 완료")
    return camera_instance

# 백그라운드에서 YOLOv8 모델 로딩
def load_detector_in_background():
    global detector, detector_loading, detector_ready
    
    if detector is None and not detector_loading:
        detector_loading = True
        detector_ready.clear()
        
        def load_model():
            global detector
            try:
                logging.info("YOLOv8 모델 로딩 시작 (백그라운드)...")
                start_time = time.time()
                detector = ObjectDetector(
                    model_path=config.MODEL_PATH,
                    confidence_threshold=config.CONFIDENCE_THRESHOLD,
                    exclude_classes=config.EXCLUDE_CLASSES,
                    max_detections=config.MAX_DETECTIONS
                )
                elapsed_time = time.time() - start_time
                logging.info(f"YOLOv8 모델 로딩 완료 (소요 시간: {elapsed_time:.2f}초)")
                detector_ready.set()
            except Exception as e:
                logging.error(f"YOLOv8 모델 로딩 실패: {str(e)}")
                detector_loading = False
        
        # 백그라운드 스레드에서 모델 로딩
        threading.Thread(target=load_model, daemon=True).start()
        logging.info("YOLOv8 모델 로딩 백그라운드 스레드 시작")

# 객체 감지기 초기화 함수 - 지연 초기화
def initialize_detector():
    global detector, detector_loading, detector_ready
    
    # 모델 로딩이 시작되지 않았으면 시작
    if detector is None and not detector_loading:
        load_detector_in_background()
    
    # 모델이 로딩 중이면 대기 메시지 반환
    if detector is None:
        if not detector_ready.is_set():
            logging.info("YOLOv8 모델 로딩 중... 잠시 기다려 주세요.")
            return None
    
    return detector

# 병 분류기 초기화 함수 - 지연 초기화
def initialize_bottle_classifier():
    global bottle_classifier
    if bottle_classifier is None:
        logging.info("병 분류기 초기화 중...")
        bottle_classifier = BottleClassifier()
        logging.info("병 분류기 초기화 완료")
    return bottle_classifier

# 웹사이트 시작 시 백그라운드에서 모델 로딩 시작
@app.before_first_request
def before_first_request():
    logging.info("첫 번째 요청 처리 시작")
    # 백그라운드에서 YOLOv8 모델 로딩 시작
    load_detector_in_background()
    logging.info("첫 번째 요청 처리 완료")

@app.route('/')
def index():
    logging.info("메인 페이지 요청")
    return render_template('index.html')

@app.route('/training')
def training():
    logging.info("학습 페이지 요청")
    return render_template('training.html')

def generate_frames():
    """비디오 스트림 생성 함수"""
    logging.info("비디오 스트림 생성 시작")
    
    # 백그라운드에서 YOLOv8 모델 로딩 시작
    load_detector_in_background()
    
    while True:
        # 카메라 객체 지연 초기화
        cam = get_camera()
        
        # 프레임 가져오기
        frame = cam.get_frame()
        
        # 객체 감지 수행 (모델이 로딩된 경우에만)
        detector_instance = initialize_detector()
        if detector_instance is not None:
            try:
                detections = detector_instance.detect(frame)
                
                # 감지된 객체 표시
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 라벨 표시
                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                logging.debug(f"감지 결과: {len(detections)} 개체")  # 디버그 레벨로 변경
                
            except Exception as e:
                logging.error(f"객체 감지 중 오류: {str(e)}")
        else:
            # 모델 로딩 중임을 프레임에 표시
            cv2.putText(frame, "YOLOv8 모델 로딩 중...", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # JPEG 인코딩
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # 프레임 전송
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # 프레임 레이트 제한
        time.sleep(0.03)  # 약 30 FPS

@app.route('/video_feed')
def video_feed():
    """비디오 스트림 라우트"""
    logging.info("비디오 피드 요청")
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/snapshot')
def take_snapshot():
    """현재 프레임의 스냅샷 저장"""
    logging.info("스냅샷 요청")
    try:
        # 카메라 객체 지연 초기화
        cam = get_camera()
        
        # 프레임 가져오기
        frame = cam.get_frame()
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        filepath = os.path.join(config.SNAPSHOT_DIR, filename)
        
        # 이미지 저장
        cv2.imwrite(filepath, frame)
        
        # 상대 경로 생성 (웹에서 접근 가능한 경로)
        relative_path = f"snapshots/{filename}"
        
        logging.info(f"스냅샷 저장 완료: {filepath}")
        return jsonify({
            "success": True,
            "message": "스냅샷이 저장되었습니다.",
            "filepath": relative_path
        })
    except Exception as e:
        logging.error(f"스냅샷 저장 중 오류: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"스냅샷 저장 중 오류가 발생했습니다: {str(e)}"
        }), 500

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
        frame = get_camera().get_frame()
        
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
    get_camera().cleanup()

if __name__ == '__main__':
    logging.info("서버 시작")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG) 