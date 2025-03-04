# Box Vision - YOLOv8 기반 실시간 객체 인식 시스템

Box Vision은 YOLOv8 모델을 활용한 실시간 객체 인식 웹 애플리케이션입니다. 웹캠을 통해 실시간으로 객체를 감지하고, 사람을 제외한 다양한 객체를 인식하여 표시합니다.

## 기술 스택

- **Python 3.8+**: 주요 프로그래밍 언어
- **OpenCV**: 이미지 처리 및 컴퓨터 비전 라이브러리
- **YOLOv8**: 객체 감지 모델
- **Flask**: 웹 서버 프레임워크
- **HTML/CSS/JavaScript**: 프론트엔드 인터페이스

## 하드웨어 요구사항

- **웹캠**: 실시간 영상 입력용
- **CPU**: 최소 Intel Core i5 또는 동급 이상 (실시간 처리를 위해)
- **RAM**: 최소 8GB
- **GPU**: NVIDIA GPU (선택 사항, 성능 향상을 위해 권장)

## 설치 및 설정

### 1. 가상 환경 생성 및 활성화

```bash
# 가상 환경 생성
python -m venv box-vision-env

# 가상 환경 활성화 (Windows)
box-vision-env\Scripts\activate

# 가상 환경 활성화 (Linux/Mac)
source box-vision-env/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. YOLOv8 모델 다운로드

```bash
# models 디렉토리가 없는 경우 생성
mkdir -p models

# YOLOv8n 모델 다운로드 (자동으로 다운로드되지만, 수동으로도 가능)
# 첫 실행 시 자동으로 다운로드됩니다.
```

## 프로젝트 구조

```
box-vision/
├── app.py                  # 메인 Flask 애플리케이션
├── src/
│   ├── __init__.py         # 패키지 초기화
│   ├── config.py           # 설정 파일
│   └── detector.py         # 객체 감지 클래스
├── models/                 # YOLOv8 모델 파일 저장
├── static/
│   ├── css/                # CSS 파일
│   ├── js/                 # JavaScript 파일
│   └── snapshots/          # 저장된 스냅샷 이미지
├── templates/              # HTML 템플릿
│   └── index.html          # 메인 페이지
├── logs/                   # 로그 파일 저장
├── requirements.txt        # 의존성 목록
└── README.md               # 프로젝트 설명
```

## 실행 방법

```bash
# 애플리케이션 실행
python app.py
```

웹 브라우저에서 `http://localhost:5000`으로 접속하여 애플리케이션을 사용할 수 있습니다.

## 주요 기능

- **실시간 객체 인식**: YOLOv8 모델을 사용하여 웹캠 영상에서 객체를 실시간으로 감지합니다.
- **웹 인터페이스**: 사용자 친화적인 웹 인터페이스를 통해 감지 결과를 확인할 수 있습니다.
- **스냅샷 저장**: 현재 프레임의 스냅샷을 저장하고 감지된 객체 정보를 함께 표시합니다.
- **통계 정보**: 감지된 객체 수, 처리된 프레임 수, FPS 등의 통계 정보를 제공합니다.
- **사람 제외 기능**: 기본적으로 사람 클래스를 제외하고 객체를 감지합니다 (설정에서 변경 가능).

## 성능 최적화

- **모델 크기 조정**: 성능이 낮은 환경에서는 YOLOv8n(nano) 모델을 사용하고, 고성능 환경에서는 YOLOv8m(medium) 또는 YOLOv8l(large) 모델을 사용할 수 있습니다.
- **해상도 조정**: 입력 이미지 해상도를 조정하여 처리 속도를 향상시킬 수 있습니다.
- **GPU 가속**: CUDA 지원 GPU가 있는 경우 자동으로 GPU 가속을 사용합니다.

## 확장성

- **클래스 필터링**: `config.py`에서 `EXCLUDE_CLASSES` 목록을 수정하여 특정 클래스를 제외하거나 포함할 수 있습니다.
- **신뢰도 임계값**: `config.py`에서 `CONFIDENCE_THRESHOLD` 값을 조정하여 감지 정확도를 조절할 수 있습니다.
- **최대 감지 객체 수**: `config.py`에서 `MAX_DETECTIONS` 값을 조정하여 한 프레임에서 감지할 최대 객체 수를 설정할 수 있습니다.

## 로깅

- 애플리케이션은 `logs` 디렉토리에 로그 파일을 생성합니다.
- 로그 파일은 날짜와 시간을 포함한 이름으로 생성됩니다 (예: `detection_log_20250303_122736.log`).
- 로그에는 서버 시작, 모델 로드, 객체 감지 결과, 오류 등의 정보가 포함됩니다.
- 로그 레벨은 INFO로 설정되어 있으며, 필요에 따라 `app.py`에서 변경할 수 있습니다.
- 로그 파일을 통해 애플리케이션의 동작 상태와 성능을 모니터링할 수 있습니다.

## 문제 해결

### 웹캠 접근 오류

- 웹캠이 다른 애플리케이션에서 사용 중인지 확인하세요.
- `config.py`에서 `CAMERA_INDEX` 값을 변경하여 다른 웹캠을 사용해 보세요.

### 모델 로드 오류

- `models` 디렉토리에 YOLOv8 모델 파일이 있는지 확인하세요.
- 인터넷 연결을 확인하여 모델이 자동으로 다운로드될 수 있도록 하세요.

### 성능 문제

- 더 작은 YOLOv8 모델(YOLOv8n)을 사용해 보세요.
- 디버그 모드를 비활성화하여 성능을 향상시킬 수 있습니다 (`config.py`에서 `DEBUG = False`).

## 참고 자료

- [YOLOv8 공식 문서](https://docs.ultralytics.com/)
- [Flask 공식 문서](https://flask.palletsprojects.com/)
- [OpenCV 공식 문서](https://docs.opencv.org/)

## 라이선스

이 프로젝트는 MIT 라이선스에 따라 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

YOLOv8 모델은 AGPL-3.0 라이선스에 따라 배포됩니다. 상업적 용도로 사용할 경우 Ultralytics에서 상업용 라이선스를 구매해야 할 수 있습니다. 