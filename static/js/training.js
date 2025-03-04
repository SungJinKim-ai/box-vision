/**
 * 병 분류 학습 인터페이스 JavaScript
 */

let isStreaming = true;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectInterval = 3000; // 3초
let capturedImageData = null;

document.addEventListener('DOMContentLoaded', function() {
    // 요소 참조
    const videoStream = document.getElementById('video-stream');
    const captureBtn = document.getElementById('capture-btn');
    const capturedImage = document.getElementById('captured-image');
    const saveSampleBtn = document.getElementById('save-sample-btn');
    const trainModelBtn = document.getElementById('train-model-btn');
    const trainingForm = document.getElementById('training-form');
    
    // 통계 요소
    const totalSamples = document.getElementById('total-samples');
    const smallBottleCount = document.getElementById('small-bottle-count');
    const mediumBottleCount = document.getElementById('medium-bottle-count');
    const largeBottleCount = document.getElementById('large-bottle-count');
    const cleanCount = document.getElementById('clean-count');
    const slightlyContaminatedCount = document.getElementById('slightly-contaminated-count');
    const heavilyContaminatedCount = document.getElementById('heavily-contaminated-count');
    
    // 초기화 함수 호출
    init();
    
    // 초기화 함수
    function init() {
        console.log('학습 인터페이스 초기화...');
        
        // 이벤트 리스너 등록
        if (captureBtn) {
            console.log('캡처 버튼 이벤트 리스너 등록');
            captureBtn.addEventListener('click', captureImage);
        }
        
        if (trainingForm) {
            trainingForm.addEventListener('submit', function(e) {
                e.preventDefault();
                saveSample(e);
            });
        }
        
        if (trainModelBtn) {
            trainModelBtn.addEventListener('click', trainModel);
        }
        
        // 스트림 토글 버튼 이벤트 리스너
        const streamToggleBtn = document.getElementById('stream-toggle');
        if (streamToggleBtn) {
            streamToggleBtn.addEventListener('click', function() {
                const button = this;
                const statusText = button.querySelector('.status-text');
                
                if (isStreaming) {
                    stopStream();
                    button.classList.add('stopped');
                    statusText.textContent = '스트림 시작';
                } else {
                    startStream();
                    button.classList.remove('stopped');
                    statusText.textContent = '스트림 중지';
                }
                
                isStreaming = !isStreaming;
            });
        }
        
        // 비디오 스트림 모니터링 시작
        if (videoStream) {
            monitorStream(videoStream);
        }
        
        // 초기 통계 로드
        loadStats();
    }
    
    // 비디오 스트림 시작
    function startStream() {
        if (videoStream) {
            videoStream.src = "/video_feed";
            reconnectAttempts = 0;
            
            // 스트림 상태 모니터링
            monitorStream(videoStream);
        }
    }
    
    // 비디오 스트림 중지
    function stopStream() {
        if (videoStream) {
            videoStream.src = "/static/img/placeholder.png";
        }
    }
    
    // 스트림 상태 모니터링
    function monitorStream(videoElement) {
        videoElement.onerror = function() {
            if (isStreaming && reconnectAttempts < maxReconnectAttempts) {
                console.log(`스트림 재연결 시도 ${reconnectAttempts + 1}/${maxReconnectAttempts}`);
                showNotification('warning', `스트림 재연결 중... (${reconnectAttempts + 1}/${maxReconnectAttempts})`);
                
                setTimeout(() => {
                    reconnectAttempts++;
                    startStream();
                }, reconnectInterval);
            } else if (reconnectAttempts >= maxReconnectAttempts) {
                showNotification('error', '스트림 연결 실패. 페이지를 새로고침하거나 나중에 다시 시도해주세요.');
                const button = document.getElementById('stream-toggle');
                if (button) {
                    button.classList.add('stopped');
                    const statusText = button.querySelector('.status-text');
                    if (statusText) {
                        statusText.textContent = '스트림 시작';
                    }
                }
                isStreaming = false;
            }
        };
    }
    
    // 이미지 캡처 함수
    function captureImage(e) {
        if (e) e.preventDefault();
        
        console.log('이미지 캡처 시도...');
        
        if (!isStreaming) {
            showNotification('error', '스트림이 중지되었습니다. 스트림을 시작한 후 다시 시도하세요.');
            return;
        }
        
        // 캡처 버튼 비활성화
        if (captureBtn) captureBtn.disabled = true;
        
        showNotification('info', '이미지를 캡처하는 중...');
        
        fetch('/api/capture', {
            method: 'POST'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('캡처 응답:', data);
            if (data.success) {
                // 캡처된 이미지 표시
                if (capturedImage) {
                    capturedImage.src = data.image_url + '?t=' + new Date().getTime();
                    capturedImageData = data.image_data;
                    
                    // 저장 버튼 활성화
                    if (saveSampleBtn) {
                        saveSampleBtn.disabled = false;
                    }
                    
                    showNotification('success', '이미지가 캡처되었습니다.');
                }
            } else {
                showNotification('error', data.message || '이미지 캡처에 실패했습니다.');
            }
        })
        .catch(error => {
            console.error('캡처 오류:', error);
            showNotification('error', '이미지 캡처 중 오류가 발생했습니다: ' + error.message);
        })
        .finally(() => {
            // 캡처 버튼 다시 활성화
            if (captureBtn) captureBtn.disabled = false;
        });
    }
    
    // 샘플 저장 함수
    function saveSample(e) {
        if (e) e.preventDefault();
        
        // 병 유형과 오염도 가져오기
        const bottleType = document.querySelector('input[name="bottle-type"]:checked');
        const contaminationLevel = document.querySelector('input[name="contamination-level"]:checked');
        
        if (!bottleType || !contaminationLevel) {
            showNotification('warning', '병 유형과 오염도를 모두 선택해주세요.');
            return;
        }
        
        if (!capturedImageData) {
            showNotification('warning', '먼저 이미지를 캡처해주세요.');
            return;
        }
        
        // 저장 버튼 비활성화
        if (saveSampleBtn) saveSampleBtn.disabled = true;
        
        showNotification('info', '샘플을 저장하는 중...');
        
        // 데이터 준비
        const data = {
            image_data: capturedImageData,
            bottle_type: bottleType.value,
            contamination_level: contaminationLevel.value
        };
        
        // API 호출
        fetch('/api/save_sample', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                showNotification('success', '샘플이 저장되었습니다.');
                
                // 통계 업데이트
                loadStats();
                
                // 폼 초기화
                resetForm();
            } else {
                showNotification('error', data.message || '샘플 저장에 실패했습니다.');
            }
        })
        .catch(error => {
            console.error('저장 오류:', error);
            showNotification('error', '샘플 저장 중 오류가 발생했습니다: ' + error.message);
        })
        .finally(() => {
            // 저장 버튼 다시 활성화 (폼이 초기화되었으므로 비활성화 상태 유지)
            if (saveSampleBtn && capturedImageData) {
                saveSampleBtn.disabled = false;
            }
        });
    }
    
    // 모델 학습 함수
    function trainModel(e) {
        if (e) e.preventDefault();
        
        const trainBtn = document.getElementById('train-model-btn');
        if (trainBtn) {
            trainBtn.disabled = true;
            trainBtn.textContent = '학습 중...';
        }
        
        showNotification('info', '모델 학습을 시작합니다...');
        
        fetch('/api/train_model', {
            method: 'POST'
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                showNotification('success', '모델 학습이 완료되었습니다.');
            } else {
                showNotification('error', data.message || '모델 학습에 실패했습니다.');
            }
        })
        .catch(error => {
            console.error('학습 오류:', error);
            showNotification('error', '모델 학습 중 오류가 발생했습니다: ' + error.message);
        })
        .finally(() => {
            if (trainBtn) {
                trainBtn.disabled = false;
                trainBtn.textContent = '학습';
            }
        });
    }
    
    // 샘플 통계 로드 함수
    function loadStats() {
        console.log('통계 로드 중...');
        
        fetch('/api/sample_stats')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('통계 데이터:', data);
            
            // 총 샘플 수 업데이트
            if (totalSamples) {
                totalSamples.textContent = data.total || 0;
            }
            
            // 병 유형별 통계 업데이트
            if (smallBottleCount) smallBottleCount.textContent = data.bottle_types?.소형 || 0;
            if (mediumBottleCount) mediumBottleCount.textContent = data.bottle_types?.중형 || 0;
            if (largeBottleCount) largeBottleCount.textContent = data.bottle_types?.대형 || 0;
            
            // 오염도별 통계 업데이트
            if (cleanCount) cleanCount.textContent = data.contamination_levels?.깨끗함 || 0;
            if (slightlyContaminatedCount) slightlyContaminatedCount.textContent = data.contamination_levels?.['약간 오염'] || 0;
            if (heavilyContaminatedCount) heavilyContaminatedCount.textContent = data.contamination_levels?.['심한 오염'] || 0;
        })
        .catch(error => {
            console.error('통계 로드 오류:', error);
            showNotification('error', '통계 로드 중 오류가 발생했습니다: ' + error.message);
        });
    }
    
    // 폼 초기화 함수
    function resetForm() {
        // 라디오 버튼 초기화
        const radioButtons = document.querySelectorAll('input[type="radio"]');
        radioButtons.forEach(radio => {
            radio.checked = false;
        });
        
        // 캡처된 이미지 초기화
        if (capturedImage) {
            capturedImage.src = '/static/img/placeholder.png';
        }
        
        // 캡처 데이터 초기화
        capturedImageData = null;
        
        // 저장 버튼 비활성화
        if (saveSampleBtn) {
            saveSampleBtn.disabled = true;
        }
    }
    
    // 알림 표시 함수
    function showNotification(type, message) {
        let notification = document.getElementById('notification');
        
        if (!notification) {
            notification = document.createElement('div');
            notification.id = 'notification';
            notification.className = 'notification hidden';
            document.body.appendChild(notification);
        }
        
        console.log(`알림 표시: [${type}] ${message}`);
        
        notification.textContent = message;
        notification.className = `notification ${type}`;
        notification.classList.remove('hidden');

        setTimeout(() => {
            notification.classList.add('hidden');
        }, 3000);
    }
}); 