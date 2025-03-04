// DOM 요소 참조
const videoFeed = document.getElementById('video-feed');
const snapshotBtn = document.getElementById('snapshot-btn');
const totalDetections = document.getElementById('total-detections');
const framesProcessed = document.getElementById('frames-processed');
const avgFps = document.getElementById('avg-fps');
const elapsedTime = document.getElementById('elapsed-time');
const detectionCountsList = document.getElementById('detection-counts-list');
const snapshotsContainer = document.getElementById('snapshots-container');
const normalCount = document.getElementById('normal-count');
const abnormalCount = document.getElementById('abnormal-count');

// 최근 스냅샷 저장 배열 (최대 6개)
const recentSnapshots = [];
const MAX_SNAPSHOTS = 6;

let isStreaming = true;
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;
const reconnectInterval = 3000; // 3초

// 페이지 로드 시 통계 정보 업데이트 시작
document.addEventListener('DOMContentLoaded', () => {
    // 초기 통계 정보 로드
    updateStats();
    
    // 5초마다 통계 정보 업데이트
    setInterval(updateStats, 5000);
    
    // 스냅샷 버튼 이벤트 리스너
    snapshotBtn.addEventListener('click', takeSnapshot);
    
    // 비디오 피드 로드 확인
    videoFeed.addEventListener('load', () => {
        console.log('비디오 피드가 로드되었습니다.');
    });
    
    videoFeed.addEventListener('error', (e) => {
        console.error('비디오 피드 로드 중 오류 발생:', e);
        alert('비디오 피드를 로드할 수 없습니다. 서버 연결을 확인하세요.');
    });

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
    const videoStream = document.getElementById('video-feed');
    if (videoStream) {
        monitorStream(videoStream);
    }
});

/**
 * 서버에서 통계 정보를 가져와 화면에 표시
 */
async function updateStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        if (response.ok) {
            // 통계 정보 업데이트
            totalDetections.textContent = data.total_detections;
            framesProcessed.textContent = data.frames_processed;
            avgFps.textContent = data.avg_fps.toFixed(2);
            elapsedTime.textContent = data.elapsed_time.toFixed(0);
            
            // 정상/비정상 병 수 업데이트
            if ('normal_count' in data && 'abnormal_count' in data) {
                normalCount.textContent = data.normal_count;
                abnormalCount.textContent = data.abnormal_count;
                
                // 비정상 병이 있으면 경고 표시
                if (data.abnormal_count > 0) {
                    abnormalCount.parentElement.classList.add('warning');
                } else {
                    abnormalCount.parentElement.classList.remove('warning');
                }
                
                // 정상/비정상 병 비율 계산
                const totalBottles = data.normal_count + data.abnormal_count;
                if (totalBottles > 0) {
                    const abnormalRatio = (data.abnormal_count / totalBottles * 100).toFixed(1);
                    // 비정상 병 비율이 30% 이상이면 경고 메시지 표시
                    if (abnormalRatio >= 30 && document.querySelector('.abnormal-warning') === null) {
                        const warningMsg = document.createElement('p');
                        warningMsg.className = 'abnormal-warning';
                        warningMsg.textContent = `경고: 비정상 병 비율이 ${abnormalRatio}%입니다!`;
                        document.querySelector('.abnormal-stats').appendChild(warningMsg);
                    } else if (abnormalRatio < 30) {
                        const warningMsg = document.querySelector('.abnormal-warning');
                        if (warningMsg) {
                            warningMsg.remove();
                        }
                    }
                }
            }
            
            // 병 유형 및 오염도 정보 업데이트
            updateBottleClassification(data);
            
            // 객체 감지 횟수 목록 업데이트
            updateDetectionCounts(data.detection_counts);
        } else {
            console.error('통계 정보를 가져오는 중 오류 발생:', data.error);
        }
    } catch (error) {
        console.error('통계 정보 요청 중 오류 발생:', error);
    }
}

/**
 * 객체 감지 횟수 목록 업데이트
 * @param {Object} counts - 객체 클래스별 감지 횟수
 */
function updateDetectionCounts(counts) {
    // 기존 목록 초기화
    detectionCountsList.innerHTML = '';
    
    // 감지 횟수가 많은 순으로 정렬
    const sortedCounts = Object.entries(counts)
        .sort((a, b) => b[1] - a[1]);
    
    // 목록 생성
    sortedCounts.forEach(([className, count]) => {
        const item = document.createElement('div');
        item.className = 'count-item';
        
        // 병(bottle) 클래스인 경우 강조 표시
        if (className === 'bottle') {
            item.classList.add('highlight');
        }
        
        item.innerHTML = `
            <span class="class-name">${className}</span>
            <span class="count">${count}</span>
        `;
        detectionCountsList.appendChild(item);
    });
    
    // 감지된 객체가 없는 경우
    if (sortedCounts.length === 0) {
        const emptyItem = document.createElement('div');
        emptyItem.className = 'count-item empty';
        emptyItem.textContent = '감지된 객체가 없습니다.';
        detectionCountsList.appendChild(emptyItem);
    }
}

/**
 * 현재 프레임의 스냅샷 저장
 */
async function takeSnapshot() {
    try {
        // 버튼 비활성화
        snapshotBtn.disabled = true;
        snapshotBtn.textContent = '저장 중...';
        
        const response = await fetch('/snapshot');
        const data = await response.json();
        
        if (response.ok) {
            // 스냅샷 정보 저장
            const snapshot = {
                filename: data.filename,
                detections: data.detections,
                objects: data.objects,
                timestamp: new Date().toLocaleTimeString()
            };
            
            // 최근 스냅샷 배열에 추가
            recentSnapshots.unshift(snapshot);
            
            // 최대 개수 유지
            if (recentSnapshots.length > MAX_SNAPSHOTS) {
                recentSnapshots.pop();
            }
            
            // 스냅샷 목록 업데이트
            updateSnapshotsList();
            
            // 성공 메시지
            alert('스냅샷이 저장되었습니다.');
        } else {
            console.error('스냅샷 저장 중 오류 발생:', data.error);
            alert(`스냅샷 저장 실패: ${data.error}`);
        }
    } catch (error) {
        console.error('스냅샷 요청 중 오류 발생:', error);
        alert('스냅샷 저장 중 오류가 발생했습니다.');
    } finally {
        // 버튼 상태 복원
        snapshotBtn.disabled = false;
        snapshotBtn.textContent = '스냅샷 저장';
    }
}

/**
 * 최근 스냅샷 목록 업데이트
 */
function updateSnapshotsList() {
    // 기존 목록 초기화
    snapshotsContainer.innerHTML = '';
    
    // 스냅샷 항목 생성
    recentSnapshots.forEach(snapshot => {
        const item = document.createElement('div');
        item.className = 'snapshot-item';
        
        // 이미지와 정보 추가
        item.innerHTML = `
            <img src="/${snapshot.filename}" alt="스냅샷" onclick="window.open('/${snapshot.filename}', '_blank')">
            <div class="snapshot-info">
                ${snapshot.detections}개 객체
                ${snapshot.abnormal_count > 0 ? `<span class="abnormal-warning">(비정상: ${snapshot.abnormal_count})</span>` : ''}
            </div>
        `;
        
        // 툴팁 추가
        const objectsList = snapshot.objects.join(', ');
        item.title = `시간: ${snapshot.timestamp}\n감지된 객체: ${objectsList}\n정상: ${snapshot.normal_count}, 비정상: ${snapshot.abnormal_count}`;
        
        snapshotsContainer.appendChild(item);
    });
    
    // 스냅샷이 없는 경우
    if (recentSnapshots.length === 0) {
        const noData = document.createElement('p');
        noData.textContent = '저장된 스냅샷이 없습니다.';
        snapshotsContainer.appendChild(noData);
    }
}

// 병 유형 및 오염도 정보 업데이트
function updateBottleClassification(data) {
    // 병 유형 업데이트
    const smallBottleCount = document.getElementById('small-bottle-count');
    const mediumBottleCount = document.getElementById('medium-bottle-count');
    const largeBottleCount = document.getElementById('large-bottle-count');
    
    smallBottleCount.textContent = data.bottle_types['소형'] || 0;
    mediumBottleCount.textContent = data.bottle_types['중형'] || 0;
    largeBottleCount.textContent = data.bottle_types['대형'] || 0;
    
    // 오염도 업데이트
    const cleanCount = document.getElementById('clean-count');
    const slightlyContaminatedCount = document.getElementById('slightly-contaminated-count');
    const heavilyContaminatedCount = document.getElementById('heavily-contaminated-count');
    
    cleanCount.textContent = data.contamination_levels['깨끗함'] || 0;
    slightlyContaminatedCount.textContent = data.contamination_levels['약간 오염'] || 0;
    heavilyContaminatedCount.textContent = data.contamination_levels['심한 오염'] || 0;
}

// 비디오 스트림 시작
function startStream() {
    const videoStream = document.getElementById('video-feed');
    if (videoStream) {
        videoStream.src = "/video_feed";
        reconnectAttempts = 0;
        
        // 스트림 상태 모니터링
        monitorStream(videoStream);
    }
}

// 비디오 스트림 중지
function stopStream() {
    const videoStream = document.getElementById('video-feed');
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

// 알림 표시 함수
function showNotification(type, message) {
    let notification = document.getElementById('notification');
    
    if (!notification) {
        notification = document.createElement('div');
        notification.id = 'notification';
        notification.className = 'notification hidden';
        document.body.appendChild(notification);
    }
    
    notification.textContent = message;
    notification.className = `notification ${type}`;
    notification.classList.remove('hidden');

    setTimeout(() => {
        notification.classList.add('hidden');
    }, 3000);
} 