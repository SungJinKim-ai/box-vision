import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# 플레이스홀더 이미지 생성 (회색 배경에 카메라 아이콘)
def create_placeholder_image(width=400, height=300):
    # 회색 배경 생성 (OpenCV)
    image = np.ones((height, width, 3), dtype=np.uint8) * 230
    
    # 카메라 아이콘 그리기
    # 카메라 본체
    cv2.rectangle(image, (width//2-50, height//2-30), (width//2+50, height//2+30), (150, 150, 150), -1)
    # 렌즈
    cv2.circle(image, (width//2, height//2), 25, (100, 100, 100), -1)
    cv2.circle(image, (width//2, height//2), 15, (70, 70, 70), -1)
    # 플래시
    cv2.rectangle(image, (width//2+30, height//2-40), (width//2+50, height//2-30), (150, 150, 150), -1)
    
    # OpenCV 이미지를 PIL 이미지로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    
    # 시스템 폰트 사용 (Windows 기본 폰트)
    try:
        # Windows 기본 폰트 경로
        font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕
        font = ImageFont.truetype(font_path, 20)
    except:
        # 폰트를 찾을 수 없는 경우 기본 폰트 사용
        font = ImageFont.load_default()
    
    # 텍스트 추가
    text = "이미지 없음"
    text_width = draw.textlength(text, font=font)
    text_x = (width - text_width) // 2
    text_y = height//2 + 50
    draw.text((text_x, text_y), text, font=font, fill=(100, 100, 100))
    
    # PIL 이미지를 다시 OpenCV 이미지로 변환
    image_with_text = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return image_with_text

# 디렉토리 확인
os.makedirs("static/img", exist_ok=True)

# 이미지 생성 및 저장
placeholder = create_placeholder_image()
cv2.imwrite("static/img/placeholder.png", placeholder)

print("플레이스홀더 이미지가 생성되었습니다: static/img/placeholder.png") 