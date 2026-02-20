import cv2
import numpy as np

# 1. 500x500 크기의 3채널(RGB) 검은색 이미지 생성
# np.zeros((세로, 가로, 채널), dtype=np.uint8)
img = np.zeros((500, 500, 3), dtype=np.uint8)

# 2. 이미지에 흰색 선 그리기 (cv2.line(이미지, 시작점, 끝점, 색상, 두께))
# cv2.line(img, (0, 0), (500, 500), (255, 255, 255), 5)

# 3. 이미지에 파란색 사각형 그리기 (cv2.rectangle(이미지, 좌상단, 우하단, 색상, 두께))
cv2.rectangle(img, (150, 150), (300, 300), (255, 255, 255), -1) # -1은 채우기



# 4. 이미지 저장
# cv2.imwrite('created_image.png', img)

# 5. 이미지 화면 표시
cv2.imshow('Made with CV2', img)
cv2.waitKey(0) # 키 입력 대기
cv2.destroyAllWindows()
