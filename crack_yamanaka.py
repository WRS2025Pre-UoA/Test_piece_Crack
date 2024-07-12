import cv2
import numpy as np

# 画像の読み込み
image = cv2.imread('IMG_1800.jpg')

#グレースケール
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#ぼかし
gray = cv2.GaussianBlur(gray, (13, 13), 0)

# エッジ検出の適用
edges = cv2.Canny(gray, 50, 100, apertureSize=3)

#エッジの画像表示(仮)
scale_percent = 10  
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized_edges = cv2.resize(edges, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('Detected Shapes and Lines', resized_edges)
cv2.waitKey(0)
cv2.destroyAllWindows()




# 線を検出
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

# 元の画像に線を描画
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

# 線の長さを計算
line_lengths = [np.sqrt((x2 - x1)**2 + (y2 - y1)**2) for line in lines]

#print(line_lengths)


# 画像をリサイズして表示
scale_percent = 10  
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

# 検出された線を描画した画像を表示
cv2.imshow('Detected Shapes and Lines', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

