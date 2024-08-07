import cv2
import numpy as np

# グローバル変数
brightness = 0
contrast = 0
canny1 = 100
canny2 = 200
square_range = 100
square_x = 0
square_y = 0
min_line_length = 50
max_line_gap = 10

# コールバック関数
def nothing(x):
    pass

# トラックバーの作成
def create_trackbars():
    cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Trackbars', 400, 400)  # トラックバーウィンドウのサイズを調整
    cv2.createTrackbar('Brightness', 'Trackbars', 0, 100, nothing)
    cv2.createTrackbar('Contrast', 'Trackbars', 0, 100, nothing)
    cv2.createTrackbar('Canny1', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('Canny2', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('Square Range', 'Trackbars', 0, 1000, nothing)
    cv2.createTrackbar('Square X', 'Trackbars', 0, 1000, nothing)
    cv2.createTrackbar('Square Y', 'Trackbars', 0, 1000, nothing)
    cv2.createTrackbar('Min Line Length', 'Trackbars', 50, 500, nothing)
    cv2.createTrackbar('Max Line Gap', 'Trackbars', 10, 100, nothing)

def get_trackbar_values():
    global brightness, contrast, canny1, canny2, square_range, square_x, square_y, min_line_length, max_line_gap
    brightness = cv2.getTrackbarPos('Brightness', 'Trackbars')
    contrast = cv2.getTrackbarPos('Contrast', 'Trackbars')
    canny1 = cv2.getTrackbarPos('Canny1', 'Trackbars')
    canny2 = cv2.getTrackbarPos('Canny2', 'Trackbars')
    square_range = cv2.getTrackbarPos('Square Range', 'Trackbars')
    square_x = cv2.getTrackbarPos('Square X', 'Trackbars')
    square_y = cv2.getTrackbarPos('Square Y', 'Trackbars')
    min_line_length = cv2.getTrackbarPos('Min Line Length', 'Trackbars')
    max_line_gap = cv2.getTrackbarPos('Max Line Gap', 'Trackbars')

def process_image(original_image):
    global brightness, contrast, canny1, canny2, square_range, square_x, square_y, min_line_length, max_line_gap
    
    # 画像のコピーを作成
    image = original_image.copy()

    # 明るさとコントラストの調整
    adjusted = cv2.convertScaleAbs(image, alpha=1 + contrast / 100.0, beta=brightness)
    
    # グレースケール変換
    gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
    
    # Cannyエッジ検出
    edges = cv2.Canny(gray, canny1, canny2)
    
    # 正方形のマスク作成
    mask = np.zeros_like(edges)
    cv2.rectangle(mask, (square_x, square_y), (square_x + square_range, square_y + square_range), 255, -1)
    
    # マスクをエッジ画像に適用
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # ハフ変換による直線検出
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # 長さを計算し、直線を描画
    length_cm = 0
    if lines is not None:
        longest_line = None
        max_length = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length > max_length:
                max_length = length
                longest_line = line[0]
        
        if longest_line is not None:
            x1, y1, x2, y2 = longest_line
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            length_cm = (max_length / (square_range * np.sqrt(2))) * 20  # 直線の長さを足す
            
            # 始点と終点の座標を画像に表示
            cv2.putText(image, f'({x1},{y1})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(image, f'({x2},{y2})', (x2, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return adjusted, edges, image, length_cm

def resize_image(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def main():
    create_trackbars()

    # 画像の読み込み
    image = cv2.imread('image.jpg')

    # 画像が読み込めたか確認
    if image is None:
        print("Error: Image not found or unable to load.")
        return

    # 画像のリサイズ
    scale_percent = 30  # 画像サイズのスケールファクター
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    display_width = 600  # 表示する画像の幅
    display_height = 600  # 表示する画像の高さ

    while True:
        get_trackbar_values()
        
        adjusted, edges, display_image, length_cm = process_image(image)
        
        # 結果を表示
        display_image = resize_image(display_image, display_width, display_height)
        adjusted = resize_image(adjusted, display_width, display_height)
        edges = resize_image(edges, display_width, display_height)
        
        cv2.rectangle(display_image, (square_x, square_y), (square_x + square_range, square_y + square_range), (0, 255, 0), 2)
        cv2.putText(display_image, f'Line Length: {length_cm:.2f} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_image, f'Brightness: {brightness}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_image, f'Contrast: {contrast}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Image', display_image)
        cv2.imshow('Adjusted Image', adjusted)
        cv2.imshow('Edges', edges)  # エッジ画像を表示
        
        if cv2.waitKey(1) & 0xFF == 27:  # Escキーで終了
            break
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
