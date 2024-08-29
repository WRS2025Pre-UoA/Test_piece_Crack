import cv2
import numpy as np

# クリックした点を格納するリスト
points_list = []

def dist(p1, p2):
    """
    2点間のユークリッド距離を計算する関数。
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def mouseEvents(event, x, y, flags, param):
    """
    マウスクリックイベントを処理する関数。
    左クリックでクリックした点をリストに追加する。
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        points_list.append([x, y])

def resize_image(image, scale_percent):
    """
    画像のサイズを指定した縮小率でリサイズする関数。
    """
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized_image

def select_points(image_path, scale_percent):
    """
    画像を表示して、ユーザーに4つの点を選択させる関数。
    """
    img = cv2.imread(image_path)
    img = resize_image(img, scale_percent)  # 画像のリサイズ
    cv2.imshow("Select the 4 points", img)
    cv2.setMouseCallback("Select the 4 points", mouseEvents)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    points = np.array(points_list, dtype="float32")
    print("Selected points:", points)
    return img, points

def perspective_transform(img, points):
    """
    4つの点を基に画像の射影変換を行う関数。
    """
    lengths = [dist(points[i], points[(i+1) % 4]) for i in range(4)]
    max_length = int(max(lengths))
    
    square = np.array([[0, 0], [max_length, 0], 
                       [max_length, max_length], [0, max_length]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(points, square)
    output_size = (max_length, max_length)
    warped = cv2.warpPerspective(img, M, output_size)
    
    return warped

def detect_and_measure_lines(img, canny_thresh1, canny_thresh2, scale=20):
    """
    画像内の黒い直線を検出し、cm単位で長さを計算する関数。
    """
    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Cannyエッジ検出
    edges = cv2.Canny(gray, canny_thresh1, canny_thresh2, apertureSize=3)

    # Hough変換を使って直線を検出
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=20, maxLineGap=5)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 検出した直線を赤色で表示
            length = dist((x1, y1), (x2, y2))
            length_cm = (length / img.shape[1]) * scale
            cv2.putText(img, f"Length: {length_cm:.2f} cm", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            print(f"Line length in cm: {length_cm:.2f}")

    return img

def update_image(val):
    """
    トラックバーの値に基づいて画像を更新するコールバック関数。
    """
    brightness = cv2.getTrackbarPos('Brightness', 'Warped Image') - 100
    contrast = cv2.getTrackbarPos('Contrast', 'Warped Image') / 100.0
    canny_threshold1 = cv2.getTrackbarPos('Canny Thresh 1', 'Warped Image')
    canny_threshold2 = cv2.getTrackbarPos('Canny Thresh 2', 'Warped Image')
    
    temp_image = cv2.convertScaleAbs(warped_img, alpha=contrast, beta=brightness)
    result_image = detect_and_measure_lines(temp_image.copy(), canny_threshold1, canny_threshold2, scale=20)
    
    # 画像を表示
    cv2.imshow('Warped Image', result_image)

def main():
    """
    主要な処理を行うメイン関数。
    """
    global warped_img

    image_path = "image14.jpg"  # 画像のパスを指定
    scale_percent = 20  # 縮小率を指定

    img, points = select_points(image_path, scale_percent)
    warped_img = perspective_transform(img, points)
    
    # ウィンドウの設定
    cv2.namedWindow('Warped Image')

    # トラックバーの設定
    cv2.createTrackbar('Brightness', 'Warped Image', 100, 200, update_image)
    cv2.createTrackbar('Contrast', 'Warped Image', 100, 300, update_image)
    cv2.createTrackbar('Canny Thresh 1', 'Warped Image', 50, 255, update_image)
    cv2.createTrackbar('Canny Thresh 2', 'Warped Image', 150, 255, update_image)
    
    # 初期画像の表示
    update_image(0)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
