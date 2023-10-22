import numpy as np
import cv2
import glob


imgs = glob.glob("Lane Detection Project/camera calib/Fisheye1_*.jpg")

o = np.zeros((6 * 8, 3), np.float32)
o[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
objPoints = []
imgPoints = []


for img in imgs:
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)
    if ret:
        imgPoints.append(corners)
        objPoints.append(o)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objPoints, imgPoints, gray.shape[::-1], None, None
)

i = 1
for img in imgs:
    img = cv2.imread(img)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite(
        "Lane Detection Project/camera calib/calibrated/undist{}.jpg".format(i), dst
    )
    i += 1
    # cv2.imshow("t", dst)
    # cv2.waitKey(0)

print(mtx)
