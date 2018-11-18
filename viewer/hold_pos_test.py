import cv2
import numpy as np

img = cv2.imread('background.jpg')
H,W,_ = img.shape
print('size:', W,H)

#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hold_pos_x_list = np.loadtxt('hold_pos_x.csv', dtype=int, delimiter=',')
hold_pos_y_list = np.loadtxt('hold_pos_y.csv', dtype=int, delimiter=',')
for x in hold_pos_x_list:
    for y in hold_pos_y_list:
        cv2.circle(img=img, center=(x,y), radius=10, color=(0,0,0), thickness=1)

#img = cv2.resize(img, dsize=None, fx=1.5, fy=1.5)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

