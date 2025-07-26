import cv2




adress = "/media/d2u25/Dont/S4S-ROF/frames/315/S3-SNr3.02_D/frames20250703_112141_DropNumber_01/000001.jpg"
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
img = cv2.imread(adress)

resized_image = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


cv2.imshow("S",resized_image)
cv2.waitKey(12000)