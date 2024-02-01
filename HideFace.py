import cv2

img_ = cv2.imread('man.jpg', 1)

desired_width = 400
desired_height = 300

original_height, original_width = img_.shape[:2]

aspect_ratio = original_width / original_height

if aspect_ratio >= 1: 
    new_width = desired_width
    new_height = int(new_width / aspect_ratio)
else:  
    new_height = desired_height
    new_width = int(new_height * aspect_ratio)


############################################################################
img = cv2.resize(img_, (new_width, new_height))
img_org = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
haar_cascade_face = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')

face_rects = haar_cascade_face.detectMultiScale(gray,
                                                scaleFactor=1.2,
                                                minNeighbors=5)

x, y, h, w = face_rects[0]
sub_face = img[y:y+h, x:x+w]
blur = cv2.blur(sub_face, (15,15))
img[y:y+blur.shape[0], x:x+blur.shape[1]] = blur

cv2.imshow("Original",img_org)
cv2.imshow("My Face",img)
cv2.waitKey(0)

cv2.destroyAllWindows()
