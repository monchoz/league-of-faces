import dlib
from PIL import Image

import cv2

vs = cv2.VideoCapture('video.mp4')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68.dat')

frame = vs.read()
fps = 60
crop_width = 108
incremental = 100
simple_crop = True
faces_dirname = 'images'

while True:
    _, frame = vs.read()

    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = []

    rects = detector(img_gray, 0)
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    print("Faces detected: %d" % (len(rects)))

    for rect in rects:
        face = {}

        shades_width = rect.right() - rect.left()
        shades_height = rect.bottom() - rect.top()

        # crop faces found
        if shades_width >= crop_width and shades_height >= crop_width:
            image_to_crop = img
            
            if simple_crop:
                crop_area = (rect.left(), rect.top(), rect.right(), rect.bottom())
            else:
                size_array = []
                size_array.append(rect.top())
                size_array.append(image_to_crop.height - rect.bottom())
                size_array.append(rect.left())
                size_array.append(image_to_crop.width - rect.right())
                size_array.sort()
                short_side = size_array[0]
                crop_area = (rect.left() - size_array[0] , rect.top() - size_array[0], rect.right() + size_array[0], rect.bottom() + size_array[0])

            cropped_image = image_to_crop.crop(crop_area)
            crop_size = (crop_width, crop_width)
            cropped_image.thumbnail(crop_size)
            cropped_image.save(faces_dirname + "/" + str(incremental) + ".jpg", "JPEG")
            incremental += 1

    cv2.imshow("League of Faces", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()