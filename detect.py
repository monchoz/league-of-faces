import cv2
import dlib
from PIL import Image as Img
from fastai.vision import *

stream = cv2.VideoCapture('video.mp4')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68.dat')

frame = stream.read()
crop_width = 100
incremental = 100
simple_crop = True
faces_dirname = 'images/'
classes = ['Ahri', 'Darius', 'Draven', 'Graves', 'Katarina', 'Leona', 'Zyra']

learn = load_learner(path='./', file='trained_model.pkl')
print('Model loaded')

model = learn.model # the Fastai model
mean_std_stats = learn.data.stats # the input means/standard deviations
class_names = learn.data.classes # the class names

while True:
    _, frame = stream.read()

    # need grayscale for dlib face detection
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(img_gray, 0)
    img = Img.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # print("Faces detected: %d" % (len(rects)))

    # loop through detected faces
    for rect in rects:
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

            # save cropped face to image locally
            cropped_image = image_to_crop.crop(crop_area)
            crop_size = (crop_width, crop_width)
            cropped_image.thumbnail(crop_size)
            cropped_name = faces_dirname + str(incremental) + ".jpg"
            cropped_image.save(cropped_name, "JPEG")
            # run prediction
            predicted_classes, y, probs = learn.predict(open_image(cropped_name))
            print(f'Found {predicted_classes} ({round(probs[y].numpy()*100,2)}%)')
            incremental += 1

    cv2.imshow("League of Faces", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()