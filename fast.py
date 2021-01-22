from fastai.vision import *

learn = load_learner(path='./', file='trained_model.pkl')

model = learn.model # the pytorch model
mean_std_stats = learn.data.stats # the input means/standard deviations
class_names = learn.data.classes # the class names

# Predict on an image
images = ['images/100.jpg']
for path in images:
    img = open_image(path)
    predicted_classes, y, probs = learn.predict(img)
    print(predicted_classes)
    print(round(probs[y].numpy()*100,2))
    print(y)