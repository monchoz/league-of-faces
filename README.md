![](./cover.png)
# League of Faces
League of Legends A New Dawn cinematic champions recognition with AI.

This project is not related with any work or colaboration with/from Riot Games. It's been created by myself with the only porpuse of exploring the capabilities of the use of AI on a video (besides I really like this game :p).

We'll use Dlib's `get_frontal_face_detector`, along with the [68 point shape prediction](https://github.com/davisking/dlib-models) model (shape_predictor_68.dat).

The script uses the face detection algorithm in Dlib to read any front faces. It uses the prediction points to create and crop the face then saves it as an image for each face detected.

The images detected where used as training data to build a Deep Learning model that is used to predict the name of the characters (Champions) on the video. Trained with the state of the art image classification model with Resnet, in PyTorch, using the fastai library.

![](Screenshot_1.png)
![](Screenshot_2.png)

## Instructions
Install `pipenv`
```
pip install pipenv
```

Activate virtual environment
```
pipenv shell
```

Install requirements
```
pipenv install -r requirements.txt
```

Run application
```
python detect.py
```