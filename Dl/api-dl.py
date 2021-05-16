from fastapi import FastAPI, UploadFile, Form, File
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.applications.densenet import DenseNet121
from keras.layers import Input, Lambda, Dense, Flatten, Conv2D, MaxPool2D
import tensorflow as tf
import keras
import cv2
import numpy as np
import time
from PIL import Image

def create_new_model():
    IMAGE_SIZE = [32, 32]
    vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    for layer in vgg.layers:
        layer.trainable = False
    x = Flatten()(vgg.output)
    prediction = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=vgg.input, outputs=prediction)
    
    model.compile(optimizer='adam',
              loss='mse',
              metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model

def dl_model(input_arr):
    
    confidence = np.random.uniform(size=3)
    confidence = confidence / np.sum(confidence)
    class_names = ["glaucoma", "normal", "other"]
    argmax = np.argmax(confidence)
    
    start = time.time()
    image_path = "test_images/t/03.jpg"

    # Predict
    g_predict = model_glaucoma.predict(input_arr)

    n_predict = model_normal.predict(input_arr)

    o_predict = model_other.predict(input_arr)

    print(g_predict)
    print(n_predict)
    print(o_predict)
    
    arrc = np.concatenate((np.transpose(g_predict)[0],np.transpose(n_predict)[0]))
    arrc = np.concatenate((arrc,np.transpose(o_predict)[0]))

    # Change it
    g_predict = (g_predict < 0.5).astype(np.float32)
    n_predict = (n_predict < 0.5).astype(np.float32)
    o_predict = (o_predict < 0.5).astype(np.float32)

    arr = np.concatenate((np.transpose(g_predict)[0],np.transpose(n_predict)[0]))
    arr = np.concatenate((arr,np.transpose(o_predict)[0]))
    if np.where(arr == 0):
        x = np.where(arr == 0)
        arrc[x] = 0
    
    end = time.time()
    print(f"Runtime of the program is {end - start}")
    conf = []
    for i in range (len(arrc)):
        x = float(str(arrc[i])[0:7])
        while x < 1 and x != 0:
            x = x * 10
        conf.append(x)

    max_value = max(conf)
    max_index = conf.index(max_value)
    
    print(class_names[max_index])
    print(confidence[argmax])

    return class_names[max_index], confidence[argmax]

app = FastAPI()

# Load Model
model_glaucoma = create_new_model()
model_glaucoma.load_weights("gsaved_models/model_2.h5")

model_normal = create_new_model()
model_normal.load_weights("nsaved_models/model_5.h5")

model_other = create_new_model()
model_other.load_weights("osaved_models/model_5.h5")

@app.get("/")
async def helloworld():
    return {"greeting": "Hello World"}


@app.post("/api/fundus")
async def upload_image(nonce: str=Form(None, title="Query Text"), 
                       image: UploadFile = File(...)):
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print("---------------------------")
    dim = (32, 32)
    image = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    input_arr = keras.preprocessing.image.img_to_array(pil_image)
    input_arr = np.array([input_arr])  # Convert single image to a batch.

    class_out, class_conf = dl_model(input_arr)
    
    return {
        "nonce": nonce,
        "classification": class_out,
        "confidence_score": np.float(class_conf),
        "debug": {
            "image_size": dict(zip(["height", "width", "channels"], img.shape)),
        }
    }
