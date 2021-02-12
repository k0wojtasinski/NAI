
"""
MakeArt - web application to create Neural Network based art from two images
Heavily inspired by: https://www.tensorflow.org/tutorials/generative/style_transfer
Author: Kacper Wojtasiński (s17460)
"""
import fastapi
import tensorflow as tf
import tensorflow_hub as tf_hub

app = fastapi.FastAPI()

import numpy as np
import PIL.Image

from fastapi import FastAPI, File, UploadFile, Request, HTTPException, status
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


from fastapi.responses import FileResponse

templates = Jinja2Templates(directory=".")


class MakeArt:
    def __init__(self, first_image_file_path: str, second_image_file_path: str):
        """ class to turn two images into completely new one

        Args:
            first_image_file_path (str): path to first file
            second_image_file_path (str):  path to second file
        """
        self.first_image = self._load_img(first_image_file_path)
        self.second_image = self._load_img(second_image_file_path)

    def _tensor_to_image(self, tensor) -> PIL.Image:
        """ helper method to turn tensor into image.
            it returns Pillow Image class, which will be saved as file

        Args:
            tensor (TensorFlow tensor): [description]

        Returns:
            PIL.Image: [description]
        """
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def _load_img(self, image_file_path: str):
        """ helper function to preprocess image to set correct size and dimensions of it

        Args:
            image_file_path (str): path to image

        """
        max_dim = 512
        img = tf.io.read_file(image_file_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(shape * scale, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def save_file(self, path: str):
        """method to save transformed file.
            It transforms file by using 'arbitrary-image-stylization-v1-256' TensorFlow Hub model

        Args:
            path (str): path to transformed file
        """
        hub_model = tf_hub.load(
            "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
        )
        stylized_image = hub_model(
            tf.constant(self.first_image), tf.constant(self.second_image)
        )[0]
        image = self._tensor_to_image(stylized_image)

        with open("result.jpg", "w+") as _f:
            image.save(path)


app = fastapi.FastAPI()


@app.post("/process_image/")
async def process_image(
    file: UploadFile = File(...), file_second: UploadFile = File(...)
) -> FileResponse:
    """ route to make art based on two source images: image to transform and image to apply style from

    Args:
        file (UploadFile, optional): file of first image. Defaults to File(...).
        file_second (UploadFile, optional): file of second image. Defaults to File(...).

    Returns:
        FileResponse: transformed file
    """
    first = await file.read()
    second = await file_second.read()

    with open(file.filename, "wb") as _f:
        _f.write(first)

    with open(file_second.filename, "wb") as _f:
        _f.write(second)

    client = MakeArt(file.filename, file_second.filename)
    client.save_file("result.jpg")

    return FileResponse("result.jpg")


@app.get("/", response_class=HTMLResponse)
async def show_index(request: Request):
    """ route to show index static file """
    return templates.TemplateResponse("index.html", context={"request": request})
