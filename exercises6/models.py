"""
Neural Networks for classification
Author: Kacper Wojtasiński (s17460)
"""
from abc import ABC, abstractmethod, abstractproperty

import sys
from typing import Tuple, Any

sys.path.insert(0, "../exercises5/")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import tensorflow as tf


class TensorflowBasedNN(ABC):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __init__(self):
        self.model = None
        (self.train_data, self.train_labels), (
            self.test_data,
            self.test_labels,
        ) = self.provide_data()

    @abstractmethod
    def provide_data(self) -> Tuple[Tuple[Any], Tuple[Any]]:
        """abstract method that needs to be implemented.
        it should return data in form (train_data, train_labels), (test_data, test_labels)
        that will be used to train and evaluate model
        """
        ...

    @abstractmethod
    def train(self, epochs: int):
        """abstract method that needs to be implemented (should implement "epochs" argument).
        it has to train model (so has to call model.compile and model.fit methods)
        """
        ...

    def evaluate(self) -> float:
        """method to compute accuracy of model

        Returns:
            float: accuracy of model
        """
        _, test_acc = self.model.evaluate(self.test_data, self.test_labels, verbose=2)
        return test_acc

    def to_keras_model(self, path: str):
        """helper method to save model as keras model

        Args:
            path (str): path where model should be saved
        """
        tf.keras.models.save_model(self.model, path)

    @classmethod
    def from_keras_model(cls, path: str) -> "TensorflowBasedNN":
        """helper method to load keras model

        Args:
            path (str): path where model is saved

        Returns:
            TensorflowBasedNN: [description]
        """
        klass = cls()
        klass.model = tf.keras.models.load_model(path)
        return klass


class TensorflowBasedNNWithImages(TensorflowBasedNN):
    @property
    @abstractmethod
    def all_classes(self) -> list:
        """abstract property that has to be implemented
            it has to return list with all the classes of data
            it is used for show_image method to set labels


        Returns:
            list: all the classes of data
        """
        ...

    def show_image(self, idx: int, from_test_data=True):
        """method to show image using matplotlib

        Args:
            idx (int): index of image to be shown
            from_test_data (bool, optional): determines if should show image from test or train data. Defaults to True.
        """
        image = self.train_data[idx] if from_test_data else self.test_data[idx]
        label = self.train_labels[idx] if from_test_data else self.test_labels[idx]

        try:
            if not isinstance(label, int):
                label = label[0]
        except:
            pass

        plt.figure()
        plt.imshow(image)
        plt.colorbar()
        plt.grid(False)
        plt.xlabel(self.all_classes[label])
        plt.show()


class Iris(TensorflowBasedNN):
    """Neural network for Iris dataset (taken from sklearn)
    adapted from https://www.kaggle.com/zahoorahmad/tensorflow-and-keras-on-iris-dataset
    """

    def provide_data(self):
        """method to get Iris dataset from sklearn API

        Returns:
            Tuple: tuple with Iris dataset
        """
        data = load_iris()
        x_train, x_test, y_train, y_test = train_test_split(
            data.data, data.target, test_size=0.1
        )
        return (x_train, y_train), (x_test, y_test)

    def train(self, epochs=50):
        """method to train Iris model

        Args:
            epochs (int, optional): epochs to train model. Defaults to 50.
        """
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model.fit(self.train_data, self.train_labels, epochs=epochs)


class BiggerIris(Iris):
    """ class which extends Iris class with more Dense and Dropout layers """

    def train(self, epochs=50):
        """method to train BiggerIris model

        Args:
            epochs (int, optional): epochs to train model. Defaults to 50.
        """
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(512, activation=tf.nn.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(10, activation=tf.nn.softmax),
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        self.model.fit(self.train_data, self.train_labels, epochs=epochs)


class FashionMnist(TensorflowBasedNNWithImages):
    """Neural network for FashionMNIST dataset
    adapted from https://www.tensorflow.org/tutorials/keras/classification
    """

    @property
    def all_classes(self):
        """property list with all the FashionMNIST classes

        Returns:
            list: list with all the FashionMNIST classes
        """
        return [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]

    def provide_data(self) -> Tuple[Tuple[Any], Tuple[Any]]:
        """method to get FashionMNIST dataset from keras API

        Returns:
            Tuple: tuple with FashionMNIST dataset
        """
        fashion_mnist = tf.keras.datasets.fashion_mnist
        return fashion_mnist.load_data()

    def train(self, epochs=10):
        """method to train FashionMNIST model

        Args:
            epochs (int, optional): epochs to train model. Defaults to 10.
        """
        self.train_data = self.train_data / 255.0
        self.test_data = self.test_data / 255.0

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        self.model.fit(self.train_data, self.train_labels, epochs=epochs)


class BiggerFashionMnist(FashionMnist):
    """ class which extends FashionMnist class with more Dense layers """

    def train(self, epochs=10):
        """method to train BiggerFashionMnist model

        Args:
            epochs (int, optional): epochs to train model. Defaults to 10.
        """
        self.train_data = self.train_data / 255.0
        self.test_data = self.test_data / 255.0

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        self.model.fit(self.train_data, self.train_labels, epochs=epochs)


class Cifar10(TensorflowBasedNNWithImages):
    """Neural network for Cifar10 dataset
    adapted from https://www.tensorflow.org/tutorials/images/cnn
    """

    @property
    def all_classes(self) -> list:
        """property list with all the Cifar10 classes

        Returns:
            list: list with all the Cifar10 classes
        """
        return [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]

    def provide_data(self) -> Tuple[Tuple[Any], Tuple[Any]]:
        """method to get Cifar10 dataset from keras API

        Returns:
            Tuple: tuple with Cifar10 dataset
        """
        return tf.keras.datasets.cifar10.load_data()

    def train(self, epochs=10):
        """method to train Cifar10 model

        Args:
            epochs (int, optional): epochs to train model. Defaults to 10.
        """
        self.train_data = self.train_data / 255.0
        self.test_data = self.test_data / 255.0

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, (3, 3), activation="relu", input_shape=(32, 32, 3)
                ),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        self.model.fit(self.train_data, self.train_labels, epochs=epochs)


class SimplerCifar10(Cifar10):
    """ class which extends Cifar10 class with less Conv2D and MaxPooling2D layers """

    def train(self, epochs=10):
        """method to train SimplerCifar10 model

        Args:
            epochs (int, optional): epochs to train model. Defaults to 10.
        """
        self.train_data = self.train_data / 255.0
        self.test_data = self.test_data / 255.0

        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, (3, 3), activation="relu", input_shape=(32, 32, 3)
                ),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(10),
            ]
        )

        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        self.model.fit(self.train_data, self.train_labels, epochs=epochs)


class Imdb(TensorflowBasedNN):
    """Neural network for IMDB dataset
    adapted from https://builtin.com/data-science/how-build-neural-network-keras
    NUM_WORDS define how many words we want to take from this dataset
    """

    NUM_WORDS = 10000

    @staticmethod
    def _vectorize(sequences, dimension: int):
        results = np.zeros((len(sequences), dimension))

        for i, sequence in enumerate(sequences):
            results[i, sequence] = 1

        return results

    def provide_data(self) -> Tuple[Tuple[Any], Tuple[Any]]:
        """method to get IMDB dataset from keras API

        Returns:
            Tuple[Tuple[Any], Tuple[Any]]: tuple with IMDB dataset
        """
        (training_data, training_targets), (
            testing_data,
            testing_targets,
        ) = tf.keras.datasets.imdb.load_data(num_words=self.NUM_WORDS)

        data = np.concatenate((training_data, testing_data), axis=0)
        targets = np.concatenate((training_targets, testing_targets), axis=0)
        data = self._vectorize(data, self.NUM_WORDS)
        targets = np.array(targets).astype("float32")
        test_x = data[: self.NUM_WORDS]
        test_y = targets[: self.NUM_WORDS]
        train_x = data[self.NUM_WORDS :]
        train_y = targets[self.NUM_WORDS :]

        return (train_x, train_y), (test_x, test_y)

    def train(self, epochs=2):
        """method to train IMDB model

        Args:
            epochs (int, optional): epochs to train model. Defaults to 2.
        """
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    50, activation="relu", input_shape=(self.NUM_WORDS,)
                ),
                tf.keras.layers.Dense(50, activation="relu"),
                tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None),
                tf.keras.layers.Dense(50, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model.fit(
            self.train_data,
            self.train_labels,
            epochs=epochs,
            batch_size=500,
        )


class BiggerImdb(Imdb):
    """ class which extends Imdb class with extra dense and dropout layers """

    def train(self, epochs=2):
        """method to train BiggerImdb model

        Args:
            epochs (int, optional): epochs to train model. Defaults to 2.
        """
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Dense(
                    50, activation="relu", input_shape=(self.NUM_WORDS,)
                ),
                tf.keras.layers.Dense(50, activation="relu"),
                tf.keras.layers.Dropout(0.2, noise_shape=None, seed=None),
                tf.keras.layers.Dense(50, activation="relu"),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        self.model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        self.model.fit(
            self.train_data,
            self.train_labels,
            epochs=epochs,
            batch_size=500,
        )
