# Simple API for classification experiments
# Author: Kacper Wojtasiński

from collections import UserDict
from datetime import datetime
import json
from typing import List, Tuple

import click

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


class MachineLearningExperimentResults(UserDict):
    def __init__(self, name: str, classifier_name: str, data: dict):
        """this is class to present results for MachineLearningExperiments
            it subclasses UserDict, so it behave similarily to dict ("get" method, etc.)
            Each dataset needs to have float/int values (except for y-column)
            Each dataset needs to have y-column as the last one
        Args:
            name (str): name of the experiment
            classifier_name (str): name of the used classifier
            data (dict): results of experiment
        """
        super().__init__(data)
        self.data["name"] = name
        self.data["classifier_name"] = classifier_name
        self.data["date"] = str(datetime.now())

        self.name = name
        self.classifier = classifier_name
        self.right = self.data.get("right")
        self.wrong = self.data.get("wrong")
        self.score = self.data.get("score")
        self.date = self.data.get("date")

    def _validate_comparison(self, other: "MachineLearningExperimentResults"):
        """it validates other - object we want to compare to
            object must be of type MachineLearningExperimentResults and have the same name

        Args:
            other (MachineLearningExperimentResults): object to compare to

        Raises:
            TypeError: if given object is not of type MachineLearningExperimentResults
            ValueError: if given MachineLearningExperimentResults's name is not the same as ours
        """
        if type(other) != type(self):
            raise TypeError(
                f'Comparison makes sense only for "{self.__class__.__name__}" type'
            )

        if self.name != other.name:
            raise ValueError(
                f'Comparison makes sense only for results for the same experiments. Comparing "{self.name}" with "{other.name}"'
            )

    def __lt__(self, other: "MachineLearningExperimentResults"):
        self._validate_comparison(other)

        return self.score < other.score

    def __gt__(self, other: "MachineLearningExperimentResults"):
        self._validate_comparison(other)

        return self.score > other.score

    def __str__(self):
        return f"{self.__class__.__name__} (name={self.name}, date={self.date}, classifier_name={self.classifier}, right={len(self.right)}, wrong={len(self.wrong)}, score={self.score})"

    def __repr__(self):
        return str(self)

    def to_json(self, path_to_json: str):
        """ method to save this class as a JSON file

        Args:
            path_to_json (str): path to created file
        """
        with open(path_to_json, "w") as _f:
            json.dump(self.data, _f)


class MachineLearningExperiment:
    def __init__(
        self, name: str, dataframe: pd.DataFrame, test_size=0.1, train_size=0.9
    ):
        """this is class to perform simple machine learning experiments using scikit-learn
            it requires that the last column of dataframe has the Y-values

        Args:
            name (str): name of the experiment
            dataframe (pd.DataFrame): dataframe with data
            test_size (float, optional): size of test set. Defaults to 0.1.
            train_size (float, optional): size of train set. Defaults to 0.9.
        """
        self.name = name
        self.classifiers = {}
        self.df = dataframe
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = self._divide_into_test_train(test_size, train_size)

    def __str__(self):
        return f"{self.__class__.__name__} (name={self.name}, classifiers={list(self.classifiers.keys())}, X_train={len(self.X_train)}, X_test={len(self.X_test)}, y_train={len(self.y_train)}, y_test={len(self.y_test)})"

    @classmethod
    def from_csv(
        cls, name: str, csv_file_path: str, test_size=0.1, train_size=0.9
    ) -> "MachineLearningExperiment":
        return cls(name, pd.read_csv(csv_file_path), test_size, train_size)

    def _raise_if_classifier_does_not_exist(self, classifier_name: str):
        """helper method to raise exception when classifier with given name
            is not available

        Args:
            classifier_name (str): name of classifier

        Raises:
            ValueError: when classifier is not available
        """
        if classifier_name not in self.classifiers:
            raise ValueError(f"{classifier_name} classifier is not present")

    def _divide_into_test_train(
        self, test_size: int, train_size: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """helper method to split data into test and train sets

        Args:
            test_size (int): size of test set
            train_size (int): size of train set

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: tuple with dataframes for all the subsets
        """
        X_train, X_test, y_train, y_test = train_test_split(
            self.df.iloc[:, :-1],
            self.df.iloc[:, -1],
            test_size=test_size,
            train_size=train_size,
        )
        return X_train, X_test, y_train, y_test

    def setup_svm(self, classifier_name="SVM", **kwargs):
        """method to setup SVM classifier.
            you can override classifier_name to compare different sets of kwargs
            all the kwargs will be provided to the scikit-learn classifier constructor

        Args:
            classifier_name (str, optional): [description]. Defaults to 'SVM'.
        """
        if not classifier_name in self.classifiers:
            clf = svm.SVC(**kwargs)
            clf.fit(self.X_train, self.y_train)
            self.classifiers[classifier_name] = clf

    def setup_ada(self, classifier_name="ADA", **kwargs):
        """method to setup ADA classifier.
            you can override classifier_name to compare different sets of kwargs
            all the kwargs will be provided to the scikit-learn classifier constructor

        Args:
            classifier_name (str, optional): [description]. Defaults to 'ADA'.
        """
        if not classifier_name in self.classifiers:
            clf = AdaBoostClassifier(**kwargs)
            clf.fit(self.X_train, self.y_train)
            self.classifiers[classifier_name] = clf

    def predict(self, classifier_name: str, _input: list) -> pd.Series:
        """method to predict output of given values

        Args:
            classifier_name (str): name of classifier to be used
            _input (list): values to be used to predict

        Returns:
            pd.Series: results
        """
        self._raise_if_classifier_does_not_exist(classifier_name)

        clf = self.classifiers.get(classifier_name)

        return pd.Series(clf.predict(_input))

    def get_results(self, classifier_name: str) -> MachineLearningExperimentResults:
        """method to get results for classifier with given name.
            it evaluates model with y_test

        Args:
            classifier_name (str): name of the classifer

        Returns:
            MachineLearningExperimentResults: results
        """
        self._raise_if_classifier_does_not_exist(classifier_name)

        results_dict = {"right": [], "wrong": [], "score": 0}
        predictions = self.predict(classifier_name, self.X_test)
        turn_pandas_series_into_list = lambda idx: self.X_test.iloc[idx, :].tolist()

        for idx, (predicted, truth) in enumerate(zip(predictions, self.y_test)):
            if predicted == truth:
                results_dict["right"].append(
                    {
                        "data": turn_pandas_series_into_list(idx),
                        "predicted": predicted,
                        "truth": truth,
                    }
                )
                results_dict["score"] += 1
            else:
                results_dict["wrong"].append(
                    {
                        "data": turn_pandas_series_into_list(idx),
                        "predicted": predicted,
                        "truth": truth,
                    }
                )

        results_dict["score"] /= len(self.y_test)

        return MachineLearningExperimentResults(
            self.name, classifier_name, results_dict
        )

    def get_results_for_all_classifiers(
        self, _sorted=True, reverse=True
    ) -> List[MachineLearningExperimentResults]:
        """method to prepare list of results for all the classifiers
           it runs get_results for all of them

        Args:
            _sorted (bool, optional): determines if list should be sorted by score. Defaults to True.
            reverse (bool, optional): determines if list should be sorted in descending order. Defaults to True.

        Returns:
            List[MachineLearningExperimentResults]: list with results
        """
        results = [
            self.get_results(classifier_name)
            for classifier_name in self.classifiers.keys()
        ]

        if _sorted:
            return sorted(results, reverse=reverse)

        return results


@click.group()
def cli():
    """ click group to create CLI powered by click """
    ...


@cli.command()
@click.option("--name", "-n", prompt="Name of the experiment", required=True)
@click.option("--path", prompt="Path to CSV file", required=True)
@click.option(
    "--save_output",
    "-s",
    prompt="Save results to json files?",
    required=True,
    type=bool,
)
def validate(name, path, save_output):
    experiment = MachineLearningExperiment.from_csv(name, path)

    experiment.setup_ada()
    experiment.setup_svm()

    results = experiment.get_results_for_all_classifiers()

    for result in results:
        print(result)

    if save_output:
        for result in results:
            result.to_json(f"{result.name}_{result.classifier}_result.json")


@cli.command()
@click.option("--name", "-n", prompt="Name of the experiment", required=True)
@click.option("--path", prompt="Path to CSV file", required=True)
@click.option("--input", prompt="Comma separated input", required=True)
def predict(name, path, input):
    experiment = MachineLearningExperiment.from_csv(name, path)

    experiment.setup_ada()
    experiment.setup_svm()

    output_for_svm = experiment.predict(
        "SVM", np.array(input.split(",")).reshape(1, -1)
    )[0]
    output_for_ada = experiment.predict(
        "ADA", np.array(input.split(",")).reshape(1, -1)
    )[0]

    print(f"Output for SVM: {output_for_svm}")
    print(f"Output for ADA: {output_for_ada}")


if __name__ == "__main__":
    cli()
