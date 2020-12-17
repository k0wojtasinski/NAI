# Authors: Kacper Wojtasiński, Piotr Palczewski
"""
Program silnika rekomendacji filmów na podstawie pliku JSON 
z danymi zebranymi od użytkowników składającego się z:
"imie_nazwisko":{
    "tytul_filmu": ocena,
    ...,
    }

Program sprawdza kompatybilność pomiędzy użytkownikami
na podstawie wspólnych filmów i ich ocen obliczając
odległość euklidesową lub Persona (zależnie od wyboru użytkownika)
na wykresie dwuwymiarowanym, wykorzystując klasteryzacje danych.
Wybiera się najbliższą osobę pod względem obejrzanych filmów 
i wybiera 6 filmów (o ile istnieją) których osoba dla której 
szukamy rekomendacji nie oglądała o ocenie powyżej 8 oraz 
6 filmów o ocenie poniżej 3. Jeśli takowe filmy nie istnieją zostają 
wyszukane u kolejnej najbliższej osoby.
"""
import json
import click
import numpy as np


class MoviesRecommendations:
    def __init__(self, data: dict) -> None:
        self._data = data
        self._closest_users = {"euclidean": {}, "pearson": {}}
        self._sort_movies_by_rating()

    @classmethod
    def from_json_file(cls, path: str) -> "MoviesRecommendations":
        """ class method to run recommendation for data in JSON file """
        with open(path) as _file:
            data = json.load(_file)
            return cls(data)

    @staticmethod
    def sort_dict_for_user(user: str, _dict: dict) -> dict:
        """helper static method to sort dicts related to given user

        Args:
            user (str): name of user
            _dict (dict): dict to be sorted (descending order)

        Returns:
            dict: sorted dict (keys are sorted by values)
        """
        _dict[user] = {
            key: value
            for key, value in sorted(
                _dict[user].items(), key=lambda i: i[1], reverse=True
            )
        }

    def validate_users(self, *users):
        """helper method to validate given users

        Raises:
            ValueError: when given user is not in data
        """
        for user in users:
            if not user in self._data.keys():
                raise ValueError(f'Cannot find "{user}" in dataset')

    def _sort_movies_by_rating(self):
        """helper method to sort all the movies by the rating (descending order) """
        for user in self._data.keys():
            self.sort_dict_for_user(user, self._data)

    def _pearson_score(self, user1: str, user2: str) -> float:
        """method to compute Pearson score between two users

        Args:
            user1 (str): name of the user
            user2 (str): name of the user

        Returns:
            float: score (the higher, the better)
        """
        self.validate_users(user1, user2)

        common_movies = {}

        # finding common movies
        for item in self._data[user1]:
            if item in self._data[user2]:
                common_movies[item] = 1

        num_ratings = len(common_movies)

        # when there are not any common movies, return 0
        if num_ratings == 0:
            return 0

        # counting sums of common movies ratings
        user1_sum = np.sum([self._data[user1][item] for item in common_movies])
        user2_sum = np.sum([self._data[user2][item] for item in common_movies])

        # counting squared sums of common movies ratings
        user1_squared_sum = np.sum(
            [np.square(self._data[user1][item]) for item in common_movies]
        )
        user2_squared_sum = np.sum(
            [np.square(self._data[user2][item]) for item in common_movies]
        )

        sum_of_products = np.sum(
            [
                self._data[user1][item] * self._data[user2][item]
                for item in common_movies
            ]
        )

        # getting Pearson's coefficients
        Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
        Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
        Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

        if Sxx * Syy == 0:
            return 0

        return Sxy / np.sqrt(Sxx * Syy)

    def _euclidean_score(self, user1, user2) -> float:
        """method to compute Euclidean score between two users

        Args:
            user1 (str): name of the user
            user2 (str): name of the user

        Returns:
            float: score (the higher, the better)
        """
        self.validate_users(user1, user2)

        common_movies = {}

        # finding common movies
        for item in self._data[user1]:
            if item in self._data[user2]:
                common_movies[item] = 1

        # when there are not any common movies, return 0
        if len(common_movies) == 0:
            return 0

        # squared difference between points (movies)
        squared_diff = []

        for item in self._data[user1]:
            if item in self._data[user2]:
                squared_diff.append(
                    np.square(self._data[user1][item] - self._data[user2][item])
                )

        return 1 / (1 + np.sqrt(np.sum(squared_diff)))

    def find_closest_users(self, name: str, method="euclidean") -> dict:
        """method to find the closest users for given user (based on score)

        Args:
            name (str): name of user to find the closest users based on
            method (str): method of scoring ("euclidean", "pearson"). Defaults to "euclidean".

        Returns:
            dict: dict with the closest users for given user
        """
        score_function = (
            self._euclidean_score if method == "euclidean" else self._pearson_score
        )

        if not name in self._closest_users[method]:
            self._closest_users[method][name] = {}

            for user in self._data.keys():
                self._closest_users[method][name][user] = score_function(name, user)

            self.sort_dict_for_user(name, self._closest_users[method])

        return self._closest_users[method][name]

    def find_unique_movies(self, for_user, from_user) -> dict:
        """method to find unique movies between users

        Args:
            for_user (str): name of user to find unique movies for
            from_user (str): name of user to find unique movies from

        Returns:
            dict: dict with unique movies with ratings (sorted by ratings, in descending order)
        """
        unique_movies_names = list(
            set(self._data[from_user].keys()) - set(self._data[for_user].keys())
        )

        unique_movies = {key: self._data[from_user][key] for key in unique_movies_names}

        return {
            key: value
            for key, value in sorted(
                unique_movies.items(), key=lambda i: i[1], reverse=True
            )
        }

    def find_recommendations(
        self,
        user: str,
        method: str,
        points_for_best: int,
        points_for_worst: int,
        amount: int,
    ) -> dict:
        """method to find movie recommendations for given user

        Args:
            user (str): name of user to find movie recommendations for
            method (str): method of scoring ("euclidean" or "pearson")
            points_for_best (int): minimum rating for "best" movie
            points_for_worst (int): maximum rating for "worst" movie
            amount (int): amount of movies (maximum), per category ("best", "worst")

        Returns:
            dict: dict with two lists ("best" and "worst")
        """
        best_movies = []
        worst_movies = []
        users = self.find_closest_users(user, method)

        # looking for the "best" and "worst" movies for given user
        for _user in users:
            unique_movies = self.find_unique_movies(user, _user)
            for movie in unique_movies:
                if (
                    self._data[_user][movie] >= points_for_best
                    and len(best_movies) < amount
                ):
                    best_movies.append(movie)

                if (
                    self._data[_user][movie] <= points_for_worst
                    and len(worst_movies) < amount
                ):
                    worst_movies.append(movie)

                if len(best_movies) == amount and len(worst_movies) == amount:
                    return {"best": list(best_movies), "worst": list(worst_movies)}

        return {"best": list(best_movies), "worst": list(worst_movies)}


@click.command()
@click.option("--path", prompt="Give path of JSON file")
@click.option(
    "--method",
    prompt="Method",
    type=click.Choice(["euclidean", "pearson"]),
    default="euclidean",
)
@click.option("--user", prompt="User")
@click.option(
    "--points_for_worst",
    prompt="Points for worst movie (1, 10)",
    type=click.IntRange(1, 10),
    default=3,
)
@click.option(
    "--points_for_best",
    prompt="Points for best movie (1, 10)",
    type=click.IntRange(1, 10),
    default=8,
)
@click.option("--amount", prompt="Amount (1, 6)", type=click.IntRange(1, 6), default=6)
def run(path, method, user, points_for_worst, points_for_best, amount):
    """ method to run interactive mode provided by click"""
    r = MoviesRecommendations.from_json_file(path)
    r.validate_users(user)

    results = r.find_recommendations(
        user, method, points_for_best, points_for_worst, amount
    )

    print(f'Movies recommended for {user}: {", ".join(results["best"])}')
    print(f'Movies not recommended for {user}: {", ".join(results["worst"])}')


if __name__ == "__main__":
    run()
