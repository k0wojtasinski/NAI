# TicTacToe game
# Author: Kacper Wojtasinski and Wojciech Kudłacik
# Inspiration: https://zulko.github.io/easyAI/examples/games.html#tic-tac-toe (heavily inspired by, made some refactoring, extra features and cli integration)

import click

from easyAI import TwoPlayersGame, Negamax
from easyAI.Player import Human_Player, AI_Player


class TicTacToe(TwoPlayersGame):
    """class to implement TicTacToe with some extra features.
    it is intended to be played by 2 users (either AI or Human)

    AVAILABLE_CHARS is a list of all possible chars to play with ('.' means empty)

    The board positions are numbered as follows:
    1 2 3
    4 5 6
    7 8 9
    """

    AVAILABLE_CHARS = [".", "O", "X"]

    def __init__(self):
        self.players = []
        self.board = [0 for _ in range(9)]
        self.nplayer = 1
        self.values_for_negamax = []

    def print_welcome(self):
        """ prints welcome message with players list and their options """
        print("\n\nYou are playing Tic-Tac-Toe")

        for idx, player in enumerate(self.players):
            print(
                f"Player {idx + 1}: {player.__class__.__name__}, value of Negamax: {self.values_for_negamax[idx]}, char: {self.AVAILABLE_CHARS[idx + 1]}"
            )

    def possible_moves(self) -> list:
        """ returns list of possible moves. When empty game should stop """
        return [idx + 1 for idx, value in enumerate(self.board) if value == 0]

    def make_move(self, move: int):
        """makes move, putting X/O in given place of the board

        Args:
            move (int): position where you want to make a move
        """
        self.board[move - 1] = self.nplayer

    def unmake_move(self, move: int):
        """undos move - used for AI optimization

        Args:
            move (int): position where you want to make a move
        """
        self.board[move - 1] = 0

    def lose(self) -> bool:
        """ returns True if any player lost """
        return any(
            [
                all([(self.board[c - 1] == self.nopponent) for c in line])
                for line in [
                    [1, 2, 3],  # horiz.
                    [4, 5, 6],
                    [7, 8, 9],
                    [1, 4, 7],  # vertical
                    [2, 5, 8],
                    [3, 6, 9],
                    [1, 5, 9],  # diagonal
                    [3, 5, 7],
                ]
            ]
        )

    def is_over(self) -> bool:
        """ returns True if game should be stopped (lack of possible moves or someone lost) """
        return not self.possible_moves() or self.lose()

    def show(self):
        """ prints board of game with current moves """
        print(
            "\n"
            + "\n".join(
                [
                    " ".join(
                        [self.AVAILABLE_CHARS[self.board[3 * j + i]] for i in range(3)]
                    )
                    for j in range(3)
                ]
            )
        )

    def scoring(self) -> int:
        """returns -100 if current player lose - used as a penalty for AI.
        returns 0 if current player does not lose
        """
        return -100 if self.lose() else 0

    def add_human_player(self):
        """adds HumanPlayer, helper function for cli integration
        it always set value_for_negamax to 1 as it does not matter
        """
        self.players.append(Human_Player())
        self.values_for_negamax.append(1)

    def add_ai_player(self, value: int):
        """adds AI player with given value for Negamax algorithm
            helper function for cli integration
        Args:
            value (int): value to be set for Negamax algorithm
        """
        self.players.append(AI_Player(Negamax(value)))
        self.values_for_negamax.append(value)

    def print_result(self):
        """ it prints results, either info about winner or just 'Draw' """
        if not self.possible_moves():
            print("Draw")
        else:
            print(
                f"Player {self.nopponent} [value of Negamax: {self.values_for_negamax[self.nopponent - 1]}, char: {self.AVAILABLE_CHARS[self.nopponent]}] won"
            )

    def play(self):
        """ overriden play function to print some info before the game and after it """
        self.print_welcome()

        super().play()

        self.print_result()


@click.command()
@click.option(
    "--first_player_type",
    required=True,
    prompt=True,
    type=click.Choice(["Human", "AI"]),
)
@click.option(
    "--second_player_type",
    required=True,
    prompt=True,
    type=click.Choice(["Human", "AI"]),
)
@click.option(
    "--first_player_value",
    required=True,
    prompt="Value for Negamax algorithm, give 1 for human players",
    type=int,
)
@click.option(
    "--second_player_value",
    required=True,
    prompt="Value for Negamax algorithm, give 1 for human players",
    type=int,
)
def start_game(
    first_player_type, second_player_type, first_player_value, second_player_value
):
    """ it starts game with defined params, uses click """
    game = TicTacToe()

    if first_player_type == "AI":
        game.add_ai_player(first_player_value)
    else:
        game.add_human_player()

    if second_player_type == "AI":
        game.add_ai_player(second_player_value)
    else:
        game.add_human_player()

    game.play()


if __name__ == "__main__":
    start_game()
