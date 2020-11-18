p### Assignment 2 - EasyAI
The goal of this assignment was to create a two player game, in this case **Tic Tac Toe**, and implement **EasyAI** library. Additionally, the solution was to be documented using **Docstring**.

#### How to run
In order to run the solution you need to have two packages installed: [EasyAI](https://zulko.github.io/easyAI/) and [Click](https://click.palletsprojects.com/en/7.x/).

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install both of them.

##### Installing EasyAI
```bash
pip install easyAI
```
##### Installing Click
```bash
pip install click
```

#### About the game
##### Setup
Before the actual game of Tic Tac Toe begins, you are asked to input a few parameters. First you need to enter the players, either **Human** or **AI**. Then you're asked to input the depth of the **Negamax** algorithm for the AI Players (for Human players please enter 1).

##### Rules
The game is played on a 3 x 3 grid represented by these values:
```
    7 8 9
    4 5 6
    1 2 3
```

First player is assigned **0**, second - **X** symbols. 
Each player is asked to input value from 1 to 9 and either **X** or **O** will be placed on the grid depening on which player's turn it was. 
For example if the first player inputs **5**, the result will be:
```
    . . .
    . 0 .
    . . .
```
First player to get 3 of the same symbols in a straight or diagonal line, wins the game.

#### Results
Below you can see the results of the game being played by different actors, be it Human vs Human, Human vs AI or AI vs AI. In case of AI vs AI examples, the results appear almost instantaneously.

Algorithm used for the AI is [Negamax](https://en.wikipedia.org/wiki/Negamax)
 
##### Human vs Human
![Imgur](https://i.imgur.com/jNm9zYj.gif)

[Link to video](https://streamable.com/x1237g)

##### Human vs AI
![Imgur](https://i.imgur.com/6oSwkaE.gif)

[Link to video](https://streamable.com/3c2avj)

##### AI vs AI - same depth
![Imgur](https://i.imgur.com/PLsIFoR.gif)

[Link to video](https://streamable.com/iaffq7)

##### AI vs AI - first depth bigger
![Imgur](https://i.imgur.com/LGzsFEH.gif)

[Link to video](https://streamable.com/ir0r70)

##### AI vs AI - second depth bigger
![Imgur](https://i.imgur.com/wNmKsSd.gif)

[Link to video](https://streamable.com/0893hd)

#### Useful Links
##### Tic Tac Toe Rules
[ENG](https://en.wikipedia.org/wiki/Tic-tac-toe)
[PL](https://pl.wikipedia.org/wiki/K%C3%B3%C5%82ko_i_krzy%C5%BCyk)

##### EasyAI
[EasyAI documentation](https://zulko.github.io/easyAI/)

##### Docstring
[Python Docstring Conventions](https://www.python.org/dev/peps/pep-0257/)
[Python Docstring Tutorial](https://www.datacamp.com/community/tutorials/docstrings-python)

##### Click
[Click](https://click.palletsprojects.com/en/7.x/p)
