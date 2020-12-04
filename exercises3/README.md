# Fuzzy Logic
## Washing Machine Time of laundry

### Goal
Goal of this prototype is to prepare simple system to predict how long can laundry take (in hours - 1 to 5, short, average, long), based on temperature (20-60), weight (grams - 1000-5000) and dirtiness (points - 1-100) - it defines how dirty the laundry is.

It is possible to import these parameters (called arguments) from JSON file, and export the as well

It is PoC with proposal on how to define custom API for fuzzy logic simulations, rather than solution with good performance (would have to tweak rules and definition of consequent).

### How to run it
First, you have to install all the dependecies (``pip install -r requirements ``)
Then you can use CLI mode, which supports two modes:

a) manual (`python washing_machine.py run`)

it takes input to gather all the arguments and run the simulation

b) from file (`python washing_machine.py from-file`)

it takes data from the JSON file to get all the arguments and run the simulation

### Examples

a) simulation with long time of laundry
![long](https://i.imgur.com/8DUXWXh.png)

b) simulation with medium time of laundry
![medium](https://i.imgur.com/lRTWQLg.png)

c) simulation with short time of laundry
![short](https://i.imgur.com/gjbmIqo.png)

