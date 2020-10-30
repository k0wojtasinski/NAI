# https://www.codingame.com/ide/puzzle/the-descent
# Author: Kacper Wojtasinski

import sys
import math

# Auto-generated code below aims at helping you parse
# the standard input according to the problem statement.

n = int(input())  # the number of temperatures to analyse
min_value = None

for i in input().split():
    # t: a temperature expressed as an integer ranging from -273 to 5526
    t = int(i)

    if not min_value:
        min_value = t

    if t == abs(min_value):
        min_value = t

    if abs(t) < abs(min_value):
        min_value = t

if not min_value:
    min_value = 0

# Write an answer using print
# To debug: print("Debug messages...", file=sys.stderr, flush=True)

print(min_value)