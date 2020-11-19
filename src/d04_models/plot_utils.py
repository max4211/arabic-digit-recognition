# importing dependencies
import random
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib import colorbar
import matplotlib.pyplot as plt # to view graphs

def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)

def all_colors():
    return ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

def random_color():
    pallette = all_colors()
    return cc(random.choice(pallette))

def all_lines():
    return ["-", "--", "-.", "."]

def all_markers():
    return [".", ",", "o", "v", "^", "1", "8", "*", "H", "d"]

def random_marker():
    return random.choice(all_markers())

def random_rgb():
    return (random.random(), random.random(), random.random())

def random_line_style():
    color = random_rgb()
    line = random.choice(all_lines())
    line_style = f"{color}{line}"
    return line_style