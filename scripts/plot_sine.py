import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

def animate(i):
    data = pd.read_csv('actuator_positions.csv')
    plt.cla()  # Clear the current axes.
    plt.plot(data)

# Create an animation
ani = FuncAnimation(plt.gcf(), animate, interval=100)

plt.tight_layout()
plt.show()