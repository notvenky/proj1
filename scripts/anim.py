import pygame
import math

# Initialize Pygame
pygame.init()

# Window size
window_width = 1200  # Adjust as needed
window_height = 800  # Adjust as needed
win = pygame.display.set_mode((window_width, window_height))

# Colors and settings
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
FPS = 60

# Snake Link Class
class SnakeLink:
    def __init__(self, x, y, length, angle):
        self.x = x
        self.y = y
        self.length = length
        self.angle = angle

    def end_point(self):
        return (self.x + self.length * math.cos(self.angle),
                self.y + self.length * math.sin(self.angle))

    def draw(self, win):
        end = self.end_point()
        pygame.draw.line(win, WHITE, (self.x, self.y), end, 2)
        self.x, self.y = end

# Robot Morphology Class
class RobotMorphology:
    def __init__(self, num_links, x, y, link_length=50):
        self.links = [SnakeLink(x, y, link_length, 0) for _ in range(num_links)]

    def update(self):
        for i, link in enumerate(self.links):
            link.angle = math.sin(pygame.time.get_ticks() / 500 + i) * math.pi / 4

    def draw(self, win):
        for link in self.links:
            link.draw(win)

# Initialize morphologies
snake_6 = RobotMorphology(6, 100, 100)
snake_8 = RobotMorphology(8, 400, 100)
snake_12 = RobotMorphology(12, 700, 100)
# Add other morphologies as needed

# Main loop
run = True
clock = pygame.time.Clock()
while run:
    clock.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    win.fill(BLACK)

    # Update and draw each morphology
    snake_6.update()
    snake_6.draw(win)
    snake_8.update()
    snake_8.draw(win)
    snake_12.update()
    snake_12.draw(win)
    # Update and draw additional morphologies

    pygame.display.update()

pygame.quit()
