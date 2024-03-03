import pygame
import numpy as np
import random

# Pygame setup
pygame.init()
window_size = 600  # Window size in pixels
grid_size = 10  # Grid size in cells
cell_size = window_size // grid_size  # Size of a grid cell in pixels
window = pygame.display.set_mode((window_size, window_size))
clock = pygame.time.Clock()

# Colors
background_color = (0, 0, 0)
organism_color = (0, 255, 0)


class Grid:
    def __init__(self, size):
        self.size = size
        self.cells = np.full((size, size), None)

    def place_organism(self, organism, x, y):
        if self.cells[x, y] is None:
            self.cells[x, y] = organism
            organism.x, organism.y = x, y

    def move_organism(self, organism, new_x, new_y):
        if (
            0 <= new_x < self.size
            and 0 <= new_y < self.size
            and self.cells[new_x, new_y] is None
        ):
            self.cells[organism.x, organism.y] = None
            self.cells[new_x, new_y] = organism
            organism.x, organism.y = new_x, new_y


class Organism:
    def __init__(self, grid, energy):
        self.grid = grid
        self.energy = energy
        self.x, self.y = None, None

    def move(self):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dir_x, dir_y = random.choice(directions)
        new_x, new_y = self.x + dir_x, self.y + dir_y
        self.grid.move_organism(self, new_x, new_y)

    def act(self):
        self.move()  # Organisms move randomly.
        self.energy -= 1  # Moving costs energy.
        if self.energy <= 0:
            self.die()

    def reproduce(self):
        if self.energy > 20:  # Threshold for reproduction.
            child_energy = self.energy // 2
            self.energy -= child_energy
            child = Organism(self.grid, child_energy)
            self.grid.place_organism(child, self.x, self.y)
            return child

    def die(self):
        self.grid.cells[self.x, self.y] = None


def draw_grid():
    for x in range(0, window_size, cell_size):
        for y in range(0, window_size, cell_size):
            rect = pygame.Rect(x, y, cell_size, cell_size)
            pygame.draw.rect(window, background_color, rect, 1)


def draw_organisms(organisms):
    for organism in organisms:
        if organism.energy > 0:
            x = organism.x * cell_size
            y = organism.y * cell_size
            pygame.draw.rect(window, organism_color, (x, y, cell_size, cell_size))


def run_simulation(grid_size, initial_organisms, steps):
    grid = Grid(grid_size)
    organisms = [Organism(grid, 10) for _ in range(initial_organisms)]
    for organism in organisms:
        x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        grid.place_organism(organism, x, y)

    running = True
    step_count = 0
    while running and step_count < steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        window.fill(background_color)
        draw_grid()
        for organism in organisms[:]:
            organism.act()
            if organism.energy > 20:
                child = organism.reproduce()
                if child:
                    organisms.append(child)
        draw_organisms(organisms)
        pygame.display.flip()
        clock.tick(10)  # Control simulation speed
        step_count += 1

    pygame.quit()
    return len([o for o in organisms if o.energy > 0])


# Run the Pygame simulation
if __name__ == "__main__":
    remaining_organisms = run_simulation(grid_size, 5, 100000000000)
    print(f"Simulation ended with {remaining_organisms} organisms.")
