import random
import numpy
import networkx as nx
import matplotlib.pyplot as plt
import pygame


class Agent:
    def __init__(
        self: object,
        shape: numpy.array,
        goals: dict = {"food": 0.5, "enemy": -0.5, "mate": 0.4},
        gender=random.choice(["female", "male"]),
    ) -> None:
        self.shape = shape
        self.goals = goals
        self.gender = gender
        self.attractiveness = random.uniform(*range(9)) / 10
        self.food = 2
        self.speed = random.uniform(1, 2.5)
        self.age = 0
        self.reproductive_urge = random.uniform(*range(9)) / 10
        self.mutationrate = 0.1

    def calculate_distance(self, position1, position2):
        # Calculate Euclidean distance between two positions
        return numpy.linalg.norm(position1 - position2)

    def mate(self, partner):
        if partner.reproductive_urge > random.randint(1, 3):
            newspeed = (self.speed + partner.speed) / 2 + random.uniform(
                int(f"-{self.mutationrate}"), self.mutationrate
            )
            newattractiveness = (
                self.attractiveness + partner.attractiveness
            ) / 2 + random.uniform(int(f"-{self.mutationrate}"), self.mutationrate)
            newmutationrate = (
                self.mutationrate + partner.mutationrate
            ) / 2 + random.uniform(int(f"-{self.mutationrate}"), self.mutationrate)
            newgoals = {
                "food": (self.goals["food"] + partner.goals["food"]) / 2
                + random.uniform(int(f"-{self.mutationrate}"), self.mutationrate),
                "enemy": (self.goals["enemy"] + partner.goals["enemy"]) / 2
                + random.uniform(int(f"-{self.mutationrate}"), self.mutationrate),
                "mate": (self.goals["mate"] + partner.goals["mate"]) / 2
                + random.uniform(int(f"-{self.mutationrate}"), self.mutationrate),
            }
            newshape = numpy.array()

            child = Agent(newshape, newgoals)
            child.attractiveness = newattractiveness
            child.mutationrate = newmutationrate
            child.parent = self
            partner.children.append(child)
            self.children.append(child)
            creatures.append(child)
            return child

    def choose_action(self, list_of_food, list_of_enemies):
        actions = ["right", "left", "up", "down"]  # Define possible actions

        action_scores = {}
        for action in actions:
            # Assume the agent's position is updated based on the action temporarily
            new_position = self.calculate_new_position(action)

            # Calculate scores based on distances to food and enemies
            food_scores = [
                1 / self.calculate_distance(new_position, food) for food in list_of_food
            ]
            enemy_scores = [
                -1 / self.calculate_distance(new_position, enemy)
                for enemy in list_of_enemies
            ]

            # Total score for this action based on distances
            total_score = sum(food_scores) + sum(enemy_scores)

            action_scores[action] = total_score

        best_action = max(action_scores, key=action_scores.get)
        self.take_action(best_action)
        result = self.find_result()

        # TODO: Implement epsilon decay over time


# Functions for environment setup
def initialize_environment():
    # Initialize environment elements (agents, food, enemies, etc.)
    pass


def create_agents(num_agents):
    # Create a specified number of agents with random attributes
    pass


def create_food_sources(num_food_sources):
    # Create food sources at random positions
    pass


# Pygame initialization and visualization functions
def render_environment():
    # Render agents, food sources, enemies, etc. on the Pygame window
    pass


def update_agents_actions():
    # Update agents' actions and interactions within the environment
    pass


# Statistics tracking and visualization
def track_statistics():
    # Track and record data such as agent population, food availability, reproductive success, etc.
    pass


def plot_graphs(G):
    # Plotting the evolutionary graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=3000,
        node_color="skyblue",
        font_weight="bold",
    )
    plt.title("Evolutionary Graph")
    plt.show()


# Main simulation function
def main_simulation():
    global creatures
    # Initialize environment elements
    initialize_environment()

    # Pygame initialization
    pygame.init()
    clock = pygame.time.Clock()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Natural Selection Simulation")

    # Run the main simulation loop
    running = True
    while running:
        screen.fill((255, 255, 255))  # Clear the screen

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update agents' actions and interactions
        update_agents_actions()

        # Render environment elements
        render_environment()

        pygame.display.flip()
        clock.tick(60)

    # After the simulation ends, track statistics and create the evolutionary graph
    track_statistics()
    evolutionary_graph = create_evolutionary_graph(
        creatures
    )  # Assuming 'all_agents' holds all agents
    plot_graphs(evolutionary_graph)

    pygame.quit()


# Helper function to create the evolutionary graph
def create_evolutionary_graph(agents):
    G = nx.DiGraph()

    for agent in agents:
        parent = agent.parent
        if parent:
            G.add_edge(parent, agent)

    return G


# Run the main simulation
if __name__ == "__main__":
    creatures = []
    main_simulation()
