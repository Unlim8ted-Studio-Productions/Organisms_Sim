from itertools import count
import random
from matplotlib import pyplot as plt
import pygame
import numpy as np
import psutil
import torch


# Define constants
WIDTH, HEIGHT = 400, 400
FPS = 30

# Additional constants
BABY_GROWTH_TIME = 120  # Frames for a baby to grow into an adult
BABY_RADIUS = 30  # Radius within which a baby can be eaten
MAX_CREATURES = psutil.cpu_count(logical=False) * 25
TIME_POINTS = 500  # Adjust the number of time points as needed

# Define colors
WHITE = (255, 255, 255)
BLACK = (25, 45, 25)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Define the Q-learning parameters
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1
epsilon_decay = 0.995  # Decay factor for epsilon

# Define the Q-tables for black and red squares
q_table_black = np.zeros((WIDTH, HEIGHT, 5))  # 5 actions (up, down, left, right, baby)
q_table_red = np.zeros((WIDTH, HEIGHT, 5))  # 5 actions (up, down, left, right, baby)

# Initialize Pygame
pygame.init()

# Set up the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("life sim")

# Clock to control the frame rate
clock = pygame.time.Clock()
time = count()
time_points = range(TIME_POINTS)


def control_creature(creature):
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        creature.state = (max(0, creature.state[0] - creature.speed), creature.state[1])
    elif keys[pygame.K_RIGHT]:
        creature.state = (
            min(WIDTH - 20, creature.state[0] + creature.speed),
            creature.state[1],
        )
    elif keys[pygame.K_UP]:
        creature.state = (creature.state[0], max(0, creature.state[1] - creature.speed))
    elif keys[pygame.K_DOWN]:
        creature.state = (
            creature.state[0],
            min(HEIGHT - 20, creature.state[1] + creature.speed),
        )


# Define the agent
class QLearningAgent:
    def __init__(self, q_table):
        self.state = (0, 0)
        self.speed = 5  # Initial speed of the agent
        self.q_table = q_table
        self.reward = 0
        self.prev_distance = None  # Store the previous distance
        self.food = 0
        self.reward = 0
        self.enemy = False
        self.attractiveness = random.randint(0, 5)

    def choose_action(self):
        if np.random.uniform(0, 1) < epsilon:
            if self.food >= 3:
                return np.random.choice(5)  # Exploration
            else:
                return np.random.choice(4)  # Exploration
        else:
            return np.argmax(self.q_table[self.state[0], self.state[1]])

    def take_action(self, action):
        x, y = self.state

        if self.food <= -100:
            self.die()

        if action == 0:  # Up
            y = max(0, y - self.speed)
            self.food - 0.001
        elif action == 1:  # Down
            y = min(HEIGHT - 20, y + self.speed)
            self.food - 0.001
        elif action == 2:  # Left
            x = max(0, x - self.speed)
            self.food - 0.001
        elif action == 3:  # Right
            x = min(WIDTH - 20, x + self.speed)
            self.food - 0.001
        elif action == 4:
            global blue_squares, baby_black_squares, baby_red_squares, adult_black_squares, adult_red_squares
            mates = [
                mate
                for mate in []
                + baby_black_squares
                + baby_red_squares
                + adult_black_squares
                + adult_red_squares
                if pygame.Rect(self.state[0], self.state[1], 20, 20).colliderect(
                    pygame.Rect(mate.state[0], mate.state[1], 20, 20)
                )
            ]
            if mates:
                self.mate = [
                    mate
                    for mate in mates
                    if mate.attractiveness
                    == sorted([mate.attractiveness for mate in mates], reverse=True)[0]
                ][0]
                self.food -= random.randint(3, 6)
                self.create_baby()

        return x, y

    def calculate_reward(self, good_positions, bad_positions, radius=50):
        self.reward = 0
        for good_pos in good_positions:
            distance = np.linalg.norm(np.array(self.state) - np.array(good_pos))
            if self.prev_distance is not None:
                if distance < self.prev_distance:
                    self.reward += 0.1  # A small positive reward for getting closer
                elif distance > self.prev_distance:
                    self.reward -= 0.1  # A small negative reward for getting further

        # Penalize for being close to bad positions
        for bad_pos in bad_positions:
            distance_to_bad = np.linalg.norm(np.array(self.state) - np.array(bad_pos))
            if distance_to_bad < radius:
                self.reward -= (
                    0.5  # A larger negative reward for being close to bad positions
                )

        self.prev_distance = np.linalg.norm(
            np.array(self.state) - np.array(self.take_action(self.choose_action()))
        )  # Update previous distance
        return self.reward

    def update_q_table(self, action, next_state, reward):
        old_value = self.q_table[self.state[0], self.state[1], action]
        next_max = np.max(self.q_table[next_state[0], next_state[1]])
        new_value = (1 - learning_rate) * old_value + learning_rate * (
            reward + discount_factor * next_max
        )
        self.q_table[self.state[0], self.state[1], action] = new_value

        self.state = next_state

    def create_baby(self):
        global baby_black_squares, baby_red_squares, adult_black_squares, adult_red_squares, MAX_CREATURES

        # print(self.q_table)

        def average_q_tables(q_table1, q_table2):
            # Convert the Q-tables to PyTorch tensors and transfer them to the GPU
            if torch.cuda.is_available():
                q_table1_tensor = torch.tensor(q_table1, dtype=torch.float32).cuda()
                q_table2_tensor = torch.tensor(q_table2, dtype=torch.float32).cuda()
            else:
                q_table1_tensor = torch.tensor(q_table1, dtype=torch.float32).cpu()
                q_table2_tensor = torch.tensor(q_table2, dtype=torch.float32).cpu()

            # Compute the average of the Q-tables on the GPU
            average_table_tensor = (
                q_table1_tensor + q_table2_tensor
            ) / 2.0 + random.choice([-0.1, 0.1])

            # Transfer the result back to the CPU and convert it to a Python list
            average_table = average_table_tensor.cpu().numpy()

            return average_table

        q_table = average_q_tables(self.q_table, self.mate.q_table)

        baby = QLearningAgent(self.q_table)
        if self.enemy == self.mate.enemy:
            baby.enemy = self.enemy
        else:
            baby.enemy = random.choice([True, False])
        baby.state = self.state
        baby.speed = (self.speed + self.mate.speed) // 2
        baby.parent = self
        if (
            len(
                baby_black_squares
                + baby_red_squares
                + adult_black_squares
                + adult_red_squares
            )
            <= MAX_CREATURES
        ):
            if baby.enemy:
                baby_red_squares.append(baby)
            else:
                baby_black_squares.append(baby)

    def die(self):
        global blue_squares, baby_black_squares, baby_red_squares, adult_black_squares, adult_red_squares
        blue_squares.append(pygame.Rect(self.state[0], self.state[1], 20, 20))
        try:
            baby_black_squares.remove(self)
        except:
            None
        try:
            baby_red_squares.remove(self)
        except:
            None
        try:
            adult_black_squares.remove(self)
        except:
            None
        try:
            adult_red_squares.remove(self)
        except:
            None


def main(
    adult_black_squares,
    baby_black_squares,
    adult_red_squares,
    baby_red_squares,
    black_life_data,
    red_life_data,
    black_life_expectancy_data,
    red_life_expectancy_data,
    baby_black_life_expectancy_data,
    baby_red_life_expectancy_data,
    baby_black_life_data,
    baby_red_life_data,
    black_speed_data,
    red_speed_data,
    plantfood_data,
    meatfood_data,
    green_squares,
    blue_squares,
    baby_frame_counter,
    controlled_creature,
):
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                # Add these print statements for debugging
                for index, agent_black in enumerate(adult_black_squares):
                    print(
                        f"Black {index} Q-values:",
                        agent_black.q_table[agent_black.state[0], agent_black.state[1]],
                    )
                for index, agent_black in enumerate(baby_black_squares):
                    print(
                        f"BabyBlack {index} Q-values:",
                        agent_black.q_table[agent_black.state[0], agent_black.state[1]],
                    )
                for index, agent_red in enumerate(adult_red_squares):
                    print(
                        f"Red {index} Q-values:",
                        agent_red.q_table[agent_red.state[0], agent_red.state[1]],
                    )
                for index, agent_red in enumerate(baby_red_squares):
                    print(
                        f"BabyRed {index} Q-values:",
                        agent_red.q_table[agent_red.state[0], agent_red.state[1]],
                    )
                    # Check for mouse click to select a creature to control
            if pygame.mouse.get_pressed()[0]:  # Left mouse button clicked
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for creature in (
                    adult_black_squares
                    + adult_red_squares
                    + baby_black_squares
                    + baby_red_squares
                ):
                    if pygame.Rect(
                        creature.state[0], creature.state[1], 20, 20
                    ).colliderect(pygame.Rect(mouse_x, mouse_y, 1, 1)):
                        controlled_creature = creature
                        break

        if controlled_creature is not None:
            # Handle keyboard input to control the selected creature
            control_creature(controlled_creature)

        # Choose an action for the black square
        for black_square in adult_black_squares + baby_black_squares:
            black_square.reward = 0  # Reset reward for each step
            black_action = black_square.choose_action()
            black_next_state = black_square.take_action(black_action)

        # Check if black square collides with red square
        for red_square in adult_red_squares:
            if pygame.Rect(
                black_next_state[0], black_next_state[1], 20, 20
            ).colliderect(
                pygame.Rect(red_square.state[0], red_square.state[1], 20, 20)
            ):
                black_square.reward -= 1
                black_square.food -= 0.1
                red_square.reward += 1
                red_square.food += 1

        # Check if black square collides with red square
        for red_square in baby_red_squares:
            for green_square in green_squares:
                if green_square.collidepoint(
                    (red_square.state[0], red_square.state[1])
                ):
                    try:
                        baby_red_squares.remove(red_square)
                    except:
                        None

            if pygame.Rect(
                black_next_state[0], black_next_state[1], 20, 20
            ).colliderect(
                pygame.Rect(red_square.state[0], red_square.state[1], 20, 20)
            ):
                black_square.reward -= 1
                try:
                    baby_black_squares.remove(black_square)
                except:
                    None
                red_square.reward += 1
                red_square.food += 2

        # Calculate rewards based on proximity to green and bad squares
        black_square.reward += black_square.calculate_reward(
            [square.center for square in green_squares],
            [red_square.state for red_square in adult_red_squares + baby_red_squares],
        )

        # Check if black square touches the green squares
        for green_square in green_squares:
            if pygame.Rect(
                black_next_state[0], black_next_state[1], 20, 20
            ).colliderect(green_square):
                black_square.reward += 1
                black_square.food += 1

        # Update the Q-table for the black square
        black_square.update_q_table(black_action, black_next_state, black_square.reward)

        # Choose an action for the red square
        for red_square in adult_red_squares + baby_red_squares:
            red_square.reward = 0  # Reset reward for each step
            red_action = red_square.choose_action()
            red_next_state = red_square.take_action(red_action)

        # Check if red square collides with black square
        for black_square in adult_black_squares + baby_black_squares:
            if pygame.Rect(red_next_state[0], red_next_state[1], 20, 20).colliderect(
                pygame.Rect(black_square.state[0], black_square.state[1], 20, 20)
            ):
                red_square.reward -= 1
                black_square.reward += 1

        for green_square in green_squares:
            if green_square.collidepoint((red_square.state[0], red_square.state[1])):
                try:
                    adult_red_squares.remove(red_square)
                except:
                    None

        # Calculate rewards based on proximity to green and bad squares
        red_square.reward += red_square.calculate_reward(
            [
                black_square.state
                for black_square in adult_black_squares + baby_black_squares
            ],
            [square.center for square in green_squares],
        )

        # Check if red square touches the green squares
        for green_square in green_squares:
            if pygame.Rect(red_next_state[0], red_next_state[1], 20, 20).colliderect(
                green_square
            ):
                red_square.reward -= 1

        # Update the Q-table for the red square
        red_square.update_q_table(red_action, red_next_state, red_square.reward)

        for creature in (
            adult_black_squares
            + adult_red_squares
            + baby_black_squares
            + baby_red_squares
        ):
            for blue in blue_squares:
                if blue.collidepoint((creature.state[0], creature.state[1])):
                    if creature.enemy:
                        creature.food += 0.3
                    else:
                        creature.food += 9
                    try:
                        blue_squares.remove(blue)
                    except:
                        None

        screen.fill(WHITE)
        # Update and draw adult black squares
        for black_square in adult_black_squares:
            pygame.draw.rect(
                screen, BLACK, (black_square.state[0], black_square.state[1], 20, 20)
            )

        # Update and draw adult red squares
        for red_square in adult_red_squares:
            pygame.draw.rect(
                screen, RED, (red_square.state[0], red_square.state[1], 20, 20)
            )

        # Update and draw adult red squares
        for blue_square in blue_squares:
            pygame.draw.rect(screen, (0, 0, 255), blue_square)

        # Update and draw baby black squares
        for baby_black in baby_black_squares:
            baby_frame_counter += 1
            if baby_frame_counter >= BABY_GROWTH_TIME:
                adult_black_squares.append(baby_black)
                baby_black_squares.remove(baby_black)
            else:
                baby_black.choose_action()
                baby_black_next_state = baby_black.take_action(
                    baby_black.choose_action()
                )
                baby_black.update_q_table(
                    0, baby_black_next_state, 0
                )  # No reward for babies
                pygame.draw.rect(
                    screen,
                    BLACK,
                    (baby_black_next_state[0], baby_black_next_state[1], 5, 5),
                )

        # Update and draw baby red squares
        for baby_red in baby_red_squares:
            baby_frame_counter += 1
            if baby_frame_counter >= BABY_GROWTH_TIME:
                adult_red_squares.append(baby_red)
                baby_red_squares.remove(baby_red)
            else:
                baby_red.choose_action()
                baby_red_next_state = baby_red.take_action(baby_red.choose_action())
                baby_red.update_q_table(
                    0, baby_red_next_state, 0
                )  # No reward for babies
                pygame.draw.rect(
                    screen, RED, (baby_red_next_state[0], baby_red_next_state[1], 5, 5)
                )

        # Draw and update green squares
        for green_square in green_squares:
            pygame.draw.rect(screen, GREEN, green_square)

        # Spawn a green square randomly
        if len(green_squares) <= MAX_CREATURES:
            if np.random.uniform(0, 1) < 0.01:
                new_square = pygame.Rect(
                    np.random.randint(0, WIDTH - 20),
                    np.random.randint(0, HEIGHT - 20),
                    20,
                    20,
                )
                green_squares.append(new_square)
        else:
            index = random.randint(0, len(green_squares))
            green_squares.pop(index)
            # Decay epsilon over time
            epsilon *= epsilon_decay
            epsilon = max(
                0.01, epsilon
            )  # Ensure epsilon doesn't go below a minimum value

        # Calculate and store data for plotting
        black_life_data.append(
            sum([creature.food for creatures in adult_black_squares])
        )
        baby_black_life_data.append(
            sum([creature.food for creatures in baby_black_squares])
        )
        baby_red_life_data.append(
            sum([creature.food for creatures in baby_red_squares])
        )
        red_life_data.append(sum([creature.food for creatures in adult_red_squares]))

        black_life_expectancy_data.append(
            np.mean([creature.food for creature in adult_black_squares])
        )
        red_life_expectancy_data.append(
            np.mean([creature.food for creature in adult_red_squares])
        )

        baby_black_life_expectancy_data.append(
            np.mean([creature.food for creature in baby_black_squares])
        )
        baby_red_life_expectancy_data.append(
            np.mean([creature.food for creature in baby_red_squares])
        )

        black_speed_data.append(
            np.mean(
                [
                    creature.speed
                    for creature in adult_black_squares + baby_black_squares
                ]
            )
        )
        red_speed_data.append(
            np.mean(
                [creature.speed for creature in adult_red_squares + baby_red_squares]
            )
        )
        plantfood_data.append(len(green_squares))
        meatfood_data.append(len(blue_squares))

        # Plotting
        plt.clf()

        # Adjust the subplot configuration to 4 rows, 1 column
        plt.subplot(4, 1, 1)  # Amount of life over time
        plt.plot(
            time_points[: len(black_life_data)],
            black_life_data,
            label="Adult Herbivore",
            color="black",
        )
        plt.plot(
            time_points[: len(red_life_data)],
            red_life_data,
            label="Adult Carnivore",
            color="red",
        )
        plt.plot(
            time_points[: len(black_life_data)],
            baby_red_life_data,
            label="Baby Carnivore",
            color="blue",
        )
        plt.plot(
            time_points[: len(black_life_data)],
            baby_black_life_data,
            label="Baby Herbivore",
            color="green",
        )
        plt.title("Amount of Life Over Time")
        plt.legend()

        plt.subplot(4, 1, 2)  # Life expectancy over time
        plt.plot(
            time_points[: len(black_life_expectancy_data)],
            black_life_expectancy_data,
            label="Adult Herbivore",
            color="black",
        )
        plt.plot(
            time_points[: len(red_life_expectancy_data)],
            red_life_expectancy_data,
            label="Adult Carnivore",
            color="red",
        )
        plt.plot(
            time_points[: len(black_life_data)],
            baby_red_life_expectancy_data,
            label="Baby Carnivore",
            color="blue",
        )
        plt.plot(
            time_points[: len(black_life_data)],
            baby_black_life_expectancy_data,
            label="Baby Herbivore",
            color="green",
        )
        plt.title("Life Expectancy Over Time")
        plt.legend()

        plt.subplot(4, 1, 3)  # Average speed over time
        plt.plot(
            time_points[: len(black_speed_data)],
            black_speed_data,
            label="Herbivore average speed",
            color="black",
        )
        plt.plot(
            time_points[: len(red_speed_data)],
            red_speed_data,
            label="Carnivore average speed",
            color="red",
        )
        plt.title("Average Speed Over Time")
        plt.legend()

        plt.subplot(4, 1, 4)  # Average food squares over time
        plt.plot(
            time_points[: len(plantfood_data)],
            plantfood_data,
            label="Plant food",
            color="green",
        )
        plt.plot(
            time_points[: len(meatfood_data)],
            meatfood_data,
            label="Non-plant food",
            color="blue",
        )
        plt.title("Average Food Squares Over Time")
        plt.legend()

        # Show the plot
        plt.tight_layout()
        plt.pause(0.01)
        # Update the display
        pygame.display.flip()

        # Control the frame rate
        # clock.tick(FPS)


if __name__ == "__main__":
    # Lists for cubes
    e = QLearningAgent(q_table_red)
    e.enemy = True
    e.speed = 10
    adult_black_squares = [QLearningAgent(q_table_black), QLearningAgent(q_table_black)]
    baby_black_squares = []
    adult_red_squares = [e]
    baby_red_squares = []

    # Score variables
    black_reward = 0
    red_reward = 0

    # Lists to store data for plotting
    black_life_data = []
    red_life_data = []
    black_life_expectancy_data = []
    red_life_expectancy_data = []
    black_speed_data = []
    red_speed_data = []
    plantfood_data = []
    meatfood_data = []

    # Variables for green squares
    green_squares = []
    blue_squares = []

    # Main game loop
    baby_frame_counter = 0
    controlled_creature = None
    main(
        adult_black_squares,
        baby_black_squares,
        adult_red_squares,
        baby_red_squares,
        black_life_data,
        red_life_data,
        black_life_expectancy_data,
        red_life_expectancy_data,
        [],
        [],
        [],
        [],
        black_speed_data,
        red_speed_data,
        plantfood_data,
        meatfood_data,
        green_squares,
        blue_squares,
        baby_frame_counter,
        controlled_creature,
    )
# Quit Pygame
pygame.quit()
