import random
import pygame
import numpy as np
import psutil


# Define constants
WIDTH, HEIGHT = 400, 400
FPS = 30

# Additional constants
BABY_GROWTH_TIME = 120  # Frames for a baby to grow into an adult
BABY_RADIUS = 30  # Radius within which a baby can be eaten
MAX_CREATURES = psutil.cpu_count(logical=False) * 25

# Define colors
WHITE = (255, 255, 255)
BLACK = (25, 45, 25)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Define the Q-learning parameters
learning_rate = 0.8
discount_factor = 0.95
epsilon = 0.1

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

        if self.enemy:
            self.speed = 4

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
        baby = QLearningAgent(self.q_table)
        baby.enemy = self.enemy
        baby.state = self.state
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


# Lists for cubes
e = QLearningAgent(q_table_red)
e.enemy = True
adult_black_squares = [QLearningAgent(q_table_black), QLearningAgent(q_table_black)]
baby_black_squares = [e]
adult_red_squares = []
baby_red_squares = []

# Score variables
black_reward = 0
red_reward = 0

# Variables for green squares
green_squares = []
blue_squares = []

# Main game loop
running = True
baby_frame_counter = 0


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
        adult_black_squares + adult_red_squares + baby_black_squares + baby_red_squares
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
            baby_black_next_state = baby_black.take_action(baby_black.choose_action())
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
            baby_red.update_q_table(0, baby_red_next_state, 0)  # No reward for babies
            pygame.draw.rect(
                screen, RED, (baby_red_next_state[0], baby_red_next_state[1], 5, 5)
            )

    # Draw and update green squares
    for green_square in green_squares:
        pygame.draw.rect(screen, GREEN, green_square)

    # Spawn a green square randomly
    if np.random.uniform(0, 1) < 0.01:
        new_square = pygame.Rect(
            np.random.randint(0, WIDTH - 20),
            np.random.randint(0, HEIGHT - 20),
            20,
            20,
        )
        green_squares.append(new_square)

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
