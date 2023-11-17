import random
import pygame
import numpy as np
import psutil

# Initialize Pygame
pygame.init()

infoObject: object = pygame.display.Info()

# Define constants
WIDTH, HEIGHT = infoObject.current_w, infoObject.current_h
FPS = 30

# Additional constants
BABY_GROWTH_TIME = 120  # Frames for a baby to grow into an adult
BABY_RADIUS = 30  # Radius within which a baby can be eaten
MAX_CREATURES = psutil.cpu_count(logical=False) * 25
WALL_COST = 10  # Food cost to create a wall
WALL_HEALTH = 3  # Initial health of the wall

# Define colors
WHITE = (255, 255, 255)
BLACK = (25, 45, 25)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)  # Color for walls

# Define the Q-learning parameters
learning_rate = 0.5
discount_factor = 0.8
epsilon = 0.2

# Define the Q-tables for black and red squares
q_table_black = np.zeros(
    (WIDTH, HEIGHT, 7)
)  # 6 actions (up, down, left, right, nothing, baby, create wall)
q_table_red = np.zeros(
    (WIDTH, HEIGHT, 7)
)  # 6 actions (up, down, left, right, nothing, baby, create wall)


# Set up the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("life sim")

# Clock to control the frame rate
clock = pygame.time.Clock()


class Wall:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.health = WALL_HEALTH


# Define the agent
class QLearningAgent:
    def __init__(self, q_table, custom_rewards=None):
        self.state = (0, 0)
        self.speed = 5  # Initial speed of the agent
        self.q_table = q_table
        self.reward = 0
        self.prev_distance = None  # Store the previous distance
        self.food = 0
        self.enemy = False
        self.custom_rewards = custom_rewards or {
            "getting_closer": 0.05,
            "getting_further": -0.05,
            "near_bad_positions": -0.3,
            "eating_green": 1,
            "additional_reward": 0.1,  # New reward for exploration
        }
        self.wall_health = 0
        self.wall_cooldown = 0

    def create_wall(self):
        global walls
        if self.food >= WALL_COST and self.wall_cooldown == 0:
            self.food -= WALL_COST
            walls.append(Wall(self.state[0], self.state[1]))  # Create a Wall object
            self.wall_cooldown = 60  # Cooldown in frames

    def attack_wall(self):
        if self.wall_health > 0:
            self.wall_health -= 1
            return True
        else:
            return False

    def update_wall(self):
        if self.wall_cooldown > 0:
            self.wall_cooldown -= 1

    def choose_action(self):
        if np.random.uniform(0, 1) < epsilon:
            if self.food >= 3:
                if self.food >= WALL_COST and not self.enemy:
                    return np.random.choice(7)  # Exploration
                else:
                    return np.random.choice(6)  # Exploration
            else:
                return np.random.choice(5)  # Exploration
        else:
            return np.argmax(self.q_table[self.state[0], self.state[1]])

    def take_action(self, action):
        x, y = self.state

        if self.enemy:
            self.speed = 4
        # else:
        #    self.speed = 5
        #
        # size = self.food // 2 + 20
        #
        # if size > 50:
        #    size = 150        #
        # self.speed -= size / 100

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
        elif action == 5:
            self.food -= random.randint(3, 6)
            self.create_baby()
        elif action == 6:
            self.create_wall()

        return x, y

    def calculate_reward(self, good_positions, bad_positions, radius=50):
        self.reward = 0
        for good_pos in good_positions:
            distance = np.linalg.norm(np.array(self.state) - np.array(good_pos))
            if self.prev_distance is not None:
                if distance < self.prev_distance:
                    self.reward += self.custom_rewards["getting_closer"]
                elif distance > self.prev_distance:
                    self.reward += self.custom_rewards["getting_further"]

        # Penalize for being close to bad positions
        for bad_pos in bad_positions:
            distance_to_bad = np.linalg.norm(np.array(self.state) - np.array(bad_pos))
            if distance_to_bad < radius:
                self.reward += self.custom_rewards["near_bad_positions"]

        self.prev_distance = np.linalg.norm(
            np.array(self.state) - np.array(self.take_action(self.choose_action()))
        )  # Update previous distance

        return self.reward + self.custom_rewards.get(
            "additional_reward", 0
        ) * np.random.uniform(-0.1, 0.1)

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
        baby = QLearningAgent(self.q_table, custom_rewards=self.custom_rewards)
        baby.enemy = self.enemy
        self.reward += 1.5
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

# Walls list to store created walls
walls = []

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
            if event.key == pygame.K_r:
                # Reset simulation when 'r' is pressed
                adult_black_squares = [QLearningAgent(q_table_black) for _ in range(2)]
                baby_black_squares = []
                adult_red_squares = [e]
                baby_red_squares = []
                green_squares = []
                blue_squares = []
                walls = []
                baby_frame_counter = 0
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
    # Check if black square collides with a wall
    for black_square in adult_black_squares + baby_black_squares:
        for red_square in adult_red_squares + baby_red_squares:
            for wall in walls:
                if pygame.Rect(
                    red_square.state[0], red_square.state[1], 20, 20
                ).colliderect(pygame.Rect(wall.x, wall.y, 20, 20)):
                    if wall.health > 0:
                        wall.health -= 1
                        red_square.reward += 0.5  # Reward for breaking a wall
                        red_square.food += 0.5

    # Update and draw walls
    for wall in walls:
        if wall.health > 0:
            pygame.draw.rect(
                screen,
                PURPLE,  # Use the PURPLE color for the wall
                (
                    wall.x,
                    wall.y,
                    20,
                    20,
                ),
            )

    # Update and draw adult black squares
    for black_square in adult_black_squares:
        size = black_square.food // 2 + 20

        if size > 50:
            size = 50
        pygame.draw.rect(
            screen,
            BLACK,
            (
                black_square.state[0],
                black_square.state[1],
                size,
                size,
            ),
        )

    # Update and draw adult red squares
    for red_square in adult_red_squares:
        size = red_square.food // 2 + 20

        if size > 50:
            size = 50

        pygame.draw.rect(
            screen,
            RED,
            (
                red_square.state[0],
                red_square.state[1],
                size,
                size,
            ),
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

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
