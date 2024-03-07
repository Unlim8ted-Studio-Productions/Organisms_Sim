import pygame
import numpy as np

# Define constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TARGET_RADIUS = 20
CREATURE_SIZE = 10
FOOD_SIZE = 5
NUM_CREATURES = 100
MUTATION_RATE = 0.1
NUM_INPUTS = 2
NUM_HIDDEN = 3
NUM_OUTPUTS = 2
NUM_GENERATIONS = 20
FOOD_SPAWN_RATE = 100  # Adjust this for desired spawn rate
HUNGER_THRESHOLD = 200  # Increase for less frequent hunger
HUNGER_RATE = 1
ATTRACTION_WEIGHT = 0.01
VISION_RADIUS = 50

# Define neural network class
class NeuralNetwork:
    def __init__(self):
        self.weights_input_hidden = np.random.randn(NUM_INPUTS, NUM_HIDDEN)
        self.weights_hidden_output = np.random.randn(NUM_HIDDEN, NUM_OUTPUTS)

    def predict(self, inputs):
        hidden_layer = np.dot(inputs, self.weights_input_hidden)
        hidden_layer = np.tanh(hidden_layer)
        output_layer = np.dot(hidden_layer, self.weights_hidden_output)
        return output_layer.reshape(1, -1)

# Define creature class
class Creature:
    def __init__(self, x, y, attractiveness, vision_radius, speed, color):
        self.x = x
        self.y = y
        self.angle = np.random.rand() * 2 * np.pi
        self.hunger = 21
        self.attractiveness = attractiveness
        self.vision_radius = vision_radius
        self.speed = speed
        self.color = color
        self.network = NeuralNetwork()

    def move(self):
        inputs = np.array([self.x / SCREEN_WIDTH, self.y / SCREEN_HEIGHT])
        outputs = self.network.predict(inputs)
        self.angle += (outputs[0][0] - outputs[0][1]) * np.pi / 8
        self.x += np.cos(self.angle) * self.speed
        self.y += np.sin(self.angle) * self.speed
        self.hunger += HUNGER_RATE
        self.hunger += HUNGER_RATE * self.speed * 2
        #new_x = self.x + np.cos(self.angle) * self.speed
        #new_y = self.y + np.sin(self.angle) * self.speed
        ## Check if the new position is within the window boundaries
        #if 0 <= new_x <= SCREEN_WIDTH and 0 <= new_y <= SCREEN_HEIGHT:
        #    self.x = new_x
        #    self.y = new_y
        #else:
        #    # If the new position is outside the boundaries, reflect the angle
        #    self.angle = np.random.rand() * 2 * np.pi

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), CREATURE_SIZE)#(500-self.hunger)/5)

    def mate(self, partner):
        child_speed = self.speed if np.random.rand() < 0.5 else partner.speed
        child_color = tuple((self.color[i] + partner.color[i]) // 2 for i in range(3))
        child_vision_radius = (self.vision_radius + partner.vision_radius) / 2
        child = Creature(self.x, self.y, (self.attractiveness + partner.attractiveness) / 2, child_vision_radius, child_speed, child_color)
        child.network.weights_input_hidden = self.crossover(self.network.weights_input_hidden, partner.network.weights_input_hidden)
        child.network.weights_hidden_output = self.crossover(self.network.weights_hidden_output, partner.network.weights_hidden_output)
        child.mutate()
        return child

    def crossover(self, parent1, parent2):
        child = np.empty_like(parent1)
        for i in range(parent1.shape[0]):
            for j in range(parent1.shape[1]):
                if np.random.rand() < 0.5:
                    child[i, j] = parent1[i, j]
                else:
                    child[i, j] = parent2[i, j]
        return child

    def mutate(self):
        # Mutate speed
        if np.random.rand() < MUTATION_RATE:
            self.speed += np.random.randn() * 0.5  # Adjust the mutation factor as needed
        # Mutate color
        if np.random.rand() < MUTATION_RATE:
            self.color = tuple(max(0, min(255, int(c + np.random.randint(-50, 50)))) for c in self.color)
        # Mutate vision
        if np.random.rand() < MUTATION_RATE:
            self.vision_radius += np.random.randn() * 10  # Adjust the mutation factor as needed

        # Ensure speed and vision radius are within reasonable bounds
        self.speed = max(1, min(10, self.speed))
        self.vision_radius = max(10, min(200, self.vision_radius))

    def eat(self, food):
        distance = np.sqrt((self.x - food.x) ** 2 + (self.y - food.y) ** 2)
        if distance < TARGET_RADIUS:
            return True
        return False

    def move_towards_food(self, food):
        direction_x = food.x - self.x
        direction_y = food.y - self.y
        magnitude = np.sqrt(direction_x ** 2 + direction_y ** 2)
        direction_x /= magnitude
        direction_y /= magnitude
        self.x += direction_x * self.speed
        self.y += direction_y * self.speed

# Define food class
class Food:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 255, 0), (int(self.x), int(self.y)), FOOD_SIZE)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Evolution Simulation")
#pygame.display.toggle_fullscreen()
# Create creatures
creatures = [Creature(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, np.random.uniform(0, 1), np.random.uniform(10, 100), np.random.uniform(1, 2), (255, 255, 255)) for _ in range(NUM_CREATURES)]

# Create food list
foods = []

# Main loop
running = True
generation = 0
food_spawn_counter = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Spawn food
    food_spawn_counter += 4
    if food_spawn_counter >= FOOD_SPAWN_RATE:
        food_spawn_counter = 0
        x = np.random.randint(0, SCREEN_WIDTH)
        y = np.random.randint(0, SCREEN_HEIGHT)
        foods.append(Food(x, y))

    # Draw everything
    screen.fill(BLACK)

    # Move creatures
    for creature in creatures:
        creature.move()
        if creature.hunger >= 500:
            creatures.remove(creature)
        elif creature.hunger <= 20:
            partner = np.random.choice(creatures)
            if partner is not creature:
                creatures.append(creature.mate(partner))
            #partners = [partner for partner in creatures if np.sqrt((creature.x - partner.x) ** 2 + (creature.y - partner.y) ** 2) <=50 and partner != creature]
            ## Define a custom key function to extract the attractiveness attribute
            #def get_attractiveness(creature):
            #    return creature.attractiveness
#
            ## Sort the list of creatures based on attractiveness, with the highest first
            #try:
            #    partner = sorted(partners, key=get_attractiveness, reverse=True)[0]
            #    creatures.append(creature.mate(partner))
            #    #reward += 20  # Arbitrary positive reward for successful reproduction
            #except:
            #    None
            
        if foods:
            nearest_food = min(foods, key=lambda f: np.sqrt((creature.x - f.x) ** 2 + (creature.y - f.y) ** 2))
            for food in foods:
                if creature.eat(food):
                    foods.remove(food)
                    creature.hunger -= 400
                    break
            else:
                creature.move_towards_food(nearest_food)
        creature.draw(screen)
    for food in foods:
        food.draw(screen)

    if len(creatures) == 1:
        creatures = [Creature(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, np.random.uniform(0, 1), np.random.uniform(10, 100), np.random.uniform(1, 10), (255, 255, 255)) for _ in range(NUM_CREATURES)]

    #if len(creatures) == 1:
    #    creatures[0].hunger = 21
    #    temp=creatures[0]
    #    creatures = [creatures[0] for _ in range(NUM_CREATURES-1)]
    #    for creature in creatures:
    #        creature.mutate()
    #    creatures.append(temp)
    # Update display
    pygame.display.flip()

    # Increment generation
    generation += 1
    if generation >= NUM_GENERATIONS:
        # Evolve creatures
        # Here you would implement the evolutionary algorithm,
        # such as selecting the fittest creatures and applying mutation
        generation = 0

# Quit Pygame
pygame.quit()
