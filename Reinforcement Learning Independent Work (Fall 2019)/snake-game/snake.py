import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle # save q table
from matplotlib import style
import time

style.use("ggplot")


# CONSTANTS

SIZE = 10  # Grid size
NO_EPISODES = 25000   # NUMBER OF EPISODES
PENALTY_MOVE = 1
PENALTY_OBSTACLE = 300  # SNAKE HITS SOME OBSTACLE
REWARD_FOOD = 25  # SNAKE FINDS FOOD
SHOW_EVERY = 1
LEARNING_RATE = 0.1
DISCOUNT = 0.95

# Dictionary representations of players
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

# Colors of players in environment (B,G,R)
d = {1: (255, 175, 0),
     2: (255, 255, 255),
     3: (0, 100, 255)}

# Q Learning Constants
epsilon = 0.9
EPS_DECAY = 0.9998

start_q_table = None  # or filename
# start_q_table = "qtable-1573110392.pickle"
episode_rewards = []

# Creating Snake class
class Snake:

    def __init__(self):

        # Starting coordinates
        self.x = np.random.randint(0, SIZE - 1)
        self.y = np.random.randint(0, SIZE - 1)


    def __str__(self):
        # for debugging
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y

    def action(self, choice):
        # player takes discrete action
        # UP, DOWN, LEFT, RIGHT
        if choice == 0:
            self.move(x=0,y=1)
        elif choice == 1:
            self.move(x=0,y=-1)
        elif choice == 2:
            self.move(x=-1,y=0)
        elif choice == 3:
            self.move(x=1,y=0)


    def move(self, x=0, y=0):

        if not x:
            self.x += np.random.randint(-1, 2)

        else:
            self.x += x

        if not y:
            self.y += np.random.randint(-1, 2)

        else:
            self.y += y


        # Accounting for wall collision
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1

        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1


# class Food:



# Making q table
# observation space will look like (x1, y1),(x2, y2)
if start_q_table is None:

    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE + 1, SIZE):
            for x2 in range(-SIZE + 1, SIZE):
                for y2 in range(-SIZE + 1, SIZE):
                    # add combinations to table
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

# Training
for episode in range(NO_EPISODES):

    player = Snake()
    food = Snake()
    enemy = Snake()

    if episode % SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon {epsilon}")  # where we are
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    # steps = 200
    for i in range(200):
        obs = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])

        else:
            action = np.random.randint(0, 4)

        player.action(action)

        # maybe later
        # enemy.move()
        # food.move()
        ########


        # reward process
        if player.x == enemy.x and player.y == enemy.y:
            reward = PENALTY_OBSTACLE
        elif player.x == food.x and player.y == food.y:
            reward = REWARD_FOOD
        else:
            reward = -PENALTY_MOVE

        # we've made the move, now we need new move
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        # Calculate Q
        if reward == REWARD_FOOD:
            new_q = REWARD_FOOD # In this case, if we grab food, episode ends.
        elif reward == -PENALTY_OBSTACLE:
            new_q = -PENALTY_OBSTACLE
        else:
            new_q = (1-LEARNING_RATE) * current_q + LEARNING_RATE*(reward + DISCOUNT*max_future_q)

        # Update Q Table
        q_table[obs][action] = new_q

        # Visualiztion stuff
        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # All black environment
            # in the arrays, seen as y by x.
            env[food.y][food.x] = d[FOOD_N]
            env[player.y][player.x] = d[PLAYER_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]

            img = Image.fromarray(env, "RGB")  # gives image
            img = img.resize((300, 300))
            cv2.imshow("", np.array(img))

            if reward == REWARD_FOOD or reward == PENALTY_OBSTACLE:
                if cv2.waitKey(500) & 0xFF == ord("q"):  # if epsiode ends, make delay
                    break

            else:
                if cv2.waitKey(1) & 0xFF == ord("q"):  # if epsiode ends, make delay
                    break



        episode_reward = reward
        # CHANGE THIS PART FOR SNAKE GAME
        if reward == REWARD_FOOD or reward == -PENALTY_OBSTACLE:
            break


    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY


# Graphing

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode="valid")
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY}ma")
plt.xlabel(f"episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)














