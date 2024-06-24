# NOTE: Code adapted from MinimalRL (URL: https://github.com/seungeunrho/minimalRL/blob/master/dqn.py)

# Imports:
# --------
import torch
import gymnasium as gym
from DQN_model import Qnet
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import ReplayBuffer, train
from padm_env import create_env
from constants import HELL_COORDINATE_POINTS
import os

from datetime import datetime

from datetime import datetime

now = datetime.now()


# User definitions:
# -----------------
train_dqn = False
test_dqn = True
render = False

#! Define env attributes (environment specific)
no_actions = 4
no_states = 2

# Hyperparameters:
# ----------------
learning_rate = 0.005
gamma = 0.99
buffer_limit = 50_0
batch_size = no_actions*16
num_episodes = 100_00
max_steps = 10

goal_coordinates = (9, 9)

hell_state_coordinates = HELL_COORDINATE_POINTS


# Main:
# -----
if train_dqn:
    env = create_env(goal_coordinates=goal_coordinates,
               hell_state_coordinates=hell_state_coordinates)

    #! Initialize the Q Net and the Q Target Net
    q_net = Qnet(no_actions=no_actions, no_states=no_states)
    q_target = Qnet(no_actions=no_actions, no_states=no_states)
    q_target.load_state_dict(q_net.state_dict())

    #! Initialize the Replay Buffer
    memory = ReplayBuffer(buffer_limit=buffer_limit)

    print_interval = 20
    episode_reward = 0.0
    optimizer = optim.Adam(q_net.parameters(),
                           lr=learning_rate)

    rewards = []

    for n_epi in range(num_episodes):
        #! Epsilon decay (Please come up with your own logic)
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200))  # ! Linear annealing from 8% to 1%

        s, _ = env.reset()
        done = False
        #! Define maximum steps per episode, here 1,000
        for _ in range(max_steps):
            #! Choose an action (Exploration vs. Exploitation)
            a = q_net.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, _ = env.step(a)

            done_mask = 0.0 if done else 1.0

            #! Save the trajectories
            memory.put((s, a, r, s_prime, done_mask))
            s = s_prime

            episode_reward += r
            if(render):
                env.render()

            if done:
                break

        # if memory.size() > 8000:
        #     train(q_net, q_target, memory, optimizer, batch_size, gamma)

        if n_epi % print_interval == 0 and n_epi != 0:
            q_target.load_state_dict(q_net.state_dict())
            print(
                f"n_episode :{n_epi}, Episode reward : {episode_reward}, n_buffer : {memory.size()}, eps : {epsilon}")

        rewards.append(episode_reward)
        episode_reward = 0.0

        #! Define a stopping condition for the game:
        if rewards[-10:] == [max_steps]*10:
            break
        
    env.close()

    #! Save the trained Q-net
    torch.save(q_net.state_dict(), "dqn.pth")

    #! Plot the training curve
    plt.plot(rewards, label='Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.savefig("training_curve.png")
    date_time = now.strftime("_%H:%M:%S_").replace(" ","")
    locationchange= os.path.join("trainings","training_curve"+str(date_time)+".png")
    plt.savefig(locationchange)
    plt.show()

# Test:
if test_dqn:
    print("Testing the trained DQN: ")
    env = create_env(goal_coordinates=goal_coordinates,
               hell_state_coordinates=hell_state_coordinates)

    dqn = Qnet(no_actions=no_actions, no_states=no_states)
    dqn.load_state_dict(torch.load("dqn.pth"))

    for _ in range(10):
        s, _ = env.reset(train=train_dqn)
        episode_reward = 0

        for _ in range(max_steps):
            #! Completely exploit
            action = dqn(torch.from_numpy(s).float())
            s_prime, reward, done, _ = env.step(action.argmax().item())
            s = s_prime

            episode_reward += reward
            env.render()

            if done:
                break
        print(f"Episode reward: {episode_reward}")

    env.close()
