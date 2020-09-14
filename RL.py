import gym
import numpy as np
import random
import argparse
from time import sleep
 
env = gym.make("Taxi-v3").env

#Hyperparameters
q_table = np.zeros([env.observation_space.n,env.action_space.n])
alpha = 0.1
gamma = 0.6
epsilon = 0.1
all_epoch= []
all_penalties = []
frames = []
def play(episodes):
   
    total_epochs, total_penalties = 0, 0

    for i in range(episodes):
       
        state = env.reset()
       
        epochs, penalties, reward = 0, 0, 0
        
        done = False
        
        while not done:
            action = np.argmax(q_table[state])
            state, reward, done, info = env.step(action)

            if reward == -10:
                penalties += 1
                
            epochs += 1
        env.render()
        total_penalties += penalties
        total_epochs += epochs
    
    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")
    
def train_model():
    #Training
    for _ in range(10000):
        state=env.reset()
        epoch, penalties, reward = 0, 0 , 0
        done = False
        
        while not done:
            if random.uniform(0,1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
            next_state, reward, done, info= env.step(action)
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])
            
            new_value = (1- alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value
            
            if reward == -10:
                penalties += 1
            state = next_state
            epoch += 1
    print("end")

def print_frames(frames):
      for i, frame in enumerate(frames):
                
                print(frame['frame'].getvalue())
                print(f"Timestep: {i + 1}")
                print(f"State: {frame['state']}")
                print(f"Action: {frame['action']}")
                print(f"Reward: {frame['reward']}")
                sleep(.1)

            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        default=False,
        type=str,
        help="Training agent"
    )
    parser.add_argument(
        "--play",
        default=False,
        type=str,
        help="Test agent performence"
    )
    parser.add_argument(
        "--episodes",
        default=1,
        type=int,
        help="Episode for agent")
    args = parser.parse_args()
   
    if args.train=="True":
        train_model()
    
    if args.play=="True":
        play(args.episodes)
        
if __name__ == "__main__":
        main()
        
    
