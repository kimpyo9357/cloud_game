"""Play Othello."""
import os
from . import othello
from . import simple_policies
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

import argparse
import numpy as np
from collections import deque
import random
import pickle
from .. import consumer

import asyncio
from prompt_toolkit import PromptSession

'''priority1 = [18,21,42,45]
priority2 = [19,20,26,29,34,37,43,44]
priority3 = [0,7,56,63]
priority4 = [2,3,4,5,16,23,24,31,32,39,40,47,57,58,59,60,61]'''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

'''gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:  # gpu가 있다면, 용량 한도를 4GB로 설정
  tf.config.experimental.set_virtual_device_configuration(gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4*1024)])

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)'''

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

'''parserDQN = argparse.ArgumentParser()
parserDQN.add_argument('--gamma', type=float, default=1)
parserDQN.add_argument('--lr', type=float, default=0.005)
parserDQN.add_argument('--batch_size', type=int, default=128)
parserDQN.add_argument('--eps', type=float, default=1.0)
parserDQN.add_argument('--eps_decay', type=float, default=0.9995)
parserDQN.add_argument('--eps_min', type=float, default=0.2)
parserDQN.add_argument('--temp_save_freq', type=int, default=1000)
parserDQN.add_argument('--target_update_freq', type=int, default=100)
parserDQN.add_argument('--temp_save', type=bool, default=True)

argsDQN = parserDQN.parse_args()'''

class ReplayBuffer:
    def __init__(self, capacity=500000000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, 128)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(128, -1)
        next_states = np.array(next_states).reshape(128, -1)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionStateModel:
    def __init__(self, state_dim, aciton_dim, env):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.epsilon = 1.0
        self.rnd = np.random.RandomState(seed=0)

        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim, activation='tanh')
        ])
        model.compile(loss='mse', optimizer=Adam(0.005))
        return model

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        '''check = list(set(self.possible_moves).intersection(priority3))
        if len(check) > 0:
            if np.random.random() < self.epsilon:
                ix = self.rnd.randint(0, len(check))
                return check[ix]
            else:
                state = np.reshape(state, [1, self.state_dim])
                q_value = self.predict(state)
                state_mask = np.full(self.state_dim, np.nan)
                for i in range(len(check)):
                    state_mask[check] = 1
                q2_value = q_value * state_mask
                nanargmax = int(np.nanargmax(q2_value))
                del q_value, state, state_mask, q2_value
                return nanargmax
        else:
            check = list(set(self.possible_moves).intersection(priority4))
            if len(check) > 0:
                if np.random.random() < self.epsilon:
                    ix = self.rnd.randint(0, len(check))
                    return check[ix]
                else:
                    state = np.reshape(state, [1, self.state_dim])
                    q_value = self.predict(state)
                    state_mask = np.full(self.state_dim, np.nan)
                    for i in range(len(check)):
                        state_mask[check] = 1
                    q2_value = q_value * state_mask
                    nanargmax = int(np.nanargmax(q2_value))
                    del q_value, state, state_mask, q2_value
                    return nanargmax
            else:
                check = list(set(self.possible_moves).intersection(priority1))
                if len(check) > 0:
                    if np.random.random() < self.epsilon:
                        ix = self.rnd.randint(0, len(check))
                        return check[ix]
                    else:
                        state = np.reshape(state, [1, self.state_dim])
                        q_value = self.predict(state)
                        state_mask = np.full(self.state_dim, np.nan)
                        for i in range(len(check)):
                            state_mask[check] = 1
                        q2_value = q_value * state_mask
                        nanargmax = int(np.nanargmax(q2_value))
                        del q_value, state, state_mask, q2_value
                        return nanargmax
                else:
                    check = list(set(self.possible_moves).intersection(priority2))
                    if len(check) > 0:
                        if np.random.random() < self.epsilon:
                            ix = self.rnd.randint(0, len(check))
                            return check[ix]
                        else:
                            state = np.reshape(state, [1, self.state_dim])
                            q_value = self.predict(state)
                            state_mask = np.full(self.state_dim, np.nan)
                            for i in range(len(check)):
                                state_mask[check] = 1
                            q2_value = q_value * state_mask
                            nanargmax = int(np.nanargmax(q2_value))
                            del q_value, state, state_mask, q2_value
                            return nanargmax
                    else:
                        if np.random.random() < self.epsilon:
                            ix = self.rnd.randint(0, len(self.possible_moves))
                            return self.possible_moves[ix]
                        state = np.reshape(state, [1, self.state_dim])
                        q_value = self.predict(state)
                        state_mask = np.full(self.state_dim,np.nan)
                        for i in range(len(self.possible_moves)):
                            state_mask[self.possible_moves] = 1
                        q2_value = q_value*state_mask
                        nanargmax = int(np.nanargmax(q2_value))
                        del q_value, state, state_mask, q2_value
                        return nanargmax'''
        if np.random.random() < self.epsilon:
            self.epsilon *= 0.9995
            self.epsilon = max(self.epsilon, 0.2)
            ix = self.rnd.randint(0, len(self.possible_moves))
            return self.possible_moves[ix]
        state = np.reshape(state, [1, self.state_dim])
        q_value = self.predict(state)
        state_mask = np.full(self.state_dim, np.nan)
        for i in range(len(self.possible_moves)):
            state_mask[self.possible_moves] = 1
        q2_value = q_value * state_mask
        nanargmax = int(np.nanargmax(q2_value))
        del q_value, state, state_mask, q2_value
        return nanargmax

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=1)

    def compile(self,loss,optimizer):
        self.model.compile(loss=loss, optimizer=optimizer)

    def save(self, path: str, model_name: str, version: str, num_trained:int, target_model_name: str = None):
        save_name = f"{path}/{model_name}_{version}_{num_trained}_trained"
        self.model.save(save_name+".h5py")
        if model_name == 'DQN':
            with open(save_name+".pickle",'wb') as f:
                pickle.dump(self.epsilon,f)

    def load(self,path: str,model_name: str,version: str,num_trained: int,target_model_name: str = None):
        save_name = f"{path}/{model_name}_{version}_{num_trained}_trained"
        self.model = tf.keras.models.load_model(save_name+".h5py")
        if model_name == 'DQN':
            with open(save_name+".pickle",'rb') as f:
                self.epsilon = pickle.load(f)

    @property
    def possible_moves(self):
        return self.env.possible_moves

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0] * self.env.observation_space.shape[1]
        self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim, self.env)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim, self.env)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(128), actions] = rewards + (1 - done) * next_q_values * 1
            self.model.train(states, targets)

    def train(self, ep: object = 0, max_episodes: object = 100) -> object:
        for ep in range(ep,max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            if self.buffer.size() >= 128:
                self.replay()
            if (ep % 1000) == 0:
                self.target_update()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            if ((ep % 1000) == 0 and True):
               self.db_save(
                path="./save",
                model_name="trans_DQN",
                version="8.8",
                num_trained=ep,
                )

    def db_save(self, path: str, model_name: str, version: str, num_trained:int, target_model_name: str = None):
        self.model.save(
            path=path,
            model_name=model_name,
            version=version,
            num_trained=num_trained
        )
        self.target_model.save(
            path=path,
            model_name="target_"+model_name,
            version=version,
            num_trained=num_trained
        )

    def db_load(self,path: str,model_name: str,version: str,num_trained: int,target_model_name: str = None):
        '''save_name = f"{path}/{model_name}_{version}_{num_trained}_trained.h5"
        target_model_name = f"target_{model_name}"
        target_save_name = (
            f"{path}/{target_model_name}_{version}_{num_trained}_trained.h5"
        )'''
        self.model.load(
            path=path,
            model_name=model_name,
            version=version,
            num_trained=num_trained
        )
        self.target_model.load(
            path=path,
            model_name="target_"+model_name,
            version=version,
            num_trained=num_trained
        )

def create_policy(policy_type='rand', board_size=8, seed=0, search_depth=1,rand_board_count=0):
    if policy_type == 'rand':
        policy = simple_policies.RandomPolicy(seed=seed+10)
    elif policy_type == 'greedy':
        policy = simple_policies.GreedyPolicy()
    elif policy_type == 'maximin':
        policy = simple_policies.MaxiMinPolicy(search_depth)
    elif policy_type == 'DQN':
        policy = simple_policies.DQNPolicy(board_size*board_size,board_size*board_size,rand_board_count)
        policy.load(path="./chat/game/save", model_name="DQN", version="8.8", num_trained=431000)
    elif policy_type == 'DQN_op':
        policy = simple_policies.DQNPolicy_op(board_size*board_size,board_size*board_size,rand_board_count)
        #policy.load(path="./save", model_name="DQN", version="8.8", num_trained=0)
    else:
        policy = simple_policies.HumanPolicy(board_size)
    return policy


async def play(protagonist = -1,
         protagonist_agent_type='human',
         opponent_agent_type='human',
         board_size=8,
         num_rounds=1,
         protagonist_search_depth=1,
         opponent_search_depth=1,
         rand_seed=0,
         env_init_rand_steps=0,
         num_disk_as_reward=False,
         render=True,
         rand_board_count=0,
         rand_board_seed=0,
         socket=None):
    print('protagonist: {}'.format(protagonist_agent_type))
    print('opponent: {}'.format(opponent_agent_type))

    ep = 0
    protagonist_policy = create_policy(
        policy_type=protagonist_agent_type,
        board_size=board_size,
        seed=ep,
        search_depth=protagonist_search_depth,
        rand_board_count=rand_board_count)
    opponent_policy = create_policy(
        policy_type=opponent_agent_type,
        board_size=board_size,
        seed=rand_seed,
        search_depth=opponent_search_depth,
        rand_board_count=rand_board_count)
    #print(id(protagonist_policy),id(opponent_policy))

    if protagonist == 1:
        white_policy = protagonist_policy
        black_policy = opponent_policy
    else:
        white_policy = opponent_policy
        black_policy = protagonist_policy

    if opponent_agent_type == 'human':
        render_in_step = True
    else:
        render_in_step = render
    env = othello.OthelloEnv(white_policy=white_policy,
                             black_policy=black_policy,
                             protagonist=protagonist,
                             board_size=board_size,
                             seed=rand_seed,
                             initial_rand_steps=env_init_rand_steps,
                             num_disk_as_reward=num_disk_as_reward,
                             render_in_step=render_in_step,
                             rand_board_count=rand_board_count,
                             socket = socket)

    '''agent = Agent(env)
    ep = 110000
    agent.db_load(path="./save", model_name="trans_DQN", version="8.8", num_trained=ep)
    agent.train(ep=ep, max_episodes=1000000)'''
    session = PromptSession()
    win_cnts = draw_cnts = lose_cnts = 0
    for i in range(num_rounds):
        print('Episode {}'.format(i + 1))
        obs = await env.reset()
        protagonist_policy.reset(env)
        if render:
            env.render()
        done = False
        while not done:
            action = await protagonist_policy.get_action(obs,env.rand_state)
            obs, reward, done, _ = await env.step(action)
            if render:
                env.render()
            if done:
                print('reward={}'.format(reward))
                if num_disk_as_reward:
                    total_disks = board_size ** 2
                    if protagonist == 1:
                        white_cnts = reward
                        black_cnts = total_disks - white_cnts
                    else:
                        black_cnts = reward
                        white_cnts = total_disks - black_cnts

                    if white_cnts > black_cnts:
                        win_cnts += 1
                    elif white_cnts == black_cnts:
                        draw_cnts += 1
                    else:
                        lose_cnts += 1
                else:
                    if reward == 1:
                        win_cnts += 1
                    elif reward == 0:
                        draw_cnts += 1
                    else:
                        lose_cnts += 1
                print('-' * 3)
    print('#ep: {} #Wins: {}, #Draws: {}, #Loses: {}'.format(
        ep, win_cnts, draw_cnts, lose_cnts))
    #f = open("check.csv",'a')
    #f.write(str(ep) + ',' + str(win_cnts) + ',' + str(draw_cnts) + ',' + str(lose_cnts) + '\n')
    #f.close()
    env.close()

def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument('--protagonist', default='human',
                        choices=['rand', 'greedy', 'maximin','DQN','DQN_op', 'human'])
    parser.add_argument('--opponent', default='human',
                        choices=['rand', 'greedy', 'maximin','DQN','DQN_op', 'human'])
    parser.add_argument('--protagonist-plays-white', default=False,
                        action='store_true')
    parser.add_argument('--num-disk-as-reward', default=False,
                        action='store_true')
    parser.add_argument('--board-size', default=8, type=int)
    parser.add_argument('--protagonist-search-depth', default=1, type=int)
    parser.add_argument('--opponent-search-depth', default=1, type=int)
    parser.add_argument('--rand-seed', default=0, type=int)
    parser.add_argument('--num-rounds', default=1, type=int)
    parser.add_argument('--init-rand-steps', default=0, type=int)
    parser.add_argument('--no-render', default=False, action='store_true')
    parser.add_argument('--rand-board-count', default=0, type=int)
    args, _ = parser.parse_known_args()

    # Run test plays.
    protagonist = 1 if args.protagonist_plays_white else -1
    protagonist_agent_type = args.protagonist
    opponent_agent_type = args.opponent
    play(protagonist=protagonist,
         protagonist_agent_type=protagonist_agent_type,
         opponent_agent_type=opponent_agent_type,
         board_size=args.board_size,
         num_rounds=args.num_rounds,
         protagonist_search_depth=args.protagonist_search_depth,
         opponent_search_depth=args.opponent_search_depth,
         rand_seed=args.rand_seed,
         env_init_rand_steps=args.init_rand_steps,
         num_disk_as_reward=args.num_disk_as_reward,
         render=not args.no_render,
         rand_board_count=args.rand_board_count,
         )


if __name__ == '__main__':
    main()


