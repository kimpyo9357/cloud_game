"""Simple policies for Othello."""

import numpy as np
import argparse
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import asyncio
from prompt_toolkit import PromptSession

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from .. import consumer

WHITE_DISK = 1
BLACK_DISK = -1
PROTAGONIST_TURN = 1
OPPONENT_TURN = -1


def copy_env(env, mute_env=True):
    new_env = env.__class__(
        board_size=env.board_size,
        sudden_death_on_invalid_move=env.sudden_death_on_invalid_move,
        mute=mute_env)
    new_env.reset()
    return new_env


class RandomPolicy(object):
    """Random policy for Othello."""

    def __init__(self, seed=0):
        self.rnd = np.random.RandomState(seed=seed)
        self.env = None

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def seed(self, seed):
        self.rnd = np.random.RandomState(seed=seed)

    async def get_action(self, obs,_,socket):
        possible_moves = self.env.possible_moves
        ix = self.rnd.randint(0, len(possible_moves))
        action = possible_moves[ix]
        return action


class GreedyPolicy(object):
    """Greed is good."""

    def __init__(self):
        self.env = None

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    async def get_action(self, obs,_,socket):
        my_perspective = self.env.player_turn
        new_env = copy_env(self.env)

        # For each move, replicate the current board and make the move.
        possible_moves = self.env.possible_moves
        disk_cnts = []
        for move in possible_moves:
            new_env.reset()
            new_env.set_board_state(
                board_state=obs, perspective=my_perspective)
            new_env.set_player_turn(my_perspective)
            new_env.step(move)
            white_disks, black_disks = new_env.count_disks()
            if my_perspective == WHITE_DISK:
                disk_cnts.append(white_disks)
            else:
                disk_cnts.append(black_disks)

        new_env.close()
        ix = np.argmax(disk_cnts)
        return possible_moves[ix]


class MaxiMinPolicy(object):
    """Maximin algorithm."""

    def __init__(self, max_search_depth=1):
        self.env = None
        self.max_search_depth = max_search_depth

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def search(self, env, depth, perspective, my_perspective):

        # Search at a node stops if
        #   1. Game terminated
        #   2. depth has reached max_search_depth
        #   3. No more possible moves
        if (
                env.terminated or
                depth >= self.max_search_depth or
                len(env.possible_moves) == 0
        ):
            white_disks, black_disks = env.count_disks()
            if my_perspective == WHITE_DISK:
                return white_disks, None
            else:
                return black_disks, None
        else:
            assert env.player_turn == perspective
            new_env = copy_env(env)

            # For each move, replicate the current board and make the move.
            possible_moves = env.possible_moves
            disk_cnts = []
            for move in possible_moves:
                new_env.reset()
                new_env.set_board_state(env.get_observation(), env.player_turn)
                new_env.set_player_turn(perspective)
                new_env.step(move)
                if (
                        not new_env.terminated and
                        new_env.player_turn == perspective
                ):
                    # The other side had no possible moves.
                    new_env.set_player_turn(-perspective)
                disk_cnt, _ = self.search(
                    new_env, depth + 1, -perspective, my_perspective)
                disk_cnts.append(disk_cnt)

            new_env.close()

            # Max-min.
            ix = int(np.argmin(disk_cnts))
            if perspective == my_perspective:
                ix = int(np.argmax(disk_cnts))
            return disk_cnts[ix], possible_moves[ix]

    async def get_action(self, obs,_,socket):
        my_perspective = self.env.player_turn
        disk_cnt, move = self.search(env=self.env,
                                     depth=0,
                                     perspective=my_perspective,
                                     my_perspective=my_perspective)                            
        return move


class DQNPolicy(object):
    def __init__(self, state_dim,action_dim,rand_board_count=0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 0.5
        self.rnd = np.random.RandomState(seed=0)
        self.env = None
        self.rand_board_count = rand_board_count
        self.rand_board_state = []

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim, activation='tanh')
        ])
        model.compile(loss='mse', optimizer=Adam(0.005))
        return model

    async def get_action(self, state, rand_state,socket):
        '''self.epsilon *= argsDQN.eps_decay
        if np.random.random() < self.epsilon:
            ix = self.rnd.randint(0, len(self.possible_moves))
            return self.possible_moves[ix]
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon = max(self.epsilon, argsDQN.eps_min)
        q_value = self.predict(state)
        state_mask = np.full(self.state_dim,np.nan)
        for i in range(len(self.possible_moves)):
            state_mask[self.possible_moves] = 1
        q2_value = q_value*state_mask
        nanargmax = int(np.nanargmax(q2_value))
        del q_value, state, state_mask, q2_value
        return nanargmax'''

        state = np.reshape(state, [1, self.state_dim])
        copy_state = state.copy()
        state_mask = np.full(self.state_dim,np.nan)
        for i in range(len(self.possible_moves)):
            state_mask[self.possible_moves] = 1
        for i in rand_state:
            copy_state[0][i] = 0
        q_value = self.predict(copy_state)
        q2_value = q_value*state_mask
        nanargmax = int(np.nanargmax(q2_value))
        del q_value, state, state_mask, q2_value
        return nanargmax

    def predict(self, state):
        return self.model.predict(state)

    def load(self,path: str,model_name: str,version: str,num_trained: int,target_model_name: str = None):
        save_name = f"{path}/{model_name}_{version}_{num_trained}_trained"
        self.model = tf.keras.models.load_model(save_name+".h5py")

    @property
    def possible_moves(self):
        return self.env.possible_moves

class DQNPolicy_op(object):
    def __init__(self, state_dim,action_dim,rand_board_count=0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = 0.5
        self.rnd = np.random.RandomState(seed=0)
        self.env = None
        self.rand_board_count = rand_board_count

    def reset(self, env):
        if hasattr(env, 'env'):
            self.env = env.env
        else:
            self.env = env

    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim, activation='tanh')
        ])
        model.compile(loss='mse', optimizer=Adam(0.005))
        return model

    async def get_action(self, state,_,socket):
        self.epsilon *= 0.75
        if np.random.random() < self.epsilon:
            ix = self.rnd.randint(0, len(self.possible_moves))
            return self.possible_moves[ix]
        state = -state
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon = max(self.epsilon, 0.01)
        q_value = self.predict(state)
        state_mask = np.full(self.state_dim,np.nan)
        for i in range(len(self.possible_moves)):
            state_mask[self.possible_moves] = 1
        q2_value = q_value*state_mask
        nanargmax = int(np.nanargmax(q2_value))
        del q_value, state, state_mask, q2_value
        return nanargmax

    def predict(self, state):
        return self.model.predict(state)

    def load(self,path: str,model_name: str,version: str,num_trained: int,target_model_name: str = None):
        save_name = f"{path}/{model_name}_{version}_{num_trained}_trained"
        self.model = tf.keras.models.load_model(save_name+".h5py")

    @property
    def possible_moves(self):
        return self.env.possible_moves
        
variable = {}
class HumanPolicy(object):
    """Human policy."""
    def __init__(self, board_size,socket):
        global variable
        self.socket = socket
        variable[socket.room_name] = -1
        self.board_size = board_size

    def reset(self, env):
        global variable
        variable[self.socket.room_name] = -1
    
    def set_value(self,action):
        global variable
        variable[socket.room_name] = action

    async def get_action(self, obs,_,socket):
        global variable
        temp = -1
        old_value = variable[socket.room_name]
        while True:
            with open('action_'+socket.room_name+'.txt','r') as f:
                temp = f.read()
            if temp != '':
                #print(temp)
                variable[socket.room_name] = int(temp)

            if variable[socket.room_name] != old_value:
                #print(variable[socket.room_name])
                return variable[socket.room_name]
            else:
                await asyncio.sleep(1.0)
        '''session = PromptSession()
        print("simple")
        input_data = await session.prompt_async('>> ')
        input_data = int(input_data)
        print(input_data + "simple")
        return input_data
        return int(input('Enter action index:'))'''

    
