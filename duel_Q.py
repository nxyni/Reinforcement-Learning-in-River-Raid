

import gym
import cv2
import keras
from keras.models import load_model, Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
from keras.layers.core import Flatten, Dense
from keras.layers import merge, Input
from keras import backend as K
from replay_buffer import ReplayBuffer
from keras.layers import concatenate
from keras.layers import Lambda
import numpy as np
import random
from keras.layers import LSTM, TimeDistributed

# Some parameters that can be adjusted by the user
DECAY_RATE = 0.99
LEARNING_RATE = 1e-6
K = 100

# Some constant used in the code
NUM_ACTIONS = 6
# TAU = 0.1

# Number of frames that will be used in the network each time
NUM_FRAMES = 3 # DQN
# NUM_FRAMES = 1 # DRQN


# Define the complete duel Q network
class DuelQ(object):

    def __init__(self):
        self.construct_q_network()

    def construct_q_network(self):
        '''Construct the deep Q network(DDQN)'''
        self.model = Sequential()
        input_layer = Input(shape = (84, 84, NUM_FRAMES))
        conv1 = Convolution2D(32, 8, 8, subsample = (4, 4), activation = 'relu')(input_layer)
        conv2 = Convolution2D(64, 4, 4, subsample = (2, 2), activation = 'relu')(conv1)
        conv3 = Convolution2D(64, 3, 3, activation = 'relu')(conv2)
        flatten = Flatten()(conv3)
        fc1 = Dense(512)(flatten)
        advantage = Dense(NUM_ACTIONS)(fc1)
        fc2 = Dense(512)(flatten)
        value = Dense(1)(fc2)
        policy = Lambda(lambda x: x[0] - K.mean(x[0])+x[1], (NUM_ACTIONS,))([advantage, value])

        # Construct the desired network
        self.model = Model(input = [input_layer], output = [policy])
        self.model.compile(loss = 'mse', optimizer = Adam(lr = LEARNING_RATE))

         # Construct the target network
        self.target_model = Model(input=[input_layer], output = [policy])
        self.target_model.compile(loss = 'mse', optimizer = Adam(lr = LEARNING_RATE))
        # The loss function is MSE, actually we can define some new loss function in the keras package, such DQfD
        # We can also use other optimizer, such as sgd

        print('Deep Q Network has been constructed.')
    
        '''Construct the deep recurrent Q network(DRQN)'''
        '''
        self.model = Sequential()
        input_layer = Input(shape = (84, 84, NUM_FRAMES))
        conv1 = TimeDistributed(Convolution2D(32, 8, 8, subsample = (4, 4), activation = 'relu')(input_layer))
        conv2 = TimeDistributed(Convolution2D(64, 4, 4, subsample = (2, 2), activation = 'relu')(conv1))
        conv3 = TimeDistributed(Convolution2D(64, 3, 3, activation = 'relu')(conv2))
        flatten = Flatten()(conv3)
        fc1 = LSTM(512, return_sequences = True)(flatten)
        advantage = Dense(NUM_ACTIONS)(fc1)
        fc2 = LSTM(512, return_sequences = True)(flatten)
        value = Dense(1)(fc2)
        policy = Lambda(lambda x: x[0] - K.mean(x[0])+x[1], (NUM_ACTIONS,))([advantage, value])

         # Construct the desired network
        self.model = Model(input = [input_layer], output = [policy])
        self.model.compile(loss = 'mse', optimizer = Adam(lr = LEARNING_RATE))

         # Construct the target network
        self.target_model = Model(input=[input_layer], output = [policy])
        self.target_model.compile(loss = 'mse', optimizer = Adam(lr = LEARNING_RATE))
        # The loss function is MSE, actually we can define some new loss function in the keras package, such DQfD
        # We can also use other optimizer, such as sgd

        print('Deep Q Network has been constructed.')

        '''

    # Use epsilon greedy to predict the corresponding action
    def predict_movement(self, data, epsilon):
        prob = np.random.random()
        q_actions = self.model.predict(data.reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
        if prob > epsilon:
            # Choose the optimal policy
            opt_policy = np.argmax(q_actions)
        else:
            # Randomly Choose an action
            opt_policy = np.random.randint(0, NUM_ACTIONS)
        return opt_policy, q_actions[0, opt_policy]


    # Define the training process
    def train(self, s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num):
        # s is current state, a is action, r is reward, d is whether the game ends, s2 is the next state

        batch_size = s_batch.shape[0] # number of samples trained in each pass
        targets = np.zeros((batch_size, NUM_ACTIONS)) 

        # Calculate the target value
        for i in range(batch_size):
            targets[i] = self.model.predict(s_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
            fut_action = self.target_model.predict(s2_batch[i].reshape(1, 84, 84, NUM_FRAMES), batch_size = 1)
            targets[i, a_batch[i]] = r_batch[i]
            if d_batch[i] == False:
                targets[i, a_batch[i]] = targets[i, a_batch[i]] + DECAY_RATE * np.max(fut_action)

        # Caluclate the loss
        loss = self.model.train_on_batch(s_batch, targets)

        # Print the loss every K iterations.
        if observation_num % K == 0:
            print("The loss is: ", loss)

    # Train the target network
    def target_train(self):
        # Directly copy the weights in the desired network to the target network
        model_weights = self.model.get_weights()
        self.target_model.set_weights(model_weights)


    # Save the trained network
    def save_network(self, path):
        self.model.save(path)
        print('Training porcess ends and the model has been saved to the declared folder.')

    # Load the network in test process
    def load_network(self, path):
        self.model = load_model(path)

