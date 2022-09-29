
import gym
import cv2
import keras
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense
from keras.optimizers import Adam
from keras.models import load_model
from replay_buffer import ReplayBuffer
import numpy as np
import random
from keras.layers import LSTM, TimeDistributed # If use rucurrency

# Some parameters that can be adjusted by the user
DECAY_RATE = 0.99
LEARNING_RATE = 1e-5
K = 100

# Some constant used in the code
NUM_ACTIONS = 6

# Number of frames that will be used in the network each time
NUM_FRAMES = 3 # DQN
# NUM_FRAMES = 1 # DRQN

# Define the complete deep Q network
class DeepQ(object):

    def __init__(self):
        self.construct_q_network()

    '''Construct the desired deep q network (DQN)'''
    def construct_q_network(self):
        self.model = Sequential()
        self.model.add(Convolution2D(32, (8, 8), subsample = (4, 4), activation = 'relu', input_shape = (84, 84, NUM_FRAMES)))
        self.model.add(Convolution2D(64, (4, 4), subsample = (2, 2), activation = 'relu'))
        self.model.add(Convolution2D(64, (3, 3), activation = 'relu'))
        self.model.add(Flatten())
        self.model.add(Dense(512, activation = 'relu'))
        self.model.add(Dense(NUM_ACTIONS))
        self.model.compile(loss = 'mse', optimizer = Adam(lr = LEARNING_RATE))
        # The loss function is MSE, actually we can define some new loss function in the keras package, such DQfD
        # We can also use other optimizer, such as sgd

        # Construct the target network which has the same structure as the desired network
        self.target_model = Sequential()
        self.target_model.add(Convolution2D(32, (8, 8), subsample = (4, 4), activation = 'relu', input_shape = (84, 84, NUM_FRAMES)))
        self.target_model.add(Convolution2D(64, (4, 4), subsample = (2, 2), activation = 'relu'))
        self.target_model.add(Convolution2D(64, (3, 3), activation = 'relu'))
        self.target_model.add(Flatten())
        self.target_model.add(Dense(512, activation = 'relu'))
        self.target_model.add(Dense(NUM_ACTIONS))
        self.target_model.compile(loss = 'mse', optimizer = Adam(lr = LEARNING_RATE))
        self.target_model.set_weights(self.model.get_weights())
        # Copy the weights in the desired network to the target network

        print('Deep Q Network has been constructed.')

    ''' Construct the desired deep recurrent q network (DRQN)'''
    '''
        def construct_q_network(self):
        self.model = Sequential()
        self.model.add(TimeDistributed(Convolution2D(32, 8, 8, subsample = (4, 4), activation = 'relu'), input_shape = (84, 84, NUM_FRAMES)))
        self.model.add(TimeDistributed(Convolution2D(64, 4, 4, subsample = (2, 2), activation = 'relu')))
        self.model.add(TimeDistributed(Convolution2D(64, 3, 3, activation = 'relu')))
        self.model.add(Flatten())
        # Change the dense layer to LSTM (use all traces to train)
        # self.model.add(LSTM(512, return_sequences = True))
        # Change the dense layer to LSTM (use the last trace to train)
        self.model.add(LSTM(512, return_sequences = 'tanh'))
        # self.model.add(Dense(512, activation = 'relu'))
        self.model.add(Dense(NUM_ACTIONS, activation = 'linear'))
        self.model.compile(loss = 'mse', optimizer = Adam(lr = LEARNING_RATE))
        # The loss function is MSE, actually we can define some new loss function in the keras package, such DQfD
        # We can also use other optimizer, such as sgd


        # Construct the target network which has the same structure as the desired network
        self.target_model = Sequential()
        self.target_model.add(TimeDistributed(Convolution2D(32, 8, 8, subsample = (4, 4), activation = 'relu'), input_shape = (84, 84, NUM_FRAMES)))
        self.target_model.add(TimeDistributed(Convolution2D(64, 4, 4, subsample = (2, 2), activation = 'relu')))
        self.target_model.add(TimeDistributed(Convolution2D(64, 3, 3, activation = 'relu')))
        self.target_model.add(Flatten())
        # Change the dense layer to LSTM (use all traces to train)
        # self.model.add(LSTM(512, return_sequences = True))
        # Change the dense layer to LSTM (use the last trace to train)
        self.model.add(LSTM(512, return_sequences = 'tanh'))
        #self.target_model.add(Dense(512, activation = 'relu'))
        self.target_model.add(Dense(NUM_ACTIONS, activation = 'linear'))
        self.target_model.compile(loss = 'mse', optimizer = Adam(lr = LEARNING_RATE))
        self.target_model.set_weights(self.model.get_weights())
        # Copy the weights in the desired network to the target network

        print('Deep Recurrent Q Network has been constructed.')
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




