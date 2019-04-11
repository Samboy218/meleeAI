import melee
import math
#import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
from collections import deque

class BasicLearner:
    def __init__(self, action_size, train_freq):
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.start = True
        self.prev_action = 0
        self.prev_state = np.zeros(8)
        #in-game percent of our agent
        self.prev_pct = 0
        self.prev_opp_pct = 0
        self.train_freq = train_freq
        self.curr_frame = 0

    def build_model(self):
        model = Sequential()
        #8 inputs because projectile lists are length 6 and we also need player x and y
        model.add(Dense(24, input_dim=8, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.array([state]))
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.array([next_state])
                #print(next_state)
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            state = np.array([state])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    #this will be called at every frame.
    def doit(self, gamestate):
        state = self.getState(gamestate)
        if not self.start:
            #if we aren't on the first frame, we must've done something last frame
            #next_state = self.getState(gamestate)
            reward = self.calcReward(gamestate)
            #disincentivize spamming shine
            #if self.prev_action != 0:
                #reward -= 1
            done = False
            self.remember(self.prev_state, self.prev_action, reward, state, done)
            #print(reward)
        else:
            self.start = False

        self.prev_action = self.act(state)
        self.prev_state = state
        if (self.curr_frame%self.train_freq == 0 and len(self.memory) > 32):
            self.replay(32)
        self.curr_frame += 1
        return self.prev_action

    #process the gamestate and get the closest projectile as well as the player x,y
    def getState(self, gamestate):
        x = np.float64(gamestate.ai_state.x)
        y = np.float64(gamestate.ai_state.y)
        if (len(gamestate.projectiles) > 0):
            closest = min(gamestate.projectiles, key=lambda p: math.sqrt((p.y-y)**2 + (p.x-x)**2))
            proj = [np.float64(x) for x in closest.tolist()]
        else:
            proj = [np.float64(0) for x in range(6)]
        state = proj
        state.append(x)
        state.append(y)
        return np.array(state, dtype=np.float64)

    #process the current state and the prev state to get the reward (percentage diff)
    def calcReward(self, gamestate):
        opp_pct = gamestate.opponent_state.percent
        pct = gamestate.ai_state.percent
        reward = (opp_pct - self.prev_opp_pct) * 3
        reward += self.prev_pct - pct
        self.prev_pct = pct
        self.prev_opp_pct = opp_pct
        #if gamestate.ai_state.action == melee.enums.Action.SHIELD_REFLECT:
        #    print("reflect")
        #    reward += 10
        return reward

    #frame 1 is okay, frame 7 is okay, frame 15 is okay
    def shine(self, gamestate, controller):
        if self.shine1 != -1:
            self.shine1 = self.curr_frame
            controller.press_button(melee.enums.Button.BUTTON_B)
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, .5, 0)
            return
        #if we're here, then we're in the middle of a multishine
        if self.curr_frame - self.shine1 == 7:
            #do shine 2
            self.shine2 = self.curr_frame
            self.shine1 = -1
            controller.press_button(melee.enums.Button.BUTTON_B)
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, .5, 0)
            return
        if self.curr_frame - self.shine2 == 8:
            #shine 3, and then we are basically at shine2 again
            self.shine2 = self.curr_frame
            self.shine1 = -1
            controller.press_button(melee.enums.Button.BUTTON_B)
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, .5, 0)
            return
        if self.curr_frame - self.shine2 > 8:
            #we've passed the multishine period
            self.shine1 = self.curr_frame
            controller.press_button(melee.enums.Button.BUTTON_B)
            controller.tilt_analog(melee.enums.Button.BUTTON_MAIN, .5, 0)
            return
        #we're in the middle of a shine, dont do anything
        controller.empty_input()




