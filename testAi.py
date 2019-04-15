import melee
from melee import Action
import math
#import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam
import random
from collections import deque
import copy

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
        self.prev_gamestate = FakeGamestate()
        self.train_freq = train_freq
        self.curr_frame = 0
        self.shine1 = 0
        self.shine2 = 0
        #0 = not in shine
        #1 = done shine 1
        #2 = done shine 2
        self.shine_state = 0
        self.reward_data = []
        self.q_data = []

    def build_model(self):
        model = Sequential()
        #8 inputs because projectile lists are length 6 and we also need player x and y
        model.add(Dense(24, input_dim=9, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(24, activation='relu'))
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
            reward = self.calcReward(gamestate)
            #try to disincentivise excessive shielding
            #if self.prev_action == 1:
                #reward -= .1
            done = False
            target = np.amax(self.model.predict(np.array([state]))[0])
            self.q_data.append(target)
            self.reward_data.append(reward)
            self.remember(self.prev_state, self.prev_action, reward, state, done)
        else:
            self.start = False

        self.prev_action = self.act(state)
        self.prev_state = state
        if (self.curr_frame%self.train_freq == 0 and len(self.memory) > 32):
            self.replay(32)
        if (self.curr_frame % (self.train_freq*20) == 0):
            self.model.save("states/ShotBlock{}".format(self.curr_frame))
            with open("data.csv", "a+") as f:
                for line in zip(self.reward_data, self.q_data):
                    f.write("{},{}\n".format(line[0], line[1]))
                self.reward_data = []
                self.q_data = []
        self.curr_frame += 1
        self.prev_gamestate.copy(gamestate)
        return self.prev_action

    def copy_gamestate(self, gamestate):
        #we really only need the projectiles
        self.prev_gamestate = copy.deepcopy(gamestate.projectiles)

    #process the gamestate and get the closest projectile as well as the player x,y
    def getState(self, gamestate):
        x = np.float64(gamestate.ai_state.x)
        y = np.float64(gamestate.ai_state.y)
        action = np.float64(gamestate.ai_state.action.value)
        if (len(gamestate.projectiles) > 0):
            closest = min(gamestate.projectiles, key=lambda p: math.sqrt((p.y-y)**2 + (p.x-x)**2))
            proj = [np.float64(x) for x in closest.tolist()]
        else:
            proj = [np.float64(0) for x in range(6)]
        state = proj
        state.append(x)
        state.append(y)
        state.append(action)
        return np.array(state, dtype=np.float64)

    #process the current state and the prev state to get the reward (percentage diff)
    def calcReward(self, gamestate):
        #filter out non-fox-laser projectiles
        projectiles = [p for p in gamestate.projectiles if p.subtype == melee.enums.ProjectileSubtype.FOX_LASER and p.x_speed != 0]
        old_projectiles = [p for p in self.prev_gamestate.projectiles if p[0].subtype == melee.enums.ProjectileSubtype.FOX_LASER and p[0].x_speed != 0]
        opp_pct = gamestate.opponent_state.percent
        pct = gamestate.ai_state.percent
        reward = (self.prev_gamestate.ai_state.percent - pct)*5
        '''
        reward = (opp_pct - self.prev_gamestate.opponent_state.percent)
        print(gamestate.ai_state.action)
        if gamestate.ai_state.action == Action.SHIELD_STUN:
            print("reflect")
            reward += 10
        if gamestate.ai_state.action in [Action.SHIELD_BREAK_FLY, Action.SHIELD_BREAK_FALL]:
            #print("stun")
            #reward -= 10
        '''
        #print("{}-old-{}:{}".format(self.curr_frame, len(old_projectiles), self.prev_gamestate))
        #print("{}-curr-{}:{}".format(self.curr_frame, len(projectiles), gamestate))

        #check if we blocked a shot
        #check if there was a projectile and now there isn't, and we haven't changed percent
        for proj1 in old_projectiles:
            found = False
            for proj2 in projectiles:
                if proj1[1] == id(proj2):
                    found = True
                    if proj1[0].x_speed != proj2.x_speed:
                        print("reflected")
                        reward += 100

            #for some reason this is not consistent
            #if an old projectile that was close to us can't be found, it must have been blocked or hit us
            #if not found and abs(gamestate.ai_state.x - proj1[0].x) < 10 and (self.prev_gamestate.ai_state.percent - pct) == 0:
            #    print("blocked")
            #    reward += 10

        '''
        if len(old_projectiles) > 0 or len(projectiles) > 0:
            #but maybe we blocked the shot, so check if the closest projectile has a negative velocity
            elif old_closest.x_speed == closest.x_speed*-1 and (self.prev_gamestate.ai_state.percent - pct) == 0:
                print("reflected")
                reward += 50
        '''
        return np.float64(reward)

class FakeGamestate:
    def __init__(self):
        self.projectiles = []
        self.ai_state = melee.gamestate.PlayerState()
        self.opponent_state = melee.gamestate.PlayerState()

    def copy(self, gamestate):
        self.projectiles = [(copy.copy(x), id(x)) for x in gamestate.projectiles]
        self.ai_state = copy.deepcopy(gamestate.ai_state)
        self.opponent_state = copy.deepcopy(gamestate.opponent_state)
