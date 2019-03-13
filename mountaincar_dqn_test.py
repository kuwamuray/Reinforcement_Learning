from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import *
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# import matplotlib as mpl
# mpl.use('agg')

import gym
import matplotlib.pyplot as plt
import numpy
import rl.callbacks
import time

from gym import wrappers

class RewardHistory(rl.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.observations = []
        self.rewards = []
    def on_step_end(self, step, logs={}):
        self.observations.append(logs.get("observation"))
        self.rewards.append(logs.get("reward"))

t1 = time.time()

env = gym.make('MountainCar-v0')
env = wrappers.Monitor(env, "/home/zzzzzz908/VIDEO")
nb_actions = env.action_space.n

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(48))
model.add(Activation('relu'))
model.add(Dense(48))
model.add(Activation('relu'))
model.add(Dense(48))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

memory = SequentialMemory(limit=50000, window_length=1)

policy = EpsGreedyQPolicy(eps=0.001)
dqn = DQNAgent(model=model, nb_actions=nb_actions,gamma=0.99, memory=memory, nb_steps_warmup=10, target_model_update=1e-2, policy=policy)

dqn.compile(Nadam(lr=1e-3, beta_1=0.9, beta_2=0.99, epsilon=1e-8, schedule_decay=1e-4), metrics=['mae'])

history = RewardHistory()
dqn.fit(env, nb_steps=50000, visualize=False, verbose=2, callbacks=[history])
O = history.observations
R = history.rewards

print(O[len(O)-1])
print(O[len(O)-2])
print(O[len(O)-3])
print(len(O))

P = numpy.array(O)[:,0].tolist()
print(len(P))
step_count = 0
OK = []
NG = []
OKNG = []

for i in range(len(P)):
    if P[i] < 0.5 :
        step_count += 1
    else :
        step_count += 1
        print("i = "+str(i)+" , step = "+str(step_count))
        OK.append(step_count)
        OKNG.append(step_count)
        step_count = 0
    if step_count == 200 :
        print("i = "+str(i)+" ### NOT CLEARED ###")
        NG.append(step_count)
        OKNG.append(step_count)
        step_count = 0

if len(OK) != 0 :
    print(min(OK))
print(OK)
print(NG)
print(sum(OK))
print(sum(NG))
print(OKNG)
print(len(OK))
print(len(OKNG))

t2 = time.time()
print(t2 - t1)

ok_1 = 0
ok_2 = 0
X = []
Y = []
AR_list = []
k1 = int(len(OKNG)/100)
k2 = int(len(OKNG)/10)

for i in range(len(OKNG)):
    if OKNG[i] != 200 :
        # print(str(i)+"th step is cleared.")
        ok_1 += 1
        ok_2 += 1
    if (i+1) % k1 == 0 :
        p = ok_1 / k1
        X.append(i+1)
        Y.append(p)
        ok_1 = 0
    if (i+1) % k2 == 0 :
        p = ok_2 / k2
        AR_list.append(p)
        ok_2 = 0
            
print(Y)
plt.plot(X,Y)
plt.savefig("NADAM_4126607.png")
print(AR_list)
