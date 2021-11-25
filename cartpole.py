import numpy as np
import gym
from collections import deque 
from keras.models import Sequential
from keras.layers import Dense

import tensorflow as tf
import random



class agent():
    def __init__(self,env):
        self.state_size =env.observation_space.shape[0]
        self.action_size=env.action_space.n
        
        self.gama = 0.95
        self.learning_rate = 0.001
        
        self.epsilon = 1
        self.epsilon_decay = 0.995 #her episode da epsilon degeri ile carpilacak
        self.epsilon_min=0.01
        
        self.memory = deque(maxlen = 1000)
        
        self.model=self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(48, input_dim = self.state_size, activation = "tanh"))
        model.add(Dense(self.action_size,activation = "linear"))
        model.compile(loss = "mse", optimizer =tf.keras.optimizers.Adam(lr = self.learning_rate))
        return model
    
    def remember(self,state,action,reward,next_state,done): 
        self.memory.append(state,action,reward,next_state,done)
    
    def action(self,state):
        if random.uniform(0,1) <=self.epsilon:
            return env.action_space.sample()
        else:
            act_values=self.model.predict(state)
            return np.argmax(act_values[0])
    
    def replay(self,batch_size):
        if len(self.memory) <batch_size:
            return
        else:
            minibatch = random.sample(self.memory,batch_size)
            for state,action,reward,next_state,done in minibatch:
                if done:
                    target = reward
                else:
                    target= reward +self.gama*np.argmax(self.model.predict(next_state)[0])
                    train_target = self.model.predict(state)
                    train_target[0][action]=target
                    self.model.fit(state,train_target, verbose = 0)
                    
                    
    def epsilon(self):
        if self.epsilon >self.epsilon_min:
            self.epsilon *=self.epsilon_decay
if __name__=="__main__":
    #env ve agenti yap
    env=gym.make("CartPole-v0")
    agent=agent(env)
    state=env.reset
    episodes=100
    batch_size=16
    for e in range(episodes):
        
        state=env.reset()
        state=np.reshape(state,[1,4])
        time=0
        
        while True:
            #act
            action=agent.action(state)
            
            #step
            next_state,reward,done,_ = env.step(action)
            next_state=np.reshape(next_state,[1,4])
            
            # remember
            agent.remember(state,action,reward,next_state,done)
      
            # update state
           
            state=next_state
            #replay
           
            agent.replay(batch_size)
            #epsilon
           
            agent.epsilon()
            time+=1
            if done:
                print("Episode:{},time:{}".format(e,time))
                break
