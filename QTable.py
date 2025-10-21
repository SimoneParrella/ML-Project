import gymnasium as gym
import numpy as np
import random 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
class RandomPolicy:
   def __init__(self,Qtable):
      self.n_actions=len(Qtable[0])
   def __call__(self, obs):
         return random.randint(0,self.n_actions-1)

class GreedyPolicy:
   def __init__(self,Qtable):
      self.Q=Qtable
   def __call__(self, obs):
      return np.argmax(self.Q[obs])
    
class EpsGreedyPolicy:
   def __init__(self,Qtable):
      self.Q=Qtable
      self.n_actions=len(Qtable[0])
   def __call__(self, obs,eps):
      greedy= random.random() > eps
      if greedy:
         return np.argmax(self.Q[obs])
      else:
         return random.randint(0,self.n_actions-1)
      
def qlearn(env:gym.Env,
          alpha:float,
          gamma:float,
          episodes:int,
          steps:int):
    Q=np.zeros([env.observation_space.n,env.action_space.n,])
    Rewards=np.zeros(episodes)
    eps=0.9
    policy=EpsGreedyPolicy(Q)
    for i in range(episodes):
     state,info=env.reset()
     for j in range(steps):
       action=policy(state,eps)
       obs,rew,done,truncated,info=env.step(action)
       Rewards[i]+=rew
       Q[state][action]+= alpha*(rew+ gamma*np.max(Q[obs])-Q[state][action])
       if(done or truncated):
          break
       state=obs
     eps = max(0.1, eps * 0.9)

    return Q,Rewards
def rolllouts(env:gym.Env,
              policy,
              gamma:float,
              episodes:int,
              Render=False)-> float:
   sum=0.0
   state,info=env.reset()
   discount=1
   if Render :
      env.render()
   for e in range(episodes):
      print("New Episode")
      state,info=env.reset()
      discount=1
      done=False
      while not done:
         action=policy(state)
         obs,rew,terminated,truncated,info=env.step(action)
         sum+=rew*discount
         discount*=gamma
         state=obs
         done=terminated or truncated
         if Render:
          print(env.render())   
   return sum/episodes
   
   
taxi_env= gym.make("Taxi-v3",render_mode="ansi")
qtable,Rewards=qlearn(taxi_env,0.1,0.95,40000,50)
for o in range(taxi_env.observation_space.n):
   print(o,":",np.argmax(qtable[o]))
x=np.array(Rewards)

plt.plot(x)
plt.show()
ret=rolllouts(taxi_env,policy=GreedyPolicy(qtable),gamma=0.95,episodes=20,Render=True)
print("Res:",ret)  