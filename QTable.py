import gymnasium as gym
import numpy as np
import random 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
actions={
   0:"South",
   1:"North",
   2:"East",
   3:"West",
   4:"Pickup",
   5:"Dropoff"
}
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
      
def plot_results(rewards:np.ndarray):
   plt.plot(rewards)
   plt.ylabel("Rewards")
   plt.xlabel("Episode")
   plt.show()
   
def show_best_actions(env:gym.Env,Qtable):
      for o in range(env.observation_space.n):
       action=actions[np.argmax(Qtable[o])]
       print("State:",o,", Action:",action,", Q value:",Qtable[o][np.argmax(Qtable[o])])
      
def qlearn(env:gym.Env,
          alpha:float,
          gamma:float,
          episodes:int,
          steps:int,
          epsilon:float):
    
    Q=np.zeros([env.observation_space.n,env.action_space.n,])
    Rewards=np.zeros(episodes)
    eps=0.9
    policy=EpsGreedyPolicy(Q)
    first=True
    for i in range(episodes):  
     state,info=env.reset()
     #print("New episode")
     #print(env.render())
     for j in range(steps):
       action=policy(state,eps)
       prev=env.render()
       obs,rew,done,truncated,info=env.step(action)
       Rewards[i]+=rew
       Q[state][action]+= alpha*(rew+ gamma*np.max(Q[obs])-Q[state][action])  
       if(done or truncated):
        break
       state=obs
       #print(env.render())
     eps =max(0.05, eps * 0.995)
     if (i + 1) % 100 == 0:
            avg100 = Rewards[max(0, i - 99):i + 1].mean()
            print(f"[train][ep={i+1:5d}] avg_return(last100)={avg100:7.2f} eps={eps:6.3f}")
    return Q,Rewards
def rolllouts(env:gym.Env,
              policy,
              gamma:float,
              episodes:int,
              Render=False)-> float:
   sum=0.0
   state,info=env.reset()
   for e in range(episodes):
      print("New Episode")
      print(env.render)
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
   
if __name__=="__main__":   
 taxi_env= gym.make("Taxi-v3",render_mode="ansi")
 alpha=0.9
 gamma=0.97
 epsilon=0.9
 epsisodes=10000
 steps=100
 
 Qtable,Rewards=qlearn(taxi_env,alpha,gamma,epsisodes,steps,epsilon)
 show_best_actions(taxi_env,Qtable)
 plot_results(Rewards)
 ret=rolllouts(taxi_env,policy=GreedyPolicy(Qtable),gamma=gamma,episodes=100,Render=False)
 print("Res:",ret)  
 
