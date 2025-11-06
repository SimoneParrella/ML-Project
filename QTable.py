import gymnasium as gym
import numpy as np
import random 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
@dataclass
class DQNConfig:
    env_id: str = "Taxi-v3"                                          
    episodes: int = 10000                  
    max_steps: int = 200                   
    gamma: float = 0.99                    
    #esplorazione ε-greedy (per-episodio)
    eps_start: float = 1.0                 
    eps_end: float = 0.05                   
    eps_decay: float = 0.995                # decadimento moltiplicativo per episodio
    
    alpha: float = 0.9                      # learning rate 
    actions={                               #descrizione azioni
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
def moving_avg(x, w):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < w: 
        return np.array([])
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x, kernel, mode="valid")     
def plot_results(rewards:np.ndarray,sucsses_rate,window=500):
    ma = moving_avg(rewards, window)
    plt.figure(); plt.plot(rewards, alpha=0.4, label="Return/ep")
    if len(ma) > 0:
        plt.plot(range(window-1, window-1+len(ma)), ma, label=f"MA({window})")
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.title("Learning curve (returns)"); plt.legend(); plt.grid(True)
    
    succ_curve = moving_avg(np.array(succ_log, dtype=np.float32), window) * 100.0
    if len(succ_curve) > 0:
        plt.figure()
        plt.plot(range(window-1, window-1+len(succ_curve)), succ_curve)
        plt.xlabel("Episode"); plt.ylabel("Success rate [%]")
        plt.title(f"Success rate (MA {window})"); plt.grid(True)
    plt.show()

   
def show_Qtable(env:gym.Env,Qtable):
      for o in range(env.observation_space.n):
       action=cfg.actions[np.argmax(Qtable[o])]
       print("State:",o,", Action:",action,", Q value:",Qtable[o][np.argmax(Qtable[o])])
      
def qlearn(cfg :DQNConfig):
    
    #Ambiente & dimensioni
    env = gym.make(cfg.env_id, render_mode="ansi")
    nS = env.observation_space.n
    nA = env.action_space.n
    #Creazione Qtable
    Q=np.zeros([nS,nA,])
    policy=EpsGreedyPolicy(Q)
    #Schedules & logging
    eps = cfg.eps_start
    ep_returns=np.zeros(cfg.episodes) 
    succ_log=[]                       # 1 se episodio chiuso con +20, altrimenti 0

    for i in range(cfg.episodes):  
     state,info=env.reset()
     ep_succeeded = False
     #print("New episode")
     #print(env.render())
     for j in range(cfg.max_steps):
       # 1) selezione azione ε-greedy
       action=policy(state,eps)

       # 2) step ambiente
       ns,r,terminated,truncated,info=env.step(action)
       done=terminated or truncated
       ep_returns[i]+=r                       #memorizza ricompensa

       # 3) memorizza valore nella Q-table
       Q[state][action]+= cfg.alpha*(r+ cfg.gamma*np.max(Q[ns])-Q[state][action])  
       # 4) termina episodio
      
       if(done):
            if r == 20:
             ep_succeeded = True
            break
       
       state=ns
    # 4) logging ogni 100 episodi
     eps = max(cfg.eps_end, eps * cfg.eps_decay)           # decay per-episodio
     succ_log.append(1 if ep_succeeded else 0)

     if (i + 1) % 100 == 0:
            avg100 = ep_returns[max(0, i - 99):i + 1].mean()
            print(f"[train][ep={i+1:5d}] avg_return(last100)={avg100:7.2f} eps={eps:6.3f}")
    return Q,ep_returns,succ_log
def evaluate_greedy(
              env_id,
              policy,
              gamma:float,
              episodes:int,
              Render=False)-> float:
   env= gym.make(env_id,render_mode="ansi")
   sum=0.0
   succ=0.0
   state,info=env.reset()

   for e in range(episodes):
      #print("New Episode")
      #print(env.render)
      state,info=env.reset()
      discount=1
      done=False
      while not done:
         action=policy(state)
         obs,r,terminated,truncated,info=env.step(action)
         sum+=r*discount
         discount*=gamma
         state=obs
         done=terminated or truncated
         if(done):
          if r == 20:
             succ+=1
          break
         #if Render:
          #print(env.render())   

   env.close()
   
   return sum/episodes,succ/episodes
   
if __name__=="__main__":   
 cfg = DQNConfig()
 Qtable,ep_returns,succ_log=qlearn(cfg)

 policy=GreedyPolicy(Qtable)           #policy ottimale
 #show_Qtable(taxi_env,Qtable)         #Stampa della Qtable

 #roullout
 returns,succs=evaluate_greedy(env_id=cfg.env_id,policy=policy,gamma=cfg.gamma,episodes=100,Render=False)
 
 #grafici
 plot_results(ep_returns,succ_log,window=500)
 print("Eval (greedy):","mean return:",returns,",success rate:",succs) 
 

