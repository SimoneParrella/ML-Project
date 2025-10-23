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
         return np.argmax(self.Q[obs])                      #scegli l'azione che massimizza la ricompensa per lo stato osservato
      else:
         return random.randint(0,self.n_actions-1)          #scegli un'azione casuale da quelle disponibili
         
def plot_results(rewards:np.ndarray):
   plt.plot(rewards)
   plt.ylabel("Rewards")
   plt.xlabel("Episode")
   plt.show()
   
def show_best_actions(env:gym.Env,Qtable):
      for o in range(env.observation_space.n):
       print(o,":",np.argmax(Qtable[o]))
         
def qlearn(env:gym.Env,
          alpha:float,                              #learning rate
          gamma:float,                              #discount factor
          episodes:int,
          steps:int):                               #numero massimo di azioni da compiere per episodio
    Q=np.zeros([env.observation_space.n,env.action_space.n,])   #inizializza la Qtable creandola come un dizionario vuoto
    Rewards=np.zeros(episodes)                                  #inizializza la lista delle ricompense totali degli episodi
    eps=0.9                                                     #probabilità di scegliere una RandomPolicy(esplorazione) invece che di una GreedyPolicy(massimizare la ricompensa)
    policy=EpsGreedyPolicy(Q)                                   #la policy viene definita come una classe che prende in input la qtable
    for i in range(episodes):
     state,info=env.reset()                                    #all'inizio dell'episodio riavvia l'ambiente e ottieni lo stato
     for j in range(steps):                                    
       action=policy(state,eps)                                #scelgli l'azione per lo stato corrente in base alla policy               
       obs,rew,done,truncated,info=env.step(action)            #esegui l'azione sull'ambiente , ottieni il nuovo stato osservato e le ricompensa 
       Rewards[i]+=rew                                         #aggiungi la ricompensa alle altre ricompense dell'episodio    
       Q[state][action]+= alpha*(rew+ gamma*np.max(Q[obs])-Q[state][action])         #aggiungi l' ricompensa alla Qtable usando la Q funcrion
       if(done or truncated):                                 # se l'agente ha raggiunto uno stato obbiettivo o qualcosa è andato storto finisci l'episodio
          break
       state=obs                                              #continua l'episodio sul nuovo stadio osservato
     eps = max(0.1, eps * 0.9)                                #alla fine di ogni episodio decrementa il valore del parametra il prametro in modo da favorire di più la GreedyPolicy

    return Q,Rewards
def rolllouts(env:gym.Env,
              policy,                            #policy scelta
              gamma:float,                       #discount factor
              episodes:int,
              Render=False)-> float:             #flag per indicare se visualizzare l'ambiente o meno
   sum=0.0
   state,info=env.reset()                       
   discount=1
   if Render :                                   
      env.render()
   for e in range(episodes):
      print("New Episode")                       #riavvia l'ambiente all'inizio di ogni episodio
      state,info=env.reset()                    
      discount=1                                 #all'inizio dell'episodio lo sconto è nullo
      done=False 
      while not done:                            # Finche l'agente non ha raggiunto uno stato obbiettivo o qualcosa va  storto continua l'episodio
         action=policy(state)                    #scelgi l'azione secondo la policy
         obs,rew,terminated,truncated,info=env.step(action)    #esegui l'azione sull'ambiente , ottieni il nuovo stato osservato e le ricompensa 
         sum+=rew*discount                                     #somma la ricompensa scontata
         discount*=gamma                                       #aumenta lo sconto
         state=obs                                             #continua l'episodio sul nuovo stadio osservato                        
         done=terminated or truncated                          
         if Render:                  
          print(env.render())   
   return sum/episodes
   
if __name__=="__main__":   
 taxi_env= gym.make("Taxi-v3",render_mode="ansi")       #inizializa l'ambiente
 alpha=0.1                                              #learning rate
 gamma=0.95                                             #sconto
   
 qtable,Rewards=qlearn(taxi_env,alpha,gamma,5000,100)   #La Qtable viene definita, e le ricompense ottenute ad ogni episodio vengono tracciate
   
 show_best_actions(taxi_env,qtable)                     #Dopo il learning vengono mostrate le azioni migiore per ogni stato

 plot_results(Rewards)                                  #Grafico delle rimcompense totali ottenute ad ogni episodio 
                                                       
 ret=rolllouts(taxi_env,policy=GreedyPolicy(qtable),gamma=gamma,episodes=20,Render=True)  #metti alla prova la Qtable con un policy(in questo caso un GreedyPolicy) per ottenere la somma media delle ricompense
 print("Res:",ret) 
   
 
