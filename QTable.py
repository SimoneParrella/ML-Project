import gymnasium as gym
import numpy as np
import random 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim

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

class GreedyPolicyNet:
    def __init__(self, qnet: nn.Module):
        self.qnet = qnet.eval()
    def __call__(self, obs: int) -> int:
        with torch.no_grad():
            q = self.qnet(torch.tensor([obs]))
            return int(q.argmax(dim=1).item())

class QNet(nn.Module):
    def __init__(self, n_states=500, n_actions=6, hid=128):
        super().__init__()
        self.fc1 = nn.Linear(n_states, hid)
        self.fc2 = nn.Linear(hid, hid)
        self.out = nn.Linear(hid, n_actions)
    def forward(self, s_idx: torch.Tensor):
        oh = F.one_hot(s_idx.long(), num_classes=500).float()
        x = F.relu(self.fc1(oh))
        x = F.relu(self.fc2(x))
        return self.out(x)  # [B, n_actions]

         
def plot_results(rewards:np.ndarray):
   plt.plot(rewards)
   plt.ylabel("Rewards")
   plt.xlabel("Episode")
   plt.show()
   
def show_best_actions(env:gym.Env,Qtable):
      for o in range(env.observation_space.n):
       print(o,":",np.argmax(Qtable[o]))

def show_best_actions_net(env:gym.Env, qnet:nn.Module):
    qnet.eval()
    for o in range(env.observation_space.n):
        with torch.no_grad():
            a = int(qnet(torch.tensor([o])).argmax(1).item())
        print(o, ":", a)
         
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

def dqnlearn(env:gym.Env,
             alpha:float, gamma:float, episodes:int, steps:int):
    nS, nA = env.observation_space.n, env.action_space.n
    qnet = QNet(n_states=nS, n_actions=nA)
    opt  = optim.Adam(qnet.parameters(), lr=alpha)   # alpha = lr
    mse  = nn.MSELoss()
    Rewards = np.zeros(episodes, dtype=np.float32)

    eps, eps_min, eps_decay = 0.9, 0.1, 0.9

    def select_action(s:int, eps:float) -> int:
        if random.random() < eps:
            return random.randrange(nA)
        with torch.no_grad():
            return int(qnet(torch.tensor([s])).argmax(1).item())

    for i in range(episodes):
        s, info = env.reset()
        ep_ret = 0.0
        for _ in range(steps):
            a = select_action(s, eps)
            ns, r, term, trunc, info = env.step(a)
            done = term or trunc
            ep_ret += r

            # TD target: r + γ (1-done) max_a' Q(ns, a')
            s_t  = torch.tensor([s])
            ns_t = torch.tensor([ns])
            a_t  = torch.tensor([[a]])
            r_t  = torch.tensor([r], dtype=torch.float32)
            d_t  = torch.tensor([float(done)], dtype=torch.float32)

            q_sa = qnet(s_t).gather(1, a_t).squeeze(1)
            with torch.no_grad():
                q_next = qnet(ns_t).max(1).values     # stessa rete (monorete)
                target = r_t + gamma * (1.0 - d_t) * q_next

            loss = mse(q_sa, target)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(qnet.parameters(), 10.0)
            opt.step()

            if done: break
            s = ns

        Rewards[i] = ep_ret
        eps = max(eps_min, eps * eps_decay)

    return qnet, Rewards

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
   alpha = 1e-3                                           #learning rate
   gamma = 0.95                                           #sconto
   episodes = 500
   steps = 200

   USE_DQN = True   # False = tabellare; True = DQN monorete
   
   if not USE_DQN:
      qtable, Rewards = qlearn(taxi_env, 0.1, 0.95, 5000, 100)   #La Qtable viene definita, e le ricompense ottenute ad ogni episodio vengono tracciate
      show_best_actions(taxi_env, qtable)                        #Dopo il learning vengono mostrate le azioni migliori per ogni stato
      plot_results(Rewards)                                      #Grafico delle rimcompense totali ottenute ad ogni episodio
      ret = rolllouts(taxi_env, policy=GreedyPolicy(qtable),
                     gamma=0.95, episodes=20, Render=True)       #metti alla prova la Qtable con un policy(in questo caso un GreedyPolicy) per ottenere la somma media delle ricompense
      print("Res:", ret)
   else:
      qnet, Rewards = dqnlearn(taxi_env, alpha, gamma, episodes, steps)
      plot_results(Rewards)
      ret = rolllouts(taxi_env, policy=GreedyPolicyNet(qnet),
                     gamma=gamma, episodes=20, Render=True)
      print("Res (DQN):", ret)
      show_best_actions_net(taxi_env, qnet)
      torch.save(qnet.state_dict(), "dqn_taxi.pt")

   
 
