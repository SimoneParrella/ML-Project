from dataclasses import dataclass
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import torch.optim as optim
import gymnasium as gym
import seaborn as sns
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import random 
import torch
import time
import math
import os

@dataclass
class DQNConfig:
    #ambiente e run 
    env_id: str = "Taxi-v3"                 # id dell'ambiente Gymnasium
    seed: int = 42                          # seme per random/pytorch/np (riproducibilità stesso seed = stessi risultati)
    episodes: int = 10000                   # numero di episodi di training
    max_steps: int = 200                    # numero massimo di step per ogni episodio
    #RL
    gamma: float = 0.99                     # discount factor (propaga ricompense future)
    #esplorazione ε-greedy (per-episodio)
    eps_start: float = 1.0                  # ε iniziale (esplorazione piena)
    eps_end: float = 0.05                   # ε finale (poca esplorazione a regime)
    eps_decay: float = 0.995                # decadimento moltiplicativo per episodio
    #ottimizzazione
    lr: float = 1e-3                        # learning rate Adam
    grad_clip: float = 10.0                 # clip del gradiente
    #replay buffer e  batch
    buffer_capacity: int = 10_000           # dimensione massima della memoria
    batch_size: int = 32                    # dimensione del mini batch
    learn_start: int = 2_000                # numero minimo di transizioni prima di iniziare ad aggiornare
    updates_per_step: int = 1               # quante ottimizzazioni per ogni step ambiente
    #target network DQN
    use_target_network: bool = True         # usa una rete target per il target TD
    target_update_every: int = 1_000        # hard update della target ogni N gradient steps
    #logging
    log_every: int = 500                    # log ogni 500 episodi

def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class ReplayBuffer:
    """Coda circolare di transizioni (s, a, r, s', done) per DQN"""
    def __init__(self, capacity: int):
        self.capacity = int(capacity)           # dimensione massima FIFO
        self.buf = deque(maxlen=self.capacity)  # quando è piena, i più vecchi escono

    def __len__(self) -> int:
        return len(self.buf)                    #per sapere quanti transizioni abbiamo
    
    def push(self, s: int, a: int, r: float, ns: int, done: bool):
        """Aggiunge una transizione nel buffer"""
        self.buf.append((s, a, r, ns, done))

    def sample(self, batch_size: int, device: torch.device):
        """Estrae un mini-batch casuale e lo converte in tensori PyTorch sul device scelto
           Ritorna: (s_t [B], a_t [B,1], r_t [B], ns_t [B], d_t[B])
        """
        batch = random.sample(self.buf, batch_size)     # campionamento senza rimpiazzo
        s, a, r, ns, d = zip(*batch)                    # unzip in 5 tuple

        s_t = torch.tensor(s, dtype=torch.long, device=device)                  # indici stato
        a_t = torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1)     # [B,1] per gather
        r_t = torch.tensor(r, dtype=torch.float32, device=device)               # reward
        ns_t = torch.tensor(ns, dtype=torch.long, device=device)                # prossimo stato
        d_t = torch.tensor(d, dtype=torch.float32, device=device)               # 1.0 se done altrimenti 0.0

        return s_t, a_t, r_t, ns_t, d_t
    
class QNet(nn.Module):
    """
    MLP per Taxi-v3:
        input: indice di stato (0..499) -> one-hot(500)
        hidden: 2 layer da 32 neuroni con ReLU
        output: 6 Q-valori (una per azione)
    """
    def __init__(self, n_states: int, n_actions: int, hid: int = 32):
        super().__init__()
        self.n_states = n_states

        #strato 1: 500 -> 32
        self.fc1 = nn.Linear(n_states, hid)
        #strato 2: 32 -> 32ù
        self.fc2 = nn.Linear(hid, hid)
        #Uscita: 32 -> 6(Q(s,·))
        self.out = nn.Linear(hid, n_actions)

        #Inizializzazione consigliata per stabilità ReLU
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        #Uscita con pesi piccoli
        nn.init.uniform_(self.out.weight, -0.003, 0.003)

    def forward(self, s_index: torch.Tensor) -> torch.Tensor:
        """
        s_index: tensor di shape [B] con interi (0..n_states-1)
        ritorna: tensor [B, n_actions] con i Q-valori per ogni azione
        """
        #1) One-hot dello stato: [B] -> [B, n_states]
        oh = F.one_hot(s_index.long(), num_classes=self.n_states).float()

        #2) Passaggi MLP + ReLU
        x = F.relu(self.fc1(oh))
        x = F.relu(self.fc2(x))

        # 3) Q(s,·) = out(x)  -> [B, n_actions]
        q = self.out(x)
        return q
    
def select_action(qnet, state: int, eps: float, n_actions: int, device: torch.device) -> int:
    """ε-greedy: con prob. eps azione random, altrimenti argmax(Q) con tie-break casuale."""
    if random.random() < eps:
        return random.randrange(n_actions)
    with torch.no_grad():
        q = qnet(torch.tensor([state], device=device))  # [1, n_actions]
        q = q.squeeze(0).cpu().numpy()                  # [n_actions]
    m = q.max()
    cand = np.flatnonzero(q >= m - 1e-8)               # tutte le azioni a massimo ~uguale
    return int(np.random.choice(cand))                 # tie-break casuale


def train_dqn(cfg: DQNConfig):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Ambiente & dimensioni
    env = gym.make(cfg.env_id, render_mode="ansi")
    nS = env.observation_space.n
    nA = env.action_space.n

    #Reti: online (qnet) e target (opzionale, DQN "standard")
    qnet = QNet(n_states=nS, n_actions=nA, hid=32).to(device)
    target = QNet(n_states=nS, n_actions=nA, hid=32).to(device)
    target.load_state_dict(qnet.state_dict())
    target.eval()

    #Ottimizzazione
    opt = optim.Adam(qnet.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    #Replay buffer
    rb = ReplayBuffer(cfg.buffer_capacity)

    #Schedules & logging
    eps = cfg.eps_start
    ep_returns = []
    succ_in_window = 0
    global_updates = 0
    loss_log = []       # loss per ogni gradient update
    eps_log  = []       # epsilon per episodio
    succ_log = []       # 1 se episodio chiuso con +20, altrimenti 0


    #TRAINING LOOP
    for ep in range(cfg.episodes):
        s, info = env.reset(seed=cfg.seed + ep)
        ep_ret = 0.0
        ep_succeeded = False

        for t in range(cfg.max_steps):
            # 1) selezione azione ε-greedy
            a = select_action(qnet, s, eps, nA, device)

            # 2) step ambiente
            ns, r, terminated, truncated, info = env.step(a)
            done = bool(terminated or truncated)
            ep_ret += r

            # 3) memorizza transizione nel replay buffer
            rb.push(s, a, r, ns, done)

            # 4) ottimizzazione se abbiamo abbastanza esperienza
            if len(rb) >= cfg.learn_start:
                for _ in range(cfg.updates_per_step):
                    # 4a) sample mini-batch
                    s_b, a_b, r_b, ns_b, d_b = rb.sample(cfg.batch_size, device)

                    # 4b) Q(s,a) online
                    q_sa = qnet(s_b).gather(1, a_b).squeeze(1)   # [B]

                    # 4c) target y = r + gamma*(1-done)*max_a' Q_target(ns, a')
                    with torch.no_grad():
                        q_next = (target if cfg.use_target_network else qnet)(ns_b).max(1).values  # [B]
                        y = r_b + cfg.gamma * (1.0 - d_b) * q_next                             # [B]

                    # 4d) loss, backward, step
                    loss = mse(q_sa, y)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(qnet.parameters(), cfg.grad_clip)
                    opt.step()
                    loss_log.append(float(loss.item()))

                    # 4e) hard update della target ogni N gradient steps
                    global_updates += 1
                    if cfg.use_target_network and (global_updates % cfg.target_update_every == 0):
                        target.load_state_dict(qnet.state_dict())

            # 5) termina episodio
            if done:
                if r == 20:
                    succ_in_window += 1
                    ep_succeeded = True
                break

            s = ns

        # 6) logging per episodio
        ep_returns.append(ep_ret)
        eps = max(cfg.eps_end, eps * cfg.eps_decay)  # decay per-episodio
        
        eps_log.append(float(eps))                          
        succ_log.append(1 if ep_succeeded else 0) 

        if (ep + 1) % cfg.log_every == 0:
            window = ep_returns[-cfg.log_every:]
            meanR = float(np.mean(window)) if window else float("nan")
            print(
                f"[DQN std] ep {ep+1}/{cfg.episodes} | meanR@{cfg.log_every}={meanR:.2f} "
                f"| lastR={ep_ret:.1f} | eps={eps:.3f} | succ@{cfg.log_every}={succ_in_window}",
                flush=True
            )
            succ_in_window = 0

    env.close()
    return qnet, ep_returns, device, {
        "loss_log": loss_log,
        "eps_log":  eps_log,
        "succ_log": succ_log,
    }



def evaluate_greedy(env_id: str, qnet: QNet, device: torch.device,
                    episodes: int = 200, max_steps: int = 200):
    """
    Esegue policy greedy (ε=0) per 'episodes' episodi e ritorna:
      - success_rate: frazione di episodi chiusi con il +20 finale
      - mean_return: ritorno medio per episodio
    """
    import gymnasium as gym
    env = gym.make(env_id, render_mode="ansi")

    qnet.eval()
    succ = 0
    total_ret = 0.0

    for _ in range(episodes):
        s, info = env.reset()
        ep_ret = 0.0
        for t in range(max_steps):
            with torch.no_grad():
                q = qnet(torch.tensor([s], device=device))  # [1,6]
                a = int(q.argmax(dim=1).item())
            ns, r, terminated, truncated, _ = env.step(a)
            ep_ret += r
            if terminated or truncated:
                if r == 20:
                    succ += 1
                break
            s = ns
        total_ret += ep_ret

    env.close()
    return {
        "success_rate": succ / max(1, episodes),
        "mean_return": total_ret / max(1, episodes),
    }

def moving_avg(x, w):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < w: 
        return np.array([])
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x, kernel, mode="valid")
def plot_learning_curves(ep_returns, eps_log, loss_log, succ_log, window=500):
    # 1) Return per episodio + media mobile
    ma = moving_avg(ep_returns, window)
    plt.figure(); plt.plot(ep_returns, alpha=0.4, label="Return/ep")
    if len(ma) > 0:
        plt.plot(range(window-1, window-1+len(ma)), ma, label=f"MA({window})")
    plt.xlabel("Episode"); plt.ylabel("Return"); plt.title("Learning curve (returns)"); plt.legend(); plt.grid(True)

    # 2) Success rate su finestra (percento)
    succ_curve = moving_avg(np.array(succ_log, dtype=np.float32), window) * 100.0
    if len(succ_curve) > 0:
        plt.figure()
        plt.plot(range(window-1, window-1+len(succ_curve)), succ_curve)
        plt.xlabel("Episode"); plt.ylabel("Success rate [%]")
        plt.title(f"Success rate (MA {window})"); plt.grid(True)

    # 3) Epsilon per episodio
    plt.figure(); plt.plot(eps_log)
    plt.xlabel("Episode"); plt.ylabel("epsilon"); plt.title("Epsilon schedule"); plt.grid(True)

    # 4) Loss per update (smoothed)
    if len(loss_log) > 0:
        lma = moving_avg(loss_log, max(1, window//5))
        plt.figure(); 
        plt.plot(loss_log, alpha=0.3, label="loss"); 
        if len(lma)>0: plt.plot(lma, label=f"MA({max(1, window//5)})")
        plt.xlabel("Update step"); plt.ylabel("Loss (MSE)"); plt.title("DQN loss"); plt.legend(); plt.grid(True)

def plot_q_regression_diagnostics(qnet, device, rb, batch_size=512, gamma=0.99, use_target=False, target=None):
    """Scatter Q_pred vs TD-target (y) e istogramma dei residui (y - Q_pred), come nel tutorial."""
    if len(rb) < batch_size:
        print("ReplayBuffer troppo piccolo per diagnostica.")
        return
    # sample
    s_b, a_b, r_b, ns_b, d_b = rb.sample(batch_size, device)
    q_pred = qnet(s_b).gather(1, a_b).squeeze(1).detach().cpu().numpy()
    with torch.no_grad():
        net_for_target = (target if (use_target and target is not None) else qnet)
        q_next = net_for_target(ns_b).max(1).values
        y = (r_b + gamma * (1.0 - d_b) * q_next).cpu().numpy()

    # Scatter "True (y) vs Predictions (Q_pred)" + diagonale
    lims = [float(min(y.min(), q_pred.min())), float(max(y.max(), q_pred.max()))]
    plt.figure()
    plt.scatter(y, q_pred, s=10)
    plt.plot(lims, lims)  # diagonale
    plt.xlabel("TD Target y")
    plt.ylabel("Q_pred(s,a)")
    plt.title("Q_pred vs TD Target (batch)")
    plt.grid(True)

    # Istogramma dei residui (y - Q_pred)
    plt.figure()
    err = y - q_pred
    plt.hist(err, bins=25)
    plt.xlabel("Residual (y - Q_pred)")
    plt.ylabel("Count")
    plt.title("Residual distribution (batch)")
    plt.grid(True)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=None, help="batch_size (es. 16/32/64/128)")
    ap.add_argument("--buffer", type=int, default=None, help="buffer_capacity (es. 2000/5000/10000/50000)")
    ap.add_argument("--tag", type=str, default=None, help="nome cartella per salvare i risultati in runs/<tag>")
    ap.add_argument("--episodes", type=int, default=None, help="override episodi (facoltativo)")
    args = ap.parse_args()

    cfg = DQNConfig()
    if args.batch is not None:
        cfg.batch_size = args.batch
    if args.buffer is not None:
        cfg.buffer_capacity = args.buffer
    if args.episodes is not None:
        cfg.episodes = args.episodes
    # regola pratica: aspetta un po' prima di imparare se il batch cresce
    cfg.learn_start = max(cfg.learn_start, 5 * cfg.batch_size)

    # TRAIN
    qnet, ep_returns, device, logs = train_dqn(cfg)

    # Salvataggi base (+ grafici se vuoi)
    if args.tag:
        out = os.path.join("runs", args.tag)
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "returns.npy"),  np.array(ep_returns, dtype=np.float32))
        np.save(os.path.join(out, "succ_log.npy"), np.array(logs["succ_log"], dtype=np.int8))
        np.save(os.path.join(out, "eps_log.npy"),  np.array(logs["eps_log"],  dtype=np.float32))
        torch.save(qnet.state_dict(), os.path.join(out, "model.pt"))
        with open(os.path.join(out, "meta.txt"), "w", encoding="utf-8") as f:
            f.write(str(cfg) + "\n")
        print(f"[OK] Risultati salvati in {out}")

    # Grafici 
    plot_learning_curves(ep_returns, logs["eps_log"], logs["loss_log"], logs["succ_log"], window=500)
    metrics = evaluate_greedy(cfg.env_id, qnet, device=device, episodes=200, max_steps=200)
    print("Eval (greedy):", metrics)
    plt.show()





    








