import gymnasium as gym
import numpy as np

class NStepSarsa():
    def __init__(
            self,
            env,
            n_steps=4,
            epsilon=1.0,
            epsilon_decay=0.9995,
            epsilon_min=0.01,
            alpha=0.4,
            gamma=0.9,
            is_training=True,
            q_table_path=None,
        ):
        self.n = n_steps

        self.env = env
        self.is_training = is_training

        if self.is_training:
            self.Q = np.zeros((env.observation_space.n, env.action_space.n)) 
            
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay
            self.epsilon_min = epsilon_min

            self.alpha = alpha
            self.gamma = gamma
        else:
            self.Q = np.load(q_table_path)

    def get_action(self, state):
        if self.is_training and np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[state])
        return action

    def exploration_decrease(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

   
    def test(self, episodes):
        for ep in range(episodes):
            state, _ = self.env.reset()

            done = False
            while not done:
                action = self.get_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                state = next_state

    def train(self, episodes):
        ep_return = 0.0
        for i in range(1, episodes + 1):
            state, _ = self.env.reset()
            action = self.get_action(state)

            states = [state]
            actions = [action]
            rewards = [0.0]

            T = float('inf')
            t = 0

            while True:
                if t < T:
                    next_state, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated

                    ep_return += reward

                    states.append(next_state)
                    rewards.append(reward)

                    if done:
                        T = t + 1
                        next_action = None
                    else:
                        next_action = self.get_action(next_state)
                        actions.append(next_action)

                tau = t - self.n + 1
                if tau >= 0:
                    G = 0.0
                    for j in range(tau + 1, min(tau + self.n, T) + 1):
                        G += (self.gamma**(j - tau - 1)) * rewards[j]
                    if tau + self.n < T:
                        s_boot = states[tau + self.n]
                        a_boot = actions[tau + self.n]
                        G += (self.gamma**self.n) * self.Q[s_boot, a_boot]

                    s_tau = states[tau]
                    a_tau = actions[tau]
                    self.Q[s_tau, a_tau] += self.alpha * (G - self.Q[s_tau, a_tau])

                if tau == T - 1:
                    break

                t+=1
                state, action = next_state, next_action

            self.exploration_decrease()

            if (i + 1) % 100000 == 0:
                print(f"Episode {i+1}, Epsilon: {self.epsilon:.3f}")
                print("total return:", ep_return)
                ep_return = 0.0

    def save(self, path):
        np.save(path, self.Q)
    
