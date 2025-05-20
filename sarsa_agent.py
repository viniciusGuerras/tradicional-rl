import gymnasium as gym
import numpy as np

class Sarsa():
    def __init__(
            self,
            env,
            epsilon=1.0,
            epsilon_decay=0.9995,
            epsilon_min=0.01,
            alpha=0.4,
            gamma=0.9,
            is_training=True,
            q_table_path=None,
        ):

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
            self.action_value_table = np.load(q_table_path)

    def get_action(self, state):
        if self.is_training and np.random.random() < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = np.argmax(self.Q[state])
        return action

    def exploration_decrease(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train_step(self, state, action, reward, state_, action_, done):
        td_target = reward + (0 if done else self.gamma * self.Q[state_, action_]) 
        td_error = td_target - self.Q[state, action] 
        self.Q[state, action] += self.alpha * td_error
    
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
        for ep in range(episodes):
            state, _ = self.env.reset()
            action = self.get_action(state)

            done = False
            total_reward = 0
            while not done:
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_action = self.get_action(next_state)
                self.train_step(state, action, reward, next_state, next_action, done)

                state, action = next_state, next_action
                total_reward += reward

            self.exploration_decrease()

            if (ep + 1) % 100000 == 0:
                print(f"Episode {ep+1}, Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")

    def save(self, path):
        np.save(path, self.Q)
    
