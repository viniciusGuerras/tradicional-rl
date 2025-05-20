from n_step_sarsa_agent import NStepSarsa
import gymnasium as gym

path = "q_tables/q_table_sarsa_n_step.npy"
env = gym.make("FrozenLake-v1", map_name="8x8", render_mode="human", is_slippery=False)
agent = NStepSarsa(env, n_steps=5, epsilon_decay=0.9999955, is_training=True, q_table_path=path)

agent.train(1_000_000)
agent.save(path)