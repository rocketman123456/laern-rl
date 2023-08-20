import gymnasium as gym

# env = gym.make('MountainCar-v0', render_mode="human")
env = gym.make("LunarLander-v2", render_mode="human")

print('观测空间 = {}'.format(env.observation_space))
print('动作空间 = {}'.format(env.action_space))
print('观测范围 = {} ~ {}'.format(env.observation_space.low, env.observation_space.high))
print('动作数 = {}'.format(env.action_space.n))

observation, info = env.reset(seed=42)
for _ in range(1000):
    action = env.action_space.sample()  # this is where you would insert your policy
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
