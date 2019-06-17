import gym
env = gym.make('CartPole-v0')

print(env.action_space.n) # 查看这个环境中可用的 action 有多少个
print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
print(env.observation_space.high)   # 查看 observation 最高取值
print(env.observation_space.low)    # 查看 observation 最低取值

observation = env.reset()
print(observation)
for t in range(100):
    env.render()
    print(observation)
    action = env.action_space.sample()
    # print(action)
    observation, reward, done, info = env.step(0)
    print("reward:", reward, "info", info)
    if done:
        print("Episode finished after {} timesteps".format(t+1))
        print("")
        break
env.close()