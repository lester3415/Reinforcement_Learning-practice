import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import matplotlib.pyplot as plt
import time

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # 學習率
EPSILON = 0.9               # greedy policy 貪婪
GAMMA = 0.9                 # y 衰減率
TARGET_REPLACE_ITER = 100   # 神經網路更新頻率
MEMORY_CAPACITY = 2000      # 記憶容量
env = gym.make('CartPole-v0')   # 匯入遊戲並取的環境
env = env.unwrapped
N_ACTIONS = env.action_space.n  # 竿子的動作總數  2
N_STATES = env.observation_space.shape[0]   # 竿子狀態變數總數 = 竿子上下端x,y軸 = 4

# using GPU
print("GPU is ->",torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

# pytorch 神經網路
class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)      # 輸入狀態
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)     # 輸出動作
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)       # 激勵函數
        actions_value = self.out(x)     # 取出action
        return actions_value # [1,2] 

# Double DQN
class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)

        self.learn_step_counter = 0                                     # target更新counter
        self.memory_counter = 0                                         # 記憶庫counter
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # 初始化記憶庫，現在狀態、下次狀態 + action + reward
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)        # 設定優化器，使用Adam
        self.loss_func = nn.MSELoss()       # 選擇損失函數

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy 90%機率選擇最優解
            actions_value = self.eval_net.forward(x)        # 取出pytorch的tensor
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()      # 選擇reward較大的
            action = action[0]  # return the argmax index，取出數字 0或1
        else:   # random 10%機率隨機
            action = np.random.randint(0, N_ACTIONS)        # 0或1 取隨機整數
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))     # 按水平(按列順序)堆疊
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY       # 容量2000筆
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:      # 神經網路每學習100次，就更新目標網路
            self.target_net.load_state_dict(self.eval_net.state_dict())     # 將網路複製過去
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)        # 從2000筆中隨機選32筆
        b_memory = self.memory[sample_index, :]     # 從記憶庫中讀取

        # 將state action reward state'數值取出
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).cuda()
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).cuda()
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).cuda()
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).cuda()

        # 從過去的action選出q_eval
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # 不反向傳遞誤差
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # 根據Q-learning公式實現
        loss = self.loss_func(q_eval, q_target)     # 帶入損失函數

        # 神經網路速度優化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn = DQN()

print('\nCollecting experience...')
rewards = []
for i_episode in range(400):
    s = env.reset()     # 獲取每回合開始的第一個state
    episode_r = 0       # episode reward
    while True:
        env.render()        # 遊戲環境重置
        a = dqn.choose_action(s)            # 帶入state，獲取action

        s_, r, done, info = env.step(a)     # 執行action，獲取行動結果

        # 修改reward
        x, x_dot, theta, theta_dot = s_
        # 竿子越偏向水平reward越小
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        # 車子越偏向旁邊reward越小
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5

        r = r1 + r2

        dqn.store_transition(s, a, r, s_)       # 儲存結果到神經網路

        episode_r += r      # 每回合reward累計
        if dqn.memory_counter > MEMORY_CAPACITY:        # 第一批空間存滿後開始學習
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(episode_r, 2))

        if done:        # 失敗，此回合結束
            rewards.append(episode_r)       # 將每回合累積的reward放入list
            break
        s = s_      # 進入下一回合

# 取得現在時間並存成檔名
localtime = time.localtime()
timeString = time.strftime("%m%d%H", localtime)
timeString = 'CartPole/DQN./' + str(timeString) + '.jpg'

# 畫出圖表並儲存結果
plt.xlabel('episodes')
plt.ylabel('total rewads')
plt.plot(rewards)
plt.savefig(timeString)
plt.show()