import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
import numpy as np


class PGNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        """ 初始化q网络，为全连接网络
            input_dim: 输入的特征数即环境的状态维度
            output_dim: 输出的动作维度
        """
        super(PGNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 输入层
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 输出层

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class PolicyGradient:

    def __init__(self, model, memory, cfg):
        self.gamma = cfg['gamma']
        self.device = torch.device(cfg['device'])
        self.memory = memory
        self.policy_net = model.to(self.device)
        self.optimizer = torch.optim.RMSprop(
            self.policy_net.parameters(), lr=cfg['lr'])

    def sample_action(self, state):

        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net(state)
        m = Bernoulli(probs)  # 伯努利分布
        action = m.sample()

        action = action.data.numpy().astype(int)[0]  # 转为标量
        return action

    def predict_action(self, state):

        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net(state)
        m = Bernoulli(probs)  # 伯努利分布
        action = m.sample()
        action = action.data.numpy().astype(int)[0]  # 转为标量
        return action

    def update(self):
        state_pool, action_pool, reward_pool = self.memory.sample()
        state_pool, action_pool, reward_pool = list(
            state_pool), list(action_pool), list(reward_pool)
        # Discount reward
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

        # Normalize reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # Gradient Desent
        self.optimizer.zero_grad()

        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]
            state = Variable(torch.from_numpy(state).float())
            probs = self.policy_net(state)
            m = Bernoulli(probs)
            # Negtive score function x reward
            loss = -m.log_prob(action) * reward
            # print(loss)
            loss.backward()
        self.optimizer.step()
        self.memory.clear()


def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境:{cfg.env_name}, 算法:{cfg.algo_name}, 设备:{cfg.device}')
    rewards = []  # 记录奖励
    for i_ep in range(cfg.train_eps):
        ep_reward = 0  # 记录每个回合的奖励
        state, info = env.reset()  # 重置环境,即开始新的回合
        action = agent.sample(state)
        while True:
            action = agent.sample(state)  # 根据算法采样一个动作
            next_state, reward, terminated, truncated, _ = env.step(
                action)  # 与环境进行一次动作交互
            next_action = agent.sample(next_state)
            agent.update(state, action, reward, next_state,
                         next_action, terminated, truncated)  # 算法更新
            state = next_state  # 更新状态
            action = next_action
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        print(
            f"回合：{i_ep+1}/{cfg.train_eps}，奖励：{ep_reward:.1f}，Epsilon：{agent.epsilon}")
    print('完成训练！')
    return {"rewards": rewards}


def test(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env_name}, 算法：{cfg.algo_name}, 设备：{cfg.device}')
    rewards = []  # 记录所有回合的奖励
    for i_ep in range(cfg.test_eps):
        ep_reward = 0  # 记录每个episode的reward
        state, info = env.reset()  # 重置环境, 重新开一局（即开始新的一个回合）
        while True:
            action = agent.predict(state)  # 根据算法选择一个动作
            next_state, reward, terminated, truncated, _ = env.step(
                action)  # 与环境进行一个交互
            state = next_state  # 更新状态
            ep_reward += reward
            if terminated or truncated:
                break
        rewards.append(ep_reward)
        print(f"回合数：{i_ep+1}/{cfg.test_eps}, 奖励：{ep_reward:.1f}")
    print('完成测试！')
    return {"rewards": rewards}
