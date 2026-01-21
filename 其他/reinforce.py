import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, state):
        logits = self.net(state)
        # 使用 Categorical 分布来处理离散动作空间
        return Categorical(logits=logits)


def train_policy_gradient():
    env = gym.make('CartPole-v1')
    policy_net = PolicyNetwork(
        env.observation_space.shape[0], env.action_space.n)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)
    gamma = 0.99  # 折扣因子

    for episode in range(1000):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        # 1. 收集一个回合的轨迹
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_dist = policy_net(state_tensor)
            action = action_dist.sample()

            log_prob = action_dist.log_prob(action)
            log_probs.append(log_prob)

            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            rewards.append(reward)

        # 2. 计算折扣回报 G_t
        returns = []
        discounted_return = 0
        for r in reversed(rewards):
            discounted_return = r + gamma * discounted_return
            returns.insert(0, discounted_return)

        returns = torch.tensor(returns)
        # 标准化回报以减小方差 (非常重要的技巧)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # 3. 计算损失并更新策略
        loss = []
        for log_prob, R in zip(log_probs, returns):
            # 目标是最大化 E[G_t * log(pi)]，所以损失是 -E[G_t * log(pi)]
            loss.append(-log_prob * R)

        optimizer.zero_grad()
        # 将一个回合的所有时间步的损失加起来
        loss = torch.cat(loss).sum()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            print(
                f'Episode {episode}, Loss: {loss.item()}, Total Reward: {sum(rewards)}')
    env.close()


train_policy_gradient()
