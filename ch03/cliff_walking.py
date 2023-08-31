# '''初始化环境'''  
# env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
# env = CliffWalkingWapper(env)
# agent = QLearning(
#     state_dim=env.observation_space.n,
#     action_dim=env.action_space.n,
#     learning_rate=cfg.policy_lr,
#     gamma=cfg.gamma,
# rewards = []  
# ma_rewards = [] # moving average reward
# for i_ep in range(cfg.train_eps): # train_eps: 训练的最大episodes数
#     ep_reward = 0  # 记录每个episode的reward
#     state = env.reset()  # 重置环境, 重新开一局（即开始新的一个episode）
#     while True:
#         action = agent.choose_action(state)  # 根据算法选择一个动作
#         next_state, reward, terminated, truncated, info = env.step(action) # 与环境进行一次动作交互
#         agent.update(state, action, reward, next_state, done)  # Q-learning算法更新
#         state = next_state  # 存储上一个观察值
#         ep_reward += reward
#         if terminated or truncated:
#             break
#     rewards.append(ep_reward)
#     if ma_rewards:
#         ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
#     else:
#         ma_rewards.append(ep_reward)
#     print("Episode:{}/{}: reward:{:.1f}".format(i_ep+1, cfg.train_eps,ep_reward))

import datetime
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import cliff_walking_warper as cww
import sarsa

def get_args():
    """ 
    """
    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name',default='Sarsa',type=str,help="name of algorithm")
    parser.add_argument('--env_name',default='CliffWalking-v0',type=str,help="name of environment")
    parser.add_argument('--train_eps',default=400,type=int,help="episodes of training") # 训练的回合数
    parser.add_argument('--test_eps',default=20,type=int,help="episodes of testing") # 测试的回合数
    parser.add_argument('--gamma',default=0.90,type=float,help="discounted factor") # 折扣因子
    parser.add_argument('--epsilon_start',default=0.95,type=float,help="initial value of epsilon") #  e-greedy策略中初始epsilon
    parser.add_argument('--epsilon_end',default=0.01,type=float,help="final value of epsilon") # e-greedy策略中的终止epsilon
    parser.add_argument('--epsilon_decay',default=300,type=int,help="decay rate of epsilon") # e-greedy策略中epsilon的衰减率
    parser.add_argument('--lr',default=0.1,type=float,help="learning rate")
    parser.add_argument('--device',default='cpu',type=str,help="cpu or cuda")          
    args = parser.parse_args([])                          
    return args

def smooth(data, weight=0.9):  
    '''用于平滑曲线，类似于Tensorboard中的smooth

    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)                    
        last = smoothed_val                                
    return smoothed

def plot_rewards(rewards,cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")
    plt.xlabel('epsiodes')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    plt.show()


# 获取参数
cfg = get_args()
# 训练
env, agent = cww.env_agent_config(cfg)
res_dic = sarsa.train(cfg, env, agent)

plot_rewards(res_dic['rewards'], cfg, tag="train")  
# 测试
res_dic = sarsa.test(cfg, env, agent)
plot_rewards(res_dic['rewards'], cfg, tag="test")  # 画出结果


