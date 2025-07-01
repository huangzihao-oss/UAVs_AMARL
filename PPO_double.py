import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !RolloutBuffer类，用于存储经验
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.masks = []
    
    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.masks[:]

# ActorCritic类，定义演员和评论家网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        self.has_continuous_action_space = has_continuous_action_space
        self.action_dim = action_dim
        
        if has_continuous_action_space:
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Mish(),
                nn.Linear(128, 128),
                nn.Mish(),
                nn.Linear(128, action_dim),
                nn.Mish()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.Mish(),
                nn.Linear(128, 128),
                nn.Mish(),
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            )
        
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Mish(),
            nn.Linear(128, 128),
            nn.Mish(),
            nn.Linear(128, 1)
        )
    
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")
    
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, mask=None, deter_action=None):
        action_probs = self.actor(state)
        
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.float32, device=action_probs.device)
            if mask.shape != (self.action_dim,):
                raise ValueError(f"Mask shape {mask.shape} does not match action_dim {self.action_dim}")
            if mask.sum() == 0:
                raise ValueError("Mask cannot be all zeros (no valid actions)")
            
            masked_probs = action_probs * mask
            masked_probs = masked_probs + 1e-25 * (1 - mask)
            masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
        else:
            masked_probs = action_probs
        
        if deter_action is not None:
            dist = Categorical(probs=masked_probs)
            action = torch.tensor(deter_action, dtype=torch.long).to(device)
            action_logprob = dist.log_prob(action)
            state_val = self.critic(state)
        else:
            dist = Categorical(probs=masked_probs)
            action = dist.sample()
            action_logprob = dist.log_prob(action)
            state_val = self.critic(state)
        
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def act_test(self, state, mask=None):
        with torch.no_grad():
            action_probs = self.actor(state)
            
            if mask is not None:
                mask = torch.as_tensor(mask, dtype=torch.float32, device=action_probs.device)
                if mask.shape != (self.action_dim,):
                    raise ValueError(f"Mask shape {mask.shape} does not match action_dim {self.action_dim}")
                if mask.sum() == 0:
                    raise ValueError("Mask cannot be all zeros (no valid actions)")
                
                masked_probs = action_probs.masked_fill(mask == 0, float('-inf'))
            else:
                masked_probs = action_probs
            
            max_prob_action = masked_probs.argmax(dim=-1)
        
        return max_prob_action.item()
    
    def evaluate(self, state, action, mask=None):
        action_probs = self.actor(state)
        
        if mask is not None:
            mask = torch.as_tensor(mask, dtype=torch.float32, device=action_probs.device)
            if mask.sum() == 0:
                raise ValueError("Mask cannot be all zeros (no valid actions)")
            
            masked_probs = action_probs * mask
            masked_probs = masked_probs + 1e-8 * (1 - mask)
            masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
        else:
            masked_probs = action_probs
        
        dist = Categorical(probs=masked_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

# PPO类，上层智能体（目标选择）
class PPO:
    def __init__(self, agent_id, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, summary_dir, entropy_ratio, gae_lambda, gae_flag, action_std_init=0.6):
        self.has_continuous_action_space = has_continuous_action_space
        if has_continuous_action_space:
            self.action_std = action_std_init
        self.id = agent_id
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_ratio = entropy_ratio
        self.gae_lambda = gae_lambda
        self.gae_flag = gae_flag
        
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.summary_dir = summary_dir
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.update_times = 0
        self.non_bs_count = 0
        self.bs_count = 0
    
    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")
    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)
        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")
        print("--------------------------------------------------------------------------------------------")
    
    def select_action(self, state, mask, deter_action=None):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob, state_val = self.policy_old.act(state, mask, deter_action)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        
        return action.item()
    
    def action_test(self, state, mask):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action = self.policy_old.act_test(state, mask)
        return action
    
    def update(self):
        print_reward = 0
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            print_reward += reward
            rewards.insert(0, discounted_reward)
                
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        len_batch = len(rewards)
        
        masks = torch.squeeze(torch.stack(self.buffer.masks[:len_batch], dim=0)).detach().to(device)
        old_states = torch.squeeze(torch.stack(self.buffer.states[:len_batch], dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions[:len_batch], dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[:len_batch], dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values[:len_batch], dim=0)).detach().to(device)
        
        if self.gae_flag:
            advantages = self.compute_gae(self.buffer.rewards, old_state_values, self.buffer.is_terminals)
        else:
            advantages = rewards.detach() - old_state_values.detach()
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, masks)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, rewards).mean()
            entropy_loss = dist_entropy.mean()
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_ratio * dist_entropy
            
            if self.id == 0:
                self.writer.add_scalar('loss/policy', policy_loss.detach().item(), self.update_times)
                self.writer.add_scalar('loss/critic', critic_loss.detach().item(), self.update_times)
                self.writer.add_scalar('stats/critic', state_values.mean(), self.update_times)
                self.writer.add_scalar('stats/entropy', entropy_loss.detach().item(), self.update_times)
                self.writer.add_scalar('reward/train', print_reward, self.update_times)
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            self.update_times += 1
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def compute_gae(self, rewards, state_values, is_terminals):
        gae = 0
        advantages = []
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(is_terminals, dtype=torch.float32).to(device)
        state_values = torch.squeeze(state_values).detach()
        
        for t in reversed(range(len(rewards))):
            if is_terminals[t]:
                delta = rewards[t] - state_values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * (t < len(rewards) - 1) * state_values[t + 1] - state_values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        return advantages
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
    
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    
    def call_2_record(self, steps, value):
        self.writer.add_scalar('reward/test', value, steps)

# LowerPPO类，下层智能体（速度选择）
class LowerPPO:
    def __init__(self, agent_id, state_dim_upper, action_dim_upper, action_dim_lower=16, lr_actor=0.0003, lr_critic=0.001, 
                 gamma=0.99, K_epochs=10, eps_clip=0.2, summary_dir='runs/lower', entropy_ratio=0.01, 
                 gae_lambda=0.95, gae_flag=True, action_std_init=0.6):
        self.has_continuous_action_space = False
        self.id = agent_id
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_ratio = entropy_ratio
        self.gae_lambda = gae_lambda
        self.gae_flag = gae_flag
        
        self.state_dim_lower = state_dim_upper + action_dim_upper
        self.action_dim_lower = action_dim_lower
        
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(self.state_dim_lower, self.action_dim_lower, self.has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        
        self.summary_dir = summary_dir
        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.policy_old = ActorCritic(self.state_dim_lower, self.action_dim_lower, self.has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        self.update_times = 0
    
    def select_action(self, state_upper, action_upper, mask=None, deter_action=None):
        with torch.no_grad():
            state_lower = torch.cat([
                torch.FloatTensor(state_upper).to(device),
                torch.FloatTensor([action_upper]).to(device)
            ], dim=-1)
            action, action_logprob, state_val = self.policy_old.act(state_lower, mask, deter_action)
        
        self.buffer.states.append(state_lower)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        
        return action.item()
    
    def action_test(self, state_upper, action_upper, mask=None):
        with torch.no_grad():
            state_lower = torch.cat([
                torch.FloatTensor(state_upper).to(device),
                torch.FloatTensor([action_upper]).to(device)
            ], dim=-1)
            action = self.policy_old.act_test(state_lower, mask)
        return action
    
    def update(self):
        print_reward = 0
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            print_reward += reward
            rewards.insert(0, discounted_reward)
                
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        len_batch = len(rewards)
        
        masks = torch.squeeze(torch.stack(self.buffer.masks[:len_batch], dim=0)).detach().to(device)
        old_states = torch.squeeze(torch.stack(self.buffer.states[:len_batch], dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions[:len_batch], dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[:len_batch], dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values[:len_batch], dim=0)).detach().to(device)
        
        if self.gae_flag:
            advantages = self.compute_gae(self.buffer.rewards, old_state_values, self.buffer.is_terminals)
        else:
            advantages = rewards.detach() - old_state_values.detach()
        
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, masks)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, rewards).mean()
            entropy_loss = dist_entropy.mean()
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_ratio * dist_entropy
            
            if self.id == 1:
                self.writer.add_scalar('loss/policy', policy_loss.detach().item(), self.update_times)
                self.writer.add_scalar('loss/critic', critic_loss.detach().item(), self.update_times)
                self.writer.add_scalar('stats/critic', state_values.mean(), self.update_times)
                self.writer.add_scalar('stats/entropy', entropy_loss.detach().item(), self.update_times)
                self.writer.add_scalar('reward/train', print_reward, self.update_times)
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            self.update_times += 1
        
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()
    
    def compute_gae(self, rewards, state_values, is_terminals):
        gae = 0
        advantages = []
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(is_terminals, dtype=torch.float32).to(device)
        state_values = torch.squeeze(state_values).detach()
        
        for t in reversed(range(len(rewards))):
            if is_terminals[t]:
                delta = rewards[t] - state_values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * (t < len(rewards) - 1) * state_values[t + 1] - state_values[t]
                gae = delta + self.gamma * self.gae_lambda * gae
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        return advantages
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
    
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    
    def call_2_record(self, steps, value):
        self.writer.add_scalar('reward/test', value, steps)


class PPO_Hierarchical:
    def __init__(self, agent_id, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, summary_dir, entropy_ratio, gae_lambda, gae_flag, action_std_init=0.6):
    
    def select_action(self, state, mask, deter_action=None):
        action_upper = upper_ppo.select_action(state, mask, deter_action[0])
        action_lower = lower_ppo.select_action(state, action_upper, deter_action[1])
    
    def action_test(self, state, mask):
        
    
    def update(self):
        

    
    def save(self, checkpoint_path):
        pass
    
    # def load(self, checkpoint_path):
        # self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        # self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    
    def call_2_record(self, steps, value):
        self.writer.add_scalar('reward/test', value, steps)
# 交互函数，展示上下层如何与环境交互
def hierarchical_rl_step(upper_ppo, lower_ppo, env, state, upper_mask=None, lower_mask=None):
    action_upper = upper_ppo.select_action(state, upper_mask)
    action_lower = lower_ppo.select_action(state, action_upper, lower_mask)
    next_state, reward, done, _ = env.step(action_lower)
    
    upper_ppo.buffer.rewards.append(reward)
    upper_ppo.buffer.is_terminals.append(done)
    upper_ppo.buffer.masks.append(upper_mask if upper_mask is not None else torch.ones(upper_ppo.policy.action_dim).to(device))
    
    lower_ppo.buffer.rewards.append(reward)
    lower_ppo.buffer.is_terminals.append(done)
    lower_ppo.buffer.masks.append(lower_mask if lower_mask is not None else torch.ones(lower_ppo.action_dim_lower).to(device))
    
    return next_state, reward, done

# 示例训练循环（需用户根据环境实现）
def train_hierarchical_rl(env, upper_ppo, lower_ppo, max_episodes=1000, update_timestep=4000):
    time_step = 0
    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0
        while True:
            state, reward, done = hierarchical_rl_step(upper_ppo, lower_ppo, env, state)
            episode_reward += reward
            time_step += 1
            if time_step % update_timestep == 0:
                upper_ppo.update()
                lower_ppo.update()
            if done:
                break
        upper_ppo.call_2_record(episode, episode_reward)
        lower_ppo.call_2_record(episode, episode_reward)