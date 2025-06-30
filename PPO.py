import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import math
import copy

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.masks = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        del self.masks[:]


# class ActorCritic(nn.Module):
#     def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
#         super(ActorCritic, self).__init__()

#         self.has_continuous_action_space = has_continuous_action_space
        
#         if has_continuous_action_space:
#             self.action_dim = action_dim
#             self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
#         # actor
#         if has_continuous_action_space :
#             self.actor = nn.Sequential(
#                             nn.Linear(state_dim, 64),
#                             nn.Tanh(),
#                             nn.Linear(64, 64),
#                             nn.Tanh(),
#                             nn.Linear(64, action_dim),
#                             nn.Tanh()
#                         )
#         else:
#             self.actor = nn.Sequential(
#                             nn.Linear(state_dim, 64),
#                             nn.Tanh(),
#                             nn.Linear(64, 64),
#                             nn.Tanh(),
#                             nn.Linear(64, action_dim),
#                             nn.Softmax(dim=-1)
#                         )
#         # critic
#         self.critic = nn.Sequential(
#                         nn.Linear(state_dim, 64),
#                         nn.Tanh(),
#                         nn.Linear(64, 64),
#                         nn.Tanh(),
#                         nn.Linear(64, 1)
#                     )
        
#     def set_action_std(self, new_action_std):
#         if self.has_continuous_action_space:
#             self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
#         else:
#             print("--------------------------------------------------------------------------------------------")
#             print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
#             print("--------------------------------------------------------------------------------------------")

#     def forward(self):
#         raise NotImplementedError
    
#     def act(self, state, mask=None):

#         action_probs = self.actor(state)
#         dist = Categorical(action_probs)

#         action = dist.sample()
#         action_logprob = dist.log_prob(action)
#         state_val = self.critic(state)

#         return action.detach(), action_logprob.detach(), state_val.detach()
    
#     def act_test(self, state, mask=None):
#         action_probs = self.actor(state)  # 假设返回动作概率分布，例如 tensor([0.2, 0.5, 0.3])
#         max_prob_action = action_probs.argmax()  # 找到最大概率的动作索引，例如 1（对应概率 0.5）
#         max_prob = action_probs[max_prob_action]  # 获取最大概率值，例如 0.5

#         return max_prob_action
    
#     def evaluate(self, state, action, mask):
#         action_probs = self.actor(state)
#         dist = Categorical(action_probs)
#         action_logprobs = dist.log_prob(action)
#         dist_entropy = dist.entropy()
#         state_values = self.critic(state)
        
#         return action_logprobs, state_values, dist_entropy

# ! 带mask的
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
        """
        Select an action during training, applying action mask if provided.
        
        Args:
            state: Current state (torch tensor, shape: [batch_size, state_dim] or [state_dim])
            mask: Action mask (torch tensor or None, shape: [action_dim]), 1 for valid, 0 for invalid
            
        Returns:
            action: Selected action (scalar)
            action_logprob: Log probability of the action
            state_val: Critic's state value
        """
        action_probs = self.actor(state)
        
        
            
        if mask is not None:
            # Ensure mask is a tensor and has correct shape
            mask = torch.as_tensor(mask, dtype=torch.float32, device=action_probs.device)
            if mask.shape != (self.action_dim,):
                raise ValueError(f"Mask shape {mask.shape} does not match action_dim {self.action_dim}")
            if mask.sum() == 0:
                raise ValueError("Mask cannot be all zeros (no valid actions)")
            
            # Apply mask: set invalid action probabilities to 0
            masked_probs = action_probs * mask
            # Add small value to avoid zero probabilities, then normalize
            masked_probs = masked_probs + 1e-25 * (1 - mask)
            masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
        else:
            masked_probs = action_probs
        
        # 如果指定动作，就查询指定动作的概率
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
        """
        Select the best action during testing, applying action mask if provided.
        
        Args:
            state: Current state (torch tensor, shape: [batch_size, state_dim] or [state_dim])
            mask: Action mask (torch tensor or None, shape: [action_dim]), 1 for valid, 0 for invalid
            
        Returns:
            max_prob_action: Action with highest probability among valid actions
        """
        with torch.no_grad():
            action_probs = self.actor(state)
            
            if mask is not None:
                # Ensure mask is a tensor and has correct shape
                mask = torch.as_tensor(mask, dtype=torch.float32, device=action_probs.device)
                if mask.shape != (self.action_dim,):
                    raise ValueError(f"Mask shape {mask.shape} does not match action_dim {self.action_dim}")
                if mask.sum() == 0:
                    raise ValueError("Mask cannot be all zeros (no valid actions)")
                
                # Apply mask: set invalid action probabilities to -inf
                masked_probs = action_probs.masked_fill(mask == 0, float('-inf'))
            else:
                masked_probs = action_probs
            
            # Select action with maximum probability
            max_prob_action = masked_probs.argmax(dim=-1)
        
        return max_prob_action.item()
    
    def evaluate(self, state, action, mask=None):
        """
        Evaluate actions for loss calculation, applying action mask if provided.
        
        Args:
            state: States (torch tensor, shape: [batch_size, state_dim])
            action: Actions (torch tensor, shape: [batch_size])
            mask: Action mask (torch tensor or None, shape: [action_dim]), 1 for valid, 0 for invalid
            
        Returns:
            action_logprobs: Log probabilities of actions
            state_values: Critic's state values
            dist_entropy: Entropy of the action distribution
        """
        action_probs = self.actor(state)
        
        if mask is not None:
            # Ensure mask is a tensor and has correct shape
            mask = torch.as_tensor(mask, dtype=torch.float32, device=action_probs.device)
            # if mask.shape != (self.action_dim,):
            #     raise ValueError(f"Mask shape {mask.shape} does not match action_dim {self.action_dim}")
            if mask.sum() == 0:
                raise ValueError("Mask cannot be all zeros (no valid actions)")
            
            # Apply mask: set invalid action probabilities to 0
            masked_probs = action_probs * mask
            # Add small value to avoid zero probabilities, then normalize
            masked_probs = masked_probs + 1e-8 * (1 - mask)
            masked_probs = masked_probs / masked_probs.sum(dim=-1, keepdim=True)
        else:
            masked_probs = action_probs

        dist = Categorical(probs=masked_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


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
            if (self.action_std <= min_action_std):
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
        # Monte Carlo estimate of returns
        print_reward = 0
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            print_reward += reward
            rewards.insert(0, discounted_reward)
                
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        len_batch = len(rewards)

        # convert list to tensor
        masks = torch.squeeze(torch.stack(self.buffer.masks[:len_batch], dim=0)).detach().to(device)
        old_states = torch.squeeze(torch.stack(self.buffer.states[:len_batch], dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions[:len_batch], dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs[:len_batch], dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values[:len_batch], dim=0)).detach().to(device)

        # calculate advantages
        if self.gae_flag:
            advantages = self.compute_gae(self.buffer.rewards, old_state_values, self.buffer.is_terminals)
            
        else:    
            advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, masks)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            policy_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.MseLoss(state_values, rewards).mean()
            entropy_loss = dist_entropy.mean()
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - self.entropy_ratio * dist_entropy
            
            # 记录
            if self.id == 0:
                self.writer.add_scalar('loss/policy', policy_loss.detach().item(),self.update_times)
                self.writer.add_scalar('loss/critic', critic_loss.detach().item(),self.update_times)
                
                self.writer.add_scalar('stats/critic', state_values.mean(), self.update_times)
                self.writer.add_scalar('stats/entropy', entropy_loss.detach().item(), self.update_times)
                
                self.writer.add_scalar('reward/train', print_reward, self.update_times)
            
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
            self.update_times += 1
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
     
    def compute_gae(self, rewards, state_values, is_terminals):
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: List or tensor of rewards.
            state_values: Tensor of state value estimates V(s_t).
            next_state_value: Value estimate of the last state V(s_{T+1}).
            is_terminals: List or tensor indicating terminal states.
        
        Returns:
            advantages: Tensor of GAE advantages.
        """
        gae = 0
        advantages = []
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        is_terminals = torch.tensor(is_terminals, dtype=torch.float32).to(device)
        state_values = torch.squeeze(state_values).detach()
        # state_values = torch.cat([state_values, torch.tensor([0.0], dtype=torch.float32, device=state_values.device)])
        
        for t in reversed(range(len(rewards))):
            
            # print(is_terminals[t], self.id)
            
            if is_terminals[t]:
                delta = rewards[t] - state_values[t]
                gae = delta
            else:
            # delta = rewards[t] + self.gamma*state_values[t + 1] - state_values[t]
                delta = rewards[t] + self.gamma * (t < len(rewards) - 1)*state_values[t + 1] - state_values[t]
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
    
    
    # !A3C
    # def update(self) :
    #     # Monte Carlo estimate of returns
    #     print_reward = 0
        
    #     # 这里开始复制
    #     rewards = copy.copy(self.buffer.rewards)
    #     len_batch = len(rewards)
    #     values = copy.copy(self.buffer.state_values[:len(rewards)])
    #     rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        
    #     # A3C式更新
    #     R = torch.zeros(1, 1).to(device)
    #     values.append(R)
    #     values = torch.tensor(values).to(device)
        
        
    #     masks = torch.squeeze(torch.stack(self.buffer.masks[:len_batch], dim=0)).detach().to(device)
    #     old_states = torch.squeeze(torch.stack(self.buffer.states[:len_batch], dim=0)).detach().to(device)
    #     old_actions = torch.squeeze(torch.stack(self.buffer.actions[:len_batch], dim=0)).detach().to(device)
    #     logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, masks)
    #     state_values = state_values.view(-1)
        
        
        
    #     policy_loss = 0
    #     value_loss = 0
    #     entropy_loss = 0
    #     gae = torch.zeros(1, 1).to(device)
    #     for i in reversed(range(len(rewards))):
    #         R = self.gamma * R + rewards[i]
    #         advantage = R - state_values[i]
    #         value_loss = value_loss + 0.5 * advantage.pow(2)

    #         # Generalized Advantage Estimation
    #         delta_t = rewards[i] + self.gamma * \
    #             values[i + 1] - values[i]
    #         gae = gae * self.gamma * self.gae_lambda + delta_t
    #         policy_loss = policy_loss - logprobs[i] * gae.detach()
    #         entropy_loss = entropy_loss - self.entropy_ratio * dist_entropy[i]
    #         print_reward += rewards[i]
        
    #     # 记录
    #     if self.id == 0:
    #         self.writer.add_scalar('loss/policy', policy_loss.detach().item(),self.update_times)
    #         self.writer.add_scalar('loss/critic', value_loss.detach().item(),self.update_times)
            
    #         self.writer.add_scalar('stats/critic', state_values.mean(), self.update_times)
    #         self.writer.add_scalar('stats/entropy', entropy_loss.detach().item(), self.update_times)
            
    #         self.writer.add_scalar('reward/train', print_reward, self.update_times)
            
            
    #     # take gradient step
    #     self.optimizer.zero_grad()
    #     (policy_loss + 0.5 * value_loss + entropy_loss).backward()
    #     self.optimizer.step()
        
    #     self.update_times += 1
            
    #     # Copy new weights into old policy
    #     self.policy_old.load_state_dict(self.policy.state_dict())

    #     # clear buffer
    #     self.buffer.clear()
        
        
       


