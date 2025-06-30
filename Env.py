import gym
import numpy as np
from gym import spaces
from scipy.spatial.distance import euclidean

import matplotlib.pyplot as plt
import os
from matplotlib import cm


class MultiDroneAoIEnv(gym.Env):
    def __init__(
        self,
        M=3,
        N=2,
        K=5,
        T=100.0,
        map_size=100.0,
        drone_speeds=None,
        sense_times=None,
        args=None,
        position_file=None,
        base_positions=None,
        sensor_positions=None
    ):
        """
        多无人机数据感知环境，目标：最小化K个感知点的平均信息年龄 (AoI)。
        异步执行：step接受uav_id和action，返回最早完成动作的无人机的obs、reward等。
        记录每架无人机在回传前的感知点访问历史，用于奖励计算。

        参数:
            M (int): 无人机数量
            N (int): 基站数量
            K (int): 感知点数量
            T (float): 时间限制（秒）
            map_size (float): 地图大小（正方形边长）
            drone_speeds (list): 每架无人机的速度（m/s），默认 [1.0, ...]
            sense_times (list): 每个感知点的感知时间（秒），默认 [1.0, ...]
            position_file (str): CSV文件路径，包含基站和感知点位置
            base_positions (np.array): 基站位置数组，形状 (N, 2)
            sensor_positions (np.array): 感知点位置数组，形状 (K, 2)
        """
        super(MultiDroneAoIEnv, self).__init__()
        self.M = M
        self.N = N
        self.K = K
        self.T = T
        self.map_size = map_size
        self.args = args

        # 无人机速度（支持异构性）
        self.drone_speeds = drone_speeds if drone_speeds is not None else [1.0] * M
        # ! 看如何计算
        self.sense_times = sense_times if sense_times is not None else [1.0] * K

        # 位置：无人机、基站、感知点
        self.drone_pos = np.zeros((M, 2))  # 初始位置 (0, 0)
        self.base_pos = np.zeros((N, 2))
        self.sensor_pos = np.zeros((K, 2))
        
        # !数据文件
        position_filename = f"./data/poi_{self.K}_bs_{self.N}_map_{int(self.map_size)}x{int(self.map_size)}.npy"
        data = np.load(position_filename, allow_pickle=True).item()
        self.sensor_pos = data['poi_positions']
        self.base_pos = data['bs_positions']

        # !全局信息
        # 信息年龄 (AoI)
        self.aoi = np.zeros(K)
        self.global_timing = 0
        # !后期需要进行修改
        self.fly_speed = 15
        self.drone_last_visited_history_at_BS = np.zeros((M, K))
        self.drone_visited_history_timing_at_BS = np.zeros((M, K))
        # self.drone_last_decision_at_BS = np.zeros(M)
        # self.drone_last_decision_time_at_BS = np.zeros(M)
        
        # !局部信息
        # 当前时刻无人机的缓存列表
        self.drone_buffer = [[] for _ in range(M)]
        # 每个缓存对应的时刻
        self.buffer_timing = [[] for _ in range(M)]
        # 每个无人机单独的时间
        self.drone_timing_now = np.zeros(M)
        self.drone_local_aoi = np.zeros((M, K))
        self.drone_last_reward = np.zeros(M)
        self.drone_step_reward = [[] for i in range(M)]
        self.drone_position_now = np.zeros((self.M, 2))
        self.other_drones_delay_rewards = [[] for i in range(M)]
        # self.drone_last_rest_energy = np.zeros(M)
        
        # !观测和动作空间
        # 动作空间：单个无人机
        self.action_space = spaces.Discrete(K + N)
        # 观测空间：单个无人机
        # obs_dim = K+(M+N+K)*2+K+1+M*K+M+1
        obs_dim = K+K+K+1+K+1+K*M+M+1
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.left = 4
        self.right = 8
        # self.bs_random_factor = np.random.randint(self.left, self.right)
        self.bs_random_factor = self.args.BS_back_times

        # 初始化环境
        self.reset()

    def reset(self):
        """重置环境"""
        # !全局信息
        # 信息年龄 (AoI)
        self.aoi = np.zeros(self.K)
        self.global_timing = 0
        self.drone_last_visited_history_at_BS = np.zeros((self.M, self.K))
        self.drone_visited_history_timing_at_BS = np.zeros((self.M, self.K))
        
        # !局部信息
        # 当前时刻无人机的缓存列表
        self.drone_buffer = [[] for _ in range(self.M)]
        # 每个缓存对应的时刻
        self.buffer_timing = [[] for _ in range(self.M)]
        # 每个无人机单独的时间
        self.drone_timing_now = np.zeros(self.M)
        self.drone_local_aoi = np.zeros((self.M, self.K))
        self.drone_last_reward = np.zeros(self.M)
        self.drone_step_reward = [[] for i in range(self.M)]
        self.drone_position_now = np.zeros((self.M, 2))
        self.other_drones_delay_rewards = [[] for i in range(self.M)]
        return self._get_obs(0)

    def _get_obs(self, uav_id):
        """获取指定无人机的观测"""
        
        # *AoI信息
        aoi_obs = self.drone_local_aoi[uav_id].copy()*10/self.T
        
        # *位置信息
        dis_2_pois = np.linalg.norm(self.sensor_pos-self.drone_position_now[uav_id], axis=-1)
        move_t_2_pois = dis_2_pois/self.fly_speed
        next_aoi_2_poi = self.drone_local_aoi[uav_id] + move_t_2_pois
        next_time_2_poi = move_t_2_pois + self.drone_timing_now[uav_id]
        
        mask = np.ones(self.K)
        mask[self.drone_buffer[uav_id]] = 0
        rewards_2_pois = next_aoi_2_poi * (self.T-next_time_2_poi)/(2*self.args.pre_reward_ratio)*mask
        rewards_2_pois /= self.args.reward_scale_size
        
        # 转换为效率，rewards/time
        rewards_2_pois *= 60
        rewards_2_pois /= (move_t_2_pois+1)
        
        rewards_2_bs = 0
        for i in range(len(self.drone_buffer[uav_id])):
            coll_timing = self.buffer_timing[uav_id][i]
            target_poi = self.drone_buffer[uav_id][i]
            curr_aoi = self.drone_timing_now[uav_id] - coll_timing
            
            rewards_2_bs += (self.drone_local_aoi[uav_id, target_poi] - curr_aoi)*(self.T - self.drone_timing_now[uav_id])/2
        rewards_2_bs -= sum(self.drone_step_reward[uav_id])
        rewards_2_bs = np.array([rewards_2_bs])*5/self.args.reward_scale_size
        
        
        
        pos_obs = np.concatenate([
            self.base_pos.flatten(),
            self.sensor_pos.flatten()
        ])/self.map_size
        
        buffer_obs = np.zeros(self.K)
        for poi in self.drone_buffer:
            buffer_obs[poi] = 1
        
        # *无人机缓存信息
        # *自己缓存的为1，别人缓存的为2，没有缓存的为0
        history_visited_sensors_and_timing = self.drone_visited_history_timing_at_BS.flatten()/self.T
            
        # *时间信息
        time_obs = np.array([self.drone_timing_now[uav_id]])/self.T
        
        # *缓存大小
        buffer_len = np.array([len(self.drone_step_reward[uav_id])])
        
        # !注意需要进行归一化 K+K+K+1+K+1+K*M+M+1
        # print(aoi_obs, dis_2_pois/500)
        return np.concatenate([aoi_obs, dis_2_pois/500, rewards_2_pois, rewards_2_bs, buffer_obs, time_obs, history_visited_sensors_and_timing, self.drone_last_reward/self.args.reward_scale_size, buffer_len])

    def step(self, uav_id, action):
        """
        异步步进：接受uav_id和action，处理最早完成动作的无人机，返回其obs、reward等。

        参数:
            uav_id (int): 执行动作的无人机ID
            action (int): 动作（0~K+N-1）

        返回:
            obs (np.array): 下一无人机的观测
            reward (float): 奖励
            done (bool): 是否终止
            info (dict): 包含drone_id、avg_aoi、current_time等
        """
        assert 0 <= uav_id < self.M, f"Invalid uav_id: {uav_id}"
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # K是感知点
        # N是基站
        # *如果是基站
        if self.K <= action <= self.K+self.N-1:
            target_position = self.base_pos[action-self.K]
            move_dis = euclidean(target_position, self.drone_position_now[uav_id])
            move_time = move_dis/self.fly_speed
            
            # 根据计算结果更新（位置、时间）
            self.drone_position_now[uav_id] = target_position
            self.drone_timing_now[uav_id] += move_time
            
            for i in range(self.K):
                self.drone_local_aoi[uav_id, i] += move_time
                self.aoi[i] += self.drone_timing_now[uav_id] - self.global_timing
            
            self.global_timing = self.drone_timing_now[uav_id]
            
            reward = 0
            if self.drone_buffer[uav_id]:
                for i in range(len(self.drone_buffer[uav_id])):
                    coll_timing = self.buffer_timing[uav_id][i]
                    target_poi = self.drone_buffer[uav_id][i]
                    curr_aoi = self.drone_timing_now[uav_id] - coll_timing
                    
                    reward += max((min(self.aoi[target_poi], self.drone_local_aoi[uav_id, target_poi]) - curr_aoi), 0)*(self.T - self.drone_timing_now[uav_id])/2
                    self.aoi[target_poi] = min(curr_aoi, self.aoi[target_poi])
                
                self.drone_local_aoi[uav_id] = self.aoi
                self.drone_visited_history_timing_at_BS[uav_id] = np.zeros(self.K)
                for i in range(len(self.drone_buffer[uav_id])):
                    self.drone_visited_history_timing_at_BS[uav_id, self.drone_buffer[uav_id][i]] = self.buffer_timing[uav_id][i]
                
                # 这一轮总的reward
                self.drone_last_reward[uav_id] = reward
                
                # 将自己当前的总奖励分给其他uav
                for i in range(self.M):
                    if i != uav_id:
                        self.other_drones_delay_rewards[i].append(reward)
                
                # 首先减去之前为了鼓励无人机飞行而产生的reward
                # 清空自己之前的reward记录，进行下一轮收集
                step_reward = np.array(self.drone_step_reward[uav_id])
                negative_reward_indx = np.argwhere(step_reward<=0)
                step_reward[negative_reward_indx] *= -1
                reward -= sum(step_reward)
                
                # reward -= sum(self.drone_step_reward[uav_id])
                self.drone_step_reward[uav_id] = []
                
                # delay rewards
                # !获取这段时间内其他人的reward，清空队列
                # reward += sum(self.other_drones_delay_rewards[uav_id])
                self.other_drones_delay_rewards[uav_id] = []
                
                # 清空buffer
                self.drone_buffer[uav_id] = []
                self.buffer_timing[uav_id] = []
                
        # *如果不是基站，只需要维护自己的内容，不知道别人的内容
        else:
            target_poi = action
            target_position = self.sensor_pos[target_poi]
            move_dis = euclidean(target_position, self.drone_position_now[uav_id])
            move_time = move_dis/self.fly_speed
            
            self.drone_timing_now[uav_id] += move_time
            self.drone_buffer[uav_id].append(target_poi)
            self.buffer_timing[uav_id].append(self.drone_timing_now[uav_id])
            self.drone_position_now[uav_id] = target_position
            
            for i in range(self.K):
                self.drone_local_aoi[uav_id, i] += move_time
                        
            # !起始这里没有处理好总时间的关系！
            # *三分之一的reward
            reward = self.drone_local_aoi[uav_id, target_poi]*(self.T - self.drone_timing_now[uav_id])/(2*self.args.pre_reward_ratio)
            if self.args.buffer_punishment:
                if len(self.drone_step_reward[uav_id]) >= self.bs_random_factor:
                    reward = -self.args.punishment_value*self.args.reward_scale_size
                    
            self.drone_step_reward[uav_id].append(reward)
        
        done = False
        # !能量或者时间很少的时候，也可以为True
        if self.drone_timing_now[uav_id] >= self.T:
            done = True
        
        # !之后加上mask
        info = np.ones(self.K+self.N)
        info[self.drone_buffer[uav_id]] = 0
        
        if len(self.drone_buffer[uav_id]) == 0:
            info[self.K:self.K+self.N] = 0
        
        # 注意时间超出时的波动很大
        if self.drone_timing_now[uav_id] > self.T:
            reward = 0
        
        if self.args.print_info:
            print("Action UAV: {}, action: {}, time now: {}, time cost: {}, local AoI: {}, position: {}, buffer: {}".format(uav_id, action, self.drone_timing_now[uav_id], move_time, self.drone_local_aoi[uav_id], self.drone_position_now[uav_id],self.drone_buffer[uav_id]))
                    
        return  self._get_obs(uav_id), reward/self.args.reward_scale_size, done, info

    def time_cost(self, uav_id, action):
        if self.K <= action <= self.K+self.N-1:
            # print(action-self.K)
            target_position = self.base_pos[action-self.K]
            move_dis = euclidean(target_position, self.drone_position_now[uav_id])
            move_time = move_dis/self.fly_speed
        else:
            target_poi = action
            # print(target_poi)
            target_position = self.sensor_pos[target_poi]
            move_dis = euclidean(target_position, self.drone_position_now[uav_id])
            move_time = move_dis/self.fly_speed
        return move_time

    def render(self, mode="human"):
        """渲染环境"""
        print(f"Time: {self.global_timing:.2f}/{self.T}")
        print(f"Average AoI: {np.mean(self.aoi):.2f}")
        print("Drone Positions:", self.drone_pos)
        print("Drone Buffers:", self.drone_buffer)
        print("Drone Visit History:", self.buffer_timing)
        print("Drone Action Times:", self.drone_timing_now)

    def seed(self, seed=None):
        """设置随机种子"""
        np.random.seed(seed)
        
    def get_pos(self, action):
        all_positions = np.concatenate([self.sensor_pos, self.base_pos], axis=0)
        return all_positions[action]
        
    def visualize_routes(self, test_actions, output_dir="output", dpi=300, file="", nums=10):
        """
        Visualize POI positions, base station positions, and UAV routes (first 10 steps).
        
        Parameters:
        - self: Object containing sensor_pos (POI) and base_pos (base stations).
        - test_actions: Array of shape (n_uavs, n_steps, 2) representing UAV positions over time.
        - output_dir: Directory to save the visualization (default: 'output').
        - dpi: Resolution of the output image (default: 300).
        """
        # Validate inputs


        # Extract data
        poi_positions = self.sensor_pos  # Shape: (n_poi, 2)
        bs_positions = self.base_pos     # Shape: (n_bs, 2)
        uav_positions = test_actions     # Shape: (n_uavs, n_steps, 2)

        n_poi = poi_positions.shape[0]
        n_bs = bs_positions.shape[0]
        n_uavs = len(uav_positions)
        n_steps = min(len(uav_positions[0]), nums)  # Limit to first 10 steps
        for i in range(self.M):
            uav_positions[i] = uav_positions[i][:n_steps]
        uav_positions = np.array(uav_positions)

        # Compute map boundaries
        all_positions = np.concatenate([poi_positions, bs_positions, uav_positions.reshape(-1, 2)], axis=0)
        x_min, y_min = np.min(all_positions, axis=0) - 10  # Add padding
        x_max, y_max = np.max(all_positions, axis=0) + 10
        map_width = x_max - x_min
        map_height = y_max - y_min

        # Create figure
        plt.cla()
        plt.clf()
        plt.figure(figsize=(8, 8))

        # Plot POIs (blue circles)
        if n_poi > 0:
            plt.scatter(poi_positions[:, 0], poi_positions[:, 1], c='blue', marker='o', label='POIs', alpha=0.6)
            for i in range(n_poi):
                plt.text(poi_positions[i, 0] + 1, poi_positions[i, 1] + 1, str(i),
                            fontsize=10, color='black', weight='bold', ha='center', va='center',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.3'))

        # Plot base stations (red triangles)
        if n_bs > 0:
            plt.scatter(bs_positions[:, 0], bs_positions[:, 1], c='red', marker='^', s=100, label='Base Stations', alpha=0.8)

        # Plot UAV routes (different colors for each UAV)
        colors = cm.rainbow(np.linspace(0, 1, n_uavs))  # Generate distinct colors
        for i in range(n_uavs):
            # Extract first 10 steps (or fewer if n_steps < 10)
            route = uav_positions[i, :n_steps, :]
            plt.plot(route[:, 0], route[:, 1], c=colors[i], marker='o', markersize=5, 
                    label=f'UAV {i+1}', alpha=0.7, linewidth=2)
            # Mark start and end points
            plt.scatter(route[0, 0], route[0, 1], c=colors[i], marker='s', s=100, edgecolors='black')  # Start
            plt.scatter(route[-1, 0], route[-1, 1], c=colors[i], marker='*', s=150, edgecolors='black')  # End

        # Set plot properties
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'POI, Base Stations, and UAV Routes\n({n_poi} POIs, {n_bs} BS, {n_uavs} UAVs, Map: {int(map_width)}x{int(map_height)})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)

        # Save plot
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = file + f"poi_{n_poi}_bs_{n_bs}_map_{int(map_width)}x{int(map_height)}_uav_routes.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
