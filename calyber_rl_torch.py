#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

# In[2]:


import numpy as np
import pandas as pd
import random
from collections import deque
import time
import os
import pickle
import matplotlib.pyplot as plt
import copy

# In[3]:


from calyber_env import Environment, populate_shared_ride_lengths

# In[4]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# In[5]:


import calyber_decision
# import decision_20240411221646 as calyber_decision

# In[6]:


seed = 42
print("random seed:", seed)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# In[7]:


weight_name = "NA.pt"

# In[8]:


def load_weights(model, path):
    if os.path.exists(path):
        weights = torch.load(path)
        model_dict = model.state_dict()
        pretrained_dict = {}
        for k, v in weights.items():
            if k in model_dict:
                if v.size() == model_dict[k].size():
                    pretrained_dict[k] = v
        model.load_state_dict(pretrained_dict, strict = False)
        print("Weights loaded successfully")

# In[9]:


max_queue_length = 25
price_options_size = 10
cluster_num = 20
queue_features = 6 + cluster_num * 2
price_starting_point = 0.4
price_end_point = 0.95
max_match_decision_num = max_queue_length + 1

# l1_lmd = 1
l2_lmd = 0

# In[10]:


online = calyber_decision.decision(max_queue_length, queue_features, price_options_size)
loss_fn = nn.MSELoss()
# loss_L1 = nn.L1Loss()
price_optimizer = optim.Adam(online.parameters(), lr = 0.001, weight_decay = l2_lmd)
match_optimizer = optim.Adam(online.parameters(), lr = 0.001, weight_decay = l2_lmd)

# In[11]:


# weight_folder = "20240318213342"
weight_path = os.path.join("variables", weight_name)
load_weights(online, weight_path)
target_model = copy.deepcopy(online)

# In[12]:


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Cuda detected")
else:
    device = torch.device("cpu")
    print("Running on CPU")

# In[13]:


target_model.to(device)
online.to(device)

# In[14]:


# scaler = {'mean_': np.array([ 1.86916373e+03,  4.18677717e+01, -8.76599503e+01,  4.18679074e+01,
#                              -8.76519859e+01,  3.63340593e+00,  5.58944205e-01]),
#         'scale_': np.array([1.03639625e+03, 7.17738253e-02, 4.60401620e-02, 6.06944593e-02,
#                             4.07789271e-02, 2.47791716e+00, 9.87676635e-02]),
#         'var_': np.array([1.07411720e+06, 5.15148200e-03, 2.11969652e-03, 3.68381739e-03,
#                           1.66292089e-03, 6.14007346e+00, 9.75505135e-03]),
#         'n_samples_seen_': 11788}
scaler = {'mean_': np.array([ 1.86916373e+03,  4.18677717e+01, -8.76599503e+01,  4.18679074e+01,
                             -8.76519859e+01,  3.63340593e+00]),
        'scale_': np.array([1.03639625e+03, 7.17738253e-02, 4.60401620e-02, 6.06944593e-02,
                            4.07789271e-02, 2.47791716e+00]),
        'var_': np.array([1.07411720e+06, 5.15148200e-03, 2.11969652e-03, 3.68381739e-03,
                          1.66292089e-03, 6.14007346e+00]),
        'n_samples_seen_': 11788}

# In[15]:


def standardize(data, scaler):
    return (data - scaler['mean_']) / scaler['scale_']

# In[16]:


def cluster_from_rider(rider, num_clusters = 20):
    plain = np.zeros(num_clusters * 2)
    pickup = rider.pickup_area
    dropoff = rider.dropoff_area
    pickup = area_code[pickup]
    dropoff = area_code[dropoff]
    plain[pickup] = 1
    plain[dropoff + num_clusters] = 1
    return plain

# In[17]:


def state_preprocess(single_state, queue_features = queue_features, dev = device):
    queue = single_state["queue"]
    rider = single_state["incoming_rider"]["rider"]
    rider_area_info = cluster_from_rider(rider)
    rider_array = np.zeros((max_queue_length, queue_features))
    rider_array1 = np.array([[
                            rider.arrival_time,
                            rider.pickup_lat,
                            rider.pickup_lon,
                            rider.dropoff_lat,
                            rider.dropoff_lon,
                            rider.solo_length
                            ]])
    rider_array1 = standardize(rider_array1, scaler)  # Using the standardize function
    rider_array1 = np.concatenate((rider_array1, rider_area_info.reshape(1, -1)), axis = 1)
    rider_array[0, :] = rider_array1

    queue_array = np.zeros((max_queue_length, queue_features))
    if len(queue) > 0:
        for idx, rider in enumerate(queue.values()):
            rider_area_info = cluster_from_rider(rider["rider"], cluster_num)
            rider_array1 = np.array([[rider["rider"].arrival_time,
                                    rider["rider"].pickup_lat,
                                    rider["rider"].pickup_lon,
                                    rider["rider"].dropoff_lat,
                                    rider["rider"].dropoff_lon,
                                    rider["rider"].solo_length
                                    ]])
            rider_array1 = standardize(rider_array1, scaler)
            rider_array1 = np.concatenate((rider_array1, rider_area_info.reshape(1, -1)), axis = 1)
            queue_array[idx, :] = rider_array1
            if idx == max_queue_length - 1:
                break
            
    riders_tensor = torch.tensor(rider_array, dtype=torch.float32).to(dev)
    queue_tensor = torch.tensor(queue_array, dtype=torch.float32).to(dev)
    return [riders_tensor, queue_tensor]

# In[18]:


price_list = np.linspace(price_starting_point + 1e-2, price_end_point - 1e-2, price_options_size)

# In[19]:


class DDQNAgent:
    def __init__(self, pricing_actions_nums, matching_actions_nums):
        self.pricing_actions_nums = pricing_actions_nums
        self.matching_actions_nums = matching_actions_nums
        self.memory = deque(maxlen = max_memory_length)  # Experience replay memory
        self.price_priority = deque(maxlen = max_memory_length)
        self.match_priority = deque(maxlen = max_memory_length)
        self.price_gamma = price_gamma  # Discount factor
        self.match_gamma = match_gamma
        # self.batch_size = batch_size
        self.sample_theta = sample_theta

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.price_priority.append(1e10)
        self.match_priority.append(1e10)
        # gc.collect()

    def sample_experience(self, optimize_type, batch_size = 32):
        if optimize_type == 0:
            priority_array = np.array(self.price_priority)
        else:
            priority_array = np.array(self.match_priority)

        priority_array = priority_array ** self.sample_theta
        sampling_probs = priority_array / np.sum(priority_array)
        indices = np.random.choice(len(self.memory), size = batch_size, p = sampling_probs)

        del priority_array
        batch = [self.memory[index] for index in indices]
        actions, rewards, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in [1, 2, 4]]
        states, next_states = [
            [experience[field_index] for experience in batch]
            for field_index in [0, 3]]
        del batch

        states = self.stack_states(states)
        next_states = self.stack_states(next_states)

        return states, actions, rewards, next_states, dones, indices, sampling_probs[indices]
    
    def stack_states(self, states):
        rider_list = []
        queue_list = []
        for state in states:
            rider_tensor, queue_tensor = state
            rider_list.append(rider_tensor)
            queue_list.append(queue_tensor)
        rider_tensor = torch.stack(rider_list).to(device)
        queue_tensor = torch.stack(queue_list).to(device)
        return rider_tensor, queue_tensor 

    def update_priority(self, optimize_type, td_errors, indices):
        if optimize_type == 0:
            for idx, error in zip(indices, td_errors):
                self.price_priority[idx] = error + 1e-2
        else:
            for idx, error in zip(indices, td_errors):
                self.match_priority[idx] = error + 1e-2

    def gradient_descent(self, rider_tensor, queue_tensor, 
                         target_val, mask, 
                         optimize_type, 
                         optimizer, weights, indices):
        online.train()
        # online.eval()
        optimizer.zero_grad()
        target_f = online(rider_tensor, queue_tensor)[optimize_type]
        Q_values = torch.sum(target_f * mask, dim = 1, keepdim = True)
        loss = torch.mean(loss_fn(Q_values, target_val) * weights)
        del rider_tensor, queue_tensor
        loss.backward()
        optimizer.step()

        td_errors = torch.abs(target_val.squeeze() - Q_values.squeeze())
        td_errors = torch.clamp(td_errors, min = 0, max = 1e10).detach().cpu().numpy()
        self.update_priority(optimize_type, td_errors, indices)


    def replay(self, optimize_type = 0, beta = 0.4, batch_size = 32):
        state, action, reward, next_state, done, indices, prob = self.sample_experience(optimize_type, batch_size)
        done = torch.tensor(done, dtype = torch.float32).to(device)
        price_target, match_target = reward[:, 0], reward[:, 1]
        price_target = torch.tensor(price_target, dtype=torch.float32).to(device)
        match_target = torch.tensor(match_target, dtype=torch.float32).to(device)
        next_rider_array, next_queue_array = next_state
        next_rider_array = next_rider_array.to(device)
        next_queue_array = next_queue_array.to(device)
        rider_tensor, queue_tensor = state
        rider_tensor = rider_tensor.to(device)
        queue_tensor = queue_tensor.to(device)
        if batch_size == 1:
            # next_rider_array = next_rider_array.unsqueeze(0)
            # next_queue_array = next_queue_array.unsqueeze(0)
            # rider_tensor = rider_tensor.unsqueeze(0)
            # queue_tensor = queue_tensor.unsqueeze(0)
            price_target = price_target.unsqueeze(0)
            match_target = match_target.unsqueeze(0)
            done = done.unsqueeze(0)
        target_model.eval()
        online.eval()
        with torch.no_grad():
            next_price_output, next_match_output = online(next_rider_array, next_queue_array)
            next_price_best_target, next_match_best_target = target_model(next_rider_array, next_queue_array)
        del next_rider_array, next_queue_array

        weights = torch.tensor((len(self.memory) * prob) ** (-beta)).to(device)

        del prob
        
        if optimize_type == 0:
            best_price = torch.argmax(next_price_output, axis = 1)
            price_next_mask = F.one_hot(best_price, self.pricing_actions_nums).to(device)
            del best_price, next_price_output
            next_price_best_target = torch.sum(next_price_best_target * price_next_mask, axis = 1)
            price_target += self.price_gamma * next_price_best_target * (1 - done)
            del next_price_best_target, done
            price_target = price_target.reshape(-1, 1)
            price_action_idx = np.argmax(action[:, 0][:, None] == price_list, axis = 1)
            price_action_idx = torch.tensor(price_action_idx, dtype=torch.long).to(device)
            price_mask = F.one_hot(price_action_idx, self.pricing_actions_nums).to(device)
            del price_action_idx
            self.gradient_descent(rider_tensor, queue_tensor, price_target, price_mask, 0, price_optimizer, weights, indices)
        else:
            best_match = torch.argmax(next_match_output, axis = 1)
            match_next_mask = F.one_hot(best_match, self.matching_actions_nums).to(device)
            del best_match, next_match_output
            next_match_best_target = torch.sum(next_match_best_target * match_next_mask, axis = 1)
            match_target += self.match_gamma * next_match_best_target * (1 - done)
            del next_match_best_target, done
            match_target = match_target.reshape(-1, 1)
            match_action_idx = torch.tensor(action[:, 1], dtype=torch.long).to(device)
            match_mask = F.one_hot(match_action_idx, self.matching_actions_nums).to(device)
            del match_action_idx
            self.gradient_descent(rider_tensor, queue_tensor, match_target, match_mask, 1, match_optimizer, weights, indices)

# In[20]:


def epsilon_greedy_policy(state, epsilon = 0):
    noise = np.random.rand()
    price_action_idx = np.random.randint(price_options_size)
    match_choices = list(range(max_match_decision_num))
    match_possibilities = [1] * (max_match_decision_num - 1) + [max_match_decision_num - 1]
    match_action_idx = random.choices(match_choices, weights = match_possibilities)[0]
    if noise >= epsilon:
        online.eval()
        with torch.no_grad():
            rider_tensor, queue_tensor = state_preprocess(state)
            rider_tensor = rider_tensor.unsqueeze(0)
            queue_tensor = queue_tensor.unsqueeze(0)
            price_Q_values, match_Q_values = online(rider_tensor, queue_tensor)
            price_Q_values = price_Q_values.squeeze()
            match_Q_values = match_Q_values.squeeze()
        del rider_tensor, queue_tensor
        price_action_idx = torch.argmax(price_Q_values).item()
        match_action_idx = torch.argmax(match_Q_values).item()

    return [price_list[price_action_idx], match_action_idx]


# In[21]:


def play_one_step(env, agent, state, epsilon, validation = False):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done = env.step(action)
    state_tensor = state_preprocess(state, dev = torch.device("cpu"))
    next_state_tensor = state_preprocess(next_state, dev = torch.device("cpu"))
    if not validation:
        agent.remember(state_tensor, action, reward, next_state_tensor, done)
        return next_state, reward, done
    else:
        return next_state, reward, done, action

# In[22]:


def training_plot(training_time, price_rewards, match_rewards, profits, profits_val, training_stage):
    length = len(profits_val)
    start = max(0, length - plot_duration)
    episode_range = np.array(range(start, length))
    best_episode = np.argmax(profits_val[training_stage:]) + training_stage
    max_profit = np.max(profits_val[training_stage:])
    mean_profit = np.mean(profits_val)
    std_profit = np.std(profits_val)
    price_rewards = (price_rewards - np.mean(price_rewards)) / np.std(price_rewards)
    match_rewards = (match_rewards- np.mean(match_rewards)) / np.std(match_rewards)
    price_rewards = price_rewards * std_profit + mean_profit
    match_rewards = match_rewards * std_profit + mean_profit
    evolve_name = "evolve_%s.jpg"%(training_time)
    evolve_path = os.path.join("pictures", evolve_name)
    plt.figure(figsize = (10, 3))
    for line, pattern, name in zip(["--", "-.", "-", "-"], 
                                [price_rewards, match_rewards, profits, profits_val], 
                                ["price", "match", "profit", "val profit"]):
        plt.plot(pattern, label = name, linestyle = line)
    plt.title("max validation profit is %.6f at episode %d"%(max_profit, best_episode))
    plt.grid(True)
    try:
        plt.savefig(evolve_path)
    except Exception as e:
        print(e)
    plt.close()

    profits = profits[start: length]
    profits_val = profits_val[start: length]
    price_rewards = price_rewards[start: length]
    match_rewards = match_rewards[start: length]
    checking_name = "check_%s.jpg"%(training_time)
    checking_path = os.path.join("pictures", checking_name)
    plt.figure(figsize = (10, 3))
    for line, pattern, name in zip(["--", "-.", "-", "-"],
                                [price_rewards, match_rewards, profits, profits_val],
                                ["price", "match", "profit", "val profit"]):
        plt.plot(episode_range, pattern, label = name, linestyle = line)
    plt.title("max validation profit is %.6f at episode %d"%(max_profit, best_episode))
    plt.grid(True)
    try:
        plt.savefig(checking_path)
    except Exception as e:
        print(e)
    plt.close()

# In[23]:


def loggings(episode, current_profit, best_profit, model_weights, training_time):
    # model_path = os.path.join("variables", "weights_%s.pt"%training_time)
    # torch.save(model_weights, model_path)
    with open("variables/logs.txt", "a") as file:
        file.write("[Running time: %s]: episode: %d, current profit: %.8f, best validation profit: %.8f\n"%(training_time, episode, current_profit, best_profit))
    # picture_name="evolve_%s.jpg"%training_time
    # training_plot(training_time, np.array(price_rewards), np.array(match_rewards), np.array(profits))

# In[24]:


price_gamma = 1 - 1e-10
match_gamma = 0.9999
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 800

price_rewards = []
match_rewards = []
profits = []
profits_val = []

# In[25]:

sample_theta = 0.65 # magnitude of how much importance should be taken into consideration when do sampling
start_beta = 0.4 # magnitude of how much to compensate for the importance sampling bias

# In[26]:


best_profit = -2
previous_stop = 0

max_time = 3600
max_memory_length = 100000

episodes = 3000
replay_per_episode = 200

# In[27]:


# interarrival_lambda = 1.831268035987099
# waiting_lambda = 69.17902760965485

# c = 0.7
# cost_saving_factor = 0.8582936955140513

# In[28]:


explore_steps = max_time + 200
# rider_nums = max_time / 1.831268035987099
training_stage = 10
train_start = max(0, previous_stop - training_stage)

# In[29]:


events_per_round = int(max_time / 1.831268035987099)
num_episode_in_memomry = int(max_memory_length / events_per_round)
plot_duration = num_episode_in_memomry * 3

# In[30]:


area_code = pd.read_csv("data/area_lat_lon_cluster.csv")
area_code = area_code.loc[:, ["area_id", "cluster"]]
area_code = area_code.set_index("area_id").to_dict()["cluster"]

# In[31]:


calyber = Environment(
    max_time = max_time,
    max_queue_length = max_queue_length,
    seed = seed
    )
calyber_val = Environment(
    max_time = max_time,
    max_queue_length = max_queue_length,
    seed = seed + 500,
    datasource = "data/env_event_val.csv"
)
calyber_robot = DDQNAgent(
    pricing_actions_nums = price_options_size,
    matching_actions_nums = max_match_decision_num
    )

# In[32]:

print("Training for double channel")
print("Training will start after episode", training_stage)
profit = best_profit
training_time = time.strftime("%Y%m%d%H%M%S")
print("Training time is", training_time)

# In[33]:


start_from = 0
price_unique_action = price_options_size
match_unique_action = max_match_decision_num
for episode in range(start_from, episodes):
    state = calyber.reset()
    state_val = calyber_val.reset()
    done_val = False
    done = False
    price_reward = 0
    match_reward = 0
    val_action_dict = {"rider_id": [], "wtp": [], "tolerance": [], 
                       "pickup_lat": [], "pickup_lon": [], "dropoff_lat": [], "dropoff_lon": [],
                       "solo_length": [], "j_solo_length": [], "price": [], "converted_or_not": [], 
                       "queue_size_before": [], "queue_size_after": [], "match": [], "match_idx": [], 
                       "trip_length": [], "shared_length":[], "solo_length_without_match": []}
    for step in range(explore_steps):
        epsilon = max(1 - (episode + train_start) / epsilon_decay, epsilon_min)
        if not done:
            state, reward, done = play_one_step(calyber, calyber_robot, state, epsilon)
            price_reward += reward[0]
            match_reward += reward[1]
        if not done_val:
            processing_rider = state_val["incoming_rider"].copy()
            val_action_dict["rider_id"].append(processing_rider["rider_id"])
            val_action_dict["wtp"].append(processing_rider["WTP"])
            val_action_dict["tolerance"].append(processing_rider["tolerance"])
            rider_info = processing_rider["rider"]
            i_pickup_lat = rider_info.pickup_lat
            i_pickup_lon = rider_info.pickup_lon
            i_dropoff_lat = rider_info.dropoff_lat
            i_dropoff_lon = rider_info.dropoff_lon
            val_action_dict["pickup_lat"].append(i_pickup_lat)
            val_action_dict["pickup_lon"].append(i_pickup_lon)
            val_action_dict["dropoff_lat"].append(i_dropoff_lat)
            val_action_dict["dropoff_lon"].append(i_dropoff_lon)
            val_action_dict["solo_length"].append(rider_info.solo_length)
            
            queue_size_before = len(state_val["queue"].keys())
            val_action_dict["queue_size_before"].append(queue_size_before)
            before_queue = state_val["queue"].copy()

            state_val, _, done_val, action = play_one_step(calyber_val, calyber_robot, state_val, 0, True)
            val_action_dict["price"].append(action[0])
            converted_or_not = int(processing_rider["WTP"] >= action[0])
            val_action_dict["converted_or_not"].append(converted_or_not)

            match_idx, matching_outcome = -1, -1
            if action[1] == max_queue_length:
                match_idx = max_queue_length
            elif (queue_size_before > 0) and (converted_or_not == 1):
                match_idx = action[1] % queue_size_before
                matching_rider = before_queue[list(before_queue.keys())[match_idx]].copy()
                matching_outcome = matching_rider["rider_id"]
            val_action_dict["match"].append(matching_outcome)
            val_action_dict["match_idx"].append(match_idx)
            trip_length, shared_length, j_solo_length, solo_length_without_match = -1, -1, -1, -1
            if (matching_outcome != -1):
                j_pickup_lat = matching_rider["rider"].pickup_lat
                j_pickup_lon = matching_rider["rider"].pickup_lon
                j_dropoff_lat = matching_rider["rider"].dropoff_lat
                j_dropoff_lon = matching_rider["rider"].dropoff_lon
                origin_i = (i_pickup_lat, i_pickup_lon)
                destination_i = (i_dropoff_lat, i_dropoff_lon)
                origin_j = (j_pickup_lat, j_pickup_lon)
                destination_j = (j_dropoff_lat, j_dropoff_lon)
                j_solo_length = matching_rider["rider"].solo_length
                solo_length_without_match = j_solo_length + rider_info.solo_length
                trip_length, shared_length, _, _, _ = populate_shared_ride_lengths(origin_i, destination_i, origin_j, destination_j)
            val_action_dict["trip_length"].append(trip_length)
            val_action_dict["shared_length"].append(shared_length)
            val_action_dict["j_solo_length"].append(j_solo_length)
            val_action_dict["solo_length_without_match"].append(solo_length_without_match)
            val_action_dict["queue_size_after"].append(len(state_val["queue"].keys()))
        if step % 400 == 0:
            print("""\rEpisode: {}, step: {}, profit: {:.3f}, unit profit: {:.3f}, best validation profit: {:.3f}, #p: {}. #m: {}""".format(
                episode, step, profit * (max_time / 60), profit, best_profit, price_unique_action, match_unique_action), end=" " * 5)
        if done and done_val:
            break
    val_action_df = pd.DataFrame(val_action_dict)
    if episode % 5 == 0:
        val_action_df.to_csv("dump/%s_tmp_decison_pattern.csv"%training_time, index = False)
    price_unique_action = val_action_df["price"].unique().shape[0]
    match_unique_action = val_action_df["match_idx"].unique().shape[0]
    price_rewards.append(price_reward)
    match_rewards.append(match_reward)
    del reward, price_reward, match_reward

    profit = calyber.profit / (max_time / 60)
    profit_val = calyber_val.profit / (max_time / 60)
    profits.append(profit)
    profits_val.append(profit_val)
    
    
    if episode >= training_stage:
        if (profit_val >= best_profit) and (episode >= training_stage + 1):
            best_weights = online.state_dict()
            best_profit = profit_val
            model_path = os.path.join("variables", "best_%s.pt"%training_time)
            torch.save(best_weights, model_path)
            val_action_df.to_csv("dump/%s_best_decison_pattern.csv"%training_time, index = False)

        batch_size = max(16, 2 ** int(8 - (episode + train_start) / 300))
        # batch_size = 1
        beta = min(1, start_beta * 1.0001 ** (episode + train_start))

        for play in range(replay_per_episode):
            calyber_robot.replay(play % 2, beta, batch_size)
            if play % 50 == 0:
                print("\rTraining-Episode: {}, replay: {}, batch size: {}, unique price action: {}, unique match action: {}".format(episode, play, batch_size, price_unique_action, match_unique_action), end = " " * 20)

        if (episode > training_stage):
            # loggings(episode, profit, best_profit, online.state_dict(), training_time)
            training_plot(training_time, np.array(price_rewards), np.array(match_rewards), np.array(profits), np.array(profits_val), training_stage + 1)
            if episode % 20 == 0:
                target_model.load_state_dict(online.state_dict())
    # print("""\rEpisode: {}, eps: {:.3f}, profit: {:.3f}, unit profit: {:.3f}, best validation profit: {:.3f}""".format(
    #     episode, epsilon, profit * rider_nums, profit, best_profit), end=" " * 20)

print("Training complete")
