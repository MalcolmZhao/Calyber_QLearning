#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# In[3]:


from calyber_env import Environment
# import calyber_decision
import sys
sys.path.append('submission_pool')

import decision_20240414111318 as calyber_decision

# In[4]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# In[5]:


weight_name = "20240414111318.pt"
print("model token:", weight_name.split(".")[0])
# In[6]:


max_queue_length = 25
price_options_size = 26
queue_features = 6 + 14
price_starting_point = 0.35
max_match_decision_num = max_queue_length + 1

# In[7]:


shang_chun_shan = calyber_decision.decision(max_queue_length, queue_features, price_options_size)

# In[8]:


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# In[9]:


weight_path = os.path.join("submission_pool", weight_name)
shang_chun_shan.load_state_dict(torch.load(weight_path, map_location = device))

# In[10]:


shang_chun_shan = shang_chun_shan.to(device)

# In[11]:


scaler = {'mean_': np.array([ 1.86916373e+03,  4.18677717e+01, -8.76599503e+01,  4.18679074e+01,
                             -8.76519859e+01,  3.63340593e+00]),
        'scale_': np.array([1.03639625e+03, 7.17738253e-02, 4.60401620e-02, 6.06944593e-02,
                            4.07789271e-02, 2.47791716e+00]),
        'var_': np.array([1.07411720e+06, 5.15148200e-03, 2.11969652e-03, 3.68381739e-03,
                          1.66292089e-03, 6.14007346e+00]),
        'n_samples_seen_': 11788}

# In[12]:


def standardize(data, scaler):
    return (data - scaler['mean_']) / scaler['scale_']

# In[13]:


def cluster_from_rider(rider):
    plain = np.zeros(7 * 2)
    pickup = rider.pickup_area
    dropoff = rider.dropoff_area
    pickup = area_code[pickup]
    dropoff = area_code[dropoff]
    plain[pickup] = 1
    plain[dropoff + 7] = 1
    return plain

# In[14]:


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
    # rider_array1 = rider_array1[:, :6]
    rider_array1 = np.concatenate((rider_array1, rider_area_info.reshape(1, 14)), axis = 1)
    # rider_array1 = np.concatenate((rider_array1, np.zeros((1, 1))), axis = 1)
    rider_array[0, :] = rider_array1

    queue_array = np.zeros((max_queue_length, queue_features))
    if len(queue) > 0:
        for idx, rider in enumerate(queue.values()):
            rider_area_info = cluster_from_rider(rider["rider"])
            rider_array1 = np.array([[rider["rider"].arrival_time,
                                    rider["rider"].pickup_lat,
                                    rider["rider"].pickup_lon,
                                    rider["rider"].dropoff_lat,
                                    rider["rider"].dropoff_lon,
                                    rider["rider"].solo_length
                                    ]])
            rider_array1 = standardize(rider_array1, scaler)
            rider_array1 = np.concatenate((rider_array1, rider_area_info.reshape(1, 14)), axis = 1)
            queue_array[idx, :] = rider_array1
            
    riders_tensor = torch.tensor(rider_array, dtype=torch.float32).to(dev)
    queue_tensor = torch.tensor(queue_array, dtype=torch.float32).to(dev)
    return [riders_tensor, queue_tensor]

# In[15]:


price_list = np.linspace(price_starting_point, 1, price_options_size)

# In[16]:


area_code = pd.read_csv("data/area_cluster.csv")
area_code = area_code.loc[:, ["area_id", "cluster"]]
area_code = area_code.set_index("area_id").to_dict()["cluster"]

# In[17]:


max_time = 3600
price_cost_penalty = 1
price_gamma = 0.9999
match_gamma = 0.975
rider_nums = max_time / 1.831268035987099
profit = -0.5

# In[18]:


calyber = Environment(
    max_time = max_time,
    max_queue_length = max_queue_length,
    price_cost_penalty = price_cost_penalty,
    match_gamma = match_gamma,
    price_gamma = price_gamma,
    seed = 225917,
    # datasource = "data/env_event.csv"
    datasource = "data/env_event_val.csv"
    )

# In[19]:


price_rewards = []
match_rewards = []
profits = []

# In[20]:


state = calyber.reset()

# In[21]:


def play_by_model(env, state):
    price_noise = np.random.rand()
    match_noise = np.random.rand()
    price_action_idx = np.random.randint(price_options_size)
    match_action_idx = np.random.randint(max_match_decision_num)
    epsilon = 0
    if not ((price_noise < epsilon) and (match_noise < epsilon)):
        shang_chun_shan.eval()
        with torch.no_grad():
            rider_tensor, queue_tensor = state_preprocess(state)
            rider_tensor = rider_tensor.unsqueeze(0)
            queue_tensor = queue_tensor.unsqueeze(0)
            price_Q_values, match_Q_values = shang_chun_shan(rider_tensor, queue_tensor)
            price_Q_values = price_Q_values.squeeze()
            match_Q_values = match_Q_values.squeeze()
        del rider_tensor, queue_tensor
        if price_noise >= epsilon:
            price_action_idx = torch.argmax(price_Q_values).item()
        if match_noise >= epsilon:
            match_action_idx = torch.argmax(match_Q_values).item()
    action = [price_list[price_action_idx], match_action_idx]
    next_state, reward, done = env.step(action)
    return next_state, reward, done

# In[22]:


for episode in range(100):
    state = calyber.reset()
    price_reward = 0
    match_reward = 0
    for step in range(max_time):
        state, reward, done = play_by_model(calyber, state)
        price_reward += reward[0]
        match_reward += reward[1]
        if step % 200 == 0:
            print("""\rEpisode: {}, step: {}, profit: {:.3f}, unit profit: {:.3f}""".format(
                episode, step, profit * rider_nums, profit), end=" " * 20)
        if done:
            break
    price_rewards.append(price_reward)
    match_rewards.append(match_reward)
    rider_nums = step
    profit = calyber.profit / rider_nums
    profits.append(profit)
    print("""\rEpisode: {}, profit: {:.3f}, unit profit: {:.3f}""".format(
        episode, profit * rider_nums, profit), end = " " * 20)

# In[23]:


def training_plot(price_rewards, match_rewards, profits):
    mean_profit = np.mean(profits)
    std_profit = np.std(profits)
    price_rewards = (price_rewards - np.mean(price_rewards)) / np.std(price_rewards)
    match_rewards = (match_rewards- np.mean(match_rewards)) / np.std(match_rewards)
    price_rewards = price_rewards * std_profit + mean_profit
    match_rewards = match_rewards * std_profit + mean_profit
    plt.figure(figsize = (10, 3))
    for line, pattern, name in zip(["--", "-.", "-"], 
                                [price_rewards, match_rewards, profits], 
                                ["price", "match", "profit"]):
        plt.plot(pattern, label = name, linestyle = line)
    # plt.title("max profit is %.6f at episode %d"%(max_profit, best_episode + 1))
    plt.title("Average profit is %.6f, std is %.6f"%(mean_profit, std_profit))
    plt.legend()
    plot_name = "simulation_result_" + weight_name.split(".")[0] + ".png"
    plt.savefig(os.path.join("pictures", plot_name))
    plt.close()

# In[24]:


training_plot(price_rewards, match_rewards, profits)

# In[ ]:



