#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from rider import rider
import pandas as pd
from shang_chun_shan import populate_shared_ride_lengths
import os
import random

# In[208]:

interarrival_lambda = 1.831268035987099
waiting_lambda = 69.17902760965485
c = 0.7
cost_saving_factor = 0.8582936955140513 * 1
satisfaction_inflation = 4
# In[224]:


class Environment:
    def __init__(self, max_time = 200,
                 max_queue_length = 20, 
                 datasource = "data/env_event.csv",
                 seed = 152837):
        np.random.seed(seed)
        random.seed(seed)
        self.env_event = pd.read_csv(datasource)
        self.max_time = max_time
        self.cur_time = 0
        self.queue = {}
        self.coming_rider = {"rider_id": 0,
                             "rider": None,
                             "WTP": None,
                             "tolerance": None}
        self.set_coming_rider(True)
        self.profit = 0
        self.queue_size = 0
        self.done = False
        self.max_queue_length = max_queue_length
    
    def set_coming_rider(self, start = False):
        cur_rider_id = np.random.choice(self.env_event["rider_id"].values)
        cur_rider_builder = self.env_event.loc[self.env_event["rider_id"] == cur_rider_id, :]
        cur_rider_info = self.env_event.loc[self.env_event["rider_id"] == cur_rider_id, :]
        cur_min_WTP = cur_rider_info["min_WTP"].values[0]
        cur_max_WTP = cur_rider_info["max_WTP"].values[0]
        cur_rider_WTP = np.random.uniform(cur_min_WTP, cur_max_WTP)
        
        cur_rider_tol = cur_rider_info["tolerance"].values[0]
        if not start:
            interarrival_time = np.random.exponential(interarrival_lambda)
            coming_rider_id = interarrival_time + self.cur_time
        else:
            interarrival_time = 0
            coming_rider_id = 0
        cur_rider_builder["arrival_time"] = coming_rider_id
        if cur_rider_tol == -1:
            cur_min_tol = cur_rider_info["min_tol"].values[0] if cur_rider_info["min_tol"].values[0] != -1 else 0
            cur_rider_tol = np.random.exponential(waiting_lambda) + cur_min_tol
        coming_rider = rider(**cur_rider_builder.iloc[0, 1:9].to_dict())
        self.coming_rider = {"rider_id": coming_rider_id,
                             "rider": coming_rider,
                             "WTP": cur_rider_WTP,
                             "tolerance": cur_rider_tol}
        return interarrival_time

    def next_event(self):
        interarrival_time = self.set_coming_rider()
        self.cur_time += interarrival_time
        renege_cost, satisfaction_cost = 0, 0

        if self.cur_time > self.max_time:
            self.done = True
            return renege_cost, satisfaction_cost
            
        leaving_rider = []
        for rider_id, rider_info in self.queue.items():
            wait_time = self.cur_time - rider_info["rider"].arrival_time
            if wait_time > rider_info["tolerance"]:
                leaving_rider.append(rider_id)

        for rider_id in leaving_rider:
            rider_info = self.queue.pop(rider_id)
            self.queue_size -= 1
            cost = rider_info["rider"].solo_length * c
            self.profit -= cost
            renege_cost += cost
            wait_time = self.cur_time - rider_info["rider"].arrival_time
            satisfaction_cost += cost * min(1, np.log(wait_time * satisfaction_inflation))
        return renege_cost, satisfaction_cost


    def clear_queue(self):
        renege_cost, satisfaction_cost = 0, 0
        for rider_id, rider_info in self.queue.items():
            cost = rider_info["rider"].solo_length * c
            self.profit -= cost
            renege_cost += cost
            wait_time = self.cur_time - rider_info["rider"].arrival_time
            satisfaction_cost += cost * min(1, np.log(wait_time * satisfaction_inflation))
        return renege_cost, satisfaction_cost
    
    def step(self, action):
        price_reward, match_reward = 0, 0
        if self.done:
            renege_cost, satisfaction_cost = self.clear_queue()
            match_reward -= satisfaction_cost
            price_reward -= renege_cost
            return {"queue": {}, "profit": self.profit, "incoming_rider": self.coming_rider}, [0, match_reward], True

        price_action, match_action = action
        rider_id = self.coming_rider["rider_id"]
        rider_WTP = self.coming_rider["WTP"]
        if price_action <= rider_WTP:
            rev = price_action * self.coming_rider["rider"].solo_length
            # price_reward += (price_action - c) * self.coming_rider["rider"].solo_length
            price_reward += price_action * self.coming_rider["rider"].solo_length
            self.profit += rev
            if (match_action != self.max_queue_length) and (self.queue_size != 0):
                match_action = match_action % self.queue_size
                match_action = list(self.queue.keys())[match_action]
                match_rider = self.queue[match_action]
                origin_i = (self.coming_rider["rider"].pickup_lat, self.coming_rider["rider"].pickup_lon)
                destination_i = (self.coming_rider["rider"].dropoff_lat, self.coming_rider["rider"].dropoff_lon)
                origin_j = (match_rider["rider"].pickup_lat, match_rider["rider"].pickup_lon)
                destination_j = (match_rider["rider"].dropoff_lat, match_rider["rider"].dropoff_lon)
                trip_length, shared_length, _, _, _ = populate_shared_ride_lengths(origin_i, destination_i, origin_j, destination_j)
                cost = c * trip_length
                price_reward -= cost
                self.profit -= cost
                i_solo_length = self.coming_rider["rider"].solo_length
                j_solo_length = match_rider["rider"].solo_length
                j_quoted_price = match_rider["quoted_price"]
                earnings = price_action * i_solo_length + j_quoted_price * j_solo_length - cost
                j_wait_time = self.cur_time - match_rider["rider"].arrival_time
                j_tolerance_utilization = j_wait_time / match_rider["tolerance"] # how much tolerance for rider j has been consumed
                
                if (shared_length > 0) and (earnings > 0):
                    reduced_cost = (i_solo_length + j_solo_length - trip_length) * c
                    match_reward += reduced_cost * np.exp(earnings ** 1.2)
                # elif (price_action < c) or (j_quoted_price < c):
                else:
                    optim_price = max(rider_WTP, match_rider["WTP"])
                    lost_sales = (optim_price - min(price_action, j_quoted_price)) * trip_length
                    # match_reward -= abs(lost_sales * np.log(j_tolerance_utilization) * 5)
                    match_reward -= lost_sales * 5

                self.queue.pop(match_action)
                self.queue_size -= 1
            else:
                push_rider = self.coming_rider.copy()
                push_rider["quoted_price"] = price_action
                self.queue[rider_id] = push_rider
                self.queue_size += 1
        else:
            # the higher the index, the more penalty would be assigned for choosing high prices
            # higher index leads to lower pricing, lower index leads to higher pricing
            # (5, 5.5)
            price_reward -= max(0, (rider_WTP - c * cost_saving_factor)) * self.coming_rider["rider"].solo_length * ((price_action / rider_WTP) ** 3)
            # price_reward -= WTP * self.coming_rider["rider"].solo_length
        reward = [price_reward, match_reward]
        renege_cost, satisfaction_cost = self.next_event()
        match_reward -= satisfaction_cost
        price_reward -= renege_cost
        self.state = {"queue": self.queue.copy(), "profit": self.profit, "incoming_rider": self.coming_rider}
        return self.state, reward, self.done

    def reset(self):
        self.cur_time = 0
        self.profit = 0
        self.queue = {}
        self.queue_size = 0
        self.set_coming_rider(True)
        self.done = False
        self.state = {"queue": self.queue.copy(), "profit": self.profit, "incoming_rider": self.coming_rider}
        return self.state


# %%
