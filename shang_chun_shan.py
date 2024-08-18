"""
This file contains three parts:
- Part 1: the skeleton code for the pricing and matching policies. You need to fill in your code of your pricing and matching policies in this part.
- Part 2: A helper function populate_shared_ride_lengths() for calculating the shortest trip length for a shared ride is also provided in this file and you can directly use it in your policy development if you need.
- Part 3: A unit test script for testing the correctness and execution time of your code. You can run this file with given examples to test your code.

This file is the only thing what you need to submit in the end.

TODO: Rename this file with <name>.py where <name> is your team name *all* in lower case with underscores
e.g. awesome_team.py
"""

# NOTE: you are allowed to use packages loaded below ONLY. If you want to use any other packages, please contact Venkatesh Ravi (venkatesh@berkeley.edu).
import numpy as np
import itertools
import math
import random
import copy
import time
import pandas as pd
import haversine
import unittest
import pickle
import collections
import torch
import torch.nn as nn

# TODO: Replace the string "TeamName" below with your own team's name in camel case. For e.g. "AwesomeTeam"
TEAM_NAME = "Shang_Chun_Shan"
cluster_dict = {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 5, 7: 5, 8: 2, 9: 1, 10: 1, 11: 1, 12: 1, 13: 5, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1, 
 21: 5, 22: 5, 23: 1, 24: 2, 25: 1, 26: 1, 27: 2, 28: 2, 29: 4, 30: 4, 31: 2, 32: 2, 33: 2, 34: 2, 35: 2, 36: 0, 37: 0, 38: 0, 39: 0, 
 40: 0, 41: 0, 42: 0, 43: 0, 44: 0, 45: 6, 46: 6, 47: 6, 48: 6, 49: 3, 50: 6, 51: 6, 52: 6, 53: 3, 54: 6, 55: 6, 56: 4, 57: 4, 58: 4, 
 59: 2, 60: 2, 61: 2, 62: 4, 63: 4, 64: 4, 65: 4, 66: 4, 67: 4, 68: 0, 69: 0, 70: 4, 71: 3, 72: 3, 73: 3, 74: 3, 75: 3, 76: 5}

# class decision(nn.Module):
#     def __init__(self, max_queue_length, queue_features, price_options_size):
#         super(decision, self).__init__()
#         self.queue_features = queue_features
#         self.dense1 = nn.Linear(self.queue_features, 16)
#         self.dense2 = nn.Linear(16, 64)
#         self.dense3 = nn.Linear(64, 256)

#         self.queue_lstm1 = nn.LSTM(input_size = queue_features, 
#                                    hidden_size = 64, num_layers = 1,
#                                    batch_first=True)
#         self.queue_lstm2 = nn.LSTM(input_size = 64, 
#                                    hidden_size = 128, num_layers = 1, 
#                                    batch_first=True)
#         self.queue_lstm3 = nn.LSTM(input_size = 128, 
#                                    hidden_size = 256, num_layers = 1, 
#                                    batch_first=True, dropout = 0.35)
        
#         self.fc1 = nn.Linear(512, 256)
#         self.fc2 = nn.Linear(256, 128)

#         self.match_fc_1 = nn.Linear(128, 64)
#         self.match_fc_2 = nn.Linear(64, 32)
#         self.match_fc_2_2 = nn.Linear(32, queue_features)
#         self.match_fc_4_1 = nn.Linear(queue_features, 32)
#         self.match_output = nn.Linear(32, max_queue_length + 1)
#         self.match_values = nn.Linear(32, 1)

#         self.price_fc_1 = nn.Linear(128, 64)
#         self.price_fc_2_1 = nn.Linear(64, 32)
#         self.price_fc_3_1 = nn.Linear(32, queue_features)
#         self.price_fc_4_1 = nn.Linear(queue_features, 32)
#         self.price_output = nn.Linear(32, price_options_size)
#         self.price_values = nn.Linear(32, 1)

#         self.dropout = nn.Dropout(p = 0.35)

#     def forward(self, current_rider_input, queue_input):
#         current_rider = torch.max_pool1d(current_rider_input.permute(0, 2, 1), kernel_size=current_rider_input.size(1)).squeeze(2)
#         rider = self.dense3(nn.ReLU()(self.dense2(nn.ReLU()(self.dense1(current_rider)))))
#         rider = self.dropout(rider)
#         rider = nn.ReLU()(rider)

#         queue, _ = self.queue_lstm1(queue_input)
#         queue, _ = self.queue_lstm2(queue)
#         queue, _ = self.queue_lstm3(queue)
#         queue = queue[:, -1, :]  # Taking the last output of the LSTM

#         concat = torch.cat((rider, queue), dim=1)

#         fc = self.fc2(nn.ReLU()(self.fc1(concat)))
#         fc = nn.ReLU()(self.dropout(fc))

#         match = nn.ELU()(self.match_fc_1(fc))
#         match = (self.match_fc_2(match))
#         match = nn.ELU()(self.dropout(match))
#         match = (self.match_fc_2_2(match))
#         match = match + current_rider
#         match = nn.ELU()(match)
#         match = nn.ELU()(self.match_fc_4_1(match))
#         match_output = self.match_output(match)
#         match_values = self.match_values(match)
#         match_output = match_output - torch.max(match_output, dim=1, keepdim=True)[0]  # Subtracting the max value to avoid overflow
#         match_output = match_output + match_values

#         price = nn.ReLU()(self.price_fc_1(fc))
#         price = (self.price_fc_2_1(price))
#         price = self.dropout(price)
#         price = nn.ELU()(price)
#         price = (self.price_fc_3_1(price))
#         price = price + current_rider
#         price = nn.ELU()(price)
#         price = nn.ELU()(self.price_fc_4_1(price))
#         price_output = self.price_output(price)
#         price_values = self.price_values(price)
#         price_output = price_output - torch.max(price_output, dim=1, keepdim=True)[0]
#         price_output = price_output + price_values
#         return price_output, match_output
    
# weight_name = "Shang_Chun_Shan.pt"
# max_queue_length = 25
# price_options_size = 26
# queue_features = 6 + 14
# price_starting_point = 0.35
# max_match_decision_num = max_queue_length + 1
# shang_chun_shan_model = decision(max_queue_length, queue_features, price_options_size)
# shang_chun_shan_model.load_state_dict(torch.load(weight_name))
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Running on the GPU")
# else:
#     device = torch.device("cpu")
#     print("Running on the CPU")
# shang_chun_shan_model.to(device)
# shang_chun_shan_model.eval()

# scaler = {'mean_': np.array([ 1.86916373e+03,  4.18677717e+01, -8.76599503e+01,  4.18679074e+01,
#                              -8.76519859e+01,  3.63340593e+00]),
#         'scale_': np.array([1.03639625e+03, 7.17738253e-02, 4.60401620e-02, 6.06944593e-02,
#                             4.07789271e-02, 2.47791716e+00]),
#         'var_': np.array([1.07411720e+06, 5.15148200e-03, 2.11969652e-03, 3.68381739e-03,
#                           1.66292089e-03, 6.14007346e+00]),
#         'n_samples_seen_': 11788}
# def standardize(data, scaler):
#     return (data - scaler['mean_']) / scaler['scale_']

# def cluster_from_rider(rider):
#     plain = np.zeros(7 * 2)
#     pickup = rider.pickup_area
#     dropoff = rider.dropoff_area
#     pickup_cluster = cluster_dict[pickup]
#     dropoff_cluster = cluster_dict[dropoff]
#     plain[pickup_cluster] = 1
#     plain[dropoff_cluster + 7] = 1
#     return plain

# def state_preprocess(state, rider, dev = device):
#     rider_area_info = cluster_from_rider(rider)
#     rider_array = np.zeros((max_queue_length, queue_features))
#     rider_array1 = np.array([[
#         rider.arrival_time,
#         rider.pickup_lat,
#         rider.pickup_lon,
#         rider.dropoff_lat,
#         rider.dropoff_lon,
#         rider.solo_length
#     ]])
#     rider_array1 = standardize(rider_array1, scaler)  # Using the standardize function
#     # rider_array1 = rider_array1[:, :6]
#     rider_array1 = np.concatenate((rider_array1, rider_area_info.reshape(1, 14)), axis = 1)
#     # rider_array1 = np.concatenate((rider_array1, np.zeros((1, 1))), axis = 1)
#     rider_array[0, :] = rider_array1

#     queue_array = np.zeros((max_queue_length, queue_features))
#     if len(state) > 0:
#         for idx, rider in enumerate(state):
#             rider_area_info = cluster_from_rider(rider)
#             rider_array1 = np.array([[
#                 rider.arrival_time,
#                 rider.pickup_lat,
#                 rider.pickup_lon,
#                 rider.dropoff_lat,
#                 rider.dropoff_lon,
#                 rider.solo_length
#                 ]])
#             rider_array1 = standardize(rider_array1, scaler)
#             rider_array1 = np.concatenate((rider_array1, rider_area_info.reshape(1, 14)), axis = 1)
#             queue_array[idx, :] = rider_array1
            
#     riders_tensor = torch.tensor(rider_array, dtype=torch.float32).to(dev)
#     queue_tensor = torch.tensor(queue_array, dtype=torch.float32).to(dev)
#     return [riders_tensor, queue_tensor]

# price_list = np.linspace(price_starting_point, 1, price_options_size)

# def model_decision(state, rider, model, dev = device):
#     with torch.no_grad():
#         rider_tensor, queue_tensor = state_preprocess(state, rider, dev)
#         rider_tensor = rider_tensor.unsqueeze(0)
#         queue_tensor = queue_tensor.unsqueeze(0)
#         price_Q_values, match_Q_values = model(rider_tensor, queue_tensor)
#     price_Q_values = price_Q_values.squeeze()
#     match_Q_values = match_Q_values.squeeze()
#     price_action_idx = torch.argmax(price_Q_values).item()
#     match_action_idx = torch.argmax(match_Q_values).item()
#     price_action = price_list[price_action_idx]
#     match_action = state[match_action_idx] if match_action_idx < len(state) else None
#     return price_action, match_action
###################################################################
################### Part 1: Write your policies ###################
###################################################################
class Shang_Chun_ShanPricingPolicy(object):
    """
    Implementation of your pricing policy
    TODO: rename this class as <Name>PricingPolicy where <Name> is your team name in camel case e.g., AwesomeTeamPricingPolicy
    """

    def __init__(self, c=0.7):
        self.c = c

    @staticmethod
    def get_name():
        """
        :return: name of the team
        """
        return TEAM_NAME

    def pricing_function(self, state, rider):
        """
        :param state: a list of rider instances waiting in the system
        :param rider: a rider instance representing the arriving rider
        :return: the quoted price for the arriving rider, a float number in [0,1]
        TODO: Fill in your code here -- right now this skeleton code makes the decision to never convert any rider by setting the price as 1.
        NOTE: Since each call of the pricing_function() has a time limit of 0.01 seconds, your pricing function should be efficient enough to make a decision within that time limit. Otherwise, the system will make the decision for you by setting the price as 1. Therefore, you may want to paste only the result of the trained pricing policy here (instead of the training process).
        """
        price, _ = model_decision(state, rider, shang_chun_shan_model)
        return price


class Shang_Chun_ShanMatchingPolicy(object):
    """
    Implementation of your matching policy
    TODO: rename this class as <Name>MatchingPolicy where <Name> is your team name in camel case e.g., AwesomeTeamMatchingPolicy
    """

    def __init__(self, c=0.7):
        self.c = c

    @staticmethod
    def get_name():
        """
        :return: name of the team
        """
        return TEAM_NAME

    def matching_function(self, state, rider):
        """
        :param state: a list of rider instances waiting in the system
        :param rider: a rider instance representing the arriving rider
        :return: a rider instance (of the waiting request to be matched with the arriving rider); if no match, return None
        TODO: Fill in your code here -- right now this skeleton code makes the decision to never match any request to the arriving rider.
        NOTE: since each call of the pricing_function() has a time limit of 0.05 seconds, your pricing function should be efficient enough to make a decision within that time limit. Otherwise, the system will make the decision for you by setting the price as `None`.
        """
        _, match = model_decision(state, rider, shang_chun_shan_model)
        return match


###################################################################
#################### Part 2: A helper function ####################
###################################################################

# Here we provide a helper function populate_shared_ride_lengths() to calculate the driving cost and cost allocation of the shared ride involving any two riders i and j.
# The following notations denote choices of the pick-up and drop-off order for a shared ride between rider i and j. They are used in the calculation of the shortest path for a shared ride.
IIJJ = 0  # pick up rider i - drop off rider i - pick up rider j - drop off rider j
IJIJ = 1  # pick up rider i - pick up rider j - drop off rider i - drop off rider j
IJJI = 2  # pick up rider i - pick up rider j - drop off rider j - drop off rider i
JIJI = 3  # pick up rider j - pick up rider i - drop off rider j - drop off rider i
JIIJ = 4  # pick up rider j - drop off rider i - pick up rider i - drop off rider j


def populate_shared_ride_lengths(origin_i, destination_i, origin_j, destination_j):
    """
    :param origin_i: (lat, lon) of rider i's origin
    :param destination_i: (lat, lon) of rider i's destination
    :param origin_j: (lat, lon) of rider j's origin
    :param destination_j: (lat, lon) of rider j's destination
    :return trip_length: the total length of the shared ride
    :return shared_length: the length of the shared part of the shared ride
    :return i_solo_length: the solo length of rider i in the shared ride
    :return j_solo_length: the solo length of rider j in the shared ride
    :return trip_order: the pick-up and drop-off order for the shared trip, denoted by 0 to 4 (IIJJ, IJIJ, IJJI, JIJI, JIIJ)
    """

    # calculate the matrix of pairwise haversine distances between the four points
    data = np.array([origin_i, destination_i, origin_j, destination_j])  # create a 4*2 shaped numpy array
    data = np.deg2rad(data)  # convert to radians
    lat = data[:, 0]  # extract the latitudes
    lon = data[:, 1]  # extract the longitudes
    # elementwise differentiations for lats & lons
    diff_lat = lat[:, None] - lat
    diff_lon = lon[:, None] - lon
    # calculate the distance matrix (in miles)
    d = np.sin(diff_lat/2)**2 + np.cos(lat[:, None])*np.cos(lat) * np.sin(diff_lon/2)**2
    distance_matrix = 2 * 6371 * np.arcsin(np.sqrt(d)) * 0.621371

    # four auxiliary distance submatrices
    # origin_i to destination_i, origin_j to destination_j
    O0D0 = np.repeat(
        np.array([distance_matrix[0, 1], distance_matrix[2, 3]])[
            :, np.newaxis, np.newaxis
        ],
        2,
        axis=1,
    )
    # (origin_i, origin_j) to (origin_i, origin_j)
    O0O1 = distance_matrix[[0, 2], :][:, [0, 2]][:, :, np.newaxis]
    # (destination_i, destination_j) to (destination_i, destination_j)
    D0D1 = distance_matrix[[1, 3], :][:, [1, 3]][:, :, np.newaxis]
    # (origin_i, origin_j) to (destination_i, destination_j)
    O0D1 = distance_matrix[[0, 2], :][:, [1, 3]][:, :, np.newaxis]

    # This method was adapted from code written by SeJIstien Martin for the paper “Detours in Shared Rides”.
    def match_efficiency_single(O0D0, O0O1, D0D1, O0D1):
        """Calculate the request length matrix, shared cost, solo cost,
        and the best pick-up and drop-off order for all rider type pairs."""

        n_riders = 2

        # Compute shortest ordering for each match
        IIJJ_triptime = O0D0 + O0D0.transpose(1, 0, 2)
        IJIJ_triptime = (
            O0O1 + O0D1.transpose(1, 0, 2) + D0D1
        )  # route IJIJ, we can transpose this matrix to get JIJI
        IJJI_triptime = (
            O0O1 + O0D0.transpose(1, 0, 2) + D0D1.transpose(1, 0, 2)
        )  # route IJJI, we can transpose this matrix to get JIIJ
        triptime_possibilities = np.stack(
            (
                IIJJ_triptime,  # 0
                IJIJ_triptime,  # 1
                IJJI_triptime,  # 2
                IJIJ_triptime.transpose(1, 0, 2),  # 3
                IJJI_triptime.transpose(1, 0, 2),  # 4
            ),
            axis=2,
        )
        best_triptime_choice = np.argmin(triptime_possibilities, axis=2)
        best_triptime = triptime_possibilities[
            np.arange(n_riders)[:, np.newaxis, np.newaxis],
            np.arange(n_riders)[np.newaxis, :, np.newaxis],
            best_triptime_choice,
            np.arange(1)[np.newaxis, np.newaxis, :],
        ]

        shared_length_matrix = np.zeros(
            (n_riders, n_riders)
        )  # matrix of the length of shared part of the trip
        i_solo_length_matrix = np.zeros(
            (n_riders, n_riders)
        )  # matrix of the length of type i's solo trip
        j_solo_length_matrix = np.zeros(
            (n_riders, n_riders)
        )  # matrix of the length of type j's solo trip
        for i, j in itertools.product(range(n_riders), range(n_riders)):
            if best_triptime_choice[i, j, 0] == IIJJ:
                shared_length_matrix[i, j] = 0  # II()JJ
                i_solo_length_matrix[i, j] = O0D1[i, i, 0]  # (II)JJ
                j_solo_length_matrix[i, j] = O0D1[j, j, 0]  # II(JJ)
            elif best_triptime_choice[i, j, 0] == IJIJ:
                shared_length_matrix[i, j] = O0D1[j, i, 0]  # I(JI)J
                i_solo_length_matrix[i, j] = O0O1[i, j, 0]  # (IJ)IJ
                j_solo_length_matrix[i, j] = D0D1[i, j, 0]  # IJ(IJ)
            elif best_triptime_choice[i, j, 0] == IJJI:
                shared_length_matrix[i, j] = O0D1[j, j, 0]  # I(JJ)I
                i_solo_length_matrix[i, j] = O0O1[i, j, 0] + D0D1[j, i, 0]  # (IJ)(JI)
                j_solo_length_matrix[i, j] = 0
            elif best_triptime_choice[i, j, 0] == JIJI:
                shared_length_matrix[i, j] = O0D1[i, j, 0]  # J(IJ)I
                i_solo_length_matrix[i, j] = D0D1[j, i, 0]  # JI(JI)
                j_solo_length_matrix[i, j] = O0O1[j, i, 0]  # (JI)JI
            elif best_triptime_choice[i, j, 0] == JIIJ:
                shared_length_matrix[i, j] = O0D1[i, i, 0]  # JIIJ
                i_solo_length_matrix[i, j] = 0
                j_solo_length_matrix[i, j] = O0O1[j, i, 0] + D0D1[i, j, 0]  # (JI)(IJ)

        # matrix of total trip length for each rider type pair
        trip_length_matrix = best_triptime[:, :, 0]
        # matrix of pick-up and drop-off order for each rider type pair,
        # denoted by 0 to 4 (IIJJ, IJIJ, IJJI, JIJI, JIIJ)
        trip_order_matrix = best_triptime_choice[:, :, 0]

        return (
            trip_length_matrix,
            shared_length_matrix,
            i_solo_length_matrix,
            j_solo_length_matrix,
            trip_order_matrix,
        )

    # calculate the optimal routing
    (
        trip_length_matrix,
        shared_length_matrix,
        i_solo_length_matrix,
        j_solo_length_matrix,
        trip_order_matrix,
    ) = match_efficiency_single(O0D0, O0O1, D0D1, O0D1)

    # extract the shortest trip length, shared length, solo length, and trip order
    trip_length = trip_length_matrix[0, 1]
    shared_length = shared_length_matrix[0, 1]
    i_solo_length = i_solo_length_matrix[0, 1]
    j_solo_length = j_solo_length_matrix[0, 1]
    trip_order = trip_order_matrix[0, 1]

    return trip_length, shared_length, i_solo_length, j_solo_length, trip_order


###################################################################
#################### Part 3: Unit test script #####################
###################################################################
"""
After you finish writing your policies in Part 1, you can run this file in the terminal (by running `python <teamname>.py`), to test the correctness and execution time of your code. You can also write your own test script if you want.

Remember to replace your policy class names in the following "TestPolicies" class (two TODOs in the class TestPolicies).

Output in the terminal if your code is correct:
The pricing decisions, matching decisions, and execution time of your pricing/matching functions will be printed for each state. In the end of the output, you will see:
'Ran 2 tests in X s'
'OK'
Otherwise, you will see an error message, and please check your code again.
"""

class TestPolicies(unittest.TestCase):
    def test_1_pricing_function(self):
        PolicyInstance = Shang_Chun_ShanPricingPolicy()
        """
        TODO: rename the name of the class while instantiating as <Name>PricingPolicy where <Name> is your team name in camel case e.g., AwesomeTeamPricingPolicy
        """

        for i, state in enumerate(states):

            # run the pricing function given state and incoming rider
            start = time.time()
            price = PolicyInstance.pricing_function(state, rider)
            end = time.time()

            # assert the price is a float between 0 and 1
            self.assertGreaterEqual(price, 0)
            self.assertLessEqual(price, 1)

            # print the pricing decision and execution time
            print('\n=============== Pricing at State {} ({} waiting requests) ==============='.format(i, len(state)))
            print('Pricing decision: {:.5f}.'.format(price))
            print('Execution time of the pricing function is {:.5f} seconds.'.format(end - start))

    def test_2_matching_function(self):
        PolicyInstance = Shang_Chun_ShanMatchingPolicy()
        """
        TODO: rename the name of the class while instantiating as <Name>MatchingPolicy where <Name> is your team name in camel case e.g., AwesomeTeamMatchingPolicy
        """

        for i, state in enumerate(states):

            # run the matching function given state and incoming rider
            start = time.time()
            matched_request = PolicyInstance.matching_function(state, rider)
            end = time.time()

            # assert matched_request is either None or a rider instance in the state
            if matched_request is not None:
                # assert matched_request is a rider instance
                self.assertEqual(matched_request.__class__.__name__, 'rider')
                # assert matched_request is in state
                self.assertIn(matched_request, state)
            else:
                self.assertEqual(matched_request, None)

            # print the matching decision and execution time
            print('\n=============== Matching at State {} ({} waiting requests) ==============='.format(i, len(state)))
            if matched_request is None:
                print('Matching decision: do not match.')
            else:
                print('Matching decision: match the incoming rider with a waiting request.')
            print('Execution time of the matching function is {:.5f} seconds.'.format(end - start))


if __name__ == '__main__':
    # Here we load the test examples including 4 states and 1 incoming rider:
    # The 4 states have 0/8/35/77 waiting requests, respectively. Each state is generated by accumulating all arriving riders over random time windows of 0s/15s/1min/2min in the training data. Hence the later two states can have more waiting requests than normal states because it is rare to have all riders converted and not matched over such a long time.
    # The incoming rider is randomly sampled from the training data.
    file = 'data/test_examples.pickle'
    with open(file, 'rb') as f:
        test_example = pickle.load(f)
    states = test_example['states']
    rider = test_example['rider']
    unittest.main()
