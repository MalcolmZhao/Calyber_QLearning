# This file contains the rider class. Each rider is an object of this class.
# In the arguments of the pricing and matching functions that you need to design, the `state` is a list of the rider objects representing the waiting requests, and the `rider` is a single rider object representing an arriving rider.

from haversine import haversine


class rider():

    def __init__(self, arrival_week, arrival_time, pickup_lat, pickup_lon, dropoff_lat, dropoff_lon, pickup_area, dropoff_area):

        # the following attributes are features of a rider that you can use in your pricing and matching functions
        self.arrival_week = arrival_week  # integer: 1 to 6 (in the training data)
        self.arrival_time = arrival_time  # arrival time (in seconds) in the one-hour time window: 0 to 3600
        self.pickup_lat = pickup_lat  # latitude of the pickup location (centroid of the area)
        self.pickup_lon = pickup_lon  # longitude of the pickup location (centroid of the area)
        self.dropoff_lat = dropoff_lat  # latitude of the dropoff location (centroid of the area)
        self.dropoff_lon = dropoff_lon  # longitude of the dropoff location (centroid of the area)
        self.pickup_area = pickup_area  # the area id of the pickup location: 1 to 76
        self.dropoff_area = dropoff_area  # the area id of the dropoff location: 1 to 76
        self.solo_length = haversine((pickup_lat, pickup_lon), (dropoff_lat, dropoff_lon), unit='mi')  # length of the solo trip (in miles)
