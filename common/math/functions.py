import numpy as np


def haversine(latitude_a, longitude_a, latitude_b, longitude_b):
    R = 6371
    latitude_a, longitude_a, latitude_b, longitude_b = map(np.radians, [latitude_a, longitude_a, latitude_b, longitude_b])
    dlat = latitude_b - latitude_a
    dlon = longitude_b - longitude_a
    a = np.sin(dlat / 2.0) ** 2 + np.cos(latitude_a) * np.cos(latitude_b) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))