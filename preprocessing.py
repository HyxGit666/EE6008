import numpy as np
import pandas as pd
import warnings
import math
from sklearn.preprocessing import StandardScaler
import pkg_resources
import sys

warnings.filterwarnings("ignore")

# read dataset and set flight date as time index
flight_data = pd.read_csv('AA-STL-MIA(20-23).csv')
# flight_data = pd.read_csv('HA17_noBlank.csv')

flight_data['FL_DATE'] = pd.to_datetime(flight_data['FL_DATE'])
flight_data['date_index'] = flight_data['FL_DATE'].dt.date
print(flight_data['date_index'].head())

flight_data.set_index('date_index', inplace=True)

print(flight_data.head())

print("--------------------------------------------------------------------------------")


def weather_vectorisation(data):
    def feature_encoder(value, threshold):
        if value > threshold:
            return 1
        else:
            return 0

    encoded_weather_features = []

    for index, row in data.iterrows():
        # obtain weather features
        wind_strength_ori = row['AWND']
        rainfall_ori = row['PRCP']
        temperature_ori = row['TAVG']
        snowfall_ori = row['SNOW']
        wind_strength_des = row['AWND']
        rainfall_des = row['PRCP']
        temperature_des = row['TAVG']

        w1 = wind_strength_ori
        w2 = feature_encoder(rainfall_ori, 0)
        w3 = temperature_ori
        w4 = feature_encoder(snowfall_ori, 0)
        w5 = wind_strength_des
        w6 = feature_encoder(rainfall_des, 0)
        w7 = temperature_des

        w = [w1, w2, w3, w4, w5, w6, w7]

        encoded_weather_features.append(w)

    return np.array(encoded_weather_features)


encoded_weather_features = weather_vectorisation(flight_data)

print("the number of flights encoding weather features:", encoded_weather_features.shape)

print("-----------------------------------------------------------------------------")


def time_stamped_vectorisation(data):
    def season_encoder(quarter):
        if quarter == 1:
            return 1
        elif quarter == 2:
            return 2
        elif quarter == 3:
            return 3
        else:
            return 4

    encoded_time_stamped_features = []

    for index, row in data.iterrows():
        # obtain time-stamped features
        day_of_month = row['DAY_OF_MONTH']
        month = row['MONTH']
        day_of_week = row['DAY_OF_WEEK']
        season = row['QUARTER']

        t1 = day_of_month
        t2 = month
        t3 = day_of_week
        t4 = season_encoder(season)

        t = [t1, t2, t3, t4]

        encoded_time_stamped_features.append(t)

    return np.array(encoded_time_stamped_features)


encoded_time_features = time_stamped_vectorisation(flight_data)

print("the number of flights encoding time features:", encoded_time_features.shape)

print("-----------------------------------------------------------------------------")


def flight_schedule_vectorisation(data):
    flight_schedule_features = []

    def dep_delay_encoder(delay):
        if np.isnan(delay):
            return 300
        else:
            return int(delay)

    for index, row in data.iterrows():
        # obtain flight schedule features

        scheduled_departure_time = row['CRS_DEP_TIME']

        scheduled_arrival_time = row['CRS_ARR_TIME']
        departure_delay = row['DEP_DELAY']

        s1 = int(scheduled_departure_time)
        s2 = int(scheduled_arrival_time)
        s3 = dep_delay_encoder(departure_delay)

        s = [s1, s2, s3]

        flight_schedule_features.append(s)

    return np.array(flight_schedule_features)


encoded_flight_schedule_features = flight_schedule_vectorisation(flight_data)
print("the number of flights encoding schedule features:", encoded_flight_schedule_features.shape)

print("--------------------------------------------------------------------------------------------")


def feature_vectorisation(data):
    max_temperature_ori = max(data['TAVG'])
    min_temperature_ori = min(data['TAVG'])
    max_temperature_des = max(data['TAVG'])
    min_temperature_des = min(data['TAVG'])

    max_wind_ori = max(data['AWND'])
    min_wind_ori = min(data['AWND'])
    max_wind_des = max(data['AWND'])
    min_wind_des = min(data['AWND'])

    mean_dep_delay = np.mean(data['DEP_DELAY'])
    mean_arr_delay = np.mean(data['ARR_DELAY'])

    def feature_encoder(value, threshold):
        if value > threshold:
            return 1
        else:
            return 0

    def season_encoder(quarter):
        if quarter == 1:
            return 1
        elif quarter == 2:
            return 2
        elif quarter == 3:
            return 3
        else:
            return 4

    def dep_delay_encoder(delay):
        if np.isnan(delay):
            return int(mean_dep_delay)
        else:
            return int(delay)

    def arr_delay_encoder(delay):
        if np.isnan(delay):
            return int(mean_arr_delay)
        else:
            return int(delay)

    def temp_ori_normalization(temp_ori):
        return (temp_ori - min_temperature_ori) / (max_temperature_ori - min_temperature_ori)

    def temp_des_normalization(temp_des):
        return (temp_des - min_temperature_des) / (max_temperature_des - min_temperature_des)

    def wind_ori_normalization(wind_ori):
        return (wind_ori - min_wind_ori) / (max_wind_ori - min_wind_ori)

    def wind_des_normalization(wind_des):
        return (wind_des - min_wind_des) / (max_wind_des - min_wind_des)

    def cyclic_coding(time, period):
        if 0 <= time < 100:
            angle = (2 * np.pi * (0 + time / 60)) / period
            sine_value = np.sin(angle)
            cosine_value = np.cos(angle)
            return sine_value, cosine_value
        elif 100 <= time < 1000:
            hour = (time // 100) % 10
            angle = (2 * np.pi * (hour + (time - 100 * hour) / 60)) / period
            sine_value = np.sin(angle)
            cosine_value = np.cos(angle)
            return sine_value, cosine_value
        else:
            hour = time // 100
            minute = time - hour * 100
            angle = (2 * np.pi * (hour + minute / 60)) / period
            sine_value = np.sin(angle)
            cosine_value = np.cos(angle)
            return sine_value, cosine_value

    def time_encoder(scheduled_time, real_time, delay):
        if np.isnan(real_time):
            return scheduled_time + delay
        else:
            return real_time

    feature_list = []

    for index, row in data.iterrows():
        # obtain weather features
        wind_strength_ori = row['AWND']
        rainfall_ori = row['PRCP']
        temperature_ori = row['TAVG']
        snowfall_ori = row['SNOW']
        wind_strength_des = row['AWND']
        rainfall_des = row['PRCP']
        temperature_des = row['TAVG']
        day_of_month = row['DAY_OF_MONTH']
        month = row['MONTH']
        day_of_week = row['DAY_OF_WEEK']
        season = row['QUARTER']

        scheduled_departure_time = row['CRS_DEP_TIME']
        scheduled_arrival_time = row['CRS_ARR_TIME']
        real_departure_time = row['DEP_TIME']
        real_arrival_time = row['ARR_TIME']
        departure_delay = row['DEP_DELAY']
        arrival_delay = row['ARR_DELAY']

        w1 = wind_ori_normalization(wind_strength_ori)
        w2 = rainfall_ori
        w3 = temp_ori_normalization(temperature_ori)
        w4 = feature_encoder(snowfall_ori, 0)
        w5 = wind_des_normalization(wind_strength_des)
        w6 = rainfall_des
        w7 = temp_des_normalization(temperature_des)

        t1 = day_of_month
        t2 = month
        t3 = day_of_week
        t4 = season_encoder(season)

        s1_1, s1_2 = cyclic_coding(scheduled_departure_time, 24)
        s2_1, s2_2 = cyclic_coding(scheduled_arrival_time, 24)

        dep_delay = dep_delay_encoder(departure_delay)
        arr_delay = arr_delay_encoder(arrival_delay)
        real_departure_time = time_encoder(scheduled_departure_time, real_departure_time, dep_delay)
        real_arrival_time = time_encoder(scheduled_arrival_time, real_arrival_time, arr_delay)

        s3_1, s3_2 = cyclic_coding(real_departure_time, 24)
        s4_1, s4_2 = cyclic_coding(real_arrival_time, 24)

        s1 = 0.5 * s1_1 + 0.5 * s1_2
        s2 = 0.5 * s2_1 + 0.5 * s2_2
        s3 = 0.5 * s3_1 + 0.5 * s3_2
        s4 = 0.5 * s4_1 + 0.5 * s4_2

        features = [s4, s2, s3, s1, t1, t2, t3, t4, w1, w2, w3, w4, w5, w6, w7]
        # features = [s4, s2, s1, t1, t2, t3, t4, w1, w2, w3, w4, w5, w6, w7]

        feature_list.append(features)

    return np.array(feature_list)


flight_features = feature_vectorisation(flight_data)
scaler = StandardScaler().fit(flight_features)
flight_features = scaler.transform(flight_features)
print(flight_features[0:5, :])
print("{} flights and each has {} features ".format(flight_features.shape[0], flight_features.shape[1]))
print("--------------------------------------------------------------------------------")


def label_extraction(data):
    train_labels = []
    mean_arr_delay = np.mean(data['ARR_DELAY'])

    def labelling(arri_delay):
        if np.isnan(arri_delay):
            return int(mean_arr_delay)
        else:
            return int(arri_delay)

    for index, row in data.iterrows():
        real_arrival_time = row['ARR_DELAY']

        label = labelling(real_arrival_time)

        train_labels.append(label)

    return train_labels


labels = label_extraction(flight_data)
print("No. of all labels", len(labels))

print("----------------------------------------------------------------------")


def dataset_generation(data, features, label, time_steps):
    data['FL_DATE'] = pd.to_datetime(data['FL_DATE'])
    data['date_index'] = data['FL_DATE'].dt.date

    date_of_step = []
    label_for_training = []
    inputs = []

    for i in range(len(data) - time_steps):
        date_of_step.append(data['date_index'][i + time_steps])
        label_for_training.append(label[i + time_steps])
        xt = features[i:i + time_steps, :]
        inputs.append(xt)

    return label_for_training, date_of_step, np.array(inputs)


label_of_step, dates_of_step, inputs = dataset_generation(flight_data, flight_features, labels, 5)
print("No. of training and test labels", len(label_of_step))
print("the shape of input vector", inputs.shape)
print("----------------------------------------------------------------------------------------")

# train_samples, test_samples, train_labels, test_labels = train_test_split(inputs, label_of_step, test_size=0.25,
#                                                                         random_state=42)

def split_train_test(X, y, y_date):
    train_nums = math.ceil(len(X) * 0.7)

    test_start = math.ceil(len(X)*0.9)

    x_train = X[:train_nums]
    x_val = X[train_nums:test_start]
    x_test = X[test_start:]

    y_train = y[:train_nums]
    y_val = y[train_nums:test_start]
    y_test = y[test_start:]

    y_date_train = y_date[:train_nums]
    y_date_val = y_date[train_nums:test_start]
    y_date_test = y_date[test_start:]

    return x_train, x_test, y_train, y_test, y_date_train, y_date_test, x_val, y_val, y_date_val


train_samples, test_samples, train_labels, test_labels, train_dates, test_dates, val_samples, val_labels, val_dates = split_train_test(inputs,
                                                                                                                                       label_of_step,
                                                                                                                                       dates_of_step)

print("No. of training samples:{}, time step:{} and feature nums:{}".format(train_samples.shape[0],
                                                                            train_samples.shape[1],
                                                                            train_samples.shape[2]))
print("No. of training labels:", len(train_labels))
print("No. of validation samples:", val_samples.shape[0])
print("No. of validation labels:", len(val_labels))
print("No. of test samples:", test_samples.shape[0])
print("No of test labels:", len(test_labels))
print("--------------------------------------------------------------------------")

print(sys.version)
required_libraries = ["numpy", "tensorflow", "matplotlib", "pandas", "sklearn.model_selection"]  # 添加需要检查版本的库

# 遍历每个库并打印其名称和版本号
for lib in required_libraries:
    try:
        version = pkg_resources.get_distribution(lib).version
        print(f"{lib} version: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{lib} is not installed")
