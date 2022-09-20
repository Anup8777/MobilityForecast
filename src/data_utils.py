"""
Data Preprocessing Utilities

Contains methods for loading and preprocessing datasets.

"""
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import math
# from sklearn.model_selection import train_test_split

def filter_outliers(df, data_type='Taxi', filter_method='StdDev3'):

    """
    Method for removing outlying data from the input dataset.

    Input:  data_frame (unfiltered pandas data frame)
    Output: filtered_data (filtered pandas data frame with outlying data removed)

    Opt:    filter_method (optional choice of filter method, default is StdDev3, which removes 
        all data outside three standard deviations of the mean)
            data_type (choice of taxi or weather dataset)
    """
    ind = []

    if filter_method == 'StdDev3':
        if data_type == 'Taxi':

            fare_outlier_max = df['fare_amount'].mean() + (df['fare_amount'].std()*3)
            fare_outlier_min = 0

            tot_fare_outlier_max = df['total_amount'].mean() + (df['total_amount'].std()*3)
            tot_fare_outlier_min = 0

            trip_dist_outlier_max = df['trip_distance'].mean() + (df['trip_distance'].std()*3)
            trip_dist_outlier_min = 0

            # Removing the outliers in the data

            fare_max_ind = df.index[df['fare_amount'] > fare_outlier_max].tolist() # fnding the indices of the outliers
            fare_min_ind = df.index[df['fare_amount'] < fare_outlier_min].tolist()

            tot_fare_max_ind = df.index[df['total_amount'] > tot_fare_outlier_max].tolist()
            tot_fare_min_ind = df.index[df['total_amount'] < tot_fare_outlier_min].tolist()

            trip_dist_ind_max = df.index[df['trip_distance'] > trip_dist_outlier_max].tolist()
            trip_dist_ind_min = df.index[df['trip_distance'] < trip_dist_outlier_min].tolist()

            passenger_count_ind = df.index[df['passenger_count'] == 0].tolist() # finding the indices of the taxi trips where there are no passengers

            ind = sorted(fare_max_ind + fare_min_ind + tot_fare_max_ind + tot_fare_min_ind + trip_dist_ind_max + trip_dist_ind_min + passenger_count_ind) # accumulating all the outlier indices to filter them out

            filtered_data = df.drop(ind) # remove all the outliers
    
    return filtered_data 



def create_ST_map(data_path, data_type='Taxi', year=2022, month=1, plotting='True'):
    """
    Returns a 3D tensor of the spatial-temporal data with shape (pickup_zones, dropoff_zones, hours)
    """
    
    if data_type == 'Taxi':
        # Creating the ST tensor for taxi data
        df = pd.read_parquet(data_path, engine='pyarrow')
        # Sort by pickup time
        df = df.sort_values(by=['tpep_pickup_datetime'], ascending=True)

        # Convert the date-time to python date-time objects to access built-in methods
        df['tpep_pickup_datetime']=pd.to_datetime(df['tpep_pickup_datetime']) 
        df['tpep_dropoff_datetime']=pd.to_datetime(df['tpep_dropoff_datetime'])

        # dropping the store and forward flag, VendorID is the company that provided the record(this does not add any value to our study)
        # RatecodeID and payment_type are categorical features affecting the price of the trip, we discard this for the moment as this does not add value to our study. 
        # We also drop the extra and mta_tax as they are not relevant to our study
        df = df.drop(columns=['store_and_fwd_flag', 'VendorID' , 'RatecodeID', 'payment_type','fare_amount', 'extra' , 'mta_tax', 'tip_amount',
        'tolls_amount', 'improvement_surcharge','total_amount', 'congestion_surcharge', 'airport_fee', 'trip_distance', 'passenger_count']) 
        # Drop the rows with missing values
        df = df.dropna(axis=0)

        # splitting the date-time objects to year, month, day, and hour
        df['pickup_year'] = df['tpep_pickup_datetime'].apply(lambda t: t.year)
        df['pickup_month'] = df['tpep_pickup_datetime'].apply(lambda t: t.month)
        df['pickup_weekday'] = df['tpep_pickup_datetime'].apply(lambda t: t.day)
        df['pickup_hour'] = df['tpep_pickup_datetime'].apply(lambda t: t.hour)

        # dropping the date-time objects
        df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime']) 

        # Remove all data not in the specified year and month
        df = df[(df.pickup_year == year)]
        df = df[(df.pickup_month == month)]
        
        # Remove year and month columns now that we have filtered out all other data
        df = df.drop(columns=['pickup_year', 'pickup_month'])

        # Create an Excel pivot table type object, aggregating the number of trips by hour and day, and changing the index to 'demand'
        features = df.pivot_table(columns=["PULocationID", "DOLocationID","pickup_weekday", "pickup_hour"], aggfunc='size').reset_index(name='demand')
        # Drop the index column
        features = pd.DataFrame(features).reset_index(drop=True)

        # splitting up the data frames by days and then hours
        rows = 265+1 #265 zones in NYC dataset
        cols = 265+1

        # Get the unique days and hours in the dataset
        days = np.unique(features.pickup_weekday)
        hours = np.unique(features.pickup_hour)

        # Create a 2D array with the features object entries for each day and hour
        features_days = []
        features_hours = []
        features_days = [features[features.pickup_weekday == i] for i in days]
        for i in range(0,len(features_days)):
            for j in hours:
                features_hours.append(features_days[i][features_days[i].pickup_hour == j])

        # Initialize the 3D ST tensor with zeros of size (pickup zones, dropoff zones, days*hours)
        ST_map = np.zeros((rows,cols, len(features_hours)))

        # Fill the ST tensor with the demand values, dropping the day and hour columns and the index
        for i in range(0,len(features_hours)):
            features_hours[i] = features_hours[i].drop(columns=["pickup_weekday","pickup_hour"])
            rind = np.array(features_hours[i].PULocationID.array)
            cind = np.array(features_hours[i].DOLocationID.array)
            demand = np.array(features_hours[i].demand.array)
            ST_map[rind,cind,i] = demand
        ST_map = ST_map[1::,1::,:]
    """ Future code for bike data:
    # if data_type == 'Bike':
    #     # Creating the ST tensor for taxi data
    #     df = pd.read_parquet(data_path, engine='pyarrow')
    #     # Sort by pickup time
    #     df = df.sort_values(by=['tpep_pickup_datetime'], ascending=True)

    #     # Convert the date-time to python date-time objects to access built-in methods
    #     df['tpep_pickup_datetime']=pd.to_datetime(df['tpep_pickup_datetime']) 
    #     df['tpep_dropoff_datetime']=pd.to_datetime(df['tpep_dropoff_datetime'])

    #     # dropping the store and forward flag, VendorID is the company that provided the record(this does not add any value to our study)
    #     # RatecodeID and payment_type are categorical features affecting the price of the trip, we discard this for the moment as this does not add value to our study. 
    #     # We also drop the extra and mta_tax as they are not relevant to our study
    #     df = df.drop(columns=['store_and_fwd_flag', 'VendorID' , 'RatecodeID', 'payment_type','fare_amount', 'extra' , 'mta_tax', 'tip_amount',
    #     'tolls_amount', 'improvement_surcharge','total_amount', 'congestion_surcharge', 'airport_fee', 'trip_distance', 'passenger_count']) 
    #     # Drop the rows with missing values
    #     df = df.dropna(axis=0)

    #     # splitting the date-time objects to year, month, day, and hour
    #     df['pickup_year'] = df['tpep_pickup_datetime'].apply(lambda t: t.year)
    #     df['pickup_month'] = df['tpep_pickup_datetime'].apply(lambda t: t.month)
    #     df['pickup_weekday'] = df['tpep_pickup_datetime'].apply(lambda t: t.day)
    #     df['pickup_hour'] = df['tpep_pickup_datetime'].apply(lambda t: t.hour)

    #     # dropping the date-time objects
    #     df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime']) 

    #     # Remove all data not in the specified year and month
    #     df = df[(df.pickup_year == year)]
    #     df = df[(df.pickup_month == month)]
        
    #     # Remove year and month columns now that we have filtered out all other data
    #     df = df.drop(columns=['pickup_year', 'pickup_month'])

    #     # Create an Excel pivot table type object, aggregating the number of trips by hour and day, and changing the index to 'demand'
    #     features = df.pivot_table(columns=["PULocationID", "DOLocationID","pickup_weekday", "pickup_hour"], aggfunc='size').reset_index(name='demand')
    #     # Drop the index column
    #     features = pd.DataFrame(features).reset_index(drop=True)

    #     # splitting up the data frames by days and then hours
    #     rows = 265+1 #265 zones in NYC dataset
    #     cols = 265+1

    #     # Get the unique days and hours in the dataset
    #     days = np.unique(features.pickup_weekday)
    #     hours = np.unique(features.pickup_hour)

    #     # Create a 2D array with the features object entries for each day and hour
    #     features_days = []
    #     features_hours = []
    #     features_days = [features[features.pickup_weekday == i] for i in days]
    #     for i in range(0,len(features_days)):
    #         for j in hours:
    #             features_hours.append(features_days[i][features_days[i].pickup_hour == j])

    #     # Initialize the 3D ST tensor with zeros of size (pickup zones, dropoff zones, days*hours)
    #     ST_map = np.zeros((rows,cols, len(features_hours)))

    #     # Fill the ST tensor with the demand values, dropping the day and hour columns and the index
    #     for i in range(0,len(features_hours)):
    #         features_hours[i] = features_hours[i].drop(columns=["pickup_weekday","pickup_hour"])
    #         rind = np.array(features_hours[i].PULocationID.array)
    #         cind = np.array(features_hours[i].DOLocationID.array)
    #         demand = np.array(features_hours[i].demand.array)
    #         ST_map[rind,cind,i] = demand
    #     ST_map = ST_map[1::,1::,:]
    """
    
    # Plot a heatmap of demand for the array of pickup and dropoff zones
    if plotting == 'True':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,12))
        sns.heatmap(ST_map[:,:,8], ax=ax1, cmap="YlGnBu")
        sns.heatmap(ST_map[:,:,18], ax=ax2, cmap="YlGnBu")
        # Additional color palletes: https://seaborn.pydata.org/tutorial/color_palettes.html
    return ST_map

def _data_loader(st_map, val_size=0.15, test_size=0.15, temporal_map='hour'):
    """
    Expects only 1 month of data
    Returns 3D ST maps for each temporal interval (e.g. every hour for a day, every hour for a week, every day for a week, all over the entire course of the month)
    Each of these ST maps are split into train, validation, and test bins
    End result is a 4D set of ST maps with dimensions [pickup_zones, dropoff_zones, hours/days, month_days/month_weeks]
    """
    train_size = 1 - val_size - test_size

    # Get the number of days in the ST tensor as an integer
    month_days = st_map.shape[2]//24
    month_hours = st_map.shape[2]

    # Split the data into training, validation, and test sets

    train_hours = int(month_hours*train_size)
    val_hours = int(month_hours*val_size)
    test_hours = int(month_hours*test_size)

    train_ST = st_map[:,:,0:train_hours]
    val_ST = st_map[:,:,train_hours:(train_hours+val_hours)]
    test_ST = st_map[:,:,(train_hours+val_hours):]
    if temporal_map=='hour':
        train_ST = train_ST
        val_ST = val_ST
        test_ST = test_ST
    elif temporal_map=='day':
        train_days = np.zeros((st_map[0], st_map[1], train_ST[2]/24))
        val_days = np.zeros((st_map[0], st_map[1], val_ST[2]/24))
        test_days = np.zeros((st_map[0], st_map[1], test_ST[2]/24))
        for day in range(month_days+1):
            if day ==0:
                train_days[day] = np.sum(train_ST[:,:,0:25])
                val_days[day] = np.sum(val_ST[:,:,0:25])
                test_days[day] = np.sum(test_ST[:,:,0:25])
            else:
                train_days[day] = np.sum(train_ST[:,:,(day*24)+1:((day+1)*24)+1]) 
                val_days[day] = np.sum(val_ST[:,:,(day*24)+1:((day+1)*24)+1])
                test_days[day] = np.sum(test_ST[:,:,(day*24)+1:((day+1)*24)+1])
        
    elif temporal_map=='week':
        train_weeks = np.zeros((st_map[0], st_map[1], np.round(train_ST[2]/(24*7))))
        val_weeks = np.zeros((st_map[0], st_map[1], np.round(val_ST[2]/(24*7))))
        test_weeks = np.zeros((st_map[0], st_map[1], np.round(test_ST[2]/(24*7))))
        for week in range(np.round((month_hours/(24*7)))+1):
            if week ==0:
                train_weeks[week] = np.sum(train_ST[:,:,0:25])
                val_weeks[week] = np.sum(val_ST[:,:,0:25])
                test_weeks[week] = np.sum(test_ST[:,:,0:25])
            else:
                train_days[day] = np.sum(train_ST[:,:,(day*24)+1:((day+1)*24)+1]) 
                val_days[day] = np.sum(val_ST[:,:,(day*24)+1:((day+1)*24)+1])
                test_days[day] = np.sum(test_ST[:,:,(day*24)+1:((day+1)*24)+1])

    return train_ST, val_ST, test_ST