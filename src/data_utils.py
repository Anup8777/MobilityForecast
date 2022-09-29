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

import shapefile

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

def load_data(data_dir, data_file, data_type='csv'):
    """
    Load data from csv or parquet file

    Args:
        data_dir (str): directory containing data file
        data_file (str): name of data file

    Returns:
        df (pandas.DataFrame): data frame containing data

    """
    if data_type == 'csv':
        df = pd.read_csv(os.path.join(data_dir, data_file))
    elif data_type == 'parquet':
        df = pd.read_parquet(os.path.join(data_dir, data_file))
    return df

def zone_to_coord(_df):
    df = _df
    coords = pd.read_csv('/Users/probook/Documents/GitHub/mobilityforecast/utils/taxi_zones/taxi_zones.csv')

    df['start_lat'] = ""
    df['start_lng'] = ""
    df['end_lat'] = ""
    df['end_lng'] = ""
    for location_id in coords['LocationID'].unique():
        df.loc[df['PULocationID'] == location_id, 'start_lng'] = coords.loc[coords['LocationID'] == location_id, 'Longitude'].values[0]
        df.loc[df['PULocationID'] == location_id, 'start_lat'] = coords.loc[coords['LocationID'] == location_id, 'Latitude'].values[0]
        df.loc[df['DOLocationID'] == location_id, 'end_lng'] = coords.loc[coords['LocationID'] == location_id, 'Longitude'].values[0]
        df.loc[df['DOLocationID'] == location_id, 'end_lat'] = coords.loc[coords['LocationID'] == location_id, 'Latitude'].values[0]

    return df

def bin_data(_df, num_bins=10):
    df = _df

    #Split the data into latitude and longitude bins of equal size
    regions = num_bins
    
    start_lat_min = df['start_lat'].min()
    start_lat_max = df['start_lat'].max()
    start_lng_min = df['start_lng'].min()
    start_lng_max = df['start_lng'].max()

    end_lat_min = df['end_lat'].min()
    end_lat_max = df['end_lat'].max()
    end_lng_min = df['end_lng'].min()
    end_lng_max = df['end_lng'].max()

    west = min(start_lat_min, end_lat_min)
    east = max(start_lat_max, end_lat_max)

    south = min(start_lng_min, end_lng_min)
    north = max(start_lng_max, end_lng_max)

    lat_bins = np.linspace(west,east,regions)
    lng_bins = np.linspace(south, north, regions)

    names = list(map(int, range(regions)))

    cord_bins = np.array(np.meshgrid(np.arange(regions), np.arange(regions))).T.reshape(-1,2)
    cord_bins = list(map(tuple, cord_bins))

    df['start_lat_regions']=pd.cut(df['start_lat'], bins=regions,labels = names, retbins=False, right=True, include_lowest=True)
    df['start_lng_regions']=pd.cut(df['start_lng'], bins=regions,labels= names, retbins=False, right=True, include_lowest=True)

    df['end_lat_regions']=pd.cut(df['end_lat'], bins=regions,labels = names, retbins=False, right=True, include_lowest=True)
    df['end_lng_regions']=pd.cut(df['end_lng'], bins=regions,labels= names, retbins=False, right=True, include_lowest=True)

    df['PU_region'] =list(zip(df.start_lat_regions, df.start_lng_regions)) #np.array([*tmp]).tolist()
    df['DO_region'] =list(zip(df.end_lat_regions, df.end_lng_regions))

    df = df.drop(columns=['start_lat_regions','start_lng_regions','end_lat_regions','end_lng_regions'])

    for i in range(0,len(cord_bins)):
        df.loc[df['PU_region'] == cord_bins[i], 'PU_region'] = i+1
        df.loc[df['DO_region'] == cord_bins[i], 'DO_region'] = i+1

    return df

def data_preprocessing(_df, data_type='Taxi', year=2022, month=1, dropna=True, num_bins=10):
    """
    Preprocess one month of data

    Args:
        df (pandas.DataFrame): data frame containing data
        data_type (str): the type of data contained in the df (i.e. Taxi, CitiBike, Metro, Weather)
        dropna (bool): whether to drop rows with missing values

    Returns:
        df (pandas.DataFrame): preprocessed data frame

    """
    df = _df
    if dropna:
        df = df.dropna()
    df = df.reset_index(drop=True)

    if data_type == 'Taxi':
        drop_cols = ['store_and_fwd_flag', 'VendorID' , 'RatecodeID', 'payment_type','fare_amount', 'extra' , 'mta_tax', 'tip_amount',
        'tolls_amount', 'improvement_surcharge','total_amount', 'congestion_surcharge', 'airport_fee', 'trip_distance', 'passenger_count']
        
        sort_by = 'tpep_pickup_datetime'

        pu_date_time = 'tpep_pickup_datetime'
        do_date_time = 'tpep_dropoff_datetime'

        # aggregate_cols = 'PULocationID', 'DOLocationID', 'pickup_weekday', 'pickup_hour'

    elif data_type == 'CitiBike':
        drop_cols = ['ride_id', 'rideable_type', 'start_station_name', 'start_station_id', 'end_station_name', 'end_station_id', 'member_casual']
        
        sort_by = 'started_at'

        pu_date_time = 'started_at'
        do_date_time = 'ended_at'

    elif data_type == 'Weather':
        drop_cols = ['coordinates (lat,lon)','model (name)','model elevation (surface)','utc_offset (hrs)']

        sort_by = 'datetime (UTC)'

        pu_date_time = 'datetime (UTC)'
        do_date_time = 'temp'
        df[do_date_time] = df[pu_date_time]

    # Prepare new date time columns
    df[pu_date_time] = pd.to_datetime(df[pu_date_time])
    df[do_date_time] = pd.to_datetime(df[do_date_time])
    df['pickup_year'] = df[pu_date_time].apply(lambda t: t.year)
    df['pickup_month'] = df[pu_date_time].apply(lambda t: t.month)
    df['pickup_day'] = df[pu_date_time].apply(lambda t: t.day)
    df['pickup_hour'] = df[pu_date_time].apply(lambda t: t.hour)
    
    # Sort by date
    df = df.sort_values(by=sort_by, ascending=True)
    # Drop columns and remove all unnecessary data (not in target year and month)
    df = df.drop(columns=drop_cols)
    df = df.drop(columns=[pu_date_time, do_date_time])
    df = df[(df['pickup_year'] == year) & (df['pickup_month'] == month)]
    df = df.drop(columns=['pickup_year', 'pickup_month'])

    # Convert zone to latitudes and longitudes (only for taxi data)
    if data_type == 'Taxi':
        df = zone_to_coord(df)
    # Bin bike and taxi data by latitude and longitude (start with 32 bins)

    if data_type == 'Taxi' or data_type == 'CitiBike':
        df['start_lat'] = pd.to_numeric(df['start_lat'])
        df['start_lng'] = pd.to_numeric(df['start_lng'])
        df['end_lat'] = pd.to_numeric(df['end_lat'])
        df['end_lng'] = pd.to_numeric(df['end_lng'])
        df = bin_data(df, num_bins)

        df = df.apply(lambda col:pd.to_numeric(col, errors='coerce'))

        ### Organize and aggregate taxi and bike demand, shape into 2D frames for each hour
        df1 = df.groupby(['PU_region','DO_region','pickup_day','pickup_hour']).size().reset_index().rename(columns={0:'demand'})

        days = np.unique(df1.pickup_day)
        rows = num_bins**2
        cols = num_bins**2
        ST_map = np.zeros((rows, cols, 24, (len(days))))
        for hour in range(24):
            for day in days:
                # print(hour,day)
                df2 = df1[(df1.pickup_day == day) & (df1.pickup_hour == hour)]
                df2 = df2.drop(columns=['pickup_day','pickup_hour'])
                df2 = df2.reset_index(drop=True, inplace=False)
                # print(len(df2))
                for i in range(len(df2)):
                    rind = int(df2.PU_region[i])
                    cind = int(df2.DO_region[i])
                    demand = df2.demand[i]
                    ST_map[rind-1, cind-1, hour-1, day-1] = demand
        #stack the st maps vertically to make one long 3D array of each hour for the month
        ST_map = ST_map.reshape((ST_map.shape[0], ST_map.shape[1], -1))

        ### Plot one hour of data
        hour = 230
        plt.imshow(ST_map[:,:,hour], cmap='hot', interpolation='None')
        plt.title('Rides in each binned zone on ' + str(year) + ' ' + str(month) + ' ' + str(int(hour/24)+1) + ' hour ' + str(hour%24))
        plt.show()


    ### Process weather data
    if data_type == 'Weather':
        df1 = df.apply(lambda col:pd.to_numeric(col, errors='coerce'))
        days = np.unique(df1.pickup_day)
        ST_map = np.zeros((1, 5, 24, (len(days))))
        for hour in range(24):
            for day in days:
                # print(hour,day)
                df2 = df1[(df1.pickup_day == day) & (df1.pickup_hour == hour)]
                df2 = df2.drop(columns=['pickup_day','pickup_hour'])
                df2 = df2.reset_index(drop=True, inplace=False)
                # print(len(df2))
                for i in range(len(df2)):
                    # rind = int(df2.PU_region[i])
                    # cind = int(df2.DO_region[i])
                    temp = df2['temperature (degC)'][i]
                    wind_speed = df2['wind_speed (m/s)'][i]
                    total_precip = df2['total_precipitation (mm of water equivalent)'][i]
                    snowfall = df2['snowfall (mm of water equivalent)'][i]
                    snow_depth = df2['snow_depth (mm of water equivalent)'][i]
                    ST_map[:, 0, hour-1, day-1] = temp
                    ST_map[:, 1, hour-1, day-1] = wind_speed
                    ST_map[:, 2, hour-1, day-1] = total_precip
                    ST_map[:, 3, hour-1, day-1] = snowfall
                    ST_map[:, 4, hour-1, day-1] = snow_depth
        #stack the st maps vertically to make one long 3D array of each hour for the month
        ST_map = ST_map.reshape((ST_map.shape[0], ST_map.shape[1], -1))

    ### TODO: Process metro data

    return ST_map

def plot_map(_ST_map):
    sum_vec = np.sum(np.sum(_ST_map, axis=0), axis=0)
    plt.plot(sum_vec)
    plt.title('Total Demand in All Zones per Hour')
    plt.xlabel('Hour')
    plt.ylabel('Total Trips')

def split_data(_ST_map, n_splits=5):
    tss = TimeSeriesSplit(n_splits=n_splits)

    split_indices = list(range(_ST_map.shape[-1]))

    for train_ind, val_ind in tss.split(split_indices):
        # print(train_ind, val_ind)
        X_train, X_val = _ST_map[:,:,train_ind], _ST_map[:,:,val_ind]

    # split_indices = list(range(X_train.shape[-1]))
    # print(X_train.shape, X_val.shape)

    # for train_ind, test_ind in tss.split(split_indices):
    #     print(train_ind, test_ind)
    #     X_train, X_test = X_train[:,:,train_ind], X_train[:,:,test_ind]

    # return X_train, X_val
    return X_train, X_val

def scale_data(_ST_map):
    scaler = MinMaxScaler(feature_range=(0, 1))
    s0, s1, s2 = _ST_map.shape[0], _ST_map.shape[1], _ST_map.shape[2]
    _ST_map = _ST_map.reshape(s0 * s1, s2)
    _ST_map = scaler.fit_transform(_ST_map)
    _ST_map = _ST_map.reshape(s0, s1, s2)
    # ST_map = scaler.fit_transform(_ST_map.reshape(-1, _ST_map.shape[-1])).reshape(_ST_map.shape)

    return _ST_map

def get_lat_lon(sf):
    sf = shapefile.Reader("shape/taxi_zones.shp")
    fields_name = [field[0] for field in sf.fields[1:]]
    shp_dic = dict(zip(fields_name, list(range(len(fields_name)))))
    attributes = sf.records()
    shp_attr = [dict(zip(fields_name, attr)) for attr in attributes]
    content = []
    for sr in sf.shapeRecords():
        shape = sr.shape
        rec = sr.record
        loc_id = rec[shp_dic['LocationID']]
        
        x = (shape.bbox[0]+shape.bbox[2])/2
        y = (shape.bbox[1]+shape.bbox[3])/2
        
        content.append((loc_id, x, y))
    return pd.DataFrame(content, columns=["LocationID", "longitude", "latitude"])