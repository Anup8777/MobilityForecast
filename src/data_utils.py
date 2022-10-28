"""
Utilities for loading, processing, and filtering data
for taxi and bike, for use in the GAN and regression models.

"""
import pandas as pd
import numpy as np
import numpy.polynomial.polynomial as poly
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import calendar
import math
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
# from sklearn.model_selection import train_test_split

import shapefile


def load_data(data_dir, data_file, data_type='parquet'):
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

def zone_to_coord(_df, _data_dir):
    df = _df
    data_dir = _data_dir
    coords = pd.read_csv((data_dir + str('taxi_zones.csv')))

    df['start station latitude'] = ""
    df['start station longitude'] = ""
    df['end station latitude'] = ""
    df['end station longitude'] = ""
    for location_id in coords['LocationID'].unique():
        df.loc[df['PULocationID'] == location_id, 'start station longitude'] = coords.loc[coords['LocationID'] == location_id, 'Longitude'].values[0]
        df.loc[df['PULocationID'] == location_id, 'start station latitude'] = coords.loc[coords['LocationID'] == location_id, 'Latitude'].values[0]
        df.loc[df['DOLocationID'] == location_id, 'end station longitude'] = coords.loc[coords['LocationID'] == location_id, 'Longitude'].values[0]
        df.loc[df['DOLocationID'] == location_id, 'end station latitude'] = coords.loc[coords['LocationID'] == location_id, 'Latitude'].values[0]

    return df

def bin_data(_df, _num_bins=5):
 
    df = _df
    regions = _num_bins

    south = 40.5
    north = 40.9

    west = -74.25
    east = -73.9

    # print('south: ', south, 'north: ', north, 'west: ', west, 'east: ', east)

    lat_bins = np.linspace(south,north,regions)
    lng_bins = np.linspace(west, east, regions)

    # print('bins: ', lat_bins, lng_bins)

    step = (north - south) / regions
    to_bin = lambda x: np.floor(x / step) * step
    df["start_lat_bin"] = to_bin(df.start_lat)
    df["start_lon_bin"] = to_bin(df.start_lon)
    df["end_lat_bin"] = to_bin(df.end_lat)
    df["end_lon_bin"] = to_bin(df.end_lon)

    names = list(map(int, range(regions)))

    cord_bins = np.array(np.meshgrid(np.arange(regions), np.arange(regions))).T.reshape(-1,2)
    cord_bins = list(map(tuple, cord_bins))

    df['start_lat_regions']=pd.cut(df['start_lat_bin'], bins=regions,labels = names, retbins=False, right=True, include_lowest=True)
    df['start_lon_regions']=pd.cut(df['start_lon_bin'], bins=regions,labels= names, retbins=False, right=True, include_lowest=True)

    df['end_lat_regions']=pd.cut(df['end_lat_bin'], bins=regions,labels = names, retbins=False, right=True, include_lowest=True)
    df['end_lon_regions']=pd.cut(df['end_lon_bin'], bins=regions,labels= names, retbins=False, right=True, include_lowest=True)

    df['PU_region'] =list(zip(df.start_lat_regions, df.start_lon_regions))
    df['DO_region'] =list(zip(df.end_lat_regions, df.end_lon_regions))

    df = df.drop(columns=['start_lat_regions','start_lon_regions','end_lat_regions','end_lon_regions'])

    for i in range(0,len(cord_bins)):
        df.loc[df['PU_region'] == cord_bins[i], 'PU_region'] = i+1
        df.loc[df['DO_region'] == cord_bins[i], 'DO_region'] = i+1
    
    df = df.apply(lambda col:pd.to_numeric(col, errors='coerce'))

    df = df.dropna()

    return df

def gan_data(_df, _data_dir, _data_type='Taxi', _year=2021, _month=1, _num_bins=5):
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
    data_dir = _data_dir
    data_type = _data_type
    year = _year
    month = _month
    dropna = True
    num_bins = _num_bins

    print('\n', data_type, year, month, num_bins)

    df = df.reset_index(drop=True)

    if data_type == 'Taxi':
        drop_cols = ['store_and_fwd_flag', 'VendorID' , 'RatecodeID', 'payment_type','fare_amount', 'extra' , 'mta_tax', 'tip_amount',
        'tolls_amount', 'improvement_surcharge','total_amount', 'congestion_surcharge', 'airport_fee', 'trip_distance', 'passenger_count']
        
        sort_by = 'tpep_pickup_datetime'

        pu_date_time = 'tpep_pickup_datetime'
        do_date_time = 'tpep_dropoff_datetime'

        # aggregate_cols = 'PULocationID', 'DOLocationID', 'pickup_weekday', 'pickup_hour'

    elif data_type == 'CitiBike':
        
        # print('first col', df.columns[0])

        if df.columns[0] == 'tripduration':
            drop_cols = ['tripduration', 'start station id', 'start station name', 'end station id', 'end station name', 'bikeid', 'usertype', 'birth year', 'gender']
            sort_by = 'starttime'
            pu_date_time = 'starttime'
            do_date_time = 'stoptime'
        elif df.columns[0] =='ride_id':
            drop_cols = ['ride_id', 'rideable_type', 'start_station_name', 'start_station_id', 'end_station_name', 'end_station_id', 'member_casual']
            df.rename(columns={'start_lat':'start station latitude','start_lng':'start station longitude','end_lat':'end station latitude','end_lng':'end station longitude'}, inplace=True)
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
    # df['pickup_day'] = df[pu_date_time].apply(lambda t: t.day)
    df['pickup_day'] = df[pu_date_time].apply(lambda t: t.weekday())
    df['pickup_hour'] = df[pu_date_time].apply(lambda t: t.hour)
    
    # Sort by date
    df = df.sort_values(by=sort_by, ascending=True)
    # print('sort',df.shape)
    # Drop columns and remove all unnecessary data (not in target year and month)
    df = df.drop(columns=drop_cols)
    df = df.drop(columns=[pu_date_time, do_date_time])
    df = df[(df['pickup_year'] == year) & (df['pickup_month'] == month)]
    df = df.drop(columns=['pickup_year'])
    if data_type == 'Taxi' or data_type == 'CitiBike':
        df = df.drop(columns=['pickup_month'])
    # print('year, month',df.shape)

    # Convert zone to latitudes and longitudes (only for taxi data)
    if data_type == 'Taxi':
        df = zone_to_coord(df, data_dir)
        # print('zone to coord', df.shape)
    # Bin bike and taxi data by latitude and longitude (start with 32 bins)


    if data_type == 'Taxi' or data_type == 'CitiBike':
        df['start_lat'] = pd.to_numeric(df['start station latitude'])
        df['start_lon'] = pd.to_numeric(df['start station longitude'])
        df['end_lat'] = pd.to_numeric(df['end station latitude'])
        df['end_lon'] = pd.to_numeric(df['end station longitude'])
        # print('to numeric',df.shape)
        # print(df.head(n=1))


        ###########################################
        # """
        df = bin_data(df, num_bins)
        # print('bin',df.shape)

        df = df.apply(lambda col:pd.to_numeric(col, errors='coerce'))
        # print('to_numeric',df.shape)
        if dropna:
            df = df.dropna()
            # print('dropna',df.shape)
        # """
        ### Organize and aggregate taxi and bike demand, shape into 2D frames for each hour
        df1 = df.groupby(['PU_region','DO_region','pickup_day','pickup_hour']).size().reset_index().rename(columns={0:'demand'})
        # df1 = df.groupby(['start_lat_bin','start_lon_bin','end_lat_bin','end_lon_bin','pickup_day','pickup_hour']).size().reset_index().rename(columns={0:'demand'})
        
        # print('df1',df1.shape)
        # days = np.unique(df1.pickup_day)
        # print(days)
        days = [*range(1,calendar.monthrange(year, month)[1]+1,1)]
        rows = num_bins**2
        cols = num_bins**2
        ST_map = np.zeros((rows, cols, 24, (len(days))))
        # print('st_map',ST_map.shape)
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

        ST_map = ST_map.T.reshape((ST_map.shape[-1], ST_map.shape[0], ST_map.shape[1], 1))

    ### Process weather data
    if data_type == 'Weather':
        coords = pd.DataFrame()
        if dropna:
            df = df.dropna()
            # print('dropna',df.shape)
        df1 = df.apply(lambda col:pd.to_numeric(col, errors='coerce'))
        days = [*range(1,calendar.monthrange(year, month)[1]+1,1)]
        ST_map = np.zeros((1, 8, 24, (len(days))))
        # print('st_map',ST_map.shape)
        for hour in range(24):
            for day in days:
                # print(hour,day)
                df2 = df1[(df1.pickup_day == day) & (df1.pickup_hour == hour)]
                # df2 = df2.drop(columns=['pickup_day','pickup_hour'])
                df2 = df2.reset_index(drop=True, inplace=False)
                # print(len(df2))
                for i in range(len(df2)):
                    # rind = int(df2.PU_region[i])
                    # cind = int(df2.DO_region[i])
                    month = df2.pickup_month[i]
                    day = df2.pickup_day[i]
                    hour = df2.pickup_hour[i]
                    temp = df2['temperature (degC)'][i]
                    wind_speed = df2['wind_speed (m/s)'][i]
                    total_precip = df2['total_precipitation (mm of water equivalent)'][i]
                    snowfall = df2['snowfall (mm of water equivalent)'][i]
                    snow_depth = df2['snow_depth (mm of water equivalent)'][i]
                    # Set temperature, wind_speed, total_precipitation, snowfall, and snow_depth to 0 or 1 if they are above or below the threshold values
                    temp_threshold = 5
                    wind_threshold = 5
                    precip_threshold = 0.5
                    snow_threshold = 0.5
                    snow_depth_threshold = 0.5 
                    if temp > temp_threshold:
                        temp = 1
                    else:
                        temp = 0
                    if wind_speed > wind_threshold:
                        wind_speed = 1
                    else:
                        wind_speed = 0
                    if total_precip > precip_threshold:
                        total_precip = 1
                    else:
                        total_precip = 0
                    if snowfall > snow_threshold:
                        snowfall = 1
                    else:
                        snowfall = 0
                    if snow_depth > snow_depth_threshold:
                        snow_depth = 1
                    else:
                        snow_depth = 0
                    ST_map[0, 0, hour-1, day-1] = month
                    ST_map[0, 1, hour-1, day-1] = day
                    ST_map[0, 2, hour-1, day-1] = hour
                    ST_map[:, 3, hour-1, day-1] = temp
                    ST_map[:, 4, hour-1, day-1] = wind_speed
                    ST_map[:, 5, hour-1, day-1] = total_precip
                    ST_map[:, 6, hour-1, day-1] = snowfall
                    ST_map[:, 7, hour-1, day-1] = snow_depth

        print(ST_map.shape)
        #stack the st maps vertically to make one long 3D array of each hour for the month
        ST_map = ST_map.reshape((ST_map.shape[0], ST_map.shape[1], -1))
        ST_map = ST_map.T.reshape((ST_map.shape[-1], ST_map.shape[0], ST_map.shape[1]))

    return ST_map

def feedforward_data(_df, _data_dir, _data_type, _year, _month, _num_bins=5):
    """
    Preprocess one month of data into a format that can be fed into the feedforward model.

    Args:
        df (pandas.DataFrame): data frame containing the loaded data
        data_dir (str): directory where the data is stored
        data_type (str): the type of data contained in the df (i.e. Taxi, CitiBike, Metro, Weather)
        year (int): the year of the data
        month (int): the month of the data
        num_bins (int): the number of bins to use for the lat/long data

    Returns:
        df (pandas.DataFrame): preprocessed data frame containing the data for the feedforward model
        
        df has the following columns: ['month', 'weekday', 'hour', 'start_lat_bin', 'start_lon_bin', ...
        ... 'end_lat_bin', 'end_lon_bin', 'weather_condition', 'taxi_demand', 'bike_demand']

    """
    df = _df
    data_dir = _data_dir
    data_type = _data_type
    year = _year
    month = _month
    num_bins = _num_bins

    dropna = True

    print('\n', data_type, year, month, num_bins)

    df = df.reset_index(drop=True)

    if data_type == 'Taxi':
        drop_cols = ['store_and_fwd_flag', 'VendorID' , 'RatecodeID', 'payment_type','fare_amount', 'extra' , 'mta_tax', 'tip_amount',
        'tolls_amount', 'improvement_surcharge','total_amount', 'congestion_surcharge', 'airport_fee', 'trip_distance', 'passenger_count']
        
        sort_by = 'tpep_pickup_datetime'

        pu_date_time = 'tpep_pickup_datetime'
        do_date_time = 'tpep_dropoff_datetime'

    elif data_type == 'CitiBike':

        if df.columns[0] == 'tripduration':
            drop_cols = ['tripduration', 'start station id', 'start station name', 'end station id', 'end station name', 'bikeid', 'usertype', 'birth year', 'gender']
            sort_by = 'starttime'
            pu_date_time = 'starttime'
            do_date_time = 'stoptime'
        elif df.columns[0] =='ride_id':
            drop_cols = ['ride_id', 'rideable_type', 'start_station_name', 'start_station_id', 'end_station_name', 'end_station_id', 'member_casual']
            df.rename(columns={'start_lat':'start station latitude','start_lng':'start station longitude','end_lat':'end station latitude','end_lng':'end station longitude'}, inplace=True)
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
    # df['pickup_day'] = df[pu_date_time].apply(lambda t: t.day)
    df['pickup_day'] = df[pu_date_time].apply(lambda t: t.weekday())
    df['pickup_hour'] = df[pu_date_time].apply(lambda t: t.hour)
    
    # Sort by date
    df = df.sort_values(by=sort_by, ascending=True)

    # Drop columns and remove all unnecessary data (not in target year and month)
    df = df.drop(columns=drop_cols)
    df = df.drop(columns=[pu_date_time, do_date_time])
    df = df[(df['pickup_year'] == year) & (df['pickup_month'] == month)]
    # df = df.drop(columns=['pickup_year', 'pickup_month'])
    df = df.drop(columns=['pickup_year'])

    # Convert zone to latitudes and longitudes (only for taxi data)
    if data_type == 'Taxi':
        df = zone_to_coord(df, data_dir)

    if data_type == 'Taxi' or data_type == 'CitiBike':
        df['start_lat'] = pd.to_numeric(df['start station latitude'])
        df['start_lon'] = pd.to_numeric(df['start station longitude'])
        df['end_lat'] = pd.to_numeric(df['end station latitude'])
        df['end_lon'] = pd.to_numeric(df['end station longitude'])

        # Bin bike and taxi data by latitude and longitude
        binned_df = bin_data(df, num_bins)
        binned_df = binned_df.apply(lambda col:pd.to_numeric(col, errors='coerce'))

        # Create new column with all rides that start and end in each combination of bins
        binned_df['start_end'] = binned_df['start_lat_bin'].astype(str) + '_' + binned_df['start_lon_bin'].astype(str) + '_' + binned_df['end_lat_bin'].astype(str) + '_' + binned_df['end_lon_bin'].astype(str)
        # Group by start_end column, then pickup_month, pickup_day, and pickup_hour and count the number of rides in each group
        df1 = binned_df.groupby(['start_end', 'pickup_month', 'pickup_day', 'pickup_hour']).size().reset_index(name='demand')
        # Convert each unique start_end string to an integer
        df1['start_end'] = df1['start_end'].astype('category').cat.codes
        # df1 = binned_df.groupby(['start_lat_bin','start_lon_bin','pickup_month','pickup_day','pickup_hour']).size().reset_index().rename(columns={0:'demand'})
        df1 = df1.sort_values(['pickup_month','pickup_day', 'pickup_hour'])

        if data_type == 'Taxi':
            df1 = df1.rename(columns={'demand':'taxi_demand'})
        elif data_type == 'CitiBike':
            df1 = df1.rename(columns={'demand':'bike_demand'})

    if data_type == 'Weather':
        if dropna:
            df = df.dropna()

        df1 = df.apply(lambda col:pd.to_numeric(col, errors='coerce'))

        # Get average of temperature (degC), wind_speed (m/s), total_precipitation (mm of water equivalent), snowfall (mm of water equivalent), and snow_depth (mm of water equivalent) for each hour of each day
        df1 = df1.groupby(['pickup_month','pickup_day','pickup_hour']).agg({'temperature (degC)':'mean', 'wind_speed (m/s)':'mean', 'total_precipitation (mm of water equivalent)':'mean', 'snowfall (mm of water equivalent)':'mean', 'snow_depth (mm of water equivalent)':'mean'}).reset_index()
        
        # Set temperature, wind_speed, total_precipitation, snowfall, and snow_depth to 0 or 1 if they are above or below the threshold values
        temp_threshold = 5
        wind_threshold = 5
        precip_threshold = 0.5
        snow_threshold = 0.5
        snow_depth_threshold = 0.5
        
        df1['temperature (degC)'] = df1['temperature (degC)'].apply(lambda x: 1 if x > temp_threshold else 0)
        df1['wind_speed (m/s)'] = df1['wind_speed (m/s)'].apply(lambda x: 1 if x > wind_threshold else 0)
        df1['total_precipitation (mm of water equivalent)'] = df1['total_precipitation (mm of water equivalent)'].apply(lambda x: 1 if x > precip_threshold else 0)
        df1['snowfall (mm of water equivalent)'] = df1['snowfall (mm of water equivalent)'].apply(lambda x: 1 if x > snow_threshold else 0)
        df1['snow_depth (mm of water equivalent)'] = df1['snow_depth (mm of water equivalent)'].apply(lambda x: 1 if x > snow_depth_threshold else 0)


        df1 = df1.sort_values(['pickup_month','pickup_day', 'pickup_hour'])
    return df1

def merge_data(taxi_df, bike_df, weather_df):
    """Merge taxi, bike, and weather data together. Apply a bike demand value of zero to all lat/lon bins that do not have bike demand data."""
    df = taxi_df.merge(bike_df, how='outer', on=['start_end', 'pickup_month', 'pickup_day', 'pickup_hour'])
    # df = taxi_df.merge(bike_df, how='outer', on=['start_lat_bin','start_lon_bin','pickup_month','pickup_day','pickup_hour'])
    # Merge the weather data by adding the weather conditions to each lat/lon bin for each hour of each day
    df = df.merge(weather_df, how='left', on=['pickup_month','pickup_day','pickup_hour'])
    df = df.fillna(0)
    return df

def transform_ff_data(df, num_bins):
    """Transform the data for the feedforward neural network."""
    # Normalize start_end column by total number of pickup to dropoff combinations
    df['start_end'] = df['start_end']/(num_bins**2)**2
    # Convert pickup_month, pickup_day, and pickup_hour to sin and cos values
    df['pickup_month_sin'] = np.sin(2*np.pi*df['pickup_month']/12)
    df['pickup_month_cos'] = np.cos(2*np.pi*df['pickup_month']/12)
    df['pickup_day_sin'] = np.sin(2*np.pi*df['pickup_day']/7)
    df['pickup_day_cos'] = np.cos(2*np.pi*df['pickup_day']/7)
    df['pickup_hour_sin'] = np.sin(2*np.pi*df['pickup_hour']/24)
    df['pickup_hour_cos'] = np.cos(2*np.pi*df['pickup_hour']/24)
    # Drop pickup_month, pickup_day, and pickup_hour columns
    df = df.drop(['pickup_month', 'pickup_day', 'pickup_hour'], axis=1)
    # Create y dataframe with taxi_demand and bike_demand columns
    y = df[['taxi_demand', 'bike_demand']]
    # Drop taxi_demand and bike_demand columns from X dataframe
    X = df.drop(['taxi_demand', 'bike_demand'], axis=1)

    # Reorder X columns
    X = X[['pickup_month_sin', 'pickup_month_cos','pickup_day_sin', 'pickup_day_cos', 'pickup_hour_sin', 'pickup_hour_cos', 'start_end', 'temperature (degC)', 'wind_speed (m/s)', 'total_precipitation (mm of water equivalent)', 'snowfall (mm of water equivalent)', 'snow_depth (mm of water equivalent)']]
    # concat X and y dataframes
    dataset_final = pd.concat([X, y], axis=1)

    return dataset_final

def normalize_data(df):
    """Normalize the data."""
    # Normalize the data using MinMaxScaler
    bike_scaler = MinMaxScaler()
    taxi_scaler = MinMaxScaler()

    df['bike_demand'] = bike_scaler.fit_transform(df['bike_demand'].values.reshape(-1,1))
    df['taxi_demand'] = taxi_scaler.fit_transform(df['taxi_demand'].values.reshape(-1,1))

    # Save the scalers
    joblib.dump(bike_scaler, 'bike_scaler.gz')
    joblib.dump(taxi_scaler, 'taxi_scaler.gz')
    return df