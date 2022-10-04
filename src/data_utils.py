"""
Data Preprocessing Utilities

Contains methods for loading and preprocessing datasets.

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
# from sklearn.model_selection import train_test_split

import shapefile

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit


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

def zone_to_coord(_df):
    df = _df
    coords = pd.read_csv('/Users/probook/Documents/GitHub/mobilityforecast/utils/taxi_zones/taxi_zones.csv')

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

def coords_to_zones(_df):
    df = _df

    return df

def bin_data(df, num_bins=5):
 
    regions = num_bins

    # start_lat_min = df['start_lat'].min()
    # start_lat_max = df['start_lat'].max()
    # start_lng_min = df['start_lng'].min()
    # start_lng_max = df['start_lng'].max()

    # end_lat_min = df['end_lat'].min()
    # end_lat_max = df['end_lat'].max()
    # end_lng_min = df['end_lng'].min()
    # end_lng_max = df['end_lng'].max()

    # south = min(start_lat_min, end_lat_min)
    south = 40.5
    north = 40.9
    # north = max(start_lat_max, end_lat_max)

    # west = min(start_lng_min, end_lng_min)
    west = -74.25
    east = -73.9
    # east = max(start_lng_max, end_lng_max)

    print('south: ', south, 'north: ', north, 'west: ', west, 'east: ', east)

    lat_bins = np.linspace(south,north,regions)
    lng_bins = np.linspace(west, east, regions)

    print('bins: ', lat_bins, lng_bins)

    step = (north - south) / regions
    to_bin = lambda x: np.floor(x / step) * step
    df["start_lat_bin"] = to_bin(df.start_lat)
    df["start_lon_bin"] = to_bin(df.start_lng)
    df["end_lat_bin"] = to_bin(df.end_lat)
    df["end_lon_bin"] = to_bin(df.end_lng)

    # print('******* binned',df.head())

    coords = df[['start_lat_bin', 'start_lon_bin', 'end_lat_bin', 'end_lon_bin']].copy()

    names = list(map(int, range(regions)))

    # print('names: ', names)


    cord_bins = np.array(np.meshgrid(np.arange(regions), np.arange(regions))).T.reshape(-1,2)
    cord_bins = list(map(tuple, cord_bins))

    # print('cord_bins: ', cord_bins)

    df['start_lat_regions']=pd.cut(df['start_lat_bin'], bins=regions,labels = names, retbins=False, right=True, include_lowest=True)
    df['start_lng_regions']=pd.cut(df['start_lon_bin'], bins=regions,labels= names, retbins=False, right=True, include_lowest=True)

    # print('start_lat_regions: ', df['start_lat_regions'])

    df['end_lat_regions']=pd.cut(df['end_lat_bin'], bins=regions,labels = names, retbins=False, right=True, include_lowest=True)
    df['end_lng_regions']=pd.cut(df['end_lon_bin'], bins=regions,labels= names, retbins=False, right=True, include_lowest=True)

    # print('******* to regions',df.head())

    df['PU_region'] =list(zip(df.start_lat_regions, df.start_lng_regions)) #np.array([*tmp]).tolist()
    df['DO_region'] =list(zip(df.end_lat_regions, df.end_lng_regions))

    # print('******* PU_region: ', df.head())
    # print('pu regions: ', df['PU_region'].unique())
    # print('df regions: ', df['DO_region'].unique())

    df = df.drop(columns=['start_lat_regions','start_lng_regions','end_lat_regions','end_lng_regions'])

    for i in range(0,len(cord_bins)):
        df.loc[df['PU_region'] == cord_bins[i], 'PU_region'] = i+1
        df.loc[df['DO_region'] == cord_bins[i], 'DO_region'] = i+1

    return df, coords

def data_preprocessing(_df, _data_type='Taxi', _year=2021, _month=1, _dropna=True, _num_bins=10):
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
    data_type = _data_type
    year = _year
    month = _month
    dropna = _dropna
    num_bins = _num_bins

    print('\n', data_type, year, month, dropna, num_bins)
    # print('orig',df.shape)
    # if dropna:
    #     df = df.dropna()
    #     print('dropna',df.shape)
    df = df.reset_index(drop=True)
    # df = df.reset_index()
    # print('reset index',df.shape)

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
    df['pickup_day'] = df[pu_date_time].apply(lambda t: t.day)
    df['pickup_hour'] = df[pu_date_time].apply(lambda t: t.hour)
    
    # Sort by date
    df = df.sort_values(by=sort_by, ascending=True)
    # print('sort',df.shape)
    # Drop columns and remove all unnecessary data (not in target year and month)
    df = df.drop(columns=drop_cols)
    df = df.drop(columns=[pu_date_time, do_date_time])
    df = df[(df['pickup_year'] == year) & (df['pickup_month'] == month)]
    df = df.drop(columns=['pickup_year', 'pickup_month'])
    # print('year, month',df.shape)

    # Convert zone to latitudes and longitudes (only for taxi data)
    if data_type == 'Taxi':
        df = zone_to_coord(df)
        # print('zone to coord', df.shape)
    # Bin bike and taxi data by latitude and longitude (start with 32 bins)


    if data_type == 'Taxi' or data_type == 'CitiBike':
        df['start_lat'] = pd.to_numeric(df['start station latitude'])
        df['start_lng'] = pd.to_numeric(df['start station longitude'])
        df['end_lat'] = pd.to_numeric(df['end station latitude'])
        df['end_lng'] = pd.to_numeric(df['end station longitude'])
        # print('to numeric',df.shape)
        # print(df.head(n=1))


        ###########################################
        # """
        df, coords = bin_data(df, num_bins)
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

    ### Process weather data
    if data_type == 'Weather':
        coords = pd.DataFrame()
        if dropna:
            df = df.dropna()
            # print('dropna',df.shape)
        df1 = df.apply(lambda col:pd.to_numeric(col, errors='coerce'))
        days = np.unique(df1.pickup_day)
        ST_map = np.zeros((1, 5, 24, (len(days))))
        # print('st_map',ST_map.shape)
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

    return ST_map, coords

def plot_map(_ST_map):
    sum_vec = np.sum(np.sum(_ST_map, axis=0), axis=0)
    x = np.arange(0, len(sum_vec), 1)
    # print(x.shape)
    # calc the trendline
    z = poly.polyfit(x, sum_vec, 3)
    # print(z.shape)
    ffit = poly.polyval(x, z)
    # print(ffit.shape)
    # plt.plot(x, ffit)
    # p = np.poly1d(z)
    plt.plot(x,ffit,"r--")
    plt.plot(sum_vec,'.')
    plt.title('Total Demand in All Zones per Hour')
    plt.xlabel('Hour')
    plt.ylabel('Total Trips')
    plt.style.use('seaborn')
    plt.show()

def plot_heatmap(_ST_map, _year, _month, _day, _hour):
        ### Plot one hour of data
        year = _year
        month = _month
        day = _day
        hour = _hour
        ax = sns.heatmap(_ST_map[:,:,hour], cmap="YlGnBu")
        # plt.imshow(_ST_map[:,:,(hour+((day-1)*24))], cmap='hot', interpolation='None')
        plt.title('Rides in each binned zone on ' + str(year) + ' ' + str(month) + ' ' + str(day) + ' hour ' + str(hour))
        plt.xlabel('Pickup Zone')
        plt.ylabel('Dropoff Zone')
        plt.show()

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


    class SpatialAttention(tf.keras.layers.Layer):
      def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        
        def build(self, input_shape):
            self.conv2d = tf.keras.layers.Conv2D(filters = 1,
                    kernel_size=self.kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)

        def call(self, inputs):
            
            # AvgPool
            avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(inputs)
            
            # MaxPool
            max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(inputs)

            attention = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])

            attention = self.conv2d(attention)


            return tf.keras.layers.multiply([inputs, attention])

def demand_scatter(_coords, _data_type='Taxi'):
    values = _coords.value_counts().reset_index(name='demand')
    temp = (1 + (1 * np.rint(values['demand'].to_numpy()/values['demand'].max())))**5
    values.plot.scatter(x='start_lon_bin', y='start_lat_bin', s=temp, c='DarkBlue')
    plt.xticks(np.arange(-74.4, -73.4, 0.2))
    plt.yticks(np.arange(40.2, 41.2, 0.2))
    plt.title(_data_type + ' Demand')
    plt.show()
    
class SelfAttention(tf.keras.layers.Layer):
    """ Adapted from the Zhang, Goodfellow, et al. paper """

    # weight_init = tf_contrib.layers.xavier_initializer()
    weight_regularizer = None
    weight_regularizer_fully = None

    def __init__(self, sess, args):
        #My new inputs
        self.channels = args.channels


        self.model_name = "SAGAN"  # name for checkpoint
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.img_size = args.img_size

        """ Generator """
        self.layer_num = int(np.log2(self.img_size)) - 3
        self.z_dim = args.z_dim  # dimension of noise-vector
        self.gan_type = args.gan_type

        """ Discriminator """
        self.n_critic = args.n_critic
        self.sn = args.sn
        self.ld = args.ld


        self.sample_num = args.sample_num  # number of generated images to be saved
        self.test_num = args.test_num


        # train
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.custom_dataset = False

    def _spectral_norm(w, iteration=1):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm    

    def _conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
        with tf.variable_scope(scope):
            if pad > 0:
                h = x.get_shape().as_list()[1]
                if h % stride == 0:
                    pad = pad * 2
                else:
                    pad = max(kernel - (h % stride), 0)

                pad_top = pad // 2
                pad_bottom = pad - pad_top
                pad_left = pad // 2
                pad_right = pad - pad_left

                if pad_type == 'zero':
                    x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                if pad_type == 'reflect':
                    x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

            if sn:
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                    regularizer=weight_regularizer)
                x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                                strides=[1, stride, stride, 1], padding='VALID')
                if use_bias:
                    bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                    x = tf.nn.bias_add(x, bias)

            else:
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                    kernel_size=kernel, kernel_initializer=weight_init,
                                    kernel_regularizer=weight_regularizer,
                                    strides=stride, use_bias=use_bias)

            return x



    def _hw_flatten(x) :
        return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

    def _attention(self, x, channels, scope='attention'):
        with tf.variable_scope(scope):

            f = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv') # [bs, h, w, c']
            g = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv') # [bs, h, w, c']
            h = conv(x, channels, kernel=1, stride=1, sn=self.sn, scope='h_conv') # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
            o = conv(o, channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')

            x = gamma * o + x

        return x

    def _build_module(self):

        attention_block = self.attention()
    """ Adapted from the Zhang, Goodfellow, et al. paper """

    # weight_init = tf_contrib.layers.xavier_initializer()
    weight_regularizer = None
    weight_regularizer_fully = None

    def __init__(self, sess, args):
        #My new inputs
        self.channels = args.channels


        self.model_name = "SAGAN"  # name for checkpoint
        self.sess = sess
        self.dataset_name = args.dataset
        self.checkpoint_dir = args.checkpoint_dir
        self.sample_dir = args.sample_dir
        self.result_dir = args.result_dir
        self.log_dir = args.log_dir

        self.epoch = args.epoch
        self.iteration = args.iteration
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.img_size = args.img_size

        """ Generator """
        self.layer_num = int(np.log2(self.img_size)) - 3
        self.z_dim = args.z_dim  # dimension of noise-vector
        self.gan_type = args.gan_type

        """ Discriminator """
        self.n_critic = args.n_critic
        self.sn = args.sn
        self.ld = args.ld


        self.sample_num = args.sample_num  # number of generated images to be saved
        self.test_num = args.test_num


        # train
        self.g_learning_rate = args.g_lr
        self.d_learning_rate = args.d_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.custom_dataset = False

    def _spectral_norm(w, iteration=1):
        w_shape = w.shape.as_list()
        w = tf.reshape(w, [-1, w_shape[-1]])

        u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

        u_hat = u
        v_hat = None
        for i in range(iteration):
            """
            power iteration
            Usually iteration = 1 will be enough
            """
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)

        return w_norm    

    def _conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
        with tf.variable_scope(scope):
            if pad > 0:
                h = x.get_shape().as_list()[1]
                if h % stride == 0:
                    pad = pad * 2
                else:
                    pad = max(kernel - (h % stride), 0)

                pad_top = pad // 2
                pad_bottom = pad - pad_top
                pad_left = pad // 2
                pad_right = pad - pad_left

                if pad_type == 'zero':
                    x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
                if pad_type == 'reflect':
                    x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

            if sn:
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                    regularizer=weight_regularizer)
                x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                                strides=[1, stride, stride, 1], padding='VALID')
                if use_bias:
                    bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                    x = tf.nn.bias_add(x, bias)

            else:
                x = tf.layers.conv2d(inputs=x, filters=channels,
                                    kernel_size=kernel, kernel_initializer=weight_init,
                                    kernel_regularizer=weight_regularizer,
                                    strides=stride, use_bias=use_bias)

            return x



    def _hw_flatten(x) :
        return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

    def _attention(self, x, channels, scope='attention'):
        with tf.variable_scope(scope):

            f = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='f_conv') # [bs, h, w, c']
            g = conv(x, channels // 8, kernel=1, stride=1, sn=self.sn, scope='g_conv') # [bs, h, w, c']
            h = conv(x, channels, kernel=1, stride=1, sn=self.sn, scope='h_conv') # [bs, h, w, c]

            # N = h * w
            s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]

            beta = tf.nn.softmax(s)  # attention map

            o = tf.matmul(beta, hw_flatten(h)) # [bs, N, C]
            gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

            o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
            o = conv(o, channels, kernel=1, stride=1, sn=self.sn, scope='attn_conv')

            x = gamma * o + x

        return x

    def _build_module(self):

        attention_block = self.attention()