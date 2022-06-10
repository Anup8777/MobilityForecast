"""
Data Preprocessing Utilities

Contains methods for loading and preprocessing datasets.

"""

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