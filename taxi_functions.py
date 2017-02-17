import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import pandas as pd
from scipy.stats import mode
import googlemaps
from datetime import datetime


time_format = '%H:%M:%S'

taxi_msrp = 24660.0    #source for Nissan NV200 www.edmunds.com 2016 Nissan NV200 Minivan
sprinter_msrp = 40745.0 #source for sprinter msrp www.mbvans.com Passenger Van 2500 Standard Roof
taxi_mpg = 24.0 #source for taxi MPG based on www.fueleconomy.org 2016 Nissan NV200 NYC taxi
sprinter_mpg = 18.0 #source for sprinter MPG based on www.fuelly.com Mercedes Benz Sprinter 2016
usd_gal = 2.626 #cost of gas per gallon in USD, source: http://www.newyorkgasprices.com/, Retrieved 02/06/2017



#get_cluster_labels takes input data X, a linkage metric, a max_dist
#between clusters, and a crit(erion) for clustering. It computes the
#cluster labels for each row N-tuple in X.
def get_cluster_labels(X, metric, max_dist, crit):
    Z = linkage(X, metric)
    return fcluster(Z, max_dist, criterion = crit)

def get_labeled_pickup_clusters(data, metric = 'ward', max_distance = 0.005, criterion = 'distance'):
    X_lat = data.loc[:,['pickup_latitude']]
    X_long = data.loc[:,['pickup_longitude']]
    X_pickup_location = np.column_stack((X_lat, X_long))
    Z_linkages = linkage(X_pickup_location, metric)
    data_clustered = fcluster(Z_linkages, max_distance, criterion)
    return data_clustered

def get_labeled_dropoff_clusters(data, metric = 'ward', max_distance = 0.005, criterion = 'distance'):
    X_lat = data.loc[:,['dropoff_latitude']]
    X_long = data.loc[:,['dropoff_longitude']]
    X_pickup_location = np.column_stack((X_lat, X_long))
    Z_linkages = linkage(X_pickup_location, metric)
    data_clustered = fcluster(Z_linkages, max_distance, criterion)
    return data_clustered

def add_dropoff_cluster_labels_to_col(data, metric = 'ward', max_distance = 0.005, criterion = 'distance'):
    cluster_column = get_labeled_dropoff_clusters(data, metric, max_distance, criterion)
    data['dropoff_cluster'] = cluster_column
    return data

def add_cluster_labels_to_col(data, metric = 'ward', max_distance = 0.005, criterion = 'distance'):
    cluster_column = get_labeled_pickup_clusters(data, metric, max_distance, criterion)
    data['cluster'] = cluster_column
    return data

#get_mode_cluster_pair
def get_mode_cluster_pair(data, metric = 'ward', max_distance = 0.005):
    data0 = add_cluster_labels_to_col(data, metric, max_distance)
    mode_value, mode_count = get_mode_cluster(data0)
    data1 = data0[data0['cluster']==mode_value]
    data2 = add_dropoff_cluster_labels_to_col(data1, metric, max_distance)
    mode_value, mode_count = get_mode_cluster(data2, col_name = 'dropoff_cluster')
    data3 = data2[data2['dropoff_cluster']==mode_value]
    return data3

#get_mode_cluster takes a taxi data dataframe that has been labeled with clusters
#in the 'clusters' column, typically by add_cluster_labels_to_col, and then
#computes the mode of the cluster data, returning the mode value and the count
def get_mode_cluster(data, col_name = 'cluster'):
    mode_result = mode(data[col_name])
    mode_value = mode_result[0][0]
    mode_count = mode_result[1][0]
    return mode_value, mode_count

#compute_revenue computes the per-passenger cost of each ride by basic division
def compute_revenue(data, pass_col_name = 'passenger_count', cost_col_name = 'total_amount'):
    data['revenue_per_passenger'] = np.divide(data[cost_col_name], data[pass_col_name])
    return data

#str_to_time is a batch processing function that converts a row (in a lambda function)
#from a str format to a Timestamp.time() format. This is applied to the CSV taxi data
def str_to_time(row,col_name):
    time_format = '%H:%M:%S'
    return pd.Timestamp(row[col_name]).time()

def get_time_range(data, t0, tf, col_name):
    import numpy as np
    for i in range(0,len(data.iloc[:,0])):
        temp = data.ix[i,col_name]
        if ~(temp >= t0 and temp <= tf):
            data = np.delete(data, (i), axis = 0)
    return data

def cols_to_datetime(data, cols):
    data[cols] = data[cols].apply(pd.to_datetime)
    return data

def seconds_from_midnight(time):
    return time.hour*3600 + time.minute*60 + time.second

def add_time_to_data(data):
    data['time'] = data['tpep_pickup_datetime'].apply(lambda row: row.time())
    data['time_seconds'] = data['time'].apply(lambda row: seconds_from_midnight(row))
    return data

def add_weekday_col_to_data(data, col_input, name_out_col):
    data[name_out_col] = data[col_input].apply(lambda row: pd.datetime.weekday(row))
    return data

#clean_taxi_data takes a pandas data frame as input that is imported
#from the CSV taxi data found at http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml
# and follows the standards guide given on the page. This function
# deletes rows where certain key values are zero.
def clean_taxi_data(data):
    zero_passenger_count = data.passenger_count == 0
    data = data[~zero_passenger_count]

    zero_trip_distance = data.trip_distance == 0
    data = data[~zero_trip_distance]

    zero_lat0 = (data.pickup_latitude == 0)
    data = data[~zero_lat0]

    zero_long0 = (data.pickup_longitude == 0)
    data = data[~zero_long0]

    zero_lat1 = (data.dropoff_latitude == 0)
    data = data[~zero_lat1]

    zero_long1 = (data.dropoff_longitude == 0)
    data = data[~zero_long1]
    return data

def clean_and_augment_taxi_data(data):
    data = clean_taxi_data(data)
    data = cols_to_datetime(data, ['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    data = add_time_to_data(data)
    data = add_weekday_col_to_data(data, 'tpep_pickup_datetime', 'weekday')
    return data

def get_average_locations(location_vector):
    return np.mean(location_vector, axis=0)

def get_directions_result(gmaps_client, pickup_lat_long, dropoff_lat_long, mode = 'driving', departure_time = datetime.now()):
    return gmaps_client.directions(pickup_lat_long, dropoff_lat_long, mode = mode, departure_time = departure_time)

#get_duration_from_dir_result takes a directions_result from googlemaps API and
#returns the duration (in seconds) from direction_result
def get_duration_from_dir_result(directions_result):
    return directions_result[0].get('legs')[0].get('duration').get('value')

#get_driving_transit_durations takes an instance of the googlemaps client and
#the pickup locations and dropoff locations (as lat,long pairs in 1x2 arrays)
#as well as the start time of the directions and computes the estimated time
#from pickup to dropoff for 'driving' and then for 'transit' (in seconds)
def get_driving_transit_durations(gmaps_client, pickup_lat_long, dropoff_lat_long, departure_time = datetime.now()):
    driving_result = get_directions_result(gmaps_client, pickup_lat_long, dropoff_lat_long, mode = 'driving', departure_time = departure_time)
    transit_result = get_directions_result(gmaps_client, pickup_lat_long, dropoff_lat_long, mode = 'transit', departure_time = departure_time)
    return [get_duration_from_dir_result(driving_result), get_duration_from_dir_result(transit_result)]

#ungroup_fares takes a taxi dataframe and for each person in the cab creates a
#copy of the entry. The whole dataframe is returned
def ungroup_fares(data):
    passenger_count = data['passenger_count']
    i = 0
    temp_count = 0
    ungrouped_data = data
    while i < len(passenger_count):
        temp_count = passenger_count.iloc[i]
        ride = data.iloc[i,:]
        ungrouped_data = ungrouped_data.append([ride], ignore_index = False)
        i += 1
    return ungrouped_data


#create_sql_connection creates an engine in postgres and a connection in psycopg2
# and returns the connection link for query usage.
def create_sql_connection(user = 'postgres', host = 'localhost', password = 'lo1120t4t!', dbname = 'jeffkahnjr'):
    db = create_engine('postgresql://%s:%s@%s:5432/%s'%(user,password,host,dbname))
    conn = None
    connect_str = "dbname='%s' user='%s' host='%s' password='%s'"%(dbname,user,host,password)
    conn = psycopg2.connect(connect_str)
    return conn

def compute_gas_costs(data, taxi_mpg = taxi_mpg, sprinter_mpg = sprinter_mpg, usd_gal = usd_gal):
    distance_miles = data['trip_distance']
    data['taxi_gas_cost'] = np.multiply((1.0/taxi_mpg)*usd_gal, distance_miles)
    data['van_gas_cost'] = np.multiply((1.0/sprinter_mpg)*usd_gal, distance_miles)
    return data



#cluster_algorithm
def cluster_algorithm(gmaps_client, conn, sql_query):
    #query sql database
    query_results = pd.read_sql_query(sql_query,conn)
    #ungroup fares
    query_results = ungroup_fares(query_results)
    #cluster pickups
    clustered_pickups = add_cluster_labels_to_col(data = query_results)
    #compute mode cluster and store
    mode_value, mode_count = get_mode_cluster(clustered_pickups)
    mode_pickup_cluster = clustered_pickups[clustered_pickups['cluster']==mode_value]
    #cluster dropoffs based on mode_pickup_cluster
    clustered_pickups_dropoffs = add_dropoff_cluster_labels_to_col(mode_pickup_cluster)
    mode_value, mode_count = get_mode_cluster(clustered_pickups_dropoffs, col_name = 'dropoff_cluster')
    mode_dropoff_cluster = clustered_pickups_dropoffs[clustered_pickups_dropoffs['dropoff_cluster']==mode_value]
    #compute revenue
    clustered_pickups_dropoffs = compute_revenue(clustered_pickups_dropoffs)
    #compute cost based on MPG
    clustered_pickups_dropoffs = compute_gas_costs(clustered_pickups_dropoffs)
    #compute travel times for driving and transit between biggest clusters
    pickup_lat_long = [clustered_pickups_dropoffs['pickup_latitude'].iloc[0], clustered_pickups_dropoffs['pickup_longitude'].iloc[0]]
    dropoff_lat_long = [clustered_pickups_dropoffs['dropoff_latitude'].iloc[0], clustered_pickups_dropoffs['dropoff_longitude'].iloc[0]]
    [driving_duration, transit_duration] = get_driving_transit_durations(gmaps_client, pickup_lat_long, dropoff_lat_long, departure_time = datetime.now())
    #return data, pickup and dropoff locations
    return clustered_pickups_dropoffs, mode_pickup_cluster, mode_dropoff_cluster, driving_duration, transit_duration

def cluster_pickups_dropoffs(conn, sql_query, metric = 'ward', max_distance = 0.005, criterion = 'distance'):
    #query sql database
    query_results = pd.read_sql_query(sql_query,conn)
    #ungroup fares
    query_results = ungroup_fares(query_results)
    #cluster pickups
    clustered_pickups = add_cluster_labels_to_col(data = query_results, metric = metric, max_distance = max_distance, criterion = criterion)
    #cluster dropoffs based on mode_pickup_cluster
    clustered_pickups_dropoffs = add_dropoff_cluster_labels_to_col(data = clustered_pickups, metric = metric, max_distance = max_distance, criterion = criterion)
    #compute revenue
    clustered_pickups_dropoffs = compute_revenue(clustered_pickups_dropoffs)
    #compute cost based on MPG
    clustered_pickups_dropoffs = compute_gas_costs(clustered_pickups_dropoffs)
    return clustered_pickups_dropoffs

def get_pickup_lat_long_from_frame(data):
    return  np.column_stack((data['pickup_latitude'].iloc[:], data['pickup_longitude'].iloc[:]))

def get_dropoff_lat_long_from_frame(data):
    return  np.column_stack((data['dropoff_latitude'].iloc[:], data['dropoff_longitude'].iloc[:]))


def get_top_n_clusters(clustered_data,label_col,n_clusters):
    cluster_probs = clustered_data.groupby(label_col).size().div(len(clustered_data))
    cluster_probs_sorted = cluster_probs.sort_values(ascending = False)
    top_n = cluster_probs_sorted.index[0:n_clusters]
    data_top_n = clustered_data[clustered_data[label_col].isin(top_n)]
    return data_top_n

def get_dbname_from_sec(dbnames, pickup_time_sec):
    key_seconds = int(pickup_time_sec/3600)
    return dbnames[key_seconds]

def get_dbname_from_time(time, dbnames):
    sec = seconds_from_midnight(time)
    return get_dbname_from_sec(dbnames, sec)

    #function that generates the columns of all pickups for use in gmaps
def get_heatmap_locations(dbnames, con, start_time = '00:00:00', wait_time = 300, weekday = 0):
    #convert time string to pd datetime object
    time_format = '%H:%M:%S'
    time_hms = pd.to_datetime(start_time, format=time_format, errors='coerce')
    #compute start_time_sec as seconds from midnight, end_time ''
    start_time_sec = seconds_from_midnight(time_hms)
    end_time_sec = start_time_sec+wait_time
    #run SQL query on database
    dbname = get_dbname_from_time(time_hms, dbnames)
    sql_query = """
          SELECT pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude FROM {0} WHERE time_seconds>={1} AND time_seconds<={2} AND weekday={3};
          """.format(dbname,start_time_sec,end_time_sec, weekday)
    trips = pd.read_sql_query(sql_query,con)
    pickup_lat_long = np.column_stack((trips['pickup_latitude'].iloc[:], trips['pickup_longitude'].iloc[:]))
    dropoff_lat_long = np.column_stack((trips['dropoff_latitude'].iloc[:], trips['dropoff_longitude'].iloc[:]))
    return pickup_lat_long, dropoff_lat_long

def get_daily_revenues_for_clusters(pickups_top5, dropoffs_top5, window_minutes=60, days=180):
    #compute mode cluster and store
    import flask_dir.taxi_functions as tf
    mode_value, mode_count = tf.get_mode_cluster(pickups_top5)
    mode_pickup_cluster = pickups_top5[pickups_top5['cluster']==mode_value]
    #cluster dropoffs based on mode_pickup_cluster
    clustered_pickups_dropoffs = tf.add_dropoff_cluster_labels_to_col(mode_pickup_cluster)
    mode_value, mode_count = tf.get_mode_cluster(clustered_pickups_dropoffs, col_name = 'dropoff_cluster')
    mode_dropoff_cluster = clustered_pickups_dropoffs[clustered_pickups_dropoffs['dropoff_cluster']==mode_value]
    #compute average estimated revenues
    rev_pickups_top5 = pickups_top5['revenue_per_passenger'].sum()/days*window_minutes
    rev_pickups_top = mode_pickup_cluster['revenue_per_passenger'].sum()/days*window_minutes
    rev_pair_top = mode_dropoff_cluster['revenue_per_passenger'].sum()/days*window_minutes
    return [rev_pickups_top5, rev_pickups_top, rev_pair_top]

#create a function to calculate the number of rides and number of passengers within the time window
def count_rides_in_cluster(clustered_data):
    #remove duplicates from clustered_data
    data_dedup = clustered_data.drop_duplicates(['label'])
    #compute ride count as a sum of all rows in cluster_data
    ride_count = data_dedup.shape[0]
    #compute passenger_count as a sum of all passenger counts in cluster_data
    passenger_count = data_dedup['passenger_count'].sum()
    return passenger_count, ride_count
