from flask import render_template
from flask_dir import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request
import numpy as np
from datetime import datetime
import flask_dir.taxi_functions as tf
from scipy.stats import mode
import gmaps
import flask_dir.taxi_functions as tf
import googlemaps

user = 'jeffkahnjr' #add your username here (same as previous postgreSQL)
host = 'localhost'
password = 'lo1120t4t!'
dbname = 'jeffkahnjr'
db = create_engine('postgresql://%s:%s@%s:5432/%s'%(user,password,host,dbname))
con = None
connect_str = "dbname='%s' user='%s' host='%s' password='%s'"%(dbname,user,host,password)
con = psycopg2.connect(connect_str)

@app.route('/')
@app.route('/index')
def index():
    return render_template("gmaps_ex.html")


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/slides')
def slides():
    return render_template("slides.html")

@app.route('/details')
def details():
    return render_template('details.html')

@app.route('/aboutme')
def about_me():
    return render_template('aboutme.html')

@app.route('/output')
def blt_output():
    #'start_time', 'wait_time', 'current_latitude', 'current_longitude'
      start_time = request.args.get('start_time')
      start_time_datetime = pd.to_datetime(start_time, format='%H:%M:%S', errors='coerce')
      wait_time_seconds = 5
      current_latitude = request.args.get('current_latitude')
      current_longitude = request.args.get('current_longitude')
      start_time_seconds_from_midnight = tf.seconds_from_midnight(start_time_datetime)
      end_time = start_time_seconds_from_midnight+wait_time_seconds

      sql_query = """
                  SELECT * FROM trips_00 WHERE time_seconds>={0} AND time_seconds<={1};
                  """.format(start_time_seconds_from_midnight,end_time)
      query_results = pd.read_sql_query(sql_query,con)
      labeled_data = tf.get_mode_cluster_pair(query_results, metric = 'ward', max_distance = 0.005)
      trips = []
      #for i in range(0,100):

#      for i in range(0,query_results.shape[0]): #use query_results.shape[0]
#          trips.append(dict(index=query_results.iloc[i]['label'],
#                            pickup_latitude=query_results.iloc[i]['pickup_latitude'],
#                            pickup_longitude=query_results.iloc[i]['pickup_longitude'],
#                            dropoff_longitude=query_results.iloc[i]['dropoff_longitude'],
#                            dropoff_latitude=query_results.iloc[i]['dropoff_longitude']))

# switch to cluster results only!!!
      for i in range(0,labeled_data.shape[0]): #use query_results.shape[0]
          trips.append(dict(index=labeled_data.iloc[i]['label'],
                            pickup_latitude=labeled_data.iloc[i]['pickup_latitude'],
                            pickup_longitude=labeled_data.iloc[i]['pickup_longitude'],
                            dropoff_longitude=labeled_data.iloc[i]['dropoff_longitude'],
                            dropoff_latitude=labeled_data.iloc[i]['dropoff_latitude']))
      return render_template("output.html", trips = trips)

@app.route('/output2')
def blt_output2():
    import datetime as datetime
    import pandas as pd
    #'start_time', 'wait_time', 'current_latitude', 'current_longitude'
    start_time = request.args.get('start_time')
    weekday = int(request.args.get('weekday'))
    start_time_datetime = pd.to_datetime(start_time, format='%H:%M:%S', errors='coerce')
    hour = start_time_datetime.hour
    #create dbnames
    dbnames = ['']*24
    for i in range(0,24):
        if i<10:
            dbnames[i] = 'trips_'+'0'+str(i)
        else:
            dbnames[i] = 'trips_'+str(i)
    #create weekday names
    day_names = ['sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat']


    #all pickups
    file_pickups = dbnames[hour]+'_p_'+day_names[weekday]+'.csv'
    pickups = pd.read_csv(file_pickups, names=['pickup_latitude','pickup_longitude'])
    #all dropoffs
    file_dropoffs = dbnames[hour]+'_d_'+day_names[weekday]+'.csv'
    dropoffs = pd.read_csv(file_dropoffs, names=['dropoff_latitude','dropoff_longitude'])
    #top5 pickups
    file_pickups_top5 = dbnames[hour]+'_top5_pickups_'+day_names[weekday]
    pickups_top5 = pd.read_csv(file_pickups_top5)
    #top5 pickups
    file_dropoffs_top5 = dbnames[hour]+'_top5_dropoffs_'+day_names[weekday]
    dropoffs_top5 = pd.read_csv(file_dropoffs_top5)

    #create variables for all pickups/dropoffs, top 5 p/d
    p = []
    d = []
    p5 = []
    d5 = []
    top = []
    for i in range(0,pickups.shape[0]): #use query_results.shape[0]
        p.append(dict(pickup_latitude=pickups.iloc[i]['pickup_latitude'],
                      pickup_longitude=pickups.iloc[i]['pickup_longitude']))
        d.append(dict(dropoff_latitude=dropoffs.iloc[i]['dropoff_latitude'],
                      dropoff_longitude=dropoffs.iloc[i]['dropoff_longitude']))
    for i in range(0,pickups_top5.shape[0]):
        p5.append(dict(pickup_latitude=pickups_top5.iloc[i]['pickup_latitude'],
                      pickup_longitude=pickups_top5.iloc[i]['pickup_longitude']))
    for i in range(0,dropoffs_top5.shape[0]):
        d5.append(dict(dropoff_latitude=dropoffs_top5.iloc[i]['dropoff_latitude'],
                      dropoff_longitude=dropoffs_top5.iloc[i]['dropoff_longitude']))

    #create variable for top 1 pickup/dropoff pair
    mode_value, mode_count = tf.get_mode_cluster(pickups_top5, col_name = 'cluster')
    mode_pickup_cluster = pickups_top5[pickups_top5['cluster']==mode_value]
    #cluster dropoffs based on mode_pickup_cluster
    clustered_pickups_dropoffs = tf.add_dropoff_cluster_labels_to_col(mode_pickup_cluster)
    mode_value, mode_count = tf.get_mode_cluster(clustered_pickups_dropoffs, col_name = 'dropoff_cluster')
    top_pair = clustered_pickups_dropoffs[clustered_pickups_dropoffs['dropoff_cluster']==mode_value]

    for i in range(0,top_pair.shape[0]):
        top.append(dict(pickup_latitude=top_pair.iloc[i]['pickup_latitude'],
                      pickup_longitude=top_pair.iloc[i]['pickup_longitude'],
                      dropoff_latitude=top_pair.iloc[i]['dropoff_latitude'],
                      dropoff_longitude=top_pair.iloc[i]['dropoff_longitude']))

    #get transit_durations
    gmaps_api = googlemaps.Client(key='AIzaSyCPmpvI8JXVxgz99HXIcssM6c2oioACfQk')
    pickup_lat_long = [top_pair['pickup_latitude'].iloc[0], top_pair['pickup_longitude'].iloc[0]]
    dropoff_lat_long = [top_pair['dropoff_latitude'].iloc[0], top_pair['dropoff_longitude'].iloc[0]]
    driving_transit = tf.get_driving_transit_durations(gmaps_api, pickup_lat_long, dropoff_lat_long, departure_time = datetime.datetime.now())
    driving_time = driving_transit[0]/60.0
    transit_time = driving_transit[1]/60.0

    #get estimates of revenues
    revenue = tf.get_daily_revenues_for_clusters(pickups_top5, dropoffs_top5)
    revenue_top_pair = revenue[2]

    #get weekday and hour
    day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
    weekday_name = day_names[weekday]

    #get number of rides and number of passengers at top pair cluster
    (passenger_count_top_pair, ride_count_top_pair) = tf.count_rides_in_cluster(top_pair)
    return render_template("output2.html",
                            pickups = p, dropoffs = d, pickups_top5 = p5,
                            dropoffs_top5 = d5, top_pair = top,
                            driving_time = driving_time, transit_time = transit_time,
                            revenue_top_pair = revenue_top_pair,
                            hour = hour, weekday_name = weekday_name,
                            passenger_count_top_pair = passenger_count_top_pair,
                            ride_count_top_pair = ride_count_top_pair)
