
<h1> <center> Hierarchical Clustering and Graphing Example </center> </h1>

The goal of this demo is to demonstrate some of the functionality of clustering on taxi data to determine shuttle routes between popular pickup and dropoff locations. It uses an example of finding pickups that occur between Midnight and 12:01AM across the dataset and identifies the largest pickup cluster and the largest associated dropoff cluster. It then demonstrates how to use some basic notebook tools to plot Google Maps.

Enjoy! 

Jeff Kahn, updated March 16, 2017.


<a id='top'></a>
# Table of Contents

* [Finding the largest pickup cluster](#findpickup)
    * [Plotting the largest pickup cluster](#plotpickup)
* [Finding the largest associated dropoff cluster](#finddropoff)
* [Compute durations for driving and transit](#transtime)
* [Conclusion](#conc)
* [Appendix](#apx)
    * [Notes](#notes)
    * [Watermark](#watermark)
---

<a id='findpickup'></a>
# Finding the largest pickup cluster

The first thing we want to do is identify the largest group of people getting picked up within a small (~200ft) radius at a time of interest. For instance, which people are getting picked up between 00:00:00 and 00:00:05 (the first 5 seconds after midnight)? Of these people getting picked up, where is the largest group of people? The pickup locations of these riders is a collection of points that we call a "cluster".

I used PostgreSQL for my querying.
The first step is to **START POSTGRESQL IN TERMINAL:** `sudo service postgresql start`


```python
#import statements - please modify for your setup
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
import numpy as np
from datetime import datetime
import flask_dir.taxi_functions as tf #REPLACE with your taxi_functions dir
from scipy.stats import mode
import psycopg2
import gmaps
import googlemaps
```

The postgres server engine is then created in advance of the query. Replace the following with your own credentials.


```python
user = 'postgres' #add your username here (same as previous postgreSQL)
host = 'localhost'
password = 'lo1120t4t!'
dbname = 'jeffkahnjr'
db = create_engine('postgresql://%s:%s@%s:5432/%s'%(user,password,host,dbname))
con = None
connect_str = "dbname='%s' user='%s' host='%s' password='%s'"%(dbname,user,host,password)
con = psycopg2.connect(connect_str)
```

Here, we can choose a start time (measured in seconds from midnight) and an end time (in seconds), to do a query. This will identify all rides that fall within the time period for the entire 6 month dataset.


```python
start_time_seconds_from_midnight = 0
waiting_time = 5
max_distance = 0.005
end_time = start_time_seconds_from_midnight+waiting_time
sql_query = """
                  SELECT * FROM trips WHERE time_seconds>={0} AND time_seconds<={1};
            """.format(start_time_seconds_from_midnight,end_time)
query_results = pd.read_sql_query(sql_query,con)
query_results = tf.ungroup_fares(query_results)
```

Now, using the `add_cluster_labels_to_col` function from the taxi functions library (found in `'taxi_functions.py'`), we apply hierarchical clustering on the taxi data's pickup GPS coordinates. People don't want to walk too far for a pickup. We can set a maximum inter-element distance (`max_distance`; in latitude units, 0.005 is just around 300 ft), which will guarantee that clusters are no larger than this distance across.

Just for reference, here is the main clustering function (for pickups):
```python
def get_labeled_pickup_clusters(data, metric = 'ward', max_distance = 0.005, criterion = 'distance'):
    X_lat = data.loc[:,['pickup_latitude']]
    X_long = data.loc[:,['pickup_longitude']]
    X_pickup_location = np.column_stack((X_lat, X_long))
    Z_linkages = linkage(X_pickup_location, metric)
    data_clustered = fcluster(Z_linkages, max_distance, criterion)
    return data_clustered
```


```python
labeled_data = tf.add_cluster_labels_to_col(data = query_results, max_distance = max_distance)
mode_value, mode_count = tf.get_mode_cluster(labeled_data)
cluster_one_locations = labeled_data[labeled_data['cluster']==mode_value]
```

And lastly, we apply the `mode` function to identify the largest pickup cluster by number of passengers. Since we've "ungrouped" the rides (creating one entry per passenger) this is actually the largest cluster by passenger count.

Here's the mode function, which is pretty simple:
```python
def get_mode_cluster(data, col_name = 'cluster'):
    mode_result = mode(data[col_name])
    mode_value = mode_result[0][0]
    mode_count = mode_result[1][0]
    return mode_value, mode_count
```


```python
mode_value, mode_count = tf.get_mode_cluster(cluster_one_locations, col_name = 'cluster')
cluster_one_locations = cluster_one_locations[cluster_one_locations['cluster']==mode_value]
cluster_one_loc_ungrouped = tf.ungroup_fares(cluster_one_locations)
```

<a id='plotpickup'></a>
## Plotting the largest pickup cluster

I really like the `gmaps` package for Jupyter Notebooks, which can be found [here](https://github.com/pbugnion/gmaps). You'll need a Google Maps API key, which can be obtained by following [this process](https://developers.google.com/maps/documentation/javascript/get-api-key). It's good for creating pretty plots with the Google Maps API and for creating basic heatmaps of data. There are more advanced tools out there, but this is simple and effective. 

Here we import and configure the API key. Add your own credentials below.


```python
import gmaps
import gmaps.datasets
gmaps.configure(api_key="AIzaSyCPmpvI8JXVxgz99HXIcssM6c2oioACfQk") # Insert your own Google Maps API Key
```


```python
loc = np.column_stack((cluster_one_locations.iloc[:,7], cluster_one_locations.iloc[:,6]))
m1 = gmaps.Map()
heatmap_layer = gmaps.heatmap_layer(loc, dissipating = True)
m1.add_layer(heatmap_layer)
#marker_layer = gmaps.marker_layer(loc) #uncomment to add markers of each pickup
#m1.add_layer(marker_layer) # ''
m1
```

[&lt;top&gt;](#top)

<a id='finddropoff'></a>
# Finding the largest associated dropoff cluster

We've seen the largest pickup cluster above, but these people aren't all going to the same places. Let's take a look at where the people from the largest pickup cluster are going. See the map plot below.


```python
loc2 = np.column_stack((cluster_one_locations.iloc[:,11], cluster_one_locations.iloc[:,10]))
m2 = gmaps.Map()
heatmap_layer2 = gmaps.heatmap_layer(loc2, dissipating = True)
m2.add_layer(heatmap_layer2)
#marker_layer = gmaps.marker_layer(loc)
#m1.add_layer(marker_layer)
m2
```

From the map above, we can see travel within Manhattan, Brooklyn, and Queens. So let's go ahead and cluster the data, but now by Dropoff GPS coordinates, rather than Pickup GPS coordinates. This way, we can follow the same process and obtain the largest dropoff cluster **associated** with this pickup cluster.


```python
clustered_data = tf.add_dropoff_cluster_labels_to_col(cluster_one_locations)
mode_value, mode_count = tf.get_mode_cluster(clustered_data, col_name = 'dropoff_cluster')
dropoff_cluster_loc = clustered_data[clustered_data['dropoff_cluster']==mode_value]
dropoff_cluster_loc = np.column_stack((dropoff_cluster_loc.iloc[:,11],dropoff_cluster_loc.iloc[:,10]))
```


```python
m3 = gmaps.Map()
heatmap_layer_2 = gmaps.heatmap_layer(dropoff_cluster_loc, dissipating = True)
m3.add_layer(heatmap_layer_2)
m3
```

And just for a final step, since we have the Google Maps API up, let's map a route from the pickup cluster to the dropoff cluster. This can be done with a simple call to `gmaps.directions_layer(pickup_latlng,dropoff_latlng)`


```python
pickup_to_dropoff_dir_layer = gmaps.directions_layer(loc[0,:], dropoff_cluster_loc[0,:])
```


```python
m3.add_layer(pickup_to_dropoff_dir_layer)
m3.add_layer(heatmap_layer2)
m3
```


```python
compute_all_ride_costs(clustered_data).head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>vendor_id</th>
      <th>tpep_pickup_datetime</th>
      <th>tpep_dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_distance</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>ratecodeid</th>
      <th>store_and_fwd_flag</th>
      <th>...</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>improvement_surcharge</th>
      <th>total_amount</th>
      <th>time</th>
      <th>time_seconds</th>
      <th>weekday</th>
      <th>cluster</th>
      <th>dropoff_cluster</th>
      <th>individual_ride_cost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>257</th>
      <td>5178031</td>
      <td>2</td>
      <td>2016-02-17 00:00:02</td>
      <td>2016-02-17 00:15:55</td>
      <td>1</td>
      <td>7.43</td>
      <td>-73.873070</td>
      <td>40.774109</td>
      <td>1</td>
      <td>False</td>
      <td>...</td>
      <td>5.00</td>
      <td>5.54</td>
      <td>0.3</td>
      <td>34.34</td>
      <td>00:00:02</td>
      <td>2</td>
      <td>2</td>
      <td>683</td>
      <td>3</td>
      <td>34.34</td>
    </tr>
    <tr>
      <th>325</th>
      <td>399915</td>
      <td>1</td>
      <td>2016-01-02 00:00:03</td>
      <td>2016-01-02 00:09:10</td>
      <td>1</td>
      <td>1.90</td>
      <td>-73.873024</td>
      <td>40.774101</td>
      <td>1</td>
      <td>False</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>10.30</td>
      <td>00:00:03</td>
      <td>3</td>
      <td>5</td>
      <td>683</td>
      <td>41</td>
      <td>10.30</td>
    </tr>
    <tr>
      <th>355</th>
      <td>951098</td>
      <td>1</td>
      <td>2016-01-04 00:00:04</td>
      <td>2016-01-04 00:14:44</td>
      <td>1</td>
      <td>11.80</td>
      <td>-73.873108</td>
      <td>40.774132</td>
      <td>1</td>
      <td>False</td>
      <td>...</td>
      <td>6.00</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>39.30</td>
      <td>00:00:04</td>
      <td>4</td>
      <td>0</td>
      <td>683</td>
      <td>46</td>
      <td>39.30</td>
    </tr>
    <tr>
      <th>366</th>
      <td>1101230</td>
      <td>1</td>
      <td>2016-01-05 00:00:04</td>
      <td>2016-01-05 00:20:53</td>
      <td>1</td>
      <td>9.70</td>
      <td>-73.872986</td>
      <td>40.774082</td>
      <td>1</td>
      <td>False</td>
      <td>...</td>
      <td>6.06</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>36.36</td>
      <td>00:00:04</td>
      <td>4</td>
      <td>1</td>
      <td>683</td>
      <td>35</td>
      <td>36.36</td>
    </tr>
    <tr>
      <th>549</th>
      <td>3958101</td>
      <td>2</td>
      <td>2016-01-11 00:00:04</td>
      <td>2016-01-11 00:20:25</td>
      <td>1</td>
      <td>12.05</td>
      <td>-73.872803</td>
      <td>40.774288</td>
      <td>1</td>
      <td>False</td>
      <td>...</td>
      <td>8.82</td>
      <td>0.00</td>
      <td>0.3</td>
      <td>44.12</td>
      <td>00:00:04</td>
      <td>4</td>
      <td>0</td>
      <td>683</td>
      <td>23</td>
      <td>44.12</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



[&lt;top&gt;](#top)

<a id='transtime'></a>
# Compute durations for driving and transit

Now that we have the pickup-dropoff shuttle route, we can look at the difference between public transit and driving times. This may be of interest from a market research perspective, where if we offer a ride sharing service, can we beat the public transit times. If we can't beat it, how can we price competitively? I don't attempt to answer these questions here, but another data science analysis could.

First, let's snag the pickup and dropoff points for the shuttle.


```python
loc = [clustered_data['pickup_latitude'].iloc[0], clustered_data['pickup_longitude'].iloc[0]]
dropoff_loc = [clustered_data['dropoff_latitude'].iloc[0], clustered_data['dropoff_longitude'].iloc[0]]
print(loc,dropoff_loc)
```

    [40.7741088867188, -73.873069763183594] [40.796089172363303, -73.949699401855497]


We're going to use the Google Maps API to compute estimates of our public transit versus driving times from pickup to dropoff locations. Now, we'll need the [official Python `googlemaps` package](https://github.com/googlemaps/google-maps-services-python), in order to call the directions service. Again, substitute your same API key below. We'll also use the Python package `datetime`.


```python
import googlemaps
from datetime import datetime
gmaps_api = googlemaps.Client(key='AIzaSyCPmpvI8JXVxgz99HXIcssM6c2oioACfQk')
```

Now, we can compute the durations for public transit and driving. And rather than import them, I show the functions here, which will return the estimated time (in seconds), between public transit and driving times. Here, I'm just using the current time (`datetime.now()`), but you could substitute whatever time is relevant to the service you'd like to offer.


```python
def get_driving_transit_durations(gmaps_client, pickup_lat_long, dropoff_lat_long, departure_time = datetime.now()):
    driving_result = get_directions_result(gmaps_client, pickup_lat_long, dropoff_lat_long, mode = 'driving', departure_time = departure_time)
    transit_result = get_directions_result(gmaps_client, pickup_lat_long, dropoff_lat_long, mode = 'transit', departure_time = departure_time)
    return [get_duration_from_dir_result(driving_result), get_duration_from_dir_result(transit_result)]
def get_directions_result(gmaps_client, pickup_lat_long, dropoff_lat_long, mode = 'driving', departure_time = datetime.now()):
    return gmaps_client.directions(pickup_lat_long, dropoff_lat_long, mode = mode, departure_time = departure_time)
now = datetime.now()
def get_duration_from_dir_result(directions_result):
    return directions_result[0].get('legs')[0].get('duration').get('value')

out = get_driving_transit_durations(gmaps_api, loc, dropoff_loc, departure_time = datetime.now())
print('Public transit (s):',out[1],'\nShuttle transit (s):',out[0])
```

    Public transit (s): 3790 
    Shuttle transit (s): 1226


[&lt;top&gt;](#top)

<a id='conc'></a>
# Conclusion
That's it for now, but hopefully you have a sense of how to use these functions to identify large pickup clusters, identify their largest associated dropoff clusters, and how to use some of the Google Maps API tools in a Jupyter notebook. You can do some interesting things with the results, as I briefly demonstrated with the public transit versus driving time comparison. Feel free to contact me with any questions, comments, or ideas for future work. I'll be developing this code further as I learn more about public transit and potential applications for the rich NYC TLC Taxi data.

[&lt;top&gt;](#top)

<a id='apx'></a>
# Appendix
<a id='notes'></a>
## Notes
The code created in this package can be applied on the [NYC Taxi and Limousine Commission Trip Record Data](http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml). For this example, and in this training set, data from Yellow Cabs during January - June 2016 were used.

Please note that the functions were created for "Yellow" cab data and the headers DO differ between datasets (Green and FHV).

[&lt;top&gt;](#top)

<a id='watermark'></a>
## Watermark
The following lists the versions of the packages I am using in this notebook. To download [watermark](https://github.com/rasbt/watermark) for iPython/Jupyter notebooks type  ```pip install watermark``` in the command line. 


```python
% load_ext watermark
% watermark -a 'Jeff Kahn' -d -t -v -m -p numpy,scipy,pandas,googlemaps,gmaps,matplotlib
```

    The watermark extension is already loaded. To reload it, use:
      %reload_ext watermark
    Jeff Kahn 2017-03-16 11:38:31 
    
    CPython 3.5.2
    IPython 5.1.0
    
    numpy 1.12.0
    scipy 0.18.1
    pandas 0.19.2
    googlemaps 2.4.5
    gmaps 0.4.0
    matplotlib 2.0.0
    
    compiler   : GCC 4.4.7 20120313 (Red Hat 4.4.7-1)
    system     : Linux
    release    : 4.4.0-21-generic
    machine    : x86_64
    processor  : x86_64
    CPU cores  : 4
    interpreter: 64bit


[&lt;top&gt;](#top)
