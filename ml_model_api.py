# all necessary imports
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly as plt
import seaborn as sns
import re
import json 
import csv
import os
import datetime 
import time
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pickle
   
### MAIN METHODS ###

'''
Load base model
'''
def load_base_model():
    with open('model.pkl', 'rb') as file:
       model = pickle.load(file)
       return model

'''
Set-up Spotfiy OAuth
'''
def SpotifyOAuth():
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                    client_id="", 
                    client_secret=""))
    
'''
Get 5 seed tracks from user_history
'''
def get_seed_tracks(df, timestamp):
    
    # split date-time to individual parts
    df['ts'] = df['ts'].apply(lambda x: convert_timestamp_to_nearest_min(x))
    df['month'] = pd.DatetimeIndex(df["ts"]).month
    df['weekday'] = pd.DatetimeIndex(df["ts"]).weekday
    df['hour'] = pd.DatetimeIndex(df["ts"]).hour
    df['min'] = pd.DatetimeIndex(df["ts"]).minute
    
    # Extract the month, weekday, and hour from the given timestamp
    given_timestamp =  pd.to_datetime(timestamp)
    given_month = given_timestamp.month()
    given_weekday = given_timestamp.weekday()
    given_hour = given_timestamp.hour()
    given_minute = given_timestamp.minute()

    # Calculate the absolute time differences between the given timestamp and each row in the dataframe
    df['time_diff'] = (abs(df['month'] - given_month) + abs(df['weekday'] - given_weekday) + 
                       abs(df['hour'] - given_hour) + abs(df['min'] - given_minute))

    # Sort the dataframe by the time differences in ascending order
    df_sorted = df.sort_values('time_diff')

    # Get the 5 rows with the closest matching timestamps
    closest_rows = df_sorted.head(5)

    # Remove created columns
    columns = ['ts','month','weekday','hour','min','time_diff']
    df.drop(labels=columns,inplace=True,axis=1)

    return closest_rows['track_id']

'''
Pre-process user listening history data

format: id // userid // latitude // longitude // spotify_track_id // timestamp
'''
def preprocess_data(user_history):

    # convert timestamp to nearest minute
    user_history['timestamp'] = user_history['timestamp'].apply(lambda x: convert_timestamp_to_nearest_min(x))
    
    # Get list of unique songs 
    track_list = user_history['spotify_track_id'].unique().tolist()

    # Get track attributes
    track_features_df = getTrackAttributes(track_list, 100)

    # Drop columns which are meta info about track features
    track_features_df.drop(labels=['type','uri','track_href','analysis_url'], 
                            axis=1, 
                            inplace=True)

    # Rename to match merged df naming convention
    track_features_df.rename(columns={'uri':'spotify_track_id'}, inplace=True)

    # Merge with main dataframe
    merged_df = pd.merge(left = user_history,
                    right = track_features_df,
                    on='spotify_track_id',
                    how='left')
    
    # Drop track_uri once merged
    merged_df.drop(labels='spotify_track_id', inplace=True, axis=1)                    
    
    return merged_df

'''
Do feature engineering on the dataframe before fit into model
'''

def perform_FE(df):
    # perform fe on timestamp
    fe_timestamp(df=df)

    # perform fe on coordinates
    fe_coordinates(df=df)

    # drop original columns
    original_columns = ['timestamp','latitude','longitude','hour']
    df.drop(labels=original_columns, inplace=True, axis=1)

    # perform OHE on FE columns
    df = pd.get_dummies(data=df, columns=['time_fe','location_fe'])

    # reset index
    df.reset_index(level=None, drop=True, inplace=True)

    return df

'''
Make new prediction given pre-processed user listening history

df = user history
model = base model
cur_request = array of current timestamp & location
'''
def predict_song_attributes(df, model, cur_request):

    # Take the features to be predicted on
    
    # fit the data to the base model
    model.partial_fit(df)
    
    # predict clusters for user data
    y_means = model.predict(df)
    
    # append clusters to user history
    df['clusters'] = y_means.astype(int)

    # predict cluster allocation for request
    i = model.predict(cur_request)

    # find average value given the predicted cluster
    cluster_data = df[df['clusters'] == i]
    average_values = np.mean(cluster_data, axis=0)

    # return average value
    return average_values

'''
Form JSON response with min/max values for track attributes with seed tracks
'''
def form_recommendation(seed_tracks, track_attributes):

    # adjust the track attribute target to follow spotify guideline
    
    # update with the min and max value for each attribute

    return 

### SUPPLEMENTARY METHODS ###
'''
Get track attributes for all unique songs from spotify API
'''
def getTrackAttributes(tracks, batch_size):
    
    # Split track list to batches of 100 unique songs due to API constraints
    track_batches = [tracks[i: i+batch_size] for i in range(0, len(tracks), batch_size)]

    # Empty list to store track features
    track_features = [] 

    # Get track features in batches and add to end of list
    for batch in track_batches:
        batch_track_features = sp.audio_features(batch)
        track_features.extend(batch_track_features)

    return pd.DataFrame(track_features)

'''
Convert timestamp to nearest minute
'''
def convert_timestamp_to_nearest_min(timestamp_str):
    timestamp_obj = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    new_format = "%Y-%m-%d %H:%M"
    formatted_timestamp = timestamp_obj.strftime(new_format)
    return formatted_timestamp

'''
OHE timestamp to blocks of 4 hour 
''' 
def fe_timestamp(df):
    
    # Convert timestamp to hour only
    def hour_block(hour):
        result = (hour + 8) % 23
        return result // 4
    
    df['hour'] = pd.DatetimeIndex(df['timestamp']).hour
    df['time_fe'] = df['hour'].apply(hour_block).astype(object)

'''
Assign cartesian coordinate (latitude, longitude) to 
bins in grid cell from Singapore  table
'''
def fe_coordinates(df):

    # Set boundaries for Singapore
    lonmin = 101.333
    lonmax = 104.412
    latmin = 1.083
    latmax = 4.0
    
    # Create grid cells
    grid_size = 0.1
    lat_bins = pd.interval_range(start=latmin,end=latmax, freq=grid_size)
    lon_bins = pd.interval_range(start=lonmin,end=lonmax, freq=grid_size)

    # Method to round the label to nearest 2 decimal
    def round_interval_label(label):
        return round(label.left, 2)
    
    # Round the label 
    rounded_lat_label = (pd.cut(df['lat'], bins=lat_bins, labels=False)
                      .apply(round_interval_label))
    rounded_lon_label = (pd.cut(df['lon'], bins=lon_bins, labels=False)
                      .apply(round_interval_label))

    # assign coordinate to grid cell
    df['location_fe'] = (rounded_lat_label.astype(str) +
                        '_' +  
                        rounded_lon_label.astype(str))
    
    # Method to remove location
    def remove_nana_from_location(df):
        df.drop(df[df['location_fe'] == 'nan_nan'].index, inplace=True)
    
    # Remove location records with null value
    remove_nana_from_location(df)