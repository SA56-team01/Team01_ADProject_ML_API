# cSpell: disable

# all necessary imports
import random
import numpy as np
import pandas as pd
import re
import json 
import csv
import os
import datetime 
import time
import requests
import spotipy
import pickle
import re
from spotipy.oauth2 import SpotifyClientCredentials
from itertools import product
from functools import reduce
from dotenv import load_dotenv
   
# ENVRIONEMNT VARIABLES
load_dotenv()
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

### MAIN METHODS ###

'''
Load base model
'''
def load_base_model():
    with open('model.pkl', 'rb') as file:
       model = pickle.load(file)
       return model
    
'''
Get 5 seed tracks from user_history
'''
def get_seed_tracks(df, timestamp): # timestamp format '2021-11-02 12:32:59'
    
    # split date-time to individual parts
    df['ts'] = df['timestamp'].apply(lambda x: pd.to_datetime(x))
    df['month'] = pd.DatetimeIndex(df["timestamp"]).month
    df['weekday'] = pd.DatetimeIndex(df["timestamp"]).weekday
    df['hour'] = pd.DatetimeIndex(df["timestamp"]).hour
    df['min'] = pd.DatetimeIndex(df["timestamp"]).minute
    
    # Extract the month, weekday, and hour from the given timestamp
    given_timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    given_month = given_timestamp.month
    given_weekday = given_timestamp.weekday()
    given_hour = given_timestamp.hour
    given_minute = given_timestamp.minute

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

    return closest_rows['spotifyTrackId'].tolist()

'''
Pre-process user listening history data

format: latitude // longitude // spotifyTrackId // timestamp
* drop id and userid columns if the intial data has it
'''
def preprocess_data(df):

    # remove all listening history data for location 
    # outside singapore borders
    lon_min = 101.333
    lon_max = 104.412
    lat_min = 1.083
    lat_max = 4.0
    df = (df[(df['longitude'] <= lon_max) & (df['longitude'] >= lon_min) &
              (df['latitude'] <= lat_max) & (df['latitude'] >= lat_min)])
    
    # convert timestamp to nearest minute
    df['timestamp'] = df['timestamp'].apply(lambda x: convert_timestamp_to_nearest_min(x))
    print(df.columns)
    # Get list of unique songs 
    track_list = df['spotifyTrackId'].unique().tolist()

    # Get track attributes as dataframe
    track_features_df = getTrackAttributes(track_list, 100)

    # Merge with main dataframe
    merged_df = pd.merge(left = df,
                    right = track_features_df,
                    on='spotifyTrackId',
                    how='left')
    
    # Drop track_uri once merged
    merged_df.drop(labels='spotifyTrackId', inplace=True, axis=1)                    
    
    return merged_df

'''
Do feature engineering on the dataframe before fit into model
'''

def perform_fe(df):
    # perform fe on timestamp
    df = fe_timestamp(df=df)

    # perform fe on coordinates
    df = fe_coordinates(df=df)

    # drop original columns
    original_columns = ['timestamp','latitude','longitude']
    df.drop(labels=original_columns, inplace=True, axis=1)

    # reset index
    df.reset_index(level=None, drop=True, inplace=True)

    return df

'''
Make new prediction given pre-processed user listening history

df = user history dataframe
cur_request = dataframe of current request 
E.g.
    {'latitude':1.360,
    'longitude':103.754,
    'timestamp':['2023-08-12 20:00:59']}
'''
def predict_song_attributes(df, cur_request):

    # load model
    model = load_base_model()
    
    # Get the location and time features to be used for prediction
    track_attributes = ['danceability', 'energy', 'key', 'loudness', 'mode',
                        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                        'valence', 'tempo', 'time_signature']
    predict_df = df.drop(labels=track_attributes, axis=1)

    # fit the data to the base model
    model.partial_fit(predict_df)
    
    # predict clusters for user data
    y_means = model.predict(predict_df)
    
    # append clusters to user history
    df['clusters'] = y_means.astype(int)

    # get only track attributes and cluster allocations
    df = df.loc[:, ['danceability', 'energy', 'key', 'loudness', 'mode',
                'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                'valence', 'tempo', 'time_signature','clusters']]

    # feature engineering for cur_request
    request_df = perform_fe(cur_request)

    # predict cluster allocation for request
    i = model.predict(request_df)

    # if predicted cluster allocation exists inside user_history
    # get average attributes value given the predicted cluster
    # else, get cluster centers from base model cluster center average
    if i[0] in df['clusters']:
        cluster_data = df[df['clusters'] == i[0]]
        average_values = np.mean(cluster_data, axis=0)
    else:
        base_model_prediction_df = pd.read_csv('base_model_cluster_prediction.csv')
        average_values = base_model_prediction_df.iloc[i[0]]
    
    # check if prediction has any null value
    if average_values.hasnans:
        base_model_prediction_df = pd.read_csv('base_model_cluster_prediction.csv')
        average_values = base_model_prediction_df.iloc[i[0]]
    
    # return predicted attributes
    return average_values

'''
Predict song attributes using base model
for user without listening  history
'''
def predict_song_attributes_without_user_history(cur_request):
    # feature engineering for cur_request
    request_df = perform_fe(cur_request)

    # load model
    model = load_base_model()

    # predict cluster allocation for request
    i = model.predict(request_df)

    # if predicted cluster allocation exists inside user_history
    # get average attributes value given the predicted cluster
    base_model_prediction_df = pd.read_csv('base_model_cluster_prediction.csv')
    average_values = base_model_prediction_df.iloc[i[0]]

    # return predicted attributes
    return average_values

'''
Form JSON response with min/max values for track attributes with seed tracks
'''
def form_recommendation(seed_tracks, track_attributes):

    # drop the 'clusters' if exists
    column_to_drop = 'clusters'
    if column_to_drop in track_attributes.index:
        track_attributes.drop(column_to_drop,inplace=True)
    
    # make empty dict for response
    response_dict = {}

    # add limit and market attributes
    response_dict['limit'] = 20
    response_dict['market'] = 'SG'
    
    #add seed tracks
    # delimiter = '%2C'
    # seed_tracks = reduce(lambda x, y: str(x) + delimiter + str(y), seed_tracks)
    response_dict['seed_tracks'] = seed_tracks
    
    # transform each attribute to min/target/max
    all_columns = track_attributes.index.tolist()
    for col in all_columns:
        # discrete_col = ['key','mode','time_signature'] # for now we do not predict for discrete variables
        # do not predict for 'instrumentalness' as value is generally very low <0.00
        continuous_col = (['danceability', 'energy','acousticness',
                        'liveness','loudness','speechiness', 'tempo', 'valence'])

        if col in continuous_col:
            # extract column value to be modified
            value = track_attributes[col]
            
            if col != 'tempo':
                # response_dict[f'min_{col}'] = f'{round(value % 1 * 0.95, 3)}'
                response_dict[f'target_{col}'] = f'{round(value % 1, 3)}'
                # response_dict[f'max_{col}'] =  f'{round(value % 1 * 1.05, 3)}'
            else:
                # response_dict[f'min_{col}'] = f'{round(value * 0.95)}'
                response_dict[f'target_{col}'] = f'{round(value)}'
                # response_dict[f'max_{col}'] =  f'{round(value * 1.05)}'

    return response_dict

def get_recommended_tracks(predicted_track_attributes):
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=os.getenv("CLIENT_ID"), 
                client_secret=os.getenv("CLIENT_SECRET")))
    
    # get recommendations using seed tracks
    if 'seed_tracks' in predicted_track_attributes:
        response = sp.recommendations(
            seed_tracks= predicted_track_attributes['seed_tracks'],
            country=predicted_track_attributes['market'],
            limit=predicted_track_attributes['limit'],
            target_danceability=predicted_track_attributes['target_danceability'],
            target_energy=predicted_track_attributes['target_energy'],
            target_loudness=predicted_track_attributes['target_loudness'],
            target_speechiness=predicted_track_attributes['target_speechiness'],
            target_acousticness=predicted_track_attributes['target_acousticness'],
            target_liveness=predicted_track_attributes['target_liveness'],
            target_valence=predicted_track_attributes['target_valence'],
            target_tempo=predicted_track_attributes['target_tempo']
        )
    else: # get recommendations using seed genre
        response = sp.recommendations(
            seed_genres= predicted_track_attributes['seed_genres'],
            country=predicted_track_attributes['market'],
            limit=predicted_track_attributes['limit'],
            target_danceability=predicted_track_attributes['target_danceability'],
            target_energy=predicted_track_attributes['target_energy'],
            target_loudness=predicted_track_attributes['target_loudness'],
            target_speechiness=predicted_track_attributes['target_speechiness'],
            target_acousticness=predicted_track_attributes['target_acousticness'],
            target_liveness=predicted_track_attributes['target_liveness'],
            target_valence=predicted_track_attributes['target_valence'],
            target_tempo=predicted_track_attributes['target_tempo']
        )

    # parse for track id in response
    def search_for_tracks(response):
        tracks = response['tracks']
        # empty list
        trackList = []

        for track in tracks:
            if 'spotify:track' in track['uri']:
                trackList.append(track['uri'])
        return trackList
    
    return search_for_tracks(response)

def form_response(pred_track_attr, rec_track_list):
    
    if 'seed_tracks' in pred_track_attr:
        # get seed tracks and append 'spotify:track:' in front of each string
        s='spotify:track:'
        seed_tracks = (s + item for item in pred_track_attr['seed_tracks'])
        
        # append to recommendation
        rec_track_list.extend(seed_tracks)
    
    #return final dict response
    pred_track_attr['playlist_songs'] = rec_track_list
    
    return pred_track_attr

### SUPPLEMENTARY METHODS ###

'''
parse top tracks for seed_tracks
'''
def parse_top_tracks(input_string):    
    pattern = r'(?<=spotify:track:)\w+'
    matches = re.findall(pattern, input_string)

    if len(matches) != 0:
        return matches
    else:
        return 'null'
'''
Get track attributes for all unique songs from spotify API
'''
def getTrackAttributes(tracks, batch_size):
    
    # Split track list to batches of 100 unique songs due to API constraints
    track_batches = [tracks[i: i+batch_size] for i in range(0, len(tracks), batch_size)]

    # Empty list to store track features
    track_features = [] 

    # SpotifyOAuth
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=os.getenv("CLIENT_ID"), 
                client_secret=os.getenv("CLIENT_SECRET")))

    # Get track features in batches and add to end of list
    for batch in track_batches:
        batch_track_features = sp.audio_features(batch)
        track_features.extend(batch_track_features)

    track_features_df = pd.DataFrame(track_features)

    # Drop columns which are meta info about track features
    track_features_df.drop(labels=['type','uri','track_href','analysis_url','duration_ms'], 
                            axis=1, 
                            inplace=True)

    # Rename to match merged df naming convention
    track_features_df.rename(columns={'id':'spotifyTrackId'}, inplace=True)

    return track_features_df

'''
Convert timestamp to nearest minute
'''
def convert_timestamp_to_nearest_min(timestamp_str):
    timestamp_obj = datetime.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    new_format = "%Y-%m-%d %H:%M"
    formatted_timestamp = timestamp_obj.strftime(new_format)
    return formatted_timestamp

'''
OHE timestamp to blocks of 4 hour intervals
''' 
def fe_timestamp(df):
    df['hour'] = pd.DatetimeIndex(df['timestamp']).hour
    
    # Create bins for the hour column in intervals of 4
    hour_bins = pd.interval_range(start=0, end=24, freq=4)
    
    # Label the bins as hour intervals
    bin_labels = [f'{i}' for i in range(1, 7)]
    
    # Assign each hour value to the corresponding bin
    df['hour'] = pd.cut(x=df['hour'], bins=hour_bins, include_lowest=True).map(dict(zip(hour_bins, bin_labels)))
    
    # Perform one-hot encoding on the hour bins
    df = pd.get_dummies(df, columns=['hour'], prefix='time_fe')
    
    return df

'''
Cluster coordinates (latitude, longitude) as cells in grid table
OHE spatials coordinates to grid cells
'''
def fe_coordinates(df):

    # Set boundaries for Singapore
    lon_min = 101.333
    lon_max = 104.412
    lat_min = 1.083
    lat_max = 4.0
    grid_size = 0.3
    
    # Create grid cells
    lat_bins = pd.interval_range(start=lat_min,end=lat_max, freq=grid_size)
    lon_bins = pd.interval_range(start=lon_min,end=lon_max, freq=grid_size)

    # Generate all possible combinations of latitude and longitude interval labels
    possible_labels = list(product(lat_bins, lon_bins))
    
    for label in possible_labels:
        label_str = '{}_{}_{}_{}'.format(round(label[0].left, 2),round(label[0].right, 2), round(label[1].left, 2), round(label[1].right, 2))
        df[label_str] = ((df['latitude'].apply(lambda x: x in label[0])) &
                                (df['longitude'].apply(lambda x: x in label[1])))
    return df

'''
get seed genres for user with no user history data
'''

def get_seed_genres():

    #spotify list of genres
    genres = ["acoustic", "afrobeat", "alt-rock", "alternative", "ambient", "anime",
               "black-metal", "bluegrass", "blues", "bossanova", "brazil", "breakbeat",
                "british", "cantopop", "chicago-house", "children", "chill", "classical",
                "club", "comedy", "country", "dance", "dancehall", "death-metal", "deep-house",
                "detroit-techno", "disco", "disney", "drum-and-bass", "dub", "dubstep", "edm",
                "electro", "electronic", "emo", "folk", "forro", "french", "funk", "garage",
                "german", "gospel", "goth", "grindcore", "groove", "grunge", "guitar",
                "happy", "hard-rock", "hardcore", "hardstyle", "heavy-metal", "hip-hop",
                "holidays", "honky-tonk", "house", "idm", "indian", "indie", "indie-pop",
                "industrial", "iranian", "j-dance", "j-idol", "j-pop", "j-rock", "jazz", 
                "k-pop", "kids", "latin", "latino", "malay", "mandopop", "metal", "metal-misc",
                "metalcore", "minimal-techno", "movies", "mpb", "new-age", "new-release", "opera",
                "pagode", "party", "philippines-opm", "piano", "pop", "pop-film", "post-dubstep",
                "power-pop", "progressive-house", "psych-rock", "punk", "punk-rock", "r-n-b",
                "rainy-day", "reggae", "reggaeton", "road-trip", "rock", "rock-n-roll",
                "rockabilly", "romance", "sad", "salsa", "samba", "sertanejo", "show-tunes",
                "singer-songwriter", "ska", "sleep", "songwriter", "soul", "soundtracks",
                "spanish", "study", "summer", "swedish", 
                "synth-pop", "tango", "techno", "trance", "trip-hop", "turkish", "work-out", "world-music"]
    
    random_genres = random.sample(genres, 5)

    return random_genres