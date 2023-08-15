import json
import os
import pandas as pd

from flask import Flask, request
from dotenv import load_dotenv
from requests import get
import ml_model_api
import aws_credentials


import boto3
from botocore.exceptions import ClientError

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

app = Flask(__name__)

load_dotenv()




#default route with no direct purpose
@app.route('/')
def index():
    return "Team 1 ML API"

#routing call by Android with parameters
@app.route('/predictTrackAttributes', methods=['GET'])
def predict_track_attributes():

    #get input from android call
    userId = request.args.get('userId')
    currentlatitude = request.args.get('latitude',type=float)
    currentlongitude = request.args.get('longitude', type=float)
    currentTimestamp = request.args.get('timestamp')
    
    cur_request = {
        'latitude':currentlatitude,
        'longitude':currentlongitude,
        'timestamp':[currentTimestamp]
    }
    request_df = pd.DataFrame(cur_request)

    #call backend to get user-history given userid
    userhistoryURL = os.getenv("USER_HISTORY_URL")
    response_result = get(userhistoryURL + userId)  
    json_response = json.loads(response_result.content)
    
    #prepare user-history as dataframe from API response
    userhistory_df = pd.DataFrame(json_response)
    
    
    if len(userhistory_df) != 0: 
        # print("The DataFrame is not empty.")
        userhistory_df = userhistory_df.drop(columns=['id','userId'])
        print("1")
        #get seed tracks 
        seed_tracks = ml_model_api.get_seed_tracks(userhistory_df,currentTimestamp)
        print("1")
        #preprocess data from user-history database
        preprocess_data = ml_model_api.preprocess_data(userhistory_df)
        print("1")
        #perform feature engineering on preprocess-data
        processed_data = ml_model_api.perform_fe(preprocess_data)
        print("1")
        #average values prediction from ml model
        average_values = ml_model_api.predict_song_attributes(processed_data, request_df)
        print("1")
        #format result to be readable to return to android
        final_prediction = ml_model_api.form_recommendation_with_seed_tracks(seed_tracks,average_values)
        dict = final_prediction


        #comment the code below if using AWS
        # sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        #         client_id=os.getenv("CLIENT_ID"), 
        #         client_secret=os.getenv("CLIENT_SECRET")))
        

        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=aws_credentials.get_spotify_id(), 
                client_secret=aws_credentials.get_spotify_secrets()))
        
        recommendations = sp.recommendations(seed_tracks=seed_tracks,limit=dict['limit'],country=dict['market'],target_acousticness=dict['target_acousticness'],
                                                  target_danceability=dict['target_danceability'],target_energy=dict['target_energy'],
                                                  target_liveness=dict['target_liveness'],target_loudness=dict['target_loudness'],
                                                  target_speechiness=dict['target_speechiness'],target_tempo=dict['target_tempo'],target_valence=dict['target_valence'])

        return recommendations
        final_prediction = ml_model_api.form_recommendation(seed_tracks,average_values)
    
    else: 
        # if user_history is empty
        # use seed_genres instead
        seed_genres = ml_model_api.get_seed_genres()

        average_values = ml_model_api.predict_song_attributes_without_user_history(request_df)

        final_prediction = ml_model_api.form_recommendation(seed_genres,average_values)

        #change the key as we are using seed_genres instead of seed_tracks
        final_prediction['seed_genres'] = final_prediction.pop('seed_tracks')
        dict = final_prediction

        #comment the code below if using AWS
        # sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        #         client_id=os.getenv("CLIENT_ID"), 
        #         client_secret=os.getenv("CLIENT_SECRET")))

        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
                client_id=aws_credentials.get_spotify_id(), 
                client_secret=aws_credentials.get_spotify_secrets()))
        
        recommendations = sp.recommendations(seed_genres=seed_genres,limit=dict['limit'],country=dict['market'],target_acousticness=dict['target_acousticness'],
                                                  target_danceability=dict['target_danceability'],target_energy=dict['target_energy'],
                                                  target_liveness=dict['target_liveness'],target_loudness=dict['target_loudness'],
                                                  target_speechiness=dict['target_speechiness'],target_tempo=dict['target_tempo'],target_valence=dict['target_valence'])
        return recommendations

    return final_response

#starting the server on any host
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5050)
            