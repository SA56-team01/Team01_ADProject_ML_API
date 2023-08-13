import json
import os
import pandas as pd

from flask import Flask, request,jsonify
from dotenv import load_dotenv
from requests import get
import ml_model_api



app = Flask(__name__)

load_dotenv()

#index route to test 
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
    currentTime = request.args.get('time')
    
    cur_request = {
        'latitude':currentlatitude,
        'longitude':currentlongitude,
        'timestamp':[currentTime]
    }
    request_df = pd.DataFrame(cur_request)

    #call backend to get user-history given userid
    userhistoryURL = os.getenv("USER_HISTORY_URL")
    response_result = get(userhistoryURL + userId)  
    json_response = json.loads(response_result.content)
    
    #prepare user-history as dataframe from API response
    userhistory_df = pd.DataFrame(json_response)
    
    if len(userhistory_df) != 0: 
        print("The DataFrame is not empty.")
        userhistory_df = userhistory_df.drop(columns=['id','userId'])

    #get seed tracks / get seed genres
    if len(userhistory_df) != 0: 
        seed_tracks = ml_model_api.get_seed_tracks(userhistory_df,currentTime)
    else:
        seed_genres = ml_model_api.get_seed_genres()

    
    #preprocess data from user-history database
    if len(userhistory_df) != 0: 
        preprocess_data = ml_model_api.preprocess_data(userhistory_df)

    #perform feature engineering on preprocess-data
    processed_data = ml_model_api.perform_fe(preprocess_data)
    
    #average values prediction from ml model
    average_values = ml_model_api.predict_song_attributes(processed_data, request_df)

    #format result to be readable to return to android
    final_prediction = ml_model_api.form_recommendation_with_seed_tracks(seed_tracks,average_values)
    
    
    return final_prediction



@app.route('/predicttest', methods=['GET'])
def test_integration():

    seed_tracks = '0c6xIDDpzE81m2q797ord+0c6xIDDpzE81m2q797ordA' 
    min_acousticness = 0
    max_acousticness = 1
    target_acousticness = 0.5
    min_danceability = 0 
    max_danceability = 1
    target_danceability = 0.5
    min_duration_ms = 140
    max_duration_ms =  150
    target_duration_ms = 145
    min_energy = 0
    max_energy = 1
    target_energy = 0.5
    min_instrumentalness = 0
    max_instrumentalness = 1
    target_instrumentalness = 0.5
    min_key =0
    max_key =11
    target_key = 6
    min_liveness = 0
    max_liveness =  1
    target_liveness = 0.5
    min_loudness = -60
    max_loudness = 0
    target_loudness = -30.1
    min_mode = 0
    max_mode = 1
    target_mode = 0.5
    min_popularity = 0
    max_popularity = 100
    target_popularity = 50
    min_speechiness =  0
    max_speechiness =  1
    target_speechiness =  0.33
    min_tempo = 105
    max_tempo = 120
    target_tempo = 110
    min_time_signature = 3
    max_time_signature = 7
    target_time_signature = 5
    min_valence = 0 
    max_valence = 0.5
    target_valence = 1

    filejson = {"seeds_tracks":seed_tracks,
                "min_acousticness" : min_acousticness,
                "max_acousticness" : max_acousticness,
                "target_acousticness" : target_acousticness,
                "min_danceability" : min_danceability,
                "max_danceability" : max_danceability,
                "target_danceability": target_danceability,
                "min_duration_ms" : min_duration_ms,
                "max_duration_ms": max_duration_ms,
                "target_duration_ms" : target_duration_ms,
                "min_energy" : min_energy,
                "max_energy" : max_energy,
                "target_energy" : target_energy,
                "min_instrumentalness" : min_instrumentalness,
                "max_instrumentalness" : max_instrumentalness,
                "target_instrumentalness" : target_instrumentalness,
                "min_key" : min_key,
                "max_key" : max_key,
                "target_key" : target_key,
                "min_liveness" : min_liveness,
                "max_liveness" : max_liveness,
                "target_liveness" : target_liveness,
                "min_loudness" : min_loudness,
                "max_loudness" : max_loudness,
                "target_loudness" : target_loudness,
                "min_mode" : min_mode,
                "max_mode" : max_mode,
                "target_mode" : target_mode,
                "min_popularity" : min_popularity,
                "max_popularity" : max_popularity,
                "target_popularity" : target_popularity,
                "min_speechiness" : min_speechiness,
                "max_speechiness" : max_speechiness,
                "target_speechiness" : target_speechiness,
                "min_tempo" : min_tempo,
                "max_tempo" : max_tempo,
                "target_tempo" : target_tempo,
                "min_time_signature" : min_time_signature,
                "max_time_signature" : max_time_signature,
                "target_time_signature" : target_time_signature,
                "min_valence" : min_valence, 
                "max_valence" : max_valence,
                "target_valence" : target_valence}


    return jsonify(filejson)

#starting the server on any host
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
            