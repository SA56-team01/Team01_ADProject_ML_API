import json
import os
import pandas as pd

from flask import Flask, request,jsonify
import pickle
from dotenv import load_dotenv
from requests import get
import ml_model_api
from datetime import datetime


app = Flask(__name__)

load_dotenv()

#index route to test 
@app.route('/')
def index():
    return "Team 1 ML API"

#routing call by Android with parameters
@app.route('/predictTrackAttributes', methods=['GET'])
def predict_track_attributes():

    # data = [
    # {
    #     "id": 1,
    #     "userId": 1,
    #     "latitude": 1.2929946056154933,
    #     "longitude": 103.77659642580656,
    #     "spotifyTrackId": "2zDt2TfQbxiSPjTVJTgbwz",
    #     "timestamp": "2023-08-12 20:52:34"
    # },
    # {
    #     "id": 2,
    #     "userId": 1,
    #     "latitude": 1.2929946056154933,
    #     "longitude": 103.77659642580656,
    #     "spotifyTrackId": "2zDt2TfQbxiSPjTVJTgbwz",
    #     "timestamp": "2023-08-12 20:52:34"
    # }
    # ]
    
    # df_mock = pd.DataFrame(data)

    #get input from android call
    currentlatitude = request.args.get('latitude',type=float)
    print(currentlatitude)
    currentlongitude = request.args.get('longitude', type=float)
    print(currentlongitude)
    userId = request.args.get('userId')
    print(userId)

    currentTime = request.args.get('time')
    #currentTime_str = datetime.strptime(currentTime,"%Y-%m-%d %H:%M:%S")  # type: ignore

    #call backend to get user-history given userid
    userhistoryURL = os.getenv("USER_HISTORY_URL")
    result = get(userhistoryURL + userId)  
    #result = get(userhistoryURL) # type: ignore
    jsonresult = json.loads(result.content)
    print(jsonresult)


    

    userhistory_df = pd.DataFrame([jsonresult])
    #print(userhistory_df)
    


    ml_model_api.SpotifyOAuth()
    
    #get seed tracks
   
    seed_tracks = ml_model_api.get_seed_tracks(userhistory_df,currentTime)

    preprocess_data = ml_model_api.preprocess_data(userhistory_df)
    
    ready_data = ml_model_api.perform_FE(preprocess_data)

    #load the model
    #modelName =  os.getenv("MODEL_NAME")
    #accessModel = pickle.load(open(modelName,'rb'))

    #dataPrediction = accessModel.predict()

    
    preprocess_result = ready_data.to_dict(orient='records')
    return json.dumps(preprocess_result)



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
            