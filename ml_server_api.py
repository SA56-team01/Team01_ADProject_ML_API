import json
import os
import pandas as pd

from flask import Flask, jsonify, request
from dotenv import load_dotenv
from requests import get
import ml_model_api


app = Flask(__name__)

load_dotenv()

#default route with no direct purpose
@app.route('/')
def index():
    return "Team 1 ML API"

#routing call by Android with parameters
@app.route('/predictTrackAttributes', methods=['GET','POST'])
def predict_track_attributes():

    #get input from android call
    userId = request.args.get('userId')
    currentLatitude = request.args.get('latitude',type=float)
    currentLongitude = request.args.get('longitude', type=float)
    currentTimestamp = request.args.get('timestamp')
    top_tracks = request.form['top_tracks']
    
    cur_request = {
        'latitude':currentLatitude,
        'longitude':currentLongitude,
        'timestamp':[currentTimestamp]
    }
    request_df = pd.DataFrame(cur_request)

    #call backend to get user-history given userid
    #userhistoryURL = os.getenv("USER_HISTORY_URL")
    #response_result = get(userhistoryURL + userId)  
    #json_response = json.loads(response_result.content)

    if userId == '1' :
        json_response = [
            {
            "id": 1,
            "userId": 1,
            "latitude": 1.2929946056154933,
            "longitude": 103.77659642580656,
            "spotifyTrackId": "2zDt2TfQbxiSPjTVJTgbwz",
            "timestamp": "2023-08-15 15:35:49"
            },
            {
            "id": 2,
            "userId": 1,
            "latitude": 1.2929946056154933,
            "longitude": 103.77659642580656,
            "spotifyTrackId": "2zDt2TfQbxiSPjTVJTgbwz",
            "timestamp": "2023-08-15 15:35:49"
            }   
        ]
    else :
        json_response = []


    
    #prepare user-history as dataframe from API response
    userhistory_df = pd.DataFrame(json_response)
    
    if len(userhistory_df) != 0: 
        print("Scenario 1")
        userhistory_df = userhistory_df.drop(columns=['id','userId'])
        
        #get seed tracks 
        seed_tracks = ml_model_api.get_seed_tracks(userhistory_df,currentTimestamp)
        
        #preprocess data from user-history database
        preprocess_data = ml_model_api.preprocess_data(userhistory_df)
        
        #perform feature engineering on preprocess-data
        processed_data = ml_model_api.perform_fe(preprocess_data)
        
        #average values prediction from ml model
        average_values = ml_model_api.predict_song_attributes(processed_data, request_df)
        
        #format result to be readable to return to android
        final_prediction = ml_model_api.form_recommendation(seed_tracks,average_values)
        
    else: # if user_history is empty
        #get seed tracks from user top tracks
        print("Scenario 2")
        seed_tracks = ml_model_api.parse_top_tracks(top_tracks)        
        if seed_tracks != 'null':
            average_values = ml_model_api.predict_song_attributes_without_user_history(request_df)
            final_prediction = ml_model_api.form_recommendation(seed_tracks,average_values)
        
        # if no top_tracks or seed tracks, use random genre
        else:
            print("Scenario 3")
            seed_genres = ml_model_api.get_seed_genres()

            average_values = ml_model_api.predict_song_attributes_without_user_history(request_df)

            final_prediction = ml_model_api.form_recommendation(seed_genres,average_values)

            #change the key as we are using seed_genres instead of seed_tracks
            final_prediction['seed_genres'] = final_prediction.pop('seed_tracks')

    # get recommended tracks based on predicted track attributes
    rec_track_list = ml_model_api.get_recommended_tracks(final_prediction)
    
    # form final response
    final_response = ml_model_api.form_response(final_prediction, rec_track_list) 

    # add metadata about playlist creation
    final_response['timestamp'] = currentTimestamp
    final_response['latitude'] = currentLatitude
    final_response['longitude'] = currentLongitude

    return final_response

# starting the server on any host
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
            