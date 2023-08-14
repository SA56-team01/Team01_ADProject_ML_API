import json
import os
import pandas as pd

from flask import Flask, request
from dotenv import load_dotenv
from requests import get
import ml_model_api

import boto3
from botocore.exceptions import ClientError

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
        print("The DataFrame is not empty.")
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
        final_prediction = ml_model_api.form_recommendation_with_seed_tracks(seed_tracks,average_values)
    
        return final_prediction
    
    else: 
        # if user_history is empty
        # use seed_genres instead
        seed_genres = ml_model_api.get_seed_genres()

        average_values = ml_model_api.predict_song_attributes_without_user_history(request_df)

        final_prediction = ml_model_api.form_recommendation_with_seed_tracks(seed_genres,average_values)

        #change the key as we are using seed_genres instead of seed_tracks
        final_prediction['seed_genres'] = final_prediction.pop('seed_tracks')
        
        return final_prediction



#starting the server on any host
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
            