import json
import boto3
import os
from botocore.exceptions import ClientError


def show_secrets():
    secrets = get_secrets()
    processed_secrets = json.loads(secrets)
    return processed_secrets
    

def get_secrets():
    secret_name = "Spotify"
    region_name = "ap-southeast-1"

    os.environ["AWS_ACCESS_KEY_ID"] = "AKIATFYKLVQGEAXVEYXA"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "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"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
        
    except ClientError as e:
            # For a list of exceptions thrown, see
            # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        raise e

    # Decrypts secret using the associated KMS key.
    secret = get_secret_value_response['SecretString']
    return secret

def get_spotify_id():
    secrets_show = show_secrets()
    Spotify_ID = secrets_show['Spotify_Client_ID']
    return Spotify_ID

def get_spotify_secrets():
    secrets_show = show_secrets()
    Spotify_Secrets = secrets_show['Spotify_Client_Secret']
    return Spotify_Secrets

