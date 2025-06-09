import boto3
from datetime import datetime

dynamodb = boto3.resource('dynamodb', region_name='ap-south-1')
table = dynamodb.Table('Users_table')

def save_transcription(user_id, filename, model, text):
    table.put_item(
        Item={
            'user_id': user_id,
            'filename': filename,
            'model': model,
            'transcription': text,
            'timestamp': datetime.utcnow().isoformat()
        }
    )
