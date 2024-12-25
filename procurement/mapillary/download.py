import json
import csv
import os
import requests
import sys
with open('./KEYS.json', 'r') as key_file:
        keys = json.load(key_file)
        access_token = keys['mapillary']


def download_image(image_id, access_token, directory):
    header = {'Authorization': f'OAuth {access_token}'}
    url = f'https://graph.mapillary.com/{image_id}?fields=thumb_2048_url'
    response = requests.get(url, headers=header)
    data = response.json()
    image_url = data.get('thumb_2048_url')

    if image_url:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f'{image_id}.jpg')
        with open(file_path, 'wb') as handler:
            image_data = requests.get(image_url, stream=True).content
            handler.write(image_data)
            
            
