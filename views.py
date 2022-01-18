from .main import process_comments
import json
from dotenv import load_dotenv
from os import getenv
import requests

load_dotenv()
YOUTUBE_API_SECRET_KEY = getenv("YOUTUBE_API_SECRET_KEY")

def getuploads(channel_id):
    response = requests.get(f"https://youtube.googleapis.com/youtube/v3/channels?part=snippet%2CcontentDetails%2Cstatistics&maxResults=1&id={channel_id}&key={YOUTUBE_API_SECRET_KEY}")
    json_data = json.loads(response.content.decode('utf-8'))
    uploads_playlist_id = json_data['items'][0]['contentDetails']['relatedPlaylists']['uploads']
    uploads_playlist = requests.get(f"https://youtube.googleapis.com/youtube/v3/playlistItems?part=snippet&maxResults=50&playlistId={uploads_playlist_id}&key={YOUTUBE_API_SECRET_KEY}")
    uploads_playlist_json_data = json.loads(uploads_playlist.content.decode('utf-8'))
    return uploads_playlist_json_data

def getcommentsfromvideo(video_id):
    response = requests.get(f"https://youtube.googleapis.com/youtube/v3/commentThreads?part=snippet&maxResults=100&videoId={video_id}&key={YOUTUBE_API_SECRET_KEY}")
    json_data = json.loads(response.content.decode('utf-8'))
    next_page_token =  json_data['nextPageToken'] if 'nextPageToken' in json_data else ""
    all_comments_text = []
    without_replies_comment_counter = json_data['pageInfo']['totalResults']
    print(json_data['pageInfo']['totalResults'])
    with_replies_comment_counter = 0

    while 'nextPageToken' in json_data:
      for item in json_data['items']:
        all_comments_text.append(item['snippet']['topLevelComment']['snippet']['textOriginal'])
        with_replies_comment_counter = with_replies_comment_counter + 1
        with_replies_comment_counter += item['snippet']['totalReplyCount']

      response = requests.get(f"https://youtube.googleapis.com/youtube/v3/commentThreads?part=snippet&maxResults=100&videoId={video_id}&key={YOUTUBE_API_SECRET_KEY}&pageToken={next_page_token}")
      json_data = json.loads(response.content.decode('utf-8'))

      without_replies_comment_counter += json_data['pageInfo']['totalResults']
      print(json_data['pageInfo']['totalResults'])
      if 'nextPageToken' in json_data: next_page_token = json_data['nextPageToken']

    for item in json_data['items']:
        all_comments_text.append(item['snippet']['topLevelComment']['snippet']['textOriginal'])
        with_replies_comment_counter = with_replies_comment_counter + 1
        with_replies_comment_counter += item['snippet']['totalReplyCount']

    return {
        "comments": all_comments_text, 
        "comment_count_without_replies": without_replies_comment_counter,
        "comment_count_with_replies": with_replies_comment_counter,
    }

