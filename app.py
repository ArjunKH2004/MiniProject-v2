import requests
import time
from colorama import Fore, Back, Style


API_KEY = "AIzaSyAiggkZTjKRGd_3pYRTJIe-08QOjfnHYgI"
video_url1 = input("Enter the YouTube video URL:")
VIDEO_ID = video_url1.split('v=')[1].split('&')[0] if 'v=' in video_url1 else video_url1.split('/')[-1]

video_url = "https://www.googleapis.com/youtube/v3/videos"
video_params = {
    "part": "liveStreamingDetails",
    "id": VIDEO_ID,
    "key": API_KEY
}
response = requests.get(video_url, params=video_params).json()
live_chat_id = response["items"][0]["liveStreamingDetails"]["activeLiveChatId"]

chat_url = "https://www.googleapis.com/youtube/v3/liveChat/messages"
chat_params = {
    "liveChatId": live_chat_id,
    "part": "snippet,authorDetails",
    "key": API_KEY
}

#importing model
import pickle
from tfIdfInheritVectorizer.feature_extraction.vectorizer import TFIDFVectorizer


with open("yt_ai_classifier_model_2.sav", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.sav", "rb") as f:
    tfidf = pickle.load(f)

Pause = False
while(Pause == False): #put Pause = False
  time.sleep(2)
  chat_response = requests.get(chat_url, params=chat_params).json()
  for item in chat_response["items"]:
        author = item["authorDetails"]["displayName"]
        message = item["snippet"]["displayMessage"]

        #classifying text
        new_text = message
        text_tfidf = tfidf.transform([new_text])
        output = model.predict(text_tfidf)

        #coloring text
        match(output):
          case 0:
            print(Fore.RED+author+" : "+message)
          case 1:
            print(Fore.YELLOW+author+" : "+message)
          case 2:
            print(Fore.GREEN+author+" : "+message)
        #when button is pressed put Pause = True
      
