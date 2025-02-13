import requests
import time
import pickle
import streamlit as st
from tfIdfInheritVectorizer.feature_extraction.vectorizer import TFIDFVectorizer

with open("yt_ai_classifier_model_2.sav", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.sav", "rb") as f:
    tfidf = pickle.load(f)

st.title("YouTube Live Chat Classifier")

API_KEY = "AIzaSyDk6Sv0xXOQyqm78sBVZSVHqrRrZHFwoGA"
video_url_input = st.text_input("Enter the YouTube video URL:")

if 'monitoring' not in st.session_state:
    st.session_state.monitoring = False

if st.button("Pause Chat Monitoring"):
    st.session_state.monitoring = False
    st.write("Chat monitoring paused. Click 'Start Chat Monitoring' to resume.")

rainbow_line = """
    <hr style="height: 5px; border: none; background: linear-gradient(to right, 
        red, orange, yellow, green, blue, indigo, violet);">
    """
st.markdown(rainbow_line, unsafe_allow_html=True)

if st.button("Start Chat Monitoring"):

    if video_url_input:
        VIDEO_ID = video_url_input.split('v=')[1].split('&')[0] if 'v=' in video_url_input else video_url_input.split('/')[-1]

        video_url = "https://www.googleapis.com/youtube/v3/videos"
        video_params = {
            "part": "liveStreamingDetails",
            "id": VIDEO_ID,
            "key": API_KEY
        }
        response = requests.get(video_url, params=video_params).json()

        if "items" in response and response["items"]:
            live_chat_id = response["items"][0]["liveStreamingDetails"]["activeLiveChatId"]
            chat_url = "https://www.googleapis.com/youtube/v3/liveChat/messages"
            chat_params = {
                "liveChatId": live_chat_id,
                "part": "snippet,authorDetails",
                "key": API_KEY
            }

            st.write("Monitoring live chat...")
            chat_placeholder = st.empty()
            all_messages = []
            
            st.session_state.monitoring = True

            while st.session_state.monitoring:
                chat_response = requests.get(chat_url, params=chat_params).json()
                if "items" in chat_response:
                    for item in chat_response["items"]:
                        author = item["authorDetails"]["displayName"]
                        message = item["snippet"]["displayMessage"]
                        
                        new_text = message
                        text_tfidf = tfidf.transform([new_text])
                        output = model.predict(text_tfidf)
                        
                        if output[0] == 0:
                            formatted_message = f"<span style='color:white;'>{author} : <span style='color:red;'>{message}</span></span>"
                        elif output[0] == 1:
                            formatted_message = f"<span style='color:white;'>{author} : <span style='color:yellow;'>{message}</span></span>"
                        elif output[0] == 2:
                            formatted_message = f"<span style='color:white;'>{author} : <span style='color:green;'>{message}</span></span>"

                        all_messages.append(formatted_message)
                    chat_placeholder.markdown("<br>".join(reversed(all_messages)), unsafe_allow_html=True)
                time.sleep(2)  
        else:
            st.error("No live chat available for this video.")
    else:
        st.error("Please enter a valid YouTube video URL.")
