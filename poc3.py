import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path
from textblob import TextBlob
import torch
import torchaudio
import torch.nn as nn
from pathlib import Path

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

class TextCNNModel(nn.Module):
    def __init__(self):
        super(TextCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.fc1 = nn.Linear(16*8*8, 3)  # Assuming 3 classes: positive/negative/neutral

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
class AudioCNNModel(nn.Module):
    def __init__(self):
        super(AudioCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.fc1 = nn.Linear(16*8*8, 3)  # Assuming 3 classes: positive/negative/neutral

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
    
cnn_text_model = TextCNNModel() 
cnn_audio_model = AudioCNNModel()

# Streamlit UI for file upload
st.title("Sentiment Analysis of Customer Review")
customer_review = st.text_area("Enter customer review text for sentiment analysis ")

uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg"])

if st.button("Process"):
    def classify_text(text):
        # For simplicity, use TextBlob sentiment analysis here
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            return "Positive"
        elif sentiment < 0:
            return "Negative"
        else:
            return "Neutral"
        
    def classify_audio(audio_path):
        # Placeholder for actual audio loading and processing
        waveform, sample_rate = torchaudio.load(audio_path)
        input_tensor = waveform.unsqueeze(0).unsqueeze(0)  # Dummy input reshaping for CNN
        output = cnn_audio_model(input_tensor)
        sentiment = torch.argmax(output, dim=1).item()
        if sentiment >0:
            return 'Positive'
        elif sentiment <0:
            return 'Negative'
        else:
            return 'Neutral'
    
    if customer_review:
        with st.spinner("Performing the sentiment analysis on customer review text..."):
            review_sentiment = classify_text(customer_review)
            st.write(f"Sentiment of review: {review_sentiment}")
        
    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        with open("uploaded_audio.mp3", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Use genai API to upload and transcribe the file
        audio_file = Path("uploaded_audio.mp3")
        
        option = st.radio("Choose how to analyze the audio:", "Transcribe the audio" ,"Classify audio directly")
        
        if option == "Transcribe the audio":
            myfile = genai.upload_file(audio_file)
            # Generate transcription
            st.spinner("Transcribing audio...")
            result = model.generate_content([myfile, "Describe this audio clip"])
            transcription = result.text
            # Display the result
            st.write(f"Transcription: {transcription}")
            
            with st.spinner("Classifying sentiment based on the transcribed text..."):
                transcribed_sentiment  = classify_text(transcription)
                st.write(f"Sentiment of transcribed text: {transcribed_sentiment}")
            
        elif option == "Classify audio directly":
            # Alternatively, classify the audio directly using the audio CNN
            st.spinner("Classifying sentiment based on audio signal...")
            audio_sentiment = classify_audio(audio_file)
            st.write(f"Sentiment of audio: {audio_sentiment}")
        
