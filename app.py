import os
import tempfile
import streamlit as st
from moviepy.editor import VideoFileClip
import whisper
from fuzzywuzzy import fuzz
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer, util

# Set custom temporary directory
os.environ["TEMP"] = "D:\\CustomTempDir"
os.environ["TMP"] = "D:\\CustomTempDir"

# Load semantic similarity model for contextual phrase detection
similarity_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Lightweight model for semantic similarity

# Function to extract audio in segments with timestamps
def extract_audio_segments(video_path, segment_duration=5):
    video = VideoFileClip(video_path)
    segments = []
    
    for start_time in range(0, int(video.duration), segment_duration):
        end_time = min(start_time + segment_duration, int(video.duration))
        audio_segment = video.subclip(start_time, end_time)
        
        # Save each segment as a temporary audio file
        segment_path = f"{os.environ['TEMP']}\\temp_segment_{start_time}-{end_time}.wav"
        audio_segment.audio.write_audiofile(segment_path, fps=16000)
        
        # Store segment information
        segments.append((segment_path, start_time, end_time))
    
    return segments

# Transcribe each audio segment and check for target statement with confidence scores and semantic similarity
def transcribe_with_timestamps(audio_segments, target_statement, selected_language):
    model = whisper.load_model("base")  # Use "base" or larger for better accuracy
    matches = []

    for segment_path, start_time, end_time in audio_segments:
        # Transcribe each segment with the specified language
        result = model.transcribe(segment_path, language=selected_language)
        transcription = result["text"]
        
        # Calculate fuzzy matching score for keyword search
        confidence_score = fuzz.partial_ratio(transcription.lower(), target_statement.lower())
        
        # Calculate semantic similarity score for contextual phrase detection
        embedding1 = similarity_model.encode(transcription, convert_to_tensor=True)
        embedding2 = similarity_model.encode(target_statement, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(embedding1, embedding2).item() * 100  # Convert to percentage

        # Store result if confidence or similarity scores are above threshold
        if confidence_score >= 80 or similarity_score >= 80:
            matches.append((start_time, end_time, transcription, confidence_score, similarity_score))
        
        # Cleanup: Remove the temporary audio segment file
        os.remove(segment_path)
    
    return matches

# Function to play audio for a specific segment
def play_audio_segment(file_path, start_time, end_time):
    audio = AudioSegment.from_file(file_path)
    segment = audio[start_time * 1000:end_time * 1000]  # Convert seconds to milliseconds
    segment_path = f"{os.environ['TEMP']}\\temp_segment_playback.wav"
    segment.export(segment_path, format="wav")
    return segment_path

# Streamlit App UI
st.title("Enhanced Video/Audio Statement Detection Tool")
st.write("Upload a video or audio file, enter the statement you want to search for, and select language.")

# File upload
uploaded_file = st.file_uploader("Upload Video or Audio", type=["mp4", "avi", "mov", "mkv", "wav", "mp3", "flac"])
statement_to_search = st.text_input("Enter the statement to search for")

# Language selection dropdown
language_options = {
    "English": "en",
    "Hausa": "ha",
    "Yoruba": "yo"
}
selected_language = st.selectbox("Select Language", list(language_options.keys()))
language_code = language_options[selected_language]

if uploaded_file and statement_to_search:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name.split('.')[-1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    # Check if the file is a video or audio
    is_video = uploaded_file.name.endswith((".mp4", ".avi", ".mov", ".mkv"))

    # Extract audio segments based on file type
    if is_video:
        with st.spinner("Extracting audio segments from video..."):
            audio_segments = extract_audio_segments(file_path, segment_duration=5)
    else:
        with st.spinner("Extracting audio segments from audio..."):
            audio_segments = extract_audio_segments_from_audio(file_path, segment_duration=5)

    # Transcribe segments and search for the statement with timestamps and confidence scores
    with st.spinner("Transcribing audio and searching for statement..."):
        matches = transcribe_with_timestamps(audio_segments, statement_to_search, language_code)

    # Display results
    if matches:
        for match in matches:
            start_time, end_time, transcription, confidence_score, similarity_score = match
            st.success(f"Statement found between {start_time}s and {end_time}s: {transcription}")
            st.write(f"Confidence Score: {confidence_score}% | Similarity Score: {similarity_score}%")

            # Play audio segment
            segment_audio_path = play_audio_segment(file_path, start_time, end_time)
            st.audio(segment_audio_path)
    else:
        st.error(f"The statement '{statement_to_search}' was not found in the audio/video.")

    # Clean up temporary file
    os.remove(file_path)
