import streamlit as st
from audio_recorder_streamlit import audio_recorder
from pydub import AudioSegment
import pandas as pd
import io
import json
from transcribe import AudioTranscriber
from diarization import DiarizationHelper
from openai_completion import OpenAIAssistant
from utils import words_per_segment


## Load classes and default variables
diarization = DiarizationHelper()
transcriber = AudioTranscriber()
if "transcription_result" not in st.session_state:
    st.session_state.transcription_result = None

if "diarization_result" not in st.session_state:
    st.session_state.diarization_result = None
# TODO add a better way to store files if user wants to save them
audio_save_path = "./audio_files/audio.wav"

st.title("🦮 Psychology Pal")

option = st.radio("Choose an option:", ("Upload Audio", "Record Audio"))

# Upload audio from the user's computer
if option == "Upload Audio":
    uploaded_file = st.file_uploader(
        "Upload your audio file (.mp3, .wav, .ogg, .m4a)",
        type=["wav", "mp3", "ogg", "m4a"],
    )
    if uploaded_file is not None:
        audio_bytes = AudioSegment.from_file(uploaded_file)
        audio_bytes.export(audio_save_path, format="wav")


# Record audio from the microphone
elif option == "Record Audio":
    # Code to record audio
    audio = audio_recorder(key="my_audio")
    if audio is not None:
        # Convert WAV to MP3 and save
        wav_audio = AudioSegment.from_file(io.BytesIO(audio), format="wav")
        wav_audio.export(audio_save_path, format="mp3")

st.write("\n" * 2)  # Adding two lines of space

# Collapsible section for playing the audio
with st.expander("Check your audio file"):
    if uploaded_file is not None:
        st.audio(audio_save_path)


transcription_button = st.button("Get transcription")
if transcription_button:
    with st.spinner("Transcribing..."):
        st.session_state.transcription_result = transcriber.transcribe(
            audio_save_path, word_timestamps=True
        )
        st.session_state.diarization_result = diarization.run_diarization(
            audio_save_path
        )

    st.success("Transcription complete!")
if (
    st.session_state.diarization_result is not None
    and st.session_state.transcription_result is not None
):
    st.write("## Result")
    final_result = words_per_segment(
        st.session_state.transcription_result, st.session_state.diarization_result
    )

    df = pd.DataFrame.from_dict(final_result, orient="index")
    df.columns = ["speaker", "text", "start", "stop"]

    st.write(df)
    st.write(df[["speaker", "text"]].to_records(index=False))
with st.expander("View breakdown"):
    st.write(st.session_state.transcription_result)
    st.write(st.session_state.diarization_result)


# Send to LLM


@st.cache_data
def get_completion(message: str) -> dict:
    assistant = OpenAIAssistant(model="gpt-3.5-turbo-1106")
    # assistant = OpenAIAssistant(model="gpt-4-1106-preview")
    response = assistant.get_openai_completion(
        system_message="You are a helpful assistant designed to output JSON",
        user_messages=[message],
        assistant_messages=[""],
    )
    return response.choices[0].message.content


template = """Based on the following conversation between a Psycologist and a Patient. Your task it to return the following in JSON format:
        "Summary": "A short  summary of the conversation (3-4 sentences)",
        "Sentiment": "The sentiment of the conversation",
        "Diagnosis": "Your diagnosis of the patient",
        "Highlights": "3,4 bullet points of the main points",
        "Recommendations": "Your recommendations for the patient (yours not the therapists). This should be a recommendation for the therapist on what to do with the patient."
                                  
        <Instructions>
            VERY IMPORTANT: Return the response in the same language as the conversation. If the conversation is in spanish your json should be in spanish.
        </Instructions>
        <Conversation>
            {CONVERSATION}
        </Conversation>"""

call_llm = st.button("Call LLM")
if call_llm:
    conversation = template.format(
        CONVERSATION=df[["speaker", "text"]].to_records(index=False)
    )
    print(conversation)
    response = get_completion(conversation)
    st.write(response)
    response_json = json.loads(response)
    st.write(response_json)
