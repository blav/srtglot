from pathlib import Path
import openai
import os
from moviepy import VideoFileClip

from dotenv import load_dotenv
load_dotenv()

# Set your OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

def extract_audio_from_video(video_path, audio_output_path):
    """
    Extract audio from a video file.
    """
    video_clip = VideoFileClip(video_path)
    video_clip.audio.write_audiofile(audio_output_path)
    video_clip.close()

def transcribe_audio_to_srt(audio_path, output_srt_path):
    """
    Transcribe audio using OpenAI's Whisper model and save to an SRT file.
    """
    with open(audio_path, "rb") as audio_file:
        response = openai.Audio.transcribe("whisper-1", audio_file)

    srt_content = convert_whisper_transcription_to_srt(response['text'])
    with open(output_srt_path, "w") as srt_file:
        srt_file.write(srt_content)

def convert_whisper_transcription_to_srt(transcription):
    """
    Convert Whisper transcription to SRT format.
    """
    srt_output = []
    segments = transcription.split('\n')
    for idx, segment in enumerate(segments):
        start_time = idx * 5  # Assuming each segment is 5 seconds apart
        end_time = start_time + 5
        start_time_srt = format_time_for_srt(start_time)
        end_time_srt = format_time_for_srt(end_time)

        srt_output.append(f"{idx + 1}")
        srt_output.append(f"{start_time_srt} --> {end_time_srt}")
        srt_output.append(segment)
        srt_output.append("")

    return "\n".join(srt_output)

def format_time_for_srt(seconds):
    """
    Format time in seconds to SRT time format (hh:mm:ss,ms).
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},000"

def main():
    video_path = "/Users/blav/Downloads/Le bureau des légendes S01.XviD/LBDL.S01E01.avi"
    audio_output_path = "LBDL.S01E01.wav"
    output_srt_path = "LBDL.S01E01.srt"

    print("Extracting audio from video...")
    # extract_audio_from_video(video_path, audio_output_path)
    
    print("Transcribing audio to SRT...")
    transcription = openai.audio.transcriptions.create(
        model="whisper-1",
        file=Path(audio_output_path),
        response_format="srt"
    )

    Path(output_srt_path).write_text(transcription.text)

    print(f"Transcription completed. SRT file saved at: {output_srt_path}")

if __name__ == "__main__":
    main()