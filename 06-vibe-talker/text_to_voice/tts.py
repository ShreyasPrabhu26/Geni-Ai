# To run this code you need to install the following dependencies:
# pip install google-genai pyaudio

import mimetypes
import os
import struct
import io
import wave
from google import genai
from google.genai import types
from dotenv import load_dotenv
import pyaudio

load_dotenv()

# Initialize client once
_client = None

def _get_client():
    """Get or create the Gemini client."""
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    return _client


def _play_audio(audio_data: bytes):
    """Play audio data directly through speakers using pyaudio.
    
    Args:
        audio_data: WAV format audio data as bytes
    """
    audio_io = io.BytesIO(audio_data)
    
    with wave.open(audio_io, 'rb') as wf:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True
        )
        
        chunk_size = 1024
        data = wf.readframes(chunk_size)
        
        while data:
            stream.write(data)
            data = wf.readframes(chunk_size)
        
        stream.stop_stream()
        stream.close()
        p.terminate()


def speak_text(text: str, voice_name: str = "Zephyr"):
    """Convert text to speech and play it through speakers.
    
    Args:
        text: The text to convert to speech
        voice_name: Voice to use (default: "Zephyr"). Options: "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Aoede"
    """
    client = _get_client()
    
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=text)],
        ),
    ]
    
    config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
    )

    audio_chunks = []
    for chunk in client.models.generate_content_stream(
        model="gemini-2.5-pro-preview-tts",
        contents=contents,
        config=config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
            
        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            
            # Convert to WAV format if needed
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            if file_extension is None or file_extension != ".wav":
                data_buffer = _convert_to_wav(inline_data.data, inline_data.mime_type)
            
            audio_chunks.append(data_buffer)
    
    # Play all collected audio chunks
    for audio_data in audio_chunks:
        _play_audio(audio_data)

def _convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters.

    Args:
        audio_data: The raw audio data as a bytes object.
        mime_type: Mime type of the audio data.

    Returns:
        A bytes object representing the WAV file header.
    """
    parameters = _parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size  # 36 bytes for header fields before data chunk size

    # http://soundfile.sapp.org/doc/WaveFormat/

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size (size of audio data)
    )
    return header + audio_data

def _parse_audio_mime_type(mime_type: str) -> dict[str, int | None]:
    """Parses bits per sample and rate from an audio MIME type string.

    Assumes bits per sample is encoded like "L16" and rate as "rate=xxxxx".

    Args:
        mime_type: The audio MIME type string (e.g., "audio/L16;rate=24000").

    Returns:
        A dictionary with "bits_per_sample" and "rate" keys. Values will be
        integers if found, otherwise None.
    """
    bits_per_sample = 16
    rate = 24000

    # Extract rate from parameters
    parts = mime_type.split(";")
    for param in parts: # Skip the main type part
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                # Handle cases like "rate=" with no value or non-integer value
                pass # Keep rate as default
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass # Keep bits_per_sample as default if conversion fails

    return {"bits_per_sample": bits_per_sample, "rate": rate}


if __name__ == "__main__":
    # Test the function
    speak_text("Hello! How are you doing today?")
