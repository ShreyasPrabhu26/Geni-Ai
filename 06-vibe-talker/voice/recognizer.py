import speech_recognition as sr


def listen_and_recognize():
    """
    Listen to audio input and convert it to text using Google Speech Recognition.
    Falls back to text input if no microphone is available.

    Returns:
        tuple: (success: bool, result: str)
            - If success is True, result contains the recognized text
            - If success is False, result contains the error message
    """
    r = sr.Recognizer()
    
    try:
        with sr.Microphone() as source:
            print("Listening... (speak now)")
            r.adjust_for_ambient_noise(source, duration=1)
            r.pause_threshold = 3 #in seconds
            audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            return True, text
        except sr.UnknownValueError:
            return False, "Could not understand the audio."
        except sr.RequestError as e:
            return False, f"Could not request results; {e}"
            
    except OSError as e:
        if "No Default Input Device Available" in str(e):
            print("No microphone detected. Please type your message:")
            text = input("> ")
            return True, text
        else:
            return False, f"Audio device error: {e}"
