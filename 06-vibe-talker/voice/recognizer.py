import speech_recognition as sr


def listen_and_recognize():
    """
    Listen to audio input and convert it to text using Google Speech Recognition.

    Returns:
        tuple: (success: bool, result: str)
            - If success is True, result contains the recognized text
            - If success is False, result contains the error message
    """
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        return True, text
    except sr.UnknownValueError:
        return False, "Could not understand the audio."
    except sr.RequestError as e:
        return False, f"Could not request results; {e}"
