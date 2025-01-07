from google_calendar import *
from deep_face import *
from capture import *
from gemini import *
from ocr import *

class SixthSense():

    def __init__(self):
        self.calendar = Calendar()
        self.facerecognition = FaceRecognition()
        self.capture = Capture()
        self.geminipro = GeminiProAssistant()
        self.geminivisionpro = GeminiVisionProAssistant(genai.GenerativeModel('gemini-pro-vision'))
        self.ocr = OCR()

    def tts(self, text):

        tts = gTTS(text=text, lang='en')
        filename = 'voice.mp3'
        tts.save(filename)
        playsound.playsound(filename)

    def stt(self):
        r = sr.Recognizer()
        with sr.Microphone() as source:
            audio = r.listen(source)
            said = ""

            try:
                said = r.recognize_google(audio)
                print(said)
            except Exception as e:
                print("Exception: " + str(e))

        return said.lower()
    
    def handle_commands(self, text):
        if "create a event" in text or "make a note" in text:
            self.calendar.find(text)
        elif "who is in front of me" in text:
            self.tts(f"{self.facerecognition.get_name(self.facerecognition.process_frame)} is in front of you and he is {self.facerecognition.get_emotion(self.facerecognition.process_frame)}")
        elif "photo" in text:
            imagepath = self.capture.camera()
            self.tts("Photo Captured")
            self.tts("What would you like to do?")
            question = self.stt()
            if 'text' in question:
                self.tts(self.ocr.extract_text_from_image(imagepath))
            else:
                self.tts(self.geminivisionpro.trigger_gemini_vision_pro(imagepath))
        else:
            # Let Gemini handle all general queries like time, weather, etc.
            response = self.geminipro.respond(text)
            self.tts(response)

def main():
    sixthsense = SixthSense()
    WAKE = "hello"

    while True:
        print("Listening")
        text = sixthsense.stt()

        if WAKE in text:
            sixthsense.tts("Yes, How can I help you?")
            text = sixthsense.stt()
            sixthsense.handle_commands(text)

if __name__ == "__main__":
    main()