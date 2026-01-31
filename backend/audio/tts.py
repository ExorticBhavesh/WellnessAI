"""
Text-to-Speech Module for Health Chatbot
Converts chatbot responses to speech audio
"""

import os
import platform
import subprocess
from pathlib import Path


class TextToSpeech:
    def __init__(self, output_dir='audio_output'):
        self.output_dir = output_dir
        self.engine = None
        self._setup_output_dir()
        self._initialize_engine()
    
    def _setup_output_dir(self):
        """Create output directory for audio files"""
        Path(self.output_dir).mkdir(exist_ok=True)
        print(f"✓ Audio output directory: {self.output_dir}")
    
    def _initialize_engine(self):
        """Initialize TTS engine based on available libraries"""
        try:
            # Try pyttsx3 (cross-platform, offline)
            import importlib
            pyttsx3 = importlib.import_module('pyttsx3')
            self.engine = pyttsx3.init()
            self.engine_type = 'pyttsx3'
            
            # Configure voice properties
            self.engine.setProperty('rate', 150)  # Speed
            self.engine.setProperty('volume', 0.9)  # Volume
            
            # Try to set a pleasant voice
            voices = self.engine.getProperty('voices')
            if voices:
                # Prefer female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower():
                        self.engine.setProperty('voice', voice.id)
                        break
            
            print(f"✓ TTS Engine initialized: pyttsx3")
            
        except ImportError:
            print("⚠️  pyttsx3 not available. Install with: pip install pyttsx3")
            self.engine_type = 'gtts'
            self._try_gtts()
    
    def _try_gtts(self):
        """Try Google Text-to-Speech as fallback"""
        try:
            import importlib
            # Lazy import to avoid static analyzer errors if gtts is not installed
            importlib.import_module('gtts')
            self.engine_type = 'gtts'
            print("✓ TTS Engine initialized: gTTS (Google)")
        except ImportError:
            print("⚠️  gTTS not available. Install with: pip install gtts")
            self.engine_type = None
            print("❌ No TTS engine available. Text-only mode.")
    
    def speak(self, text, save_to_file=None):
        """
        Convert text to speech
        
        Args:
            text (str): Text to convert to speech
            save_to_file (str): Optional filename to save audio
        """
        if not text:
            print("⚠️  No text provided for TTS")
            return False
        
        # Clean text for better speech
        clean_text = self._clean_text(text)
        
        if self.engine_type == 'pyttsx3':
            return self._speak_pyttsx3(clean_text, save_to_file)
        elif self.engine_type == 'gtts':
            return self._speak_gtts(clean_text, save_to_file)
        else:
            print(f"Text: {text}")
            return False
    
    def _clean_text(self, text):
        """Clean text for better speech output"""
        # Remove emojis and special characters
        import re
        
        # Remove emoji
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub('', text)
        
        # Remove excessive formatting
        text = re.sub(r'[=\-_*]{3,}', '', text)
        text = re.sub(r'[•✓✗❌]', '', text)
        
        # Clean up whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _speak_pyttsx3(self, text, save_to_file=None):
        """Speak using pyttsx3 (creates a fresh engine each call to avoid driver issues)"""
        try:
            import importlib
            pyttsx3 = importlib.import_module('pyttsx3')

            # Create fresh engine every time (helps avoid backend issues)
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)

            # Prefer a female voice if available
            try:
                voices = engine.getProperty('voices')
                if voices:
                    for voice in voices:
                        if 'female' in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
            except Exception:
                # If voice selection fails, continue with defaults
                pass

            if save_to_file:
                filepath = os.path.join(self.output_dir, save_to_file)
                # Save to file using pyttsx3's save_to_file and flush the engine
                engine.save_to_file(text, filepath)
                engine.runAndWait()
                print(f"✓ Audio saved to: {filepath}")
            else:
                # Speak directly
                engine.say(text)
                engine.runAndWait()

            return True

        except Exception as e:
            print(f"❌ pyttsx3 TTS Error: {e}")
            try:
                engine.stop()
            except Exception:
                pass
            return False
    def _speak_gtts(self, text, save_to_file=None):
        """Speak using Google TTS (lazy-import to avoid analyzer errors if gtts is not installed)"""
        try:
            import importlib
            try:
                gtts_mod = importlib.import_module('gtts')
                gTTS = getattr(gtts_mod, 'gTTS')
            except Exception:
                print("⚠️  gTTS not available. Install with: pip install gtts")
                return False

            # Create audio
            tts = gTTS(text=text, lang='en', slow=False)
            
            if save_to_file:
                filepath = os.path.join(self.output_dir, save_to_file)
            else:
                filepath = os.path.join(self.output_dir, 'temp_audio.mp3')
            
            # Save audio file
            tts.save(filepath)
            print(f"✓ Audio saved to: {filepath}")
            
            # Play audio if not just saving
            if not save_to_file:
                self._play_audio(filepath)
            
            return True

        except Exception as e:
            print(f"❌ TTS Error: {e}")
            return False
            print(f"❌ TTS Error: {e}")
            return False
    
    def _play_audio(self, filepath):
        """Play audio file based on OS"""
        try:
            system = platform.system()
            
            if system == 'Darwin':  # macOS
                subprocess.call(['afplay', filepath])
            elif system == 'Linux':
                subprocess.call(['mpg123', filepath])
            elif system == 'Windows':
                os.startfile(filepath)
            else:
                print(f"⚠️  Audio saved but cannot auto-play on {system}")
                
        except Exception as e:
            print(f"⚠️  Could not play audio: {e}")
    
    def speak_chatbot_response(self, response, save=False):
        """
        Speak chatbot response with optional save
        
        Args:
            response (str): Chatbot response text
            save (bool): Whether to save audio file
        """
        # Split long responses into chunks
        chunks = self._split_response(response)
        
        for i, chunk in enumerate(chunks):
            filename = f"response_{i+1}.mp3" if save else None
            self.speak(chunk, filename)
    
    def _split_response(self, text, max_chars=500):
        """Split long text into speech-friendly chunks"""
        # Split by double newlines (paragraphs)
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) < max_chars:
                current_chunk += para + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def read_health_info(self, topic):
        """Read predefined health information aloud"""
        health_topics = {
            'diabetes': """
                Diabetes is a chronic condition affecting blood sugar regulation.
                Type 1 diabetes means your body doesn't produce insulin.
                Type 2 diabetes is more common and means your body doesn't use insulin properly.
                Key prevention steps include maintaining a healthy weight, exercising regularly,
                and eating a balanced diet.
            """,
            'heart': """
                Heart disease affects the heart and blood vessels.
                Common risk factors include high blood pressure, high cholesterol, smoking,
                diabetes, and obesity.
                You can prevent heart disease through regular exercise, healthy eating,
                not smoking, and managing stress.
            """,
            'exercise': """
                Regular exercise is crucial for good health.
                Adults should aim for at least 150 minutes of moderate exercise per week.
                Exercise strengthens your heart and lungs, helps maintain a healthy weight,
                and reduces disease risk.
            """
        }
        
        if topic.lower() in health_topics:
            self.speak(health_topics[topic.lower()])
            return True
        else:
            print(f"⚠️  Topic '{topic}' not found")
            return False


class AudioChatbot:
    """Chatbot with audio output capability"""
    
    def __init__(self):
        self.tts = TextToSpeech()
        
    def greet_with_audio(self):
        """Greet user with audio"""
        greeting = """
            Welcome to the Health Assessment Chatbot!
            I can help assess your health status and provide information.
            Would you like to start the health assessment?
        """
        print(greeting)
        self.tts.speak(greeting)
    
    def respond_with_audio(self, text_response):
        """Respond with both text and audio"""
        print(f"\nBot: {text_response}")
        self.tts.speak_chatbot_response(text_response)


def demo_tts():
    """Demo TTS functionality"""
    print("="*60)
    print("Text-to-Speech Demo")
    print("="*60)
    
    tts = TextToSpeech()
    
    # Test basic speech
    print("\n[Test 1] Basic speech:")
    tts.speak("Hello! Welcome to the health chatbot.")
    
    # Test saving audio
    print("\n[Test 2] Saving audio file:")
    tts.speak("This is a test of the audio saving feature.", 
              save_to_file="test_audio.mp3")
    
    # Test health info
    print("\n[Test 3] Reading health information:")
    tts.read_health_info('diabetes')
    
    print("\n✓ TTS Demo complete!")


if __name__ == "__main__":
    # Check for required packages using importlib to avoid static import resolution errors
    try:
        import importlib
        importlib.import_module('pyttsx3')
        print("✓ pyttsx3 is installed")
    except Exception:
        print("⚠️  pyttsx3 not available. Install with: pip install pyttsx3")
    
    try:
        import importlib
        importlib.import_module('gtts')
        print("✓ gTTS is installed")
    except Exception:
        print("⚠️  gTTS not available. Install with: pip install gtts")
    
    print("\nRunning TTS demo...\n")
    demo_tts()