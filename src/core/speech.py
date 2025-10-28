"""Text-to-speech functionality for voice alerts."""

import os
import queue
import shutil
import tempfile
import threading
import time
from typing import Optional

from src.config import POSTURE_CONFIG, TTS_MESSAGES

# Try to import TTS dependencies
try:
    import pygame
    from gtts import gTTS

    TTS_AVAILABLE = True
except ImportError as e:
    print(f"TTS imports failed: {e}")
    gTTS = None
    pygame = None
    TTS_AVAILABLE = False


class SpeechManager:
    """Manages text-to-speech functionality for voice alerts."""

    def __init__(self):
        """Initialize the speech manager."""
        self.speech_queue = queue.Queue(maxsize=1)
        self.speech_worker_running = False
        self.tts_available = False
        self.last_speech_time = 0.0
        self.speech_cooldown = POSTURE_CONFIG.speech_cooldown
        self.temp_dir: Optional[str] = None

        self._initialize_tts()

    def _initialize_tts(self) -> None:
        """Initialize TTS system with Google TTS."""
        if not TTS_AVAILABLE:
            print("TTS libraries not available - voice alerts disabled")
            return

        try:
            # Create temporary directory for audio files
            self.temp_dir = tempfile.mkdtemp()

            # Initialize pygame mixer for audio playback
            if pygame is not None:
                pygame.mixer.init()

            self.tts_available = True
            self._start_speech_worker()
            print("Voice alerts enabled with high-quality TTS")

        except Exception as e:
            print(f"TTS initialization failed: {e}")
            self.tts_available = False

    def _start_speech_worker(self) -> None:
        """Start the speech worker thread."""
        if self.speech_worker_running:
            return

        def speech_worker():
            while self.speech_worker_running:
                try:
                    # Wait for speech requests with timeout
                    text = self.speech_queue.get(timeout=1.0)

                    if (
                        text
                        and self.tts_available
                        and gTTS is not None
                        and pygame is not None
                    ):
                        current_time = time.time()

                        # Enforce cooldown to prevent overlapping speech
                        if current_time - self.last_speech_time >= self.speech_cooldown:
                            try:
                                self._generate_and_play_speech(text, current_time)
                                self.last_speech_time = current_time
                            except Exception as e:
                                print(f"Speech error: {e}")

                    self.speech_queue.task_done()

                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Speech worker error: {e}")
                    try:
                        self.speech_queue.task_done()
                    except ValueError:
                        pass

        self.speech_worker_running = True
        worker_thread = threading.Thread(target=speech_worker, daemon=True)
        worker_thread.start()

    def _generate_and_play_speech(self, text: str, current_time: float) -> None:
        """Generate and play TTS audio.

        Args:
            text: Text to synthesize
            current_time: Current timestamp for temp file naming
        """
        if gTTS is None or pygame is None or self.temp_dir is None:
            return

        # Generate speech with Google TTS (natural female voice)
        tts = gTTS(text=text, lang="en", slow=False)

        # Create temporary file
        temp_file = os.path.join(self.temp_dir, f"speech_{int(current_time)}.mp3")
        tts.save(temp_file)

        # Play the audio using pygame
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()

        # Wait for playback to complete
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        # Clean up the temporary file
        try:
            os.remove(temp_file)
        except OSError:
            pass

    def speak(self, text: str) -> None:
        """Queue text for speech synthesis.

        Args:
            text: Text to speak
        """
        if not self.tts_available or not self.speech_worker_running:
            return

        # Check cooldown before queuing
        current_time = time.time()
        if current_time - self.last_speech_time < self.speech_cooldown:
            return

        # Try to add speech request (non-blocking)
        try:
            # Clear any pending speech first
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                pass

            # Queue new speech
            self.speech_queue.put_nowait(text)
        except queue.Full:
            pass

    def speak_bad_posture_alert(self) -> None:
        """Speak the standard bad posture alert message."""
        self.speak(TTS_MESSAGES["bad_posture"])

    def cleanup(self) -> None:
        """Clean up resources and stop speech worker."""
        try:
            # Stop speech worker
            self.speech_worker_running = False

            # Clean up pygame and temp files
            if self.tts_available and pygame is not None:
                pygame.mixer.quit()

            if self.temp_dir:
                try:
                    shutil.rmtree(self.temp_dir)
                except OSError:
                    pass
        except Exception:
            pass

    @property
    def is_available(self) -> bool:
        """Check if TTS is available and initialized."""
        return self.tts_available
