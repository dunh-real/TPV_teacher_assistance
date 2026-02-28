from src.models.vietnormalizer.normalizer import VietnameseNormalizer
from src.services.tts_service import TTS_Service
import soundfile as sf
from pathlib import Path

# Flow: Input Text -> Normalize -> TTS Model -> Output Audio

# Load service
tts_service = TTS_Service()

def process_text(text):
    normalizer = VietnameseNormalizer()
    normalized_text = normalizer.normalize(text)

    output_audio, sample_rate = tts_service.text2speech(normalized_text)

    # Save audio
    out_path = Path("./output.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), output_audio, sample_rate)

text = "chào bạn, tôi là chú mèo máy đến từ tương lai, tên tôi là đô-ra-ê-mon."

process_text(text)
