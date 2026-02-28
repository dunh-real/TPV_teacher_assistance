from huggingface_hub import snapshot_download
from voxcpm.core import VoxCPM
import librosa

LOCAL_MODEL_DIR = "src/models/VoxCPM/VoxCPM-1.5-VN"
MODEL_NAME = "JayLL13/VoxCPM-1.5-VN"

#CONFIG PARAMS
cfg_value_default = 2.0
inference_timesteps_default = 10
max_len_default = 600
speed_rate_default = 1.0

class TTS_Service:
    def __init__(self):
        print("Loading snapshot ...")
        snapshot_download(MODEL_NAME, local_dir=LOCAL_MODEL_DIR)

        print("Loading model ...")
        self.model = VoxCPM.from_pretrained(
            hf_model_id=LOCAL_MODEL_DIR,
            load_denoiser=False,
            optimize=True,
        )
        print("Load complete !!!")

    def text2speech(self, input_text):
        audio_np = self.model.generate(
            text=input_text,
            prompt_wav_path=None,
            prompt_text=None,
            cfg_value=cfg_value_default,
            inference_timesteps=inference_timesteps_default,
            max_len=max_len_default,
            normalize=False,
            denoise=False,
        )

        speed_rate = speed_rate_default
        output_audio = librosa.effects.time_stretch(audio_np, rate=speed_rate)

        return output_audio, self.model.tts_model.sample_rate
    