
import time
from TTS.api import TTS
import numpy as np
import sounddevice as sd
from functools import partial
import sounddevice as sd
from TTS.api import TTS
from TTS.utils.radam import RAdam
import soundfile as sf
import torch
import tempfile
import noisereduce as nr
import librosa

model_path = "/home/pnx/Desktop/paddle/models/xtts_v2/"  # 修正路径
config_path = "/home/pnx/Desktop/paddle/models/xtts_v2/config.json"
vocab_path = "/home/pnx/Desktop/paddle/models/xtts_v2/vocab.json"  # 必须提供 vocab.json
root_path = "/home/pnx/Desktop/paddle/"

torch.serialization.add_safe_globals([RAdam])
torch.load = partial(torch.load, pickle_module=torch.serialization.pickle, mmap=None)

def load_reference_audio(reference_audio_path):
    # 使用 librosa 加载参考音频（返回音频数据及采样率）
    audio, sr = librosa.load(reference_audio_path, sr=None)
    return audio, sr

tts = TTS(
    model_path=model_path,
    config_path=config_path,
    gpu=True
)


def stream_tts(text, reference, chunk_length=10):
    # 分割文本为短句（按标点或固定长度）
    chunks = split_text_into_chunks(text, chunk_length)
    
    for chunk in chunks:
        # 生成当前块的音频
        audio_np = tts.tts(chunk, speaker_wav=reference_audio, language="zh-cn")
        audio_np = np.array(audio_np).astype(np.float32)
        
        # 实时播放当前块
        sd.play(audio_np, samplerate=22050)
        sd.wait()  # 等待播放完成（或异步处理）


def split_text_into_chunks(text, max_length):
    # 简单按标点分割（可优化为 NLP 分句）
    chunks = []
    buffer = ""
    for char in text:
        buffer += char
        if char in ".!?。！？" and (len(buffer) > max_length):
            chunks.append(buffer.strip())
            buffer = ""
    if buffer:
        chunks.append(buffer.strip())
    return chunks

def speak(text, reference_audio, save_path=root_path+"output/"):
    try:
        # 生成音频 numpy 数组 (采样率默认为 24000)
        wav = tts.tts(
            text=text,
            speaker_wav=reference_audio,
            language="zh-cn",
            temperature=0.7,
            speed=1.0,
            top_p=0.9,             # 限制采样范围（默认0.85）
            length_penalty=1.5,

        )

        sample_rate = tts.synthesizer.output_sample_rate
        now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time())) 
        file_name = save_path+now+".wav"
        # 如果提供了保存路径，则保存音频文件
        if save_path:
            sf.write(file_name, wav, sample_rate)
            print(f"音频已保存到: {file_name}")
        
        # 实时播放
        sample_rate = tts.synthesizer.output_sample_rate  # 获取模型采样率 (通常为 24000)
        print(f"正在播放 (采样率: {sample_rate}Hz)...")
        sd.play(wav, samplerate=sample_rate, blocking=True)  # blocking=True 等待播放完成
        
    except Exception as e:
        print(f"错误: {e}")



text = "随着人工智能技术的发展，语音合成技术（Text-to-Speech, TTS）在多个领域得到了广泛应用，如智能助手、有声阅读、自动播报等。为了推动语音合成技术的进一步发展，本赛题旨在挑战参赛者设计并实现一个高效、自然、准确的TTS系统。在数字化时代，教育行业正经历着前所未有的技术革新。语音合成（TTS）技术和声音克隆技术作为人工智能领域的重要分支，在多个领域得到了广泛应用，如智能助手、有声阅读、自动播报等。在教育领域也展现出巨大的潜力，能够提供个性化的学习体验，增强教学内容的互动性和可访问性，从而提高学习效率和质量。"
reference_audio = root_path + "data/2.wav"



reference_audio_data, sr = load_reference_audio(reference_audio)

# 调用 stream_tts，传递参考音频
stream_tts(text, reference_audio)