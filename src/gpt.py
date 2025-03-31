from TTS.api import TTS
import sounddevice as sd
from functools import partial
from TTS.utils.radam import RAdam
import torch
import queue
import threading
import time

# 模型和资源路径设置
model_path = "/home/pnx/Desktop/paddle/models/xtts_v2/"  # 修改为正确的模型路径
config_path = "/home/pnx/Desktop/paddle/models/xtts_v2/config.json"
vocab_path = "/home/pnx/Desktop/paddle/models/xtts_v2/vocab.json"  # vocab.json 必须提供
root_path = "/home/pnx/Desktop/paddle/"

# 配置 torch 安全加载参数
torch.serialization.add_safe_globals([RAdam])
torch.load = partial(torch.load, pickle_module=torch.serialization.pickle, mmap=None)

# 初始化 TTS 模型
tts = TTS(
    model_path=model_path,
    config_path=config_path,
    gpu=True
)

# 全局播放锁，确保同一时刻只有一个播放线程
play_lock = threading.Lock()

def play_audio_queue(audio_queue, sample_rate):
    def audio_callback(outdata, frames, time_info, status):
        if status:
            print(status)
        try:
            # 获取队列数据
            data = audio_queue.get(timeout=1)
            if data is None:
                raise sd.CallbackStop()
            # 如果数据不足 frames 长度则补 0
            if len(data) < frames:
                outdata[:len(data), 0] = data
                outdata[len(data):, 0] = 0
            else:
                outdata[:, 0] = data[:frames]
        except queue.Empty:
            outdata.fill(0)

    with sd.OutputStream(samplerate=sample_rate, channels=1, callback=audio_callback):
        # 等待播放完毕（队列中最后会放入 None 作为结束标志）
        while not audio_queue.empty():
            sd.sleep(100)

def speak_stream(text, reference_audio, chunk_size=8, sentence_pause=0.5):
    # 使用全局锁，防止多线程同时调用播放
    with play_lock:
        sample_rate = tts.synthesizer.output_sample_rate  # 获取模型输出采样率（通常为24000）
        # 按“。”拆分文本，逐句合成并播放
        sentences = text.split("。")
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence += "。"
                print(f"正在合成句子: {sentence}")
                try:
                    # 调整 speed 参数为 0.8 以降低播放速度
                    wav = tts.tts(
                        text=sentence,
                        speaker_wav=reference_audio,
                        language="zh-cn",
                        temperature=0.7,
                        speed=1.0,   # 将语速从1.0降低到0.8
                        top_p=0.9,
                        length_penalty=1.5,
                    )
                except Exception as e:
                    print(f"合成错误: {e}")
                    continue

                # 创建一个队列，用于存储当前句子的音频数据块
                audio_queue = queue.Queue()
                for i in range(0, len(wav), chunk_size):
                    time.sleep(0.1)
                    audio_queue.put(wav[i:i+chunk_size])
                # 放入结束标记
                audio_queue.put(None)

                print(f"播放句子: {sentence}")
                play_audio_queue(audio_queue, sample_rate)
                # 播放完毕后增加暂停，确保听清每句
                time.sleep(sentence_pause)

# 示例文本和参考音频路径
text = ("随着人工智能技术的发展，语音合成技术（Text-to-Speech, TTS）在多个领域得到了广泛应用，"
        "如智能助手、有声阅读、自动播报等。为了推动语音合成技术的进一步发展，本赛题旨在挑战参赛者设计并实现一个高效、自然、准确的TTS系统。"
        "在数字化时代，教育行业正经历着前所未有的技术革新。语音合成（TTS）技术和声音克隆技术作为人工智能领域的重要分支，"
        "在多个领域得到了广泛应用，如智能助手、有声阅读、自动播报等。在教育领域也展现出巨大的潜力，能够提供个性化的学习体验，"
        "增强教学内容的互动性和可访问性，从而提高学习效率和质量。")
reference_audio = root_path + "data/2.wav"

# 调用逐句合成并播放的函数
speak_stream(text, reference_audio)
