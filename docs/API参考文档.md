# IndexTTS API 参考文档

## 核心类和方法

### IndexTTS 主类

#### 类定义
```python
class IndexTTS:
    """IndexTTS 主要推理类
    
    这是 IndexTTS 系统的核心接口，提供了完整的文本转语音功能。
    """
```

#### 构造函数
```python
def __init__(
    self,
    cfg_path: str = "checkpoints/config.yaml",
    model_dir: str = "checkpoints", 
    is_fp16: bool = True,
    device: Optional[str] = None,
    use_cuda_kernel: Optional[bool] = None
)
```

**参数说明:**
- `cfg_path` (str): 配置文件路径，默认为 "checkpoints/config.yaml"
- `model_dir` (str): 模型文件目录，默认为 "checkpoints"
- `is_fp16` (bool): 是否使用半精度推理，默认为 True
- `device` (Optional[str]): 指定设备 ('cuda:0', 'cpu', 'mps')，None时自动选择
- `use_cuda_kernel` (Optional[bool]): 是否使用CUDA自定义内核，None时自动判断

**示例:**
```python
# 基础初始化
tts = IndexTTS()

# 自定义配置
tts = IndexTTS(
    cfg_path="custom/config.yaml",
    model_dir="custom/checkpoints",
    is_fp16=True,
    device="cuda:0"
)

# CPU推理
tts = IndexTTS(device="cpu", is_fp16=False)
```

#### 主要方法

### infer() - 标准推理

```python
def infer(
    self,
    audio_prompt: str,
    text: str, 
    output_path: str,
    verbose: bool = False,
    max_text_tokens_per_sentence: int = 120,
    **generation_kwargs
) -> str
```

**功能:** 执行标准的文本转语音推理

**参数说明:**
- `audio_prompt` (str): 参考音频文件路径 (.wav格式)
- `text` (str): 要合成的文本内容
- `output_path` (str): 输出音频文件路径
- `verbose` (bool): 是否显示详细的处理信息，默认False
- `max_text_tokens_per_sentence` (int): 每句最大token数，用于文本分句，默认120
- `**generation_kwargs`: 生成参数，详见下文

**返回值:** 
- `str`: 输出音频文件路径

**生成参数 (generation_kwargs):**
```python
generation_kwargs = {
    "do_sample": True,              # 是否启用采样
    "top_p": 0.8,                  # Top-p (nucleus) 采样阈值 [0.0, 1.0]
    "top_k": 30,                   # Top-k 采样候选数量 [1, 100]
    "temperature": 1.0,            # 采样温度 [0.1, 2.0]
    "length_penalty": 0.0,         # 长度惩罚 [-2.0, 2.0]
    "num_beams": 3,                # Beam search 束宽 [1, 10] 
    "repetition_penalty": 10.0,    # 重复惩罚 [0.1, 20.0]
    "max_mel_tokens": 600,         # 最大生成mel token数 [50, 2000]
}
```

**示例:**
```python
# 基础使用
tts.infer(
    audio_prompt="reference.wav",
    text="你好，这是一个测试。",
    output_path="output.wav"
)

# 带参数调用
tts.infer(
    audio_prompt="reference.wav", 
    text="长文本内容...",
    output_path="output.wav",
    verbose=True,
    max_text_tokens_per_sentence=100,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=8.0
)
```

### infer_fast() - 批量优化推理

```python
def infer_fast(
    self,
    audio_prompt: str,
    text: str,
    output_path: str, 
    verbose: bool = False,
    max_text_tokens_per_sentence: int = 100,
    sentences_bucket_max_size: int = 4,
    **generation_kwargs
) -> str
```

**功能:** 执行批量优化的推理，适合长文本，性能更高

**参数说明:**
- 基础参数与 `infer()` 相同
- `sentences_bucket_max_size` (int): 分句分桶的最大容量，影响批处理效率，默认4

**适用场景:**
- 长文本合成 (>500字符)
- 需要高吞吐量的场景
- 内存充足的环境

**示例:**
```python
# 长文本快速推理
long_text = "这是一段很长的文本..." * 100
tts.infer_fast(
    audio_prompt="reference.wav",
    text=long_text,
    output_path="output.wav",
    sentences_bucket_max_size=6
)
```

### 工具方法

#### remove_long_silence()
```python
def remove_long_silence(
    self, 
    codes: torch.Tensor, 
    silent_token: int = 52, 
    max_consecutive: int = 30
) -> Tuple[torch.Tensor, torch.Tensor]
```

**功能:** 移除生成音频中的长静音片段

**参数:**
- `codes` (torch.Tensor): 输入的音频token序列
- `silent_token` (int): 静音token的ID
- `max_consecutive` (int): 允许的最大连续静音token数

#### bucket_sentences()
```python
def bucket_sentences(
    self, 
    sentences: List[str], 
    bucket_max_size: int = 4
) -> List[List[Dict]]
```

**功能:** 将句子按长度分桶，优化批处理效率

## 工具模块 API

### TextNormalizer - 文本标准化

```python
from indextts.utils.front import TextNormalizer

normalizer = TextNormalizer()
normalizer.load()

# 标准化文本
normalized = normalizer.normalize("原始文本")
```

**主要功能:**
- 数字转文字
- 符号标准化  
- 拼音处理
- 多音字处理

### TextTokenizer - 文本分词

```python
from indextts.utils.front import TextTokenizer

tokenizer = TextTokenizer(bpe_model_path, normalizer)

# 分词
tokens = tokenizer.tokenize("文本内容")

# 分句
sentences = tokenizer.split_sentences(
    tokens, 
    max_tokens_per_sentence=120
)
```

### MelSpectrogramFeatures - 特征提取

```python
from indextts.utils.feature_extractors import MelSpectrogramFeatures

mel_extractor = MelSpectrogramFeatures(
    sample_rate=22050,
    n_mel_channels=80,
    n_fft=1024,
    hop_length=256,
    win_length=1024
)

# 提取mel频谱
mel_spec = mel_extractor.extract(audio_path)
```

## 配置系统

### 配置文件结构 (config.yaml)

```yaml
# 模型版本
version: "1.5"

# GPT模型配置
gpt:
  model_dim: 1024
  max_text_tokens: 402
  max_mel_tokens: 600
  stop_mel_token: 8192
  # ... 其他参数

# BigVGAN声码器配置  
bigvgan:
  sampling_rate: 22050
  hop_length: 256
  # ... 其他参数

# VQ-VAE配置
vqvae:
  dim: 512
  codebook_size: 8192
  # ... 其他参数

# 数据集配置
dataset:
  bpe_model: "bpe.model"
  # ... 其他参数

# 检查点路径
gpt_checkpoint: "gpt.pth"
bigvgan_checkpoint: "bigvgan_generator.pth" 
dvae_checkpoint: "dvae.pth"
```

### 配置加载

```python
from omegaconf import OmegaConf

# 加载配置
cfg = OmegaConf.load("config.yaml")

# 访问配置
model_dim = cfg.gpt.model_dim
sampling_rate = cfg.bigvgan.sampling_rate

# 修改配置
cfg.gpt.temperature = 0.8
```

## 命令行接口

### indextts 命令

```bash
indextts <text> [OPTIONS]
```

**基础参数:**
- `<text>`: 要合成的文本 (必需)
- `-v, --voice <path>`: 参考音频文件路径 (必需)
- `-o, --output_path <path>`: 输出文件路径，默认"gen.wav"

**高级参数:**
- `-c, --config <path>`: 配置文件路径
- `--model_dir <path>`: 模型目录路径
- `--fp16`: 启用FP16推理
- `-f, --force`: 强制覆盖输出文件
- `-d, --device <device>`: 指定设备

**示例:**
```bash
# 基础使用
indextts "你好世界" -v reference.wav -o output.wav

# 完整参数
indextts "测试文本" \
  --voice ref.wav \
  --output_path out.wav \
  --config custom/config.yaml \
  --model_dir custom/checkpoints \
  --fp16 \
  --device cuda:0
```

## Web UI API

### 启动 Web 界面

```bash
python webui.py [OPTIONS]
```

**参数:**
- `--port <int>`: 端口号，默认7860
- `--host <str>`: 主机地址，默认"127.0.0.1" 
- `--model_dir <path>`: 模型目录
- `--verbose`: 详细输出模式

### Web API 端点

启动WebUI后，可通过HTTP API调用：

```python
import requests

# 文本转语音API调用示例
url = "http://localhost:7860/api/predict"
data = {
    "data": [
        "reference.wav",    # 参考音频
        "要合成的文本",      # 文本内容  
        "普通推理",         # 推理模式
        # ... 其他参数
    ]
}

response = requests.post(url, json=data)
result = response.json()
```

## 错误处理

### 常见异常类型

```python
# 文件不存在
FileNotFoundError: 音频文件或模型文件不存在

# CUDA内存不足  
torch.cuda.OutOfMemoryError: GPU内存不足

# 模型加载失败
RuntimeError: 模型权重加载失败

# 文本处理错误
ValueError: 文本为空或格式不正确
```

### 异常处理示例

```python
try:
    tts = IndexTTS(model_dir="checkpoints")
    result = tts.infer("ref.wav", "测试文本", "output.wav")
    print(f"生成成功: {result}")
    
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
    
except torch.cuda.OutOfMemoryError:
    print("GPU内存不足，尝试使用CPU模式")
    tts = IndexTTS(device="cpu", is_fp16=False)
    result = tts.infer("ref.wav", "测试文本", "output.wav")
    
except Exception as e:
    print(f"未知错误: {e}")
```

## 性能调优参数

### 推理速度优化

```python
# 最快速度配置 (质量略降)
fast_kwargs = {
    "do_sample": True,
    "top_k": 10,           # 减少候选
    "temperature": 0.7,    # 降低随机性
    "num_beams": 1,        # 贪心搜索
    "max_mel_tokens": 400  # 限制长度
}

# 最高质量配置 (速度较慢)
quality_kwargs = {
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 50,
    "temperature": 1.0,
    "num_beams": 5,        # 更大的束搜索
    "repetition_penalty": 12.0,
    "max_mel_tokens": 800
}
```

### 内存优化

```python
# 低内存配置
low_memory_tts = IndexTTS(
    device="cuda:0",
    is_fp16=True,          # 启用半精度
    use_cuda_kernel=True   # 使用优化内核
)

# 分句处理长文本
result = low_memory_tts.infer_fast(
    audio_prompt="ref.wav",
    text=long_text,
    output_path="output.wav",
    max_text_tokens_per_sentence=80,  # 较小分句
    sentences_bucket_max_size=2       # 较小批次
)
```

## 高级用法

### 自定义模型加载

```python
class CustomIndexTTS(IndexTTS):
    def __init__(self, custom_gpt_path=None, **kwargs):
        super().__init__(**kwargs)
        
        if custom_gpt_path:
            # 加载自定义GPT模型
            from indextts.utils.checkpoint import load_checkpoint
            load_checkpoint(self.gpt, custom_gpt_path)
            print(f"已加载自定义GPT模型: {custom_gpt_path}")

# 使用自定义模型
custom_tts = CustomIndexTTS(
    custom_gpt_path="custom_gpt.pth",
    model_dir="checkpoints"
)
```

### 批量处理接口

```python
def batch_process(tts_instance, task_list, max_workers=2):
    """批量处理任务
    
    Args:
        tts_instance: IndexTTS实例
        task_list: 任务列表，每个任务为dict包含audio_prompt, text, output_path
        max_workers: 最大并发数
    """
    import concurrent.futures
    
    def process_single(task):
        return tts_instance.infer(**task)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single, task_list))
    
    return results

# 使用示例
tasks = [
    {"audio_prompt": "ref1.wav", "text": "文本1", "output_path": "out1.wav"},
    {"audio_prompt": "ref2.wav", "text": "文本2", "output_path": "out2.wav"},
]

results = batch_process(tts, tasks, max_workers=2)
```

这份API参考文档涵盖了IndexTTS的所有主要接口和使用方法，可以作为开发时的快速参考手册。 