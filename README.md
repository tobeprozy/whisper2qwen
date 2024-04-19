# Whisper <!-- omit in toc -->

## 目录 <!-- omit in toc -->
- [1. 简介](#1-简介)
- [2. 特性](#2-特性)
- [3. 准备模型与数据](#3-准备模型与数据)
- [4. 模型编译](#4-模型编译)
  - [4.1 TPU-MLIR环境搭建](#41-tpu-mlir环境搭建)
    - [4.1.1 安装docker](#411-安装docker)
    - [4.1.2 下载并解压TPU-MLIR](#412-下载并解压tpu-mlir)
    - [4.1.3 创建并进入docker](#413-创建并进入docker)
  - [4.2 获取onnx](#42-获取onnx)
  - [4.3 bmodel编译](#43-bmodel编译)
- [5. 例程测试](#5-例程测试)
- [6. 精度测试](#6-精度测试)
  - [6.1 测试方法](#61-测试方法)
  - [6.2 测试结果](#62-测试结果)
- [7. 性能测试](#7-性能测试)

## 1. 简介
Whisper 是一个开源的深度学习语音识别模型，由 OpenAI 开发，它能够实现实时、多语言的语音识别，并支持跨多种环境和设备的灵活部署。本例程对[Whisper官方开源仓库](https://github.com/openai/whisper)中的算法进行移植，使之能在SOPHON BM1684X上进行推理。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持FP16(BM1684X)模型编译和推理
* 支持基于SAIL推理的Python例程

## 3. 准备模型与数据
该模型目前只支持在1684X上运行，已提供编译好的bmodel，​同时，您需要准备用于测试的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`。

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684X
│   ├── bmwhisper_base_1684x_f16.bmodel # whisper-medium模型，模型参数量为769 M
│   ├── bmwhisper_medium_1684x_f16.bmodel # whisper-small模型，模型参数量为244 M
│   └── bmwhisper_small_1684x_f16.bmodel # whisper-base模型，模型参数量为74 M
└── onnx # whisper的onnx原始模型
    ├── decoder_loop_with_kvcache_base_5beam_448pad.onnx
    ├── decoder_loop_with_kvcache_medium_5beam_448pad.onnx
    ├── decoder_loop_with_kvcache_small_5beam_448pad.onnx
    ├── decoder_main_with_kvcache_base_5beam_448pad.onnx
    ├── decoder_main_with_kvcache_medium_5beam_448pad.onnx
    ├── decoder_main_with_kvcache_small_5beam_448pad.onnx
    ├── decoder_post_base_5beam_448pad.onnx
    ├── decoder_post_medium_5beam_448pad.onnx
    ├── decoder_post_small_5beam_448pad.onnx
    ├── encoder_base_5beam_448pad.onnx
    ├── encoder_medium_5beam_448pad.onnx
    ├── encoder_small_5beam_448pad.onnx
    ├── kvcache_rearrange_base_5beam_448pad.onnx
    ├── kvcache_rearrange_medium_5beam_448pad.onnx
    ├── kvcache_rearrange_small_5beam_448pad.onnx
    ├── logits_decoder_base_5beam_448pad.onnx
    ├── logits_decoder_medium_5beam_448pad.onnx
    └── logits_decoder_small_5beam_448pad.onnx
```

下载的数据包括：
```
./datasets
|── aishell_S0764                             # 从aishell数据集中抽取的用于测试的音频文件
|   └── *.wav
├── aishell_S0764.list                        # 从aishell数据集的文件列表
├── ground_truth.txt                          # 从aishell数据集的预测真实值
└── test                                      # 测试使用的音频文件
    └── demo.wav
```
## 4. 模型编译
此部分请参考[Whisper模型的导出与编译](./docs/ChatGLM3_Export_Guide.md)

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法
首先，参考[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测结果至result路径，注意修改数据集(datasets/aishell_S0764)和相关参数。
然后，使用`tools`目录下的`eval_aishell.py`脚本，将测试生成的txt文件与测试集标签txt文件进行对比，计算出语音识别的评价指标，命令如下：
```bash
# 请根据实际情况修改程序路径和txt文件路径
python3 tools/eval_aishell.py --char=1 --v=1 datasets/ground_truth.txt python/result  > online_wer
cat online_wer | grep "Overall"
```

### 6.2 测试结果
在aishell数据集上，精度测试结果如下：
|   测试平台    |    测试程序   |              测试模型                                 | WER    |
| ------------ | ------------ | ----------------------------------------------------- | ------ |
|   SE7-32     | whisper.py   | bmwhisper_base_1684x_f16.bmodel                       | 17.80% |
|   SE7-32     | whisper.py   | bmwhisper_small_1684x_f16.bmodel                      | 9.44%  |
|   SE7-32     | whisper.py   | bmwhisper_medium_1684x_f16.bmodel                     | 5.88%  |

> **测试说明**：
1. 在使用的模型相同的情况下，wer在不同的测试平台上是相同的。
2. 由于SDK版本之间的差异，实测的wer与本表有1%以内的差值是正常的。

## 7. 性能测试
|    测试平台   |     测试程序      |           测试模型                  |  Preprocess time(ms) |    Inference time(ms)   |
| -----------  | ---------------- | -----------------------------------| --------------------- | ----------------------- |
|   SE7-32     | whisper.py       | bmwhisper_base_1684x_f16.bmodel    | 247.61                | 61.70                   |
|   SE7-32     | whisper.py       | bmwhisper_small_1684x_f16.bmodel   | 268.22                | 179.44                  |
|   SE7-32     | whisper.py       | bmwhisper_medium_1684x_f16.bmodel  | 300.66                | 451.54                  |

> **测试说明**：
> 1. 该性能使用datasets/test/demo.wav音频进行测试，计算后得出平均每秒音频所需推理时间。
> 2. whisper模型的预处理主要包括加载语音，特征提取等，推理后的结果可直接转换为自然语言，时间可忽略不计，因此无后处理部分时间统计。
> 3. 性能测试结果具有一定的波动性，实测结果与该表结果有误差属正常现象，建议多次测试取平均值。
> 4. BM1684X SoC的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz。




# Qwen

## 目录
  - [1. 简介](#1-简介)
  - [2. 特性](#2-特性)
  - [3. 运行环境准备](#3-运行环境准备)
  - [4. 准备模型](#4-准备模型)
  - [5. 例程测试](#5-例程测试)
  - [3. 程序性能测试](#6-程序性能测试)

## 1. 简介
Qwen 是开源中英双语对话模型，关于它的特性，请前往源repo查看：https://huggingface.co/Qwen。本例程对Qwen进行移植，使之能在SOPHON BM1684X上进行推理测试。

该例程支持在V23.07.01(libsophon_0.4.9)及以上的SDK上运行，支持在插有1684X加速卡(SC7系列)的x86主机上运行，也可以在1684X SoC设备（如SE7、SM7、Airbox等）上运行。在SoC上运行需要额外进行环境配置，请参照[运行环境准备](#3-运行环境准备)完成环境部署。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)
* 支持FP16、INT8、INT4模型编译和推理
* 支持基于SAIL推理的Python例程
* 支持多轮对话


## 3. 运行环境准备
在PCIe上无需修改内存，以下为soc模式相关：
对于1684X系列设备（如SE7/SM7），都可以通过这种方式完成环境准备，使得满足Qwen运行条件。首先，在1684x SoC环境上，参考如下命令修改设备内存。
```bash
cd /data/
mkdir memedit && cd memedit
wget -nd https://sophon-file.sophon.cn/sophon-prod-s3/drive/23/09/11/13/DeviceMemoryModificationKit.tgz
tar xvf DeviceMemoryModificationKit.tgz
cd DeviceMemoryModificationKit
tar xvf memory_edit_{vx.x}.tar.xz #vx.x是版本号
cd memory_edit
./memory_edit.sh -p #这个命令会打印当前的内存布局信息
./memory_edit.sh -c -npu 7615 -vpu 3072 -vpp 3072 #npu也可以访问vpu和vpp的内存
sudo cp /data/memedit/DeviceMemoryModificationKit/memory_edit/boot.itb /boot/boot.itb && sync
sudo reboot
```
> **注意：**
> 1. tpu总内存为npu/vpu/vpp三者之和，fp16模型应满足tpu内存 >= 12800 MB，int8应满足tpu内存 >= 7168MB，int4应满足tpu内存 >= 4608MB。
> 2. 更多教程请参考[SoC内存修改工具](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/SophonSDK_doc/zh/html/appendix/2_mem_edit_tools.html)

## 4. 准备模型
该模型目前只支持在1684X上运行，已提供编译好的bmodel。
### 4.1 使用提供的模型

​本例程在`scripts`目录下提供了下载脚本`download.sh`

**注意：**在运行前，应该保证存储空间大于10G (Qwen-1.8B), 50GB(Qwen-7B), 100G(Qwen-14B)。

```bash
# 安装unzip，若已安装请跳过，非ubuntu系统视情况使用yum或其他方式安装
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

执行下载脚本后，当前目录下的文件如下：
```bash
├── docs
│   └── Qwen_Export_Guide.md        #Qwen onnx导出和bmodel编译指南
├── models
│   └── BM1684X                     #download.sh下载的bmodel
│       ├── qwen-7b_int4_1dev.bmodel
│       └── qwen-7b_int8_1dev.bmodel
├── python
│   ├── qwen.py                     #Qwen python推理脚本
│   ├── README.md                   #python例程执行指南
│   ├── tokenization_util.py        #tokenizer分词处理脚本
│   ├── requirements.txt            #python例程的依赖模块
│   └── token_config                #download.sh下载的tokenizer
│       ├── tokenization_qwen.py
│       ├── tokenizer_config.json
│       └── qwen.tiktoken 
├── README.md                       #Qwen例程指南
├── scripts                         
│   ├── download.sh                 #下载脚本
│   └── gen_bmodel.sh               #模型编译脚本
└── tools
    ├── Qwen-1_8B-Chat               #修改过的Qwen-1.8B源码
    │   ├── config.json
    │   └── modeling_qwen.py
    ├── Qwen-7B-Chat                 #修改过的Qwen-7B源码
    │   ├── config.json
    │   └── modeling_qwen.py
    ├── Qwen-14B-Chat                #修改过的Qwen-14B源码
    │   ├── config.json
    │   └── modeling_qwen.py
    └── export_onnx.py               #Qwen导出onnx脚本。
```

### 4.2 自行编译模型

此部分请参考[Qwen模型导出与编译](./docs/Qwen_Export_Guide.md)

## 5. 例程测试

- [Python例程](./python/README.md)

## 6. 程序性能测试

这里的测试输入为："请使用C++写一段冒泡排序算法。"
|    测试平台   |     测试程序       |           测试模型             |first token latency(s)|token per second(tokens/s)| 
| -----------  | ---------------- | ---------------------------     | --------------------- | ----------------------- | 
| SE7-32  | qwen.py      | qwen-7b_int4_1dev.bmodel         |    0.678              |    9.392          | 
| SE7-32  | qwen.py      | qwen-7b_int8_1dev.bmodel         |    0.682              |    6.162         | 

> **测试说明**：  
> 1. 性能测试结果具有一定的波动性，建议多次测试取平均值；
> 2. SE7-32的主控处理器为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于处理器的不同可能存在较大差异；
> 3. 这里使用的SDK版本是V23.10.01；
