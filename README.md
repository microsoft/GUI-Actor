<!-- # GUI-Actor -->

<div align="center">
<img src="assets/images/title.png?raw=true" width="80%" style="margin-bottom: 40px;">

[Qianhui Wu](https://qianhuiwu.github.io/)<sup>*1</sup>&nbsp;
[Kanzhi Cheng](https://scholar.google.com/citations?user=S2IPVnwAAAAJ&hl=en&oi=ao/)<sup>*2</sup>&nbsp;
[Rui Yang](https://yangrui2015.github.io/)<sup>*3</sup>&nbsp;
[Chaoyun Zhang](https://vyokky.github.io/)<sup>1</sup>&nbsp;
[Jianwei Yang](https://jwyang.github.io/)<sup>1</sup>&nbsp;
[Huiqiang Jiang](https://hqjiang.com/)<sup>1</sup><br>
[Jian Mu]()<sup>2</sup>&nbsp;
[Baolin Peng](https://scholar.google.com/citations?user=u1CNjgwAAAAJ&hl=zh-CN)<sup>1</sup>&nbsp;
[Bo Qiao](https://scholar.google.com/citations?user=_6ugrdYAAAAJ&hl=en)<sup>1</sup>&nbsp;
[Reuben Tan](https://cs-people.bu.edu/rxtan/)<sup>1</sup>&nbsp;
[Si Qin](https://sqin860.github.io/)<sup>1</sup>&nbsp;
[Lars Liden](https://sites.google.com/site/larsliden)<sup>1</sup><br>
[Qingwei Lin](https://scholar.google.com/citations?user=W9fdsxMAAAAJ&hl=zh-CN)<sup>1</sup>&nbsp;
[Huan Zhang](https://huan-zhang.com/)<sup>3</sup>&nbsp;
[Tong Zhang](https://tongzhang-ml.org/)<sup>3</sup>&nbsp;
[Jianbing Zhang](https://cs.nju.edu.cn/zhangjb/index.htm)<sup>2</sup>&nbsp;
[Dongmei Zhang](https://scholar.google.com/citations?user=jLlBBl4AAAAJ&hl=en)<sup>1</sup>&nbsp;
[Jianfeng Gao](https://scholar.google.com/citations?user=CQ1cqKkAAAAJ&hl=en)<sup>1</sup><sup>†</sup> 

<sup>1</sup> Microsoft Research&nbsp;&nbsp;<sup>2</sup> Nanjing University&nbsp;&nbsp;<sup>3</sup> University of Illinois Urbana-Champaign<br>
<sup>*</sup> Equal Contribution&nbsp;&nbsp;&nbsp;&nbsp;<sup>†</sup> Leadership  

<h4>
<a href="https://www.arxiv.org/pdf/2502.13130">📄 arXiv Paper</a> &nbsp; 
<a href="https://aka.ms/GUI-Actor/">🌐 Project Page</a> &nbsp; 
<a href="https://huggingface.co/microsoft/GUI-Actor-7B-Qwen2-VL">🤗 Hugging Face Models</a>
</h4>

</div>

<div align="center">
<img src="assets/images/main_figure.png?raw=true" width="100%">
</div>
Figure 1. Left: Model performance vs. training data scale on the ScreenSpot-Pro benchmark. Higher and more left is better; larger points indicate models with more parameters. We only show GUI-Actor models built upon Qwen2-VL here for fair comparison. With Qwen2.5-VL as the backbone, GUI-Actor-3B/7B reaches scores up to 42.2/44.6 (without Verifier). Right: Illustration of action attention. GUI-Actor grounds target elements by attending to the most relevant visual regions.

## :sparkles: Highlights
🤔 **We identify several limitations in coordinate-generation based methods** (_i.e._, output screen positions as text tokens x=…, y=…) for GUI grounding, including (1) weak spatial-semantic alignment, (2) ambiguous supervision signals, and (3) granularity mismatch between vision and action space.

💡 **Rethink how humans interact with digital interfaces**: humans do NOT calculate precise screen coordinates before acting—they perceive the target element and interact with it directly.

🚀 **We propose _GUI-Actor_, a VLM enhanced by an action head, to mitigate the above limitations.** The attention-based action head not only enables GUI-Actor to peform coordinate-free GUI grounding that more closely aligns with human behavior, but also can generate multiple candidate regions in a single forward pass, offering flexibility for downstream modules such as search strategies.

➕ **We design a _grounding verifier_ to evaluate and select the most plausible action region** among the candidates proposed from the action attention map. We show that this verifier can be easily integrated with other grounding methods for a further performance boost.

🎯 **GUI-Actor achieves state-of-the-art performance on multiple GUI action grounding benchmarks** with the same Qwen2-VL backbone, demonstrating its effectiveness and generalization to unseen screen resolutions and layouts. Notably,GUI-Actor-7B even surpasses UI-TARS-72B (38.1) on **ScreenSpot-Pro**, achieving scores of **40.7** with Qwen2-VL and **44.6** with Qwen2.5-VL as backbones.

<!-- ## :fire: News
* **[2025.06.03]**  We released the GUI-Actor training/inference code and model checkpoints!
-->

## :bookmark_tabs: Todos
We will be releasing all the following contents:
- [x] Model training and evaluation based on Qwen2-VL
- [x] Model checkpoint
- [ ] Support for Qwen2.5-VL
- [ ] Processed training data
- [ ] Demo

## :bar_chart: Main Results
Table 1. Main results on the ScreenSpot-Pro benchmark with Qwen2-VL as the backbone. † indicates scores obtained from our own evaluation of the official models on Huggingface.
| Method           | Backbone VLM | ScreenSpot-Pro | ScreenSpot | ScreenSpot-v2 |
|------------------|--------------|----------------|------------|----------------|
| **_72B models:_**
| AGUVIS-72B       | Qwen2-VL     | -              | 89.2       | -              |
| UGround-V1-72B   | Qwen2-VL     | 34.5           | **89.4**   | -              |
| UI-TARS-72B      | Qwen2-VL     | **38.1**       | 88.4       | **90.3**       |
| **_7B models:_**
| OS-Atlas-7B      |              | 18.9           | 82.5       | 84.1           |
| AGUVIS-7B        | Qwen2-VL     | 22.9           | 84.4       | 86.0†          |
| UGround-V1-7B    | Qwen2-VL     | 31.1           | 86.3       | 87.6†          |
| UI-TARS-7B       | Qwen2-VL     | 35.7           | **89.5**   | **91.6**       |
| GUI-Actor-7B     | Qwen2-VL     | **40.7**       | 88.3       | 89.5           |
| **_2B models:_**
| UGround-V1-2B    | Qwen2-VL     | 26.6           | 77.1       | -              |
| UI-TARS-2B       | Qwen2-VL     | 27.7           | 82.3       | 84.7           |
| GUI-Actor-2B     | Qwen2-VL     | **36.7**       | **86.5**   | **88.6**       |

Table 2. Main results on the ScreenSpot-Pro benchmark with Qwen2.5-VL as the backbone.
| Method         | Backbone VLM | ScreenSpot-Pro | ScreenSpot-v2 |
|----------------|---------------|----------------|----------------|
| **_7B models:_**
| Qwen2.5-VL-7B  | Qwen2.5-VL    | 27.6           | 80.9           |
| Jedi-7B        | Qwen2.5-VL    | 36.1           | 88.6           |
| GUI-Actor-7B   | Qwen2.5-VL    | **42.2**       | **91.0**       |
| **_3B models:_**
| Qwen2.5-VL-3B  | Qwen2.5-VL    | 25.9           | 88.8           |
| Jedi-3B        | Qwen2.5-VL    | 39.5           | 91.7           |
| GUI-Actor-3B   | Qwen2.5-VL    | **44.6**       | **92.1**       |

## :rescue_worker_helmet: Installation
1. Clone this repo to your local machine:
```bash
git clone https://github.com/microsoft/GUI-Actor.git
cd GUI-Actor
```
2. Create a conda environment and install the dependencies:
```bash
conda create -n gui_actor python=3.10
conda activate gui_actor
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
pip install -e .
```
## :minidisc: Data Preparation
1. Download the processed data from [here](https://huggingface.co/datasets/microsoft/GUI-Actor-Data).
2. Modify the paths in the [data_config.py](./data/data_config.yaml) file to point to the downloaded data.

## :building_construction: Model Training
1. Warmup stage:
```bash
bash scripts/warmup.sh
```
2. Full-parameter training stage:
```bash
bash scripts/train.sh
```

## :memo: Evaluation on GUI Grounding Benchmarks
For evaluation on ScreenSpot and ScreenSpot-v2, you can directly run the scripts under the `scripts/` folder like `python eval/screenSpot.py` or `python eval/screenSpot_v2.py`.

For evaluation on ScreenSpot-Pro, you first need to download the data from [here](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro), then run the following command:
```bash
python eval/screenSpot_pro.py --save_path <path_to_save_results> --data_path <path_to_data_dir>
```

## :+1: Acknowledgements

This project is built upon the following projects. Thanks for their great work!
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [AGUVIS](https://github.com/xlang-ai/aguvis)

## :checkered_flag: Citation
If you find this work useful in your research, please consider citing:
```bibtex
@article{wu2025guiactor,
    title={GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents}, 
    author={Qianhui Wu and Kanzhi Cheng and Rui Yang and Chaoyun Zhang and Jianwei Yang and Huiqiang Jiang and Jian Mu and Baolin Peng and Bo Qiao and Reuben Tan and Si Qin and Lars Liden and Qingwei Lin and Huan Zhang and Tong Zhang and Jianbing Zhang and Dongmei Zhang and Jianfeng Gao},
    year={2025},
    eprint={},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={},
}
```