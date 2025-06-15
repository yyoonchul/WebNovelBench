# Code for WebNovelBench
<!-- NOTE: Chinese text translated to English in example snippet -->


This repository contains the official Pytorch implementation for the paper "WebNovelBench: Placing LLM Novelists on the Web Novel Distribution". We provide the code for reproducing our experiments on novel generation and scoring.

## Prerequisites
### 1. Scoring Data Preparation
The full data used can be found at [![Hugging Face Model: bert-base-uncased](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-webnovelbench-yellow.svg)](https://huggingface.co/datasets/Oedon42/webnovelbench)

Data structure
```
novels_dir/
├── novel1.json
├── novel2.json
└── novel3.json
```
Raw novel scoring
```bash
python novel_original_critic.py --dir absolute/path/to/novels_dir
```
Scoring results preview `scoring_results.json`:
```json
[
  {
    "novel": "novel1",
    "original score": [
      3.5,
      3.5,
      4.0,
      3.8,
      4.1,
      3.9,
      4.1,
      4.0
    ],
    "normalized score": 0.8749,
    "percentile": 99.31,
    "chapters": [
      "chapter 24, xxxx",
      "..."
    ],
    "novel_info": [
      "<main characters> [Zhang San, Li Si,...] </main characters> \n<main plot> \n(1)...\n(2)...</main plot> \n<important scenes> [scene1, scene2,...] </important scenes>",
      "..."
    ],
    "request_id": "4e14bf73-7e0d-4e16-8450-xxxxxxxx"
  },
  {"..."}
]
```
### 2. Configuration for Generation Task

Before running the generation task, you need to set up your configuration file.

1.  Copy the example configuration file:
    ```bash
    cp config_example.json config.json
    ```
2.  Edit `config.json` with your specific settings. Pay close attention to the following fields:
    *   `input_name`: Path to your input data.
    *   `api_key`: Your API key for the generation service (e.g., SiliconFlow).
    *   Adjust other parameters like `num_threads`, `model_name`, `temperature`, `max_tokens` as needed.

    **Example `config.json` structure (based on `config_example.json`):**
    ```json
    {
        "generator": {
            "num_threads": 5,
            "input_name": "/path/to/your/scoring_results.json",
            "url": "https://api.myapi.cn/v1",
            "model_name": "Qwen/Qwen2.5-72B-Instruct",
            "note": "Qwen2-5-72B-Instruct",
            "api_key": "sk-YOUR_SILICONFLOW_API_KEY",
            "temperature": 0.6,
            "max_tokens": 4096
        },
        "critic": {
            "num_threads": 2,
            "url": "https://api.myapi.cn/v1",
            "model_name": "deepseek-ai/DeepSeek-V3",
            "note": "DeepSeek-v3",
            "api_key": "sk-xxxxxxxxxxxxxxxxxxx",
            "temperature": 0.6,
            "max_tokens": 1024
        }
    }
    ```
    **Note:** Replace placeholder values (like `/path/to/your/scoring_results.json` and `sk-YOUR_SILICONFLOW_API_KEY`) with your actual information.

### 3. Environment Setup for Scoring Task

Before running the scoring task, configure your environment as follows:

1.  Install the necessary Python package:
    ```bash
    pip install --upgrade "volcengine-python-sdk[ark]"
    ```
2.  Set the required environment variables. You can add these to your shell configuration file (e.g., `.bashrc`, `.zshrc`) or export them in your current terminal session:
    ```bash
    export ARK_API_ID="ep-bi-YOUR_ARK_API_ID"
    export ARK_API_KEY="YOUR_ARK_API_KEY"
    ```
    **Note:** Replace `ep-bi-YOUR_ARK_API_ID` and `YOUR_ARK_API_KEY` with your actual Volcengine Ark API credentials.

## Running the Pipeline

Once the prerequisites are met, you can run the main pipeline script.
Make sure your `config.json` (created in Step 1 of Prerequisites) is correctly configured.

```bash
python novel_gands_pipeline.py --config config.json
```

## Notes

Currently, the novel scoring function can only be completed through batch inference using the Volcano Ark API.
