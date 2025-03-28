# A.R.O.N.A

## Overview
A.R.O.N.A is an open-source AI project based on Tiny Llama, specifically designed for Blue Archive on Linux platforms. The project features an AI named Arona with custom training capabilities.

## Features
- Tiny Llama-based AI assistant
- Linux compatibility
- Custom training functionality
- Multiple interaction modes

## Prerequisites
- Python 3
- CUDA-compatible GPU (recommended)
- Google Colab (recommended for training)
- Nvidia T4 GPU (optimal for training)

## Installation
```bash
pip install --upgrade pip
pip install torch transformers peft accelerate bitsandbytes datasets
```

## Colab Training Setup

### Dependencies
```python
!pip install torch transformers peft accelerate bitsandbytes datasets
from google.colab import drive
drive.mount('/content/drive')
```

### Training Configuration
- Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
- Dataset: `data.json` (500+ entries)
- Training Method: LoRA (Low-Rank Adaptation)

### Training Process
1. Mount Google Drive
2. Load TinyLlama model
3. Preprocess dataset
4. Apply LoRA configuration
5. Train model
6. Save LoRA adapter

## Project Structure
- `trainbot.py`: Advanced user training script
- `chatbot.py`: AI testing script
- `main.py`: Primary interaction interface
- `data.json`: Included dataset (500+ entries)

## Usage
1. Test AI: `python3 chatbot.py`
2. Chat Interface: `python3 main.py`

## Recommended Hardware
- CUDA-compatible GPU
- Nvidia T4 (optimal for training)
- Google Colab Pro recommended

## Training Tips
- Use Google Colab for best performance
- Ensure sufficient GPU memory
- Regularly save model checkpoints

## License
[Add your license information here]

## Contributors
[Add contributor information]

## Acknowledgments
- TinyLlama Project
- Hugging Face Transformers
