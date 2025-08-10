# BERT Fine-tuning for Sentiment Analysis

This repository contains a comprehensive implementation of fine-tuning a BERT-based model (DistilBERT) for sentiment analysis using Parameter Efficient Fine-Tuning (PEFT) with LoRA (Low-Rank Adaptation).


## 🎯 Overview

This project demonstrates how to fine-tune a pre-trained DistilBERT model for binary sentiment classification (Positive/Negative) using LoRA, a parameter-efficient fine-tuning technique. The implementation achieves high accuracy while training only 0.93% of the model's parameters, making it highly efficient and cost-effective.

## ✨ Features

- **Parameter Efficient Fine-tuning**: Uses LoRA to fine-tune only 0.93% of model parameters
- **High Performance**: Achieves ~88-89% accuracy on validation data
- **Memory Efficient**: Significantly reduces memory requirements compared to full fine-tuning
- **Easy to Use**: Simple implementation with Hugging Face Transformers
- **Reproducible Results**: Complete training pipeline with evaluation metrics

## 📊 Dataset

The project uses the `shawhin/imdb-truncated` dataset:
- **Training Set**: 1,000 movie reviews
- **Validation Set**: 1,000 movie reviews
- **Classes**: Binary classification (Positive/Negative sentiment)
- **Format**: Text reviews with corresponding sentiment labels

## 🏗️ Model Architecture

- **Base Model**: DistilBERT (`distilbert-base-uncased`)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Task**: Sequence Classification
- **Target Modules**: Query linear layers (`q_lin`)
- **LoRA Configuration**:
  - Rank (r): 4
  - Alpha: 32
  - Dropout: 0.01

## 🔧 Installation

### Prerequisites
- Python 3.7+
- PyTorch
- CUDA-compatible GPU (recommended)

### Install Dependencies

```bash
pip install torch transformers datasets evaluate peft numpy
```

Or install from requirements.txt:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/prakhar175/fine-tuned-BERT.git
cd fine-tuned-BERT
```

2. **Run the Jupyter notebook**:
```bash
jupyter notebook Fine_tuning_BERT.ipynb
```


### Training Configuration

The model is trained with the following hyperparameters:

```python
# Training Parameters
learning_rate = 1e-3
batch_size = 4
num_epochs = 10
weight_decay = 0.01
max_length = 512

# LoRA Configuration
lora_config = LoraConfig(
    task_type="SEQ_CLS",
    r=4,
    lora_alpha=32,
    lora_dropout=0.01,
    bias="none",
    target_modules=["q_lin"]
)
```

## 📈 Results

### Training Metrics

| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|-----------------|----------|
| 1     | -             | 0.393           | 88.3%    |
| 2     | 0.412         | 0.482           | 86.9%    |
| 5     | 0.181         | 0.754           | 89.6%    |
| 10    | 0.006         | 0.691           | 90.6%    |

### Model Performance

- **Best Validation Accuracy**: 90.6% (Epoch 10)
- **Final Validation Accuracy**: 89.8%
- **Trainable Parameters**: 628,994 out of 67,584,004 (0.93%)
- **Training Time**: ~7.5 minutes on T4 GPU

### Sample Predictions

| Text | Predicted Sentiment |
|------|-------------------|
| "It was good." | Positive ✅ |
| "Not a fan, don't recommend." | Negative ✅ |
| "Better than the first one." | Positive ✅ |
| "This is not worth watching even once." | Negative ✅ |
| "This one is a pass." | Negative ✅ |


## 🔬 Technical Details

### LoRA Implementation
Low-Rank Adaptation (LoRA) works by:
1. Freezing the original model weights
2. Adding small trainable matrices to specific layers
3. Learning a low-rank decomposition of weight updates
4. Significantly reducing the number of trainable parameters

### Key Components

- **Tokenization**: Uses DistilBERT tokenizer with left truncation for long sequences
- **Data Collation**: Dynamic padding for efficient batch processing
- **Evaluation**: Accuracy metric with proper prediction aggregation
- **Training**: Standard sequence classification with cross-entropy loss

### Performance Optimizations

- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Dynamic Padding**: Reduces computational overhead
- **Gradient Accumulation**: Enables larger effective batch sizes
- **Learning Rate Scheduling**: Optimal learning rate decay
---

⭐ If you found this project helpful, please consider giving it a star!
