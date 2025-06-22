# pytorch-paligemma

# VLM Training Launch Guide

## Prerequisites

### 1. Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets pillow fire tqdm wandb pandas requests
```

### 2. Verify GPU Setup (if using CUDA)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

## Step 1: Prepare Your Dataset

### Option A: Use Conceptual Captions Dataset
```bash
# Download Conceptual Captions TSV file
wget https://ai.google.com/research/ConceptualCaptions/download -O conceptual_captions.tsv

# Prepare the dataset (downloads images and creates JSONL)
python dataset_preparation.py conceptual_captions \
    --csv_path=conceptual_captions.tsv \
    --output_dir=./data/conceptual_captions \
    --output_jsonl=./data/conceptual_captions.jsonl \
    --max_samples=50000 \
    --min_image_size=224
```

### Option B: Use COCO Captions Dataset
```bash
# Download COCO dataset
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Extract
unzip train2017.zip -d ./data/coco/
unzip annotations_trainval2017.zip -d ./data/coco/

# Prepare dataset
python dataset_preparation.py coco_captions \
    --images_dir=./data/coco/train2017 \
    --annotations_file=./data/coco/annotations/captions_train2017.json \
    --output_jsonl=./data/coco_train.jsonl
```

### Option C: Combine Multiple Datasets
```bash
# Combine datasets
python dataset_preparation.py combine \
    --input_jsonl_files='["./data/conceptual_captions.jsonl", "./data/coco_train.jsonl"]' \
    --output_jsonl=./data/combined_dataset.jsonl \
    --shuffle=True
```

## Step 2: Create Train/Validation Split
```bash
python dataset_preparation.py split \
    --input_jsonl=./data/combined_dataset.jsonl \
    --train_jsonl=./data/train.jsonl \
    --val_jsonl=./data/val.jsonl \
    --val_split=0.1 \
    --shuffle=True
```

## Step 3: Verify Dataset Integrity
```bash
python dataset_preparation.py verify \
    --jsonl_file=./data/train.jsonl \
    --check_images=True
```

## Step 4: Launch Training

### Basic Training Launch
```bash
python train_vlm.py \
    --model_path="google/paligemma-3b-pt-224" \
    --train_data_path="./data/train.jsonl" \
    --val_data_path="./data/val.jsonl" \
    --output_dir="./checkpoints" \
    --batch_size=8 \
    --gradient_accumulation_steps=4 \
    --learning_rate=2e-5 \
    --num_epochs=3 \
    --warmup_steps=1000 \
    --max_length=512
```

### Training with Weights & Biases Logging
```bash
python train_vlm.py \
    --model_path="google/paligemma-3b-pt-224" \
    --train_data_path="./data/train.jsonl" \
    --val_data_path="./data/val.jsonl" \
    --output_dir="./checkpoints" \
    --batch_size=8 \
    --gradient_accumulation_steps=4 \
    --learning_rate=2e-5 \
    --num_epochs=3 \
    --warmup_steps=1000 \
    --use_wandb=True \
    --wandb_project="vlm-training" \
    --wandb_run_name="paligemma-finetune-v1"
```

### Multi-GPU Training (using torchrun)
```bash
torchrun --nproc_per_node=4 train_vlm.py \
    --model_path="google/paligemma-3b-pt-224" \
    --train_data_path="./data/train.jsonl" \
    --val_data_path="./data/val.jsonl" \
    --output_dir="./checkpoints" \
    --batch_size=4 \
    --gradient_accumulation_steps=8 \
    --learning_rate=2e-5 \
    --num_epochs=3 \
    --warmup_steps=1000
```

### Resume Training from Checkpoint
```bash
python train_vlm.py \
    --model_path="google/paligemma-3b-pt-224" \
    --train_data_path="./data/train.jsonl" \
    --val_data_path="./data/val.jsonl" \
    --output_dir="./checkpoints" \
    --batch_size=8 \
    --gradient_accumulation_steps=4 \
    --learning_rate=2e-5 \
    --num_epochs=5 \
    --resume_from_checkpoint="./checkpoints/checkpoint-epoch-2.pt"
```

## Step 5: Monitor Training

### Check Training Progress
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# View training logs
tail -f training.log

# Check checkpoint directory
ls -la ./checkpoints/
```

### Using Weights & Biases Dashboard
1. Go to https://wandb.ai/
2. Navigate to your project
3. Monitor loss curves, learning rate, and other metrics

## Step 6: Evaluate Trained Model

### Test Inference with Trained Model
```bash
python inference.py \
    --model_path="./checkpoints/best_model.pt" \
    --prompt="What is in this image?" \
    --image_file_path="test_image.jpg" \
    --max_tokens_to_generate=100 \
    --temperature=0.8 \
    --do_sample=True
```

## Recommended Hyperparameters

### For Small-Scale Training (< 100K samples)
```bash
--batch_size=16
--gradient_accumulation_steps=2
--learning_rate=5e-5
--num_epochs=5
--warmup_steps=500
```

### For Medium-Scale Training (100K - 1M samples)
```bash
--batch_size=8
--gradient_accumulation_steps=4
--learning_rate=2e-5
--num_epochs=3
--warmup_steps=1000
```

### For Large-Scale Training (> 1M samples)
```bash
--batch_size=4
--gradient_accumulation_steps=8
--learning_rate=1e-5
--num_epochs=2
--warmup_steps=2000
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory Error
```bash
# Reduce batch size and increase gradient accumulation
--batch_size=4
--gradient_accumulation_steps=8

# Or enable CPU offloading (if implemented)
--only_cpu=True
```

#### 2. Slow Data Loading
```bash
# Increase number of workers
--num_workers=8

# Pre-resize images to consistent size
python dataset_preparation.py resize_images \
    --input_dir=./data/images \
    --output_dir=./data/images_resized \
    --size=224
```

#### 3. Model Not Learning
```bash
# Check if vision encoder is properly frozen
# Increase learning rate
--learning_rate=5e-5

# Reduce warmup steps
--warmup_steps=100
```

#### 4. Training Instability
```bash
# Reduce learning rate
--learning_rate=1e-5

# Increase gradient clipping
--max_grad_norm=0.5

# Use smaller batch size
--batch_size=4
```

## Directory Structure After Setup
```
project/
├── train_vlm.py                 # Main training script
├── dataset_preparation.py       # Dataset preparation script
├── inference.py                 # Original inference script
├── modeling_gemma.py           # Model architecture
├── processing_paligemma.py     # Processor
├── utils.py                    # Utilities
├── data/
│   ├── train.jsonl            # Training data
│   ├── val.jsonl              # Validation data
│   └── images/                # Image files
├── checkpoints/               # Model checkpoints
│   ├── checkpoint-epoch-1.pt
│   ├── checkpoint-epoch-2.pt
│   ├── best_model.pt
│   └── final_model.pt
└── logs/                     # Training logs
```

## Performance Optimization Tips

1. **Use Mixed Precision Training**: Automatically enabled for CUDA
2. **Optimize Data Loading**: Use multiple workers and pin memory
3. **Gradient Accumulation**: Use larger effective batch sizes
4. **Learning Rate Scheduling**: Use cosine annealing with warmup
5. **Early Stopping**: Monitor validation loss and stop if not improving
6. **Checkpoint Regularly**: Save checkpoints every N steps
7. **Monitor GPU Memory**: Use `nvidia-smi` to check utilization

## Next Steps After Training

1. **Evaluate on Test Set**: Run inference on held-out test data
2. **Compare with Baseline**: Benchmark against original model
3. **Fine-tune Further**: Continue training on specific domains
4. **Deploy Model**: Export to ONNX or other deployment formats
5. **A/B Test**: Compare with existing models in production