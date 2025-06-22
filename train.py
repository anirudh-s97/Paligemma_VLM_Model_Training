import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import os
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer
import wandb
from typing import Dict, Optional

from modeling_gemma import PaliGemmaForConditionalGeneration, PaliGemmaConfig
from modeling_siglip import SiglipVisionConfig
from processing_paligemma import PaliGemmaProcessor
from dataset import VisionLanguageDataset, collate_fn, CC3MDataset


class PaliGemmaTrainer:
    def __init__(
        self,
        model: PaliGemmaForConditionalGeneration,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        log_wandb: bool = False,
        freeze_vision_encoder: bool = True
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.save_dir = save_dir
        self.log_wandb = log_wandb
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Freeze vision encoder if specified
        if freeze_vision_encoder:
            self.freeze_vision_encoder()
        
        # Setup optimizer and scheduler
        if optimizer is None:
            self.optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=1e-4,
                weight_decay=0.01
            )
        else:
            self.optimizer = optimizer
            
        if scheduler is None:
            self.scheduler = CosineAnnealingLR(
                self.optimizer, 
                T_max=len(train_dataloader) * 3,  # 3 epochs
                eta_min=1e-6
            )
        else:
            self.scheduler = scheduler
        
        # Initialize wandb if specified
        if self.log_wandb:
            wandb.init(
                project="paligemma-training",
                config={
                    "learning_rate": self.optimizer.param_groups[0]['lr'],
                    "batch_size": train_dataloader.batch_size,
                    "freeze_vision_encoder": freeze_vision_encoder
                }
            )
    
    def freeze_vision_encoder(self):
        """Freeze the vision encoder parameters"""
        print("Freezing vision encoder...")
        for param in self.model.vision_tower.parameters():
            param.requires_grad = False
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {epoch+1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to wandb
            if self.log_wandb and batch_idx % 100 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "learning_rate": self.scheduler.get_last_lr()[0],
                    "epoch": epoch,
                    "step": epoch * num_batches + batch_idx
                })
        
        return {"train_loss": total_loss / num_batches}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model"""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validating", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    pixel_values=batch['pixel_values'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                total_loss += outputs['loss'].item()
        
        avg_loss = total_loss / num_batches
        
        if self.log_wandb:
            wandb.log({
                "val_loss": avg_loss,
                "epoch": epoch
            })
        
        return {"val_loss": avg_loss}
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics
        }, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self, num_epochs: int, save_every: int = 1):
        """Main training loop"""
        print(f"Starting training for {num_epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Print epoch summary
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                print(f"Val Loss: {val_metrics['val_loss']:.4f}")
                
                # Save best model
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    best_checkpoint_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.model.state_dict(), best_checkpoint_path)
                    print(f"New best model saved: {best_checkpoint_path}")
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(epoch, metrics)
        
        print("Training completed!")


def load_config(config_path: str) -> PaliGemmaConfig:
    """Load model configuration from JSON file"""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    # Create vision config
    vision_config = SiglipVisionConfig(**config_dict['vision_config'])
    
    # Create PaliGemma config
    config = PaliGemmaConfig(
        vision_config=config_dict['vision_config'],
        text_config=config_dict['text_config'],
        **{k: v for k, v in config_dict.items() if k not in ['vision_config', 'text_config']}
    )
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train PaliGemma Vision-Language Model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to training data JSON file')
    parser.add_argument('--val_data_path', type=str, help='Path to validation data JSON file')
    parser.add_argument('--config_path', type=str, required=True, help='Path to model config JSON file')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to tokenizer')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--freeze_vision', action='store_true', help='Freeze vision encoder')
    parser.add_argument('--wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--image_dir', type=str, help='Base directory for images')
    parser.add_argument('--dataset_type', type=str, choices=['json', 'cc3m'], default='json', help='Dataset type')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    config = load_config(args.config_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Create processor
    processor = PaliGemmaProcessor(
        tokenizer=tokenizer,
        num_image_tokens=config.text_config.num_image_tokens,
        image_size=config.vision_config.image_size
    )
    
    # Create datasets
    if args.dataset_type == 'cc3m':
        train_dataset = CC3MDataset(
            data_path=args.data_path,
            processor=processor,
            max_length=args.max_length
        )
    else:
        train_dataset = VisionLanguageDataset(
            data_path=args.data_path,
            processor=processor,
            max_length=args.max_length,
            image_dir=args.image_dir
        )
    
    val_dataset = None
    if args.val_data_path:
        if args.dataset_type == 'cc3m':
            val_dataset = CC3MDataset(
                data_path=args.val_data_path,
                processor=processor,
                max_length=args.max_length
            )
        else:
            val_dataset = VisionLanguageDataset(
                data_path=args.val_data_path,
                processor=processor,
                max_length=args.max_length,
                image_dir=args.image_dir
            )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
    
    # Create model
    model = PaliGemmaForConditionalGeneration(config)
    
    # Create optimizer
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=len(train_dataloader) * args.num_epochs,
        eta_min=1e-6
    )
    
    # Create trainer
    trainer = PaliGemmaTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=args.save_dir,
        log_wandb=args.wandb,
        freeze_vision_encoder=args.freeze_vision
    )
    
    # Start training
    trainer.train(num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()