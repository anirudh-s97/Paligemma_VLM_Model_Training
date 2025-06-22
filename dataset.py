import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from typing import List, Dict, Any
import requests
from io import BytesIO

class VisionLanguageDataset(Dataset):
    """
    Dataset class for Vision-Language training using image-text pairs.
    Supports both local images and URLs.
    
    Expected data format (JSON):
    [
        {"image": "path/to/image.jpg", "text": "A description of the image"},
        {"image": "http://example.com/image.jpg", "text": "Another description"},
        ...
    ]
    """
    
    def __init__(
        self, 
        data_path: str, 
        processor, 
        max_length: int = 512,
        image_dir: str = None
    ):
        """
        Args:
            data_path: Path to JSON file containing image-text pairs
            processor: PaliGemmaProcessor instance
            max_length: Maximum sequence length for text
            image_dir: Base directory for images (if using relative paths)
        """
        self.processor = processor
        self.max_length = max_length
        self.image_dir = image_dir or ""
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
            
        print(f"Loaded {len(self.data)} image-text pairs")
    
    def __len__(self):
        return len(self.data)
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load image from local path or URL"""
        try:
            if image_path.startswith(('http://', 'https://')):
                # Load from URL
                response = requests.get(image_path, timeout=10)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
            else:
                # Load from local path
                full_path = os.path.join(self.image_dir, image_path)
                image = Image.open(full_path)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            return image
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank RGB image as fallback
            return Image.new('RGB', (224, 224), color='white')
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Load image
        image = self.load_image(item['image'])
        text = item['text']
        
        # Process through PaliGemmaProcessor
        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            truncation=True
        )
        
        # Extract tensors and remove batch dimension (added by processor)
        input_ids = inputs['input_ids'].squeeze(0)  # [seq_len]
        attention_mask = inputs['attention_mask'].squeeze(0)  # [seq_len]
        pixel_values = inputs['pixel_values'].squeeze(0)  # [3, H, W]
        
        # Truncate if needed
        if input_ids.size(0) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pixel_values': pixel_values,
            'labels': input_ids.clone()  # For causal LM, labels are same as input_ids
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle variable length sequences
    """
    # Get max sequence length in this batch
    max_len = max(item['input_ids'].size(0) for item in batch)
    
    # Pad sequences
    input_ids = []
    attention_masks = []
    labels = []
    pixel_values = []
    
    for item in batch:
        seq_len = item['input_ids'].size(0)
        padding_len = max_len - seq_len
        
        # Pad input_ids and labels with pad_token_id (assuming 0)
        padded_input_ids = torch.cat([
            item['input_ids'], 
            torch.zeros(padding_len, dtype=torch.long)
        ])
        
        padded_labels = torch.cat([
            item['labels'], 
            torch.full((padding_len,), -100, dtype=torch.long)  # -100 is ignored in loss
        ])
        
        # Pad attention_mask with 0s
        padded_attention_mask = torch.cat([
            item['attention_mask'], 
            torch.zeros(padding_len, dtype=torch.long)
        ])
        
        input_ids.append(padded_input_ids)
        labels.append(padded_labels)
        attention_masks.append(padded_attention_mask)
        pixel_values.append(item['pixel_values'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'pixel_values': torch.stack(pixel_values),
        'labels': torch.stack(labels)
    }


class CC3MDataset(VisionLanguageDataset):
    """
    Specialized dataset class for Conceptual Captions 3M dataset
    """
    
    def __init__(self, data_path: str, processor, max_length: int = 512):
        """
        Args:
            data_path: Path to CC3M TSV file
            processor: PaliGemmaProcessor instance
            max_length: Maximum sequence length
        """
        self.processor = processor
        self.max_length = max_length
        
        # Load CC3M data (TSV format: caption \t url)
        self.data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    caption, url = parts[0], parts[1]
                    self.data.append({'text': caption, 'image': url})
        
        print(f"Loaded {len(self.data)} CC3M pairs")


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """
    Create a sample dataset file for testing
    """
    sample_data = []
    
    # Using placeholder images from picsum for demo
    for i in range(num_samples):
        sample_data.append({
            "image": f"https://picsum.photos/224/224?random={i}",
            "text": f"This is a sample image description number {i}. It shows various objects and scenes."
        })
    
    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"Created sample dataset with {num_samples} entries at {output_path}")


if __name__ == "__main__":
    # Create a sample dataset for testing
    create_sample_dataset("sample_data.json", 50)