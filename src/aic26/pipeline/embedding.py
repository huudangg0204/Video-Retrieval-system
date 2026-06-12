"""
EVA-02 embedding extraction from keyframes.

Extracts CLIP embeddings for keyframe images and captions
using the EVA-02 model with batch processing.

Migrated from: get_embedding_eva02.py, get_embedding_eva02_from_caption.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import open_clip

from loguru import logger

from aic26.config import Settings
from aic26.utils.file_utils import ensure_dir


class EmbeddingExtractor:
    """
    Extract EVA-02 embeddings from keyframe images or captions.
    
    Supports:
    - Image embedding extraction
    - Caption/text embedding extraction
    - Batch processing for GPU efficiency
    - Resume capability
    
    Usage:
        extractor = EmbeddingExtractor(settings)
        extractor.process_images("./keyframes", "./embeddings")
    """
    
    def __init__(self, config: Optional['Settings'] = None) -> None:
        """
        Initialize the embedding extractor.
        
        Args:
            config: Settings instance
        """
        if config is None:
            from aic26.config import settings as global_settings
            config = global_settings
        
        self.config = config
        self.device = torch.device("cpu")
        
        # Load model
        logger.info(f"Loading EVA-02 model on {self.device}...")
        self.model, self.preprocess_train, self.preprocess_val = (
            open_clip.create_model_and_transforms(
                self.config.eva02_model_name,
                pretrained=self.config.eva02_pretrained,
            )
        )
        self.tokenizer = open_clip.get_tokenizer(self.config.eva02_model_name)
        self.model = self.model.to(self.device).eval()
        logger.info("EVA-02 model ready.")
    
    def process_images(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        batch_size: int = 8,
    ) -> dict:
        """
        Extract embeddings from keyframe images.
        
        Args:
            input_dir: Directory containing video subfolders with keyframes
            output_dir: Output directory for .npy embedding files
            batch_size: Batch size for GPU processing
            
        Returns:
            Dict with processing statistics
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_path = Path(output_dir) if output_dir else Path(self.config.embedding_dir)
        ensure_dir(output_path)
        
        # Find video folders
        folders = sorted(
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        )
        
        logger.info(f"Processing {len(folders)} folder(s) for image embeddings...")
        
        stats = {"folders_total": len(folders), "folders_processed": 0, "images_total": 0}
        
        for folder in tqdm(folders, desc="Embedding folders", unit="folder"):
            folder_path = os.path.join(input_dir, folder)
            save_file = os.path.join(str(output_path), f"{folder}.npy")
            
            if os.path.exists(save_file):
                logger.debug(f"Skipping (exists): {folder}")
                continue
            
            try:
                n_images = self._embed_image_folder(
                    folder_path, save_file, batch_size
                )
                stats["folders_processed"] += 1
                stats["images_total"] += n_images
                
            except Exception as e:
                logger.error(f"Failed to process {folder}: {e}")
        
        logger.info(
            f"Image embeddings complete: "
            f"{stats['folders_processed']}/{stats['folders_total']} folders, "
            f"{stats['images_total']} images"
        )
        
        return stats
    
    def process_captions(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        batch_size: int = 1,
    ) -> dict:
        """
        Extract embeddings from caption JSON files.
        
        Args:
            input_dir: Directory containing video subfolders with caption JSONs
            output_dir: Output directory for .npy embedding files
            batch_size: Batch size for GPU processing
            
        Returns:
            Dict with processing statistics
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_path = Path(output_dir) if output_dir else Path(
            str(self.config.embedding_dir).replace("embeddings", "caption_embeddings")
        )
        ensure_dir(output_path)
        
        folders = sorted(
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        )
        
        logger.info(f"Processing {len(folders)} folder(s) for caption embeddings...")
        
        stats = {"folders_total": len(folders), "folders_processed": 0, "captions_total": 0}
        
        for folder in tqdm(folders, desc="Caption folders", unit="folder"):
            folder_path = os.path.join(input_dir, folder)
            save_file = os.path.join(str(output_path), f"{folder}.npy")
            
            if os.path.exists(save_file):
                logger.debug(f"Skipping (exists): {folder}")
                continue
            
            try:
                n_captions = self._embed_caption_folder(
                    folder_path, save_file, batch_size
                )
                stats["folders_processed"] += 1
                stats["captions_total"] += n_captions
                
            except Exception as e:
                logger.error(f"Failed to process {folder}: {e}")
        
        logger.info(
            f"Caption embeddings complete: "
            f"{stats['folders_processed']}/{stats['folders_total']} folders, "
            f"{stats['captions_total']} captions"
        )
        
        return stats
    
    def _embed_image_folder(
        self,
        folder_path: str,
        save_path: str,
        batch_size: int,
    ) -> int:
        """
        Extract embeddings for all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            save_path: Path to save .npy file
            batch_size: Batch size
            
        Returns:
            Number of images processed
        """
        image_files = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        
        if not image_files:
            return 0
        
        embeddings = []
        batch_images = []
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            
            try:
                image = Image.open(img_path).convert("RGB")
                batch_images.append(self.preprocess_val(image))
                
                if len(batch_images) == batch_size:
                    batch_tensor = torch.stack(batch_images).to(self.device)
                    with torch.no_grad():
                        feats = self.model.encode_image(batch_tensor)
                        feats = F.normalize(feats, dim=-1)
                    embeddings.append(feats.cpu().numpy())
                    batch_images = []
                    
            except Exception as e:
                logger.warning(f"Error with {img_file}: {e}")
        
        # Process remaining batch
        if batch_images:
            batch_tensor = torch.stack(batch_images).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_image(batch_tensor)
                feats = F.normalize(feats, dim=-1)
            embeddings.append(feats.cpu().numpy())
        
        if embeddings:
            combined = np.concatenate(embeddings, axis=0)
            np.save(save_path, combined)
            return len(combined)
        
        return 0
    
    def _embed_caption_folder(
        self,
        folder_path: str,
        save_path: str,
        batch_size: int,
    ) -> int:
        """
        Extract embeddings for captions in a folder.
        
        Args:
            folder_path: Path to folder containing JSON caption files
            save_path: Path to save .npy file
            batch_size: Batch size
            
        Returns:
            Number of captions processed
        """
        # Find JSON files
        json_files = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith('.json')
        )
        
        if not json_files:
            return 0
        
        json_file = os.path.join(folder_path, json_files[0])
        
        # Load captions
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                text = f.read()
            
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    data = [data]
            except json.JSONDecodeError:
                # Try JSON Lines format
                data = []
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                for line in lines:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                
                if not data:
                    # Try concatenated objects
                    try:
                        data = json.loads(
                            "[" + text.replace("}\n{", "},{").replace("}{", "},{") + "]"
                        )
                    except json.JSONDecodeError:
                        raise ValueError(f"Failed to parse JSON: {json_file}")
            
        except Exception as e:
            logger.error(f"Failed to load {json_file}: {e}")
            return 0
        
        embeddings = []
        batch_captions = []
        
        for item in data:
            try:
                caption = item.get("caption", "")
                if not isinstance(caption, str):
                    caption = str(caption)
                batch_captions.append(caption)
                
                if len(batch_captions) == batch_size:
                    tokens = self.tokenizer(batch_captions).to(self.device)
                    with torch.no_grad():
                        feats = self.model.encode_text(tokens)
                        feats = F.normalize(feats, dim=-1)
                    embeddings.append(feats.cpu().numpy())
                    batch_captions = []
                    
            except Exception as e:
                logger.warning(f"Error with caption item: {e}")
        
        # Process remaining
        if batch_captions:
            tokens = self.tokenizer(batch_captions).to(self.device)
            with torch.no_grad():
                feats = self.model.encode_text(tokens)
                feats = F.normalize(feats, dim=-1)
            embeddings.append(feats.cpu().numpy())
        
        if embeddings:
            combined = np.concatenate(embeddings, axis=0)
            np.save(save_path, combined)
            return len(combined)
        
        return 0
