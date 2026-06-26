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
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        
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
    
    def process_images_dual(
        self,
        input_dir: str,
        eva02_output_dir: Optional[str] = None,
        blip2_output_dir: Optional[str] = None,
        batch_size: int = 8,
    ) -> dict:
        """
        Extract embeddings with BOTH EVA-02 and BLIP-2 (Q-Former) in one pass.

        For each keyframe folder, writes:
        - {eva02_output_dir}/{video}.npy    — shape (N, 768), EVA-02 image features
        - {blip2_output_dir}/{video}.npy    — shape (N, 768), BLIP-2 Q-Former features

        Resume: a folder is skipped only if BOTH outputs exist. The two streams
        stay in lockstep (same N, same image order) because they share the
        same image list per folder.

        Args:
            input_dir: Directory containing keyframe subfolders
            eva02_output_dir: Output dir for EVA-02 .npy (defaults to settings)
            blip2_output_dir: Output dir for BLIP-2 .npy (defaults to settings)
            batch_size: Batch size (smaller than single-stream — BLIP-2 is heavy)

        Returns:
            Dict with processing statistics
        """
        from aic26.core.models import ModelRegistry  # local import to avoid cycle

        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        eva02_out = Path(eva02_output_dir) if eva02_output_dir else Path(self.config.embedding_dir)
        blip2_out = Path(blip2_output_dir) if blip2_output_dir else Path(self.config.blip2_embedding_dir)
        ensure_dir(eva02_out)
        ensure_dir(blip2_out)

        # Lazy-load BLIP-2 via the shared ModelRegistry path used by core.
        # We use a private instance here so pipeline can run without spinning
        # up the full search-side registry (EVA-02 is already loaded in self).
        blip2 = ModelRegistry.__new__(ModelRegistry)
        blip2.device = self.device
        blip2._load_blip2()
        # We pass the ModelRegistry instance to _flush_dual_batch because
        # encode_images_blip2_batch lives there (needs the processor and the
        # shared Q-Former model together).

        folders = sorted(
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        )
        logger.info(
            f"Dual embedding: {len(folders)} folder(s), "
            f"eva02 -> {eva02_out}, blip2 -> {blip2_out}"
        )

        stats = {
            "folders_total": len(folders),
            "folders_processed": 0,
            "images_total": 0,
            "dual": True,
        }

        for folder in tqdm(folders, desc="Dual embedding folders", unit="folder"):
            eva02_file = eva02_out / f"{folder}.npy"
            blip2_file = blip2_out / f"{folder}.npy"
            if eva02_file.exists() and blip2_file.exists():
                logger.debug(f"Skipping (exists): {folder}")
                continue

            try:
                n_imgs = self._embed_image_folder_dual(
                    input_path / folder, eva02_file, blip2_file,
                    blip2, batch_size,
                )
                stats["folders_processed"] += 1
                stats["images_total"] += n_imgs
            except Exception as e:
                logger.error(f"Failed dual-embed {folder}: {e}")

        logger.info(
            f"Dual image embeddings complete: "
            f"{stats['folders_processed']}/{stats['folders_total']} folders, "
            f"{stats['images_total']} images"
        )
        return stats

    def _embed_image_folder_dual(
        self,
        folder_path: Path,
        eva02_save: Path,
        blip2_save: Path,
        models_registry: "ModelRegistry",
        batch_size: int,
    ) -> int:
        """
        Run EVA-02 + BLIP-2 over a single folder, writing both .npy files.

        The two arrays are written in lockstep (same row order = same image).
        """
        image_files = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        if not image_files:
            return 0

        eva02_chunks: list = []
        blip2_chunks: list = []
        batch_images_eva: list = []
        batch_paths_blip: list = []

        for img_file in image_files:
            img_path = folder_path / img_file
            try:
                # EVA-02 branch
                image = Image.open(img_path).convert("RGB")
                batch_images_eva.append(self.preprocess_val(image))

                # BLIP-2 branch — keep paths for deferred open (PIL is cheap,
                # but we want the two batches to align on the same images).
                batch_paths_blip.append(img_path)

                flush = len(batch_images_eva) >= batch_size
                if flush:
                    self._flush_dual_batch(
                        batch_images_eva, batch_paths_blip,
                        eva02_chunks, blip2_chunks,
                        models_registry,
                    )
                    batch_images_eva, batch_paths_blip = [], []
            except Exception as e:
                logger.warning(f"Dual embed error with {img_file}: {e}")

        if batch_images_eva:
            self._flush_dual_batch(
                batch_images_eva, batch_paths_blip,
                eva02_chunks, blip2_chunks,
                models_registry,
            )

        if eva02_chunks and blip2_chunks:
            eva02_arr = np.concatenate(eva02_chunks, axis=0)
            blip2_arr = np.concatenate(blip2_chunks, axis=0)
            assert eva02_arr.shape[0] == blip2_arr.shape[0], (
                f"Stream length mismatch: eva02={eva02_arr.shape} blip2={blip2_arr.shape}"
            )
            np.save(eva02_save, eva02_arr)
            np.save(blip2_save, blip2_arr)
            return eva02_arr.shape[0]
        return 0

    def _flush_dual_batch(
        self,
        eva02_images,
        blip2_paths,
        eva02_chunks: list,
        blip2_chunks: list,
        models_registry: "ModelRegistry",
    ) -> None:
        """Run one EVA-02 + BLIP-2 batch and append to chunk lists."""
        import torch

        # EVA-02
        batch_tensor = torch.stack(eva02_images).to(self.device)
        with torch.no_grad():
            feats_eva = self.model.encode_image(batch_tensor)
            feats_eva = F.normalize(feats_eva, dim=-1)
        eva02_chunks.append(feats_eva.cpu().numpy())

        # BLIP-2 — batch via ModelRegistry helper for consistent pooling
        feats_blip = models_registry.encode_images_blip2_batch(blip2_paths)
        blip2_chunks.append(feats_blip)

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
