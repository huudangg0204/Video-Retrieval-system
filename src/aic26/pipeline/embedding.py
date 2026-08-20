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


class KeyframeImageDataset(torch.utils.data.Dataset):
    """
    Dataset for loading and preprocessing keyframes across multiple CPU workers.
    """
    def __init__(self, image_paths: list[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            return image
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 224, 224), dtype=torch.float32)


class DualKeyframeImageDataset(torch.utils.data.Dataset):
    """
    Dataset for dual EVA-02 + CLIP-32 keyframe extraction.
    Loads raw PIL images and transforms for EVA-02 while keeping image paths for CLIP-32.
    """
    def __init__(self, image_paths: list[str], eva_transform=None):
        self.image_paths = image_paths
        self.eva_transform = eva_transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            eva_tensor = self.eva_transform(image) if self.eva_transform is not None else image
            return eva_tensor, img_path
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 224, 224), dtype=torch.float32), img_path


class EmbeddingExtractor:
    """
    Extract EVA-02 embeddings from keyframe images or captions.
    
    Supports:
    - Image embedding extraction using multi-worker DataLoader per folder
    - Caption/text embedding extraction
    - Batch processing for GPU efficiency
    - Resume capability
    
    Usage:
        extractor = EmbeddingExtractor(settings)
        extractor.process_images("./keyframes", "./embeddings", num_workers=4)
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
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> dict:
        """
        Extract embeddings from keyframe images sequentially folder-by-folder,
        using DataLoader with num_workers to parallelize preprocessing per folder.
        
        Args:
            input_dir: Directory containing video subfolders with keyframes
            output_dir: Output directory for .npy embedding files
            batch_size: Batch size for GPU processing
            num_workers: Number of CPU worker processes per folder
            
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
        
        # Filter folders that already have embedding files
        pending_folders = [
            f for f in folders
            if not os.path.exists(os.path.join(str(output_path), f"{f}.npy"))
        ]
        already_done = len(folders) - len(pending_folders)
        
        logger.info(
            f"Found {len(folders)} folder(s): {already_done} already done (skipped), "
            f"{len(pending_folders)} to process ({num_workers} CPU workers/folder)..."
        )
        
        stats = {
            "folders_total": len(folders),
            "folders_processed": already_done,
            "images_total": 0,
        }
        
        if not pending_folders:
            logger.info("All embeddings already exist. Nothing to do.")
            return stats
        
        for idx, folder in enumerate(pending_folders, start=1):
            folder_path = os.path.join(input_dir, folder)
            save_file = os.path.join(str(output_path), f"{folder}.npy")
            try:
                n_images = self._embed_image_folder(
                    folder_path,
                    save_file,
                    batch_size,
                    num_workers=num_workers,
                    folder_idx=idx,
                    total_folders=len(pending_folders),
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
        clip32_output_dir: Optional[str] = None,
        batch_size: int = 8,
        num_workers: int = 4,
    ) -> dict:
        """
        Extract embeddings with BOTH EVA-02 and CLIP ViT-B-32 in one pass.

        For each keyframe folder, writes:
        - {eva02_output_dir}/{video}.npy    — shape (N, 768), EVA-02 image features
        - {clip32_output_dir}/{video}.npy   — shape (N, 512), CLIP ViT-B-32 image features

        Resume: a folder is skipped only if BOTH outputs exist.

        Args:
            input_dir: Directory containing keyframe subfolders
            eva02_output_dir: Output dir for EVA-02 .npy (defaults to settings)
            clip32_output_dir: Output dir for CLIP-32 .npy (defaults to settings)
            batch_size: Batch size
            num_workers: Number of CPU worker processes per folder

        Returns:
            Dict with processing statistics
        """
        from aic26.core.models import ModelRegistry  # local import to avoid cycle
        from transformers import CLIPModel, CLIPProcessor

        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        eva02_out = Path(eva02_output_dir) if eva02_output_dir else Path(self.config.embedding_dir)
        clip32_out = Path(clip32_output_dir) if clip32_output_dir else Path(self.config.clip32_embedding_dir)
        ensure_dir(eva02_out)
        ensure_dir(clip32_out)

        # Load CLIP ViT-B-32 for the second stream
        logger.info(f"Loading CLIP ViT-B-32: {self.config.clip32_model_name}")
        clip32_processor = CLIPProcessor.from_pretrained(self.config.clip32_model_name)
        clip32_model = (
            CLIPModel.from_pretrained(self.config.clip32_model_name)
            .to(self.device)
            .eval()
        )
        folders = sorted(
            d for d in os.listdir(input_dir)
            if os.path.isdir(os.path.join(input_dir, d))
        )
        
        pending_folders = [
            f for f in folders
            if not ((eva02_out / f"{f}.npy").exists() and (clip32_out / f"{f}.npy").exists())
        ]
        already_done = len(folders) - len(pending_folders)

        logger.info(
            f"Dual embedding: {len(folders)} folder(s): {already_done} already done, "
            f"{len(pending_folders)} to process ({num_workers} CPU workers/folder), "
            f"eva02 -> {eva02_out}, clip32 -> {clip32_out}"
        )

        stats = {
            "folders_total": len(folders),
            "folders_processed": already_done,
            "images_total": 0,
            "dual": True,
        }

        if not pending_folders:
            logger.info("All dual embeddings already exist. Nothing to do.")
            return stats

        for idx, folder in enumerate(pending_folders, start=1):
            eva02_file = eva02_out / f"{folder}.npy"
            clip32_file = clip32_out / f"{folder}.npy"
            try:
                n_imgs = self._embed_image_folder_dual(
                    input_path / folder,
                    eva02_file,
                    clip32_file,
                    clip32_model,
                    clip32_processor,
                    batch_size,
                    num_workers=num_workers,
                    folder_idx=idx,
                    total_folders=len(pending_folders),
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
        clip32_save: Path,
        clip32_model: "CLIPModel",
        clip32_processor: "CLIPProcessor",
        batch_size: int,
        num_workers: int = 4,
        folder_idx: int = 1,
        total_folders: int = 1,
    ) -> int:
        """
        Run EVA-02 + CLIP ViT-B-32 over a single folder using DataLoader, writing both .npy files.
        """
        image_files = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        if not image_files:
            return 0

        image_paths = [str(folder_path / f) for f in image_files]
        dataset = DualKeyframeImageDataset(image_paths, eva_transform=self.preprocess_val)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        eva02_chunks: list = []
        clip32_chunks: list = []

        folder_name = folder_path.name
        with tqdm(
            total=len(image_files),
            desc=f"[{folder_idx}/{total_folders}] {folder_name}",
            unit="img",
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
            for eva_batch, paths in loader:
                # EVA-02
                eva_batch = eva_batch.to(self.device, non_blocking=True)
                with torch.no_grad():
                    feats_eva = self.model.encode_image(eva_batch)
                    feats_eva = F.normalize(feats_eva, dim=-1)
                eva02_chunks.append(feats_eva.cpu().numpy())

                # CLIP ViT-B-32
                images = [Image.open(p).convert("RGB") for p in paths]
                inputs = clip32_processor(images=images, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    feats_clip = clip32_model.get_image_features(**inputs)
                    if hasattr(feats_clip, 'pooler_output'):
                        feats_clip = feats_clip.pooler_output
                    elif isinstance(feats_clip, tuple):
                        feats_clip = feats_clip[0]
                    feats_clip = F.normalize(feats_clip, dim=-1)
                clip32_chunks.append(feats_clip.cpu().numpy())
                pbar.update(len(eva_batch))

        if eva02_chunks and clip32_chunks:
            eva02_arr = np.concatenate(eva02_chunks, axis=0)
            clip32_arr = np.concatenate(clip32_chunks, axis=0)
            assert eva02_arr.shape[0] == clip32_arr.shape[0], (
                f"Stream length mismatch: eva02={eva02_arr.shape} clip32={clip32_arr.shape}"
            )
            np.save(eva02_save, eva02_arr)
            np.save(clip32_save, clip32_arr)
            return eva02_arr.shape[0]
        return 0

    def _embed_image_folder(
        self,
        folder_path: str,
        save_path: str,
        batch_size: int = 16,
        num_workers: int = 4,
        folder_idx: int = 1,
        total_folders: int = 1,
    ) -> int:
        """
        Extract embeddings for all images in a folder using DataLoader with CPU workers.
        
        Args:
            folder_path: Path to folder containing images
            save_path: Path to save .npy file
            batch_size: Batch size
            num_workers: Number of CPU workers for image loading and preprocessing
            folder_idx: Current folder index
            total_folders: Total number of folders to process
            
        Returns:
            Number of images processed
        """
        image_files = sorted(
            f for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        )
        if not image_files:
            return 0

        image_paths = [os.path.join(folder_path, f) for f in image_files]
        dataset = KeyframeImageDataset(image_paths, transform=self.preprocess_val)
        
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

        folder_name = os.path.basename(folder_path)
        embeddings = []
        with tqdm(
            total=len(image_files),
            desc=f"[{folder_idx}/{total_folders}] {folder_name}",
            unit="img",
            dynamic_ncols=True,
            leave=True,
        ) as pbar:
            with torch.no_grad():
                for batch_tensor in loader:
                    batch_tensor = batch_tensor.to(self.device, non_blocking=True)
                    feats = self.model.encode_image(batch_tensor)
                    feats = F.normalize(feats, dim=-1)
                    embeddings.append(feats.cpu().numpy())
                    pbar.update(len(batch_tensor))

        if embeddings:
            combined = np.concatenate(embeddings, axis=0)
            np.save(save_path, combined)
            return len(combined)

        return 0
    

