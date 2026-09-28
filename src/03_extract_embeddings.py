#!/usr/bin/env python3
"""
Extract Mammo-CLIP embeddings for longitudinal cohort.

Pipeline:
1. Load final cohort from step 2 (already includes image paths)
2. Load Mammo-CLIP EfficientNet-B5 model with pretrained weights
3. Extract class embeddings (2048-dim pooled features from B5)
4. Save embeddings organized by patient/study/laterality

Preprocessing (from Mammo-CLIP tutorial):
- Image size: [1520, 912] (H x W)
- Normalization: mean=0.3089279, std=0.25053555
- Output: [batch, 2048] feature vectors

Input: Cohort dataframes from step 2 with image paths already merged
Output: {output_dir}/{PATIENTID}/{STUDYID}/{SOPINSTANCEUID}.pt

Note: This script should be run on GPU via SLURM (see slurm_extract_embeddings.sh)
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, List, Dict
import argparse
from tqdm import tqdm
import numpy as np
from io import BytesIO
from PIL import Image
from efficientnet_pytorch import EfficientNet


# Local image loading functions (copied from hoppr to avoid complex dependencies)
def _normalize_dicom_tensor(dicom, range_min, range_max):
    """Min-max normalize tensor to [0, 1]."""
    if (range_max - range_min) == 0:
        return dicom - range_min
    img_tensor = (dicom - range_min) / (range_max - range_min)
    return img_tensor


def _local_read_jp2_file(file_path, normalize=True):
    """Read .jph/.jp2 file and return 3-channel normalized tensor."""
    if file_path.endswith(".dcm"):
        import pydicom
        ds = pydicom.dcmread(file_path)
        image_array = ds.pixel_array.astype(np.float32)
    else:
        with open(file_path, 'rb') as file:
            file_stream = BytesIO(file.read())
        jp2 = Image.open(file_stream)
        image_array = np.array(jp2).astype(np.float32)

    img_tensor = torch.tensor(image_array)

    if normalize:
        img_tensor = _normalize_dicom_tensor(img_tensor, img_tensor.min(), img_tensor.max())

    if len(img_tensor.shape) == 2:
        img_tensor = torch.stack([img_tensor, img_tensor, img_tensor])
    else:
        img_tensor = torch.permute(img_tensor, [2, 0, 1])

    return img_tensor


# Use torchvision EfficientNet (same as Mammo-CLIP base)


class MammogramEmbeddingDataset(Dataset):
    """Dataset for loading mammogram images for embedding extraction."""

    def __init__(
        self,
        cohort_df: pd.DataFrame,
        target_size: Tuple[int, int] = (1520, 912),
        mean: float = 0.3089279,
        std: float = 0.25053555408335154
    ):
        """
        Args:
            cohort_df: DataFrame with image paths and metadata
            target_size: Target size (H, W) for model input
                Default: (1520, 912) from Mammo-CLIP checkpoint config
            mean: Normalization mean from Mammo-CLIP training
            std: Normalization std from Mammo-CLIP training
        """
        self.cohort_df = cohort_df.reset_index(drop=True)
        self.target_size = target_size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.cohort_df)

    def __getitem__(self, idx):
        row = self.cohort_df.iloc[idx]
        image_path = row['fullPath_cropped_resampledBy2']

        load_failed = False
        load_error = None
        try:
            # Load image (normalized by _local_read_jp2_file)
            image = _local_read_jp2_file(image_path, normalize=True)

            # Ensure it's a torch tensor
            if not isinstance(image, torch.Tensor):
                image = torch.from_numpy(image).float()

            # Handle single channel (expand to 3 channels for RGB)
            if image.ndim == 2:
                image = image.unsqueeze(0).repeat(3, 1, 1)
            elif image.ndim == 3 and image.shape[0] == 1:
                image = image.repeat(3, 1, 1)

            # Resize to target size (always resize for consistency)
            # Using bilinear interpolation with antialias as in Mammo-CLIP training
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=self.target_size,
                mode='bilinear',
                align_corners=False,
                antialias=True
            ).squeeze(0)

            # Apply Mammo-CLIP normalization (mean=0.3089279, std=0.25053555)
            # Note: _local_read_jp2_file already normalizes to [0, 1], so we apply z-score normalization
            image = (image - self.mean) / self.std

        except Exception as e:
            # Placeholder so the DataLoader can still stack the batch; this
            # sample is flagged for exclusion downstream in process_cohort.
            # NEVER save its embedding — it would be a zero-image artifact.
            print(f"Error loading {image_path}: {e}")
            image = torch.zeros(3, *self.target_size)
            load_failed = True
            load_error = str(e)

        # Return image and metadata
        metadata = {
            'patient_id': row['PATIENTID'],
            'study_id': row['STUDYID'],
            'laterality': row.get('IMAGELATERALITY', 'UNKNOWN'),
            'view_position': row.get('VIEWPOSITION', 'UNKNOWN'),
            'sop_instance_uid': row['SOPINSTANCEUID'],
            'image_path': image_path,
            'load_failed': load_failed,
            'load_error': load_error,
        }

        return image, metadata


def custom_collate_fn(batch):
    """Custom collate function to handle metadata dicts with strings."""
    images = torch.stack([item[0] for item in batch])
    metadata = [item[1] for item in batch]  # Keep metadata as list of dicts
    return images, metadata


def load_mammo_clip_model(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Load Mammo-CLIP EfficientNet-B5 model with pretrained weights.

    Args:
        checkpoint_path: Path to checkpoint file
        device: torch device (cuda/cpu)

    Returns:
        Model in eval mode that outputs 2048-dim features
    """
    print(f"Loading Mammo-CLIP model from {checkpoint_path}...")

    # Create EfficientNet-B5 using efficientnet_pytorch (same as Mammo-CLIP)
    model = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
    model.out_dim = 2048

    # Remove final classifier to get 2048-dim features (as Mammo-CLIP does)
    model._fc = nn.Identity()

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract image_encoder weights
    image_encoder_weights = {}
    for k in ckpt['model'].keys():
        if k.startswith('image_encoder.'):
            new_key = '.'.join(k.split('.')[1:])
            image_encoder_weights[new_key] = ckpt['model'][k]

    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(image_encoder_weights, strict=False)

    print(f"Loaded {len(image_encoder_weights)} weights from checkpoint")
    if missing_keys:
        print(f"Note: {len(missing_keys)} keys not in checkpoint (expected for removed _fc layer)")

    print("Checkpoint loaded successfully!")
    print("Model outputs 2048-dim features directly")

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model




def filter_existing_embeddings(
    cohort_df: pd.DataFrame,
    output_dir: Path,
    cohort_name: str,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Drop rows whose embedding file already exists at
    {output_dir}/{PATIENTID}/{STUDYID}/{SOPINSTANCEUID}.pt.

    Filters the dataframe BEFORE the DataLoader so we skip the .jph load,
    resize, and GPU forward — saves the bulk of the work, not just the
    save-to-disk step.

    Returns
    -------
    (filtered_df, skipped_records) — `skipped_records` is a list of dicts
    suitable for writing to a manifest CSV.
    """
    if cohort_df.empty:
        return cohort_df, []

    paths = (
        output_dir
        / cohort_df["PATIENTID"].astype(str)
        / cohort_df["STUDYID"].astype(str)
        / (cohort_df["SOPINSTANCEUID"].astype(str) + ".pt")
    )
    # Path division above doesn't broadcast through pandas; build paths in a
    # plain Python list comprehension for portability.
    paths = [
        output_dir / str(pid) / str(sid) / f"{sop}.pt"
        for pid, sid, sop in zip(
            cohort_df["PATIENTID"].to_numpy(),
            cohort_df["STUDYID"].to_numpy(),
            cohort_df["SOPINSTANCEUID"].to_numpy(),
        )
    ]
    exists = np.fromiter((p.is_file() for p in paths), dtype=bool, count=len(paths))

    n_skip = int(exists.sum())
    n_new = int((~exists).sum())
    print(f"\n  --skip-existing: {cohort_name} — {n_skip} already extracted, "
          f"{n_new} new to process.")

    skipped_records = [
        {"PATIENTID": pid, "STUDYID": sid, "SOPINSTANCEUID": sop, "path": str(p)}
        for pid, sid, sop, p, ok in zip(
            cohort_df["PATIENTID"].to_numpy(),
            cohort_df["STUDYID"].to_numpy(),
            cohort_df["SOPINSTANCEUID"].to_numpy(),
            paths,
            exists,
        ) if ok
    ]
    return cohort_df[~exists].reset_index(drop=True), skipped_records


def validate_cohort_has_images(cohort_df: pd.DataFrame, cohort_name: str) -> pd.DataFrame:
    """
    Validate that cohort has image paths (should be added in step 2).

    Args:
        cohort_df: DataFrame from step 2
        cohort_name: Name for logging

    Returns:
        Validated DataFrame
    """
    print(f"\nValidating {cohort_name} cohort...")
    print(f"  Total records: {len(cohort_df)}")

    # Check for required columns
    required_cols = ['PATIENTID', 'STUDYID', 'IMAGELATERALITY', 'fullPath_cropped_resampledBy2']
    missing_cols = [col for col in required_cols if col not in cohort_df.columns]

    if missing_cols:
        raise ValueError(
            f"Missing required columns in {cohort_name}: {missing_cols}\n"
            f"Make sure step 2 merged image paths from DICOM catalogue."
        )

    # Check for missing image paths
    n_missing = cohort_df['fullPath_cropped_resampledBy2'].isna().sum()
    if n_missing > 0:
        print(f"  Warning: {n_missing} records missing image paths")
        cohort_df = cohort_df[cohort_df['fullPath_cropped_resampledBy2'].notna()].copy()

    print(f"  Valid records with images: {len(cohort_df)}")

    return cohort_df


def process_cohort(
    cohort_with_images: pd.DataFrame,
    model: nn.Module,
    device: torch.device,
    output_dir: Path,
    cohort_name: str,
    batch_size: int = 16,
    num_workers: int = 4
):
    """
    Process a cohort and extract embeddings for all images using DataLoader.

    Args:
        cohort_with_images: DataFrame with image paths
        model: Mammo-CLIP model
        device: torch device
        output_dir: Output directory for embeddings
        cohort_name: Name for logging
        batch_size: Batch size for inference
        num_workers: Number of workers for data loading
    """
    print(f"\n{'='*80}")
    print(f"EXTRACTING EMBEDDINGS: {cohort_name}")
    print(f"{'='*80}")
    print(f"  Total images: {len(cohort_with_images)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers}")

    # Create dataset and dataloader
    dataset = MammogramEmbeddingDataset(cohort_with_images)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=custom_collate_fn  # Custom collate to handle metadata strings
    )

    # Extract embeddings
    all_embeddings = []
    all_metadata = []

    model.eval()
    with torch.no_grad():
        for batch_images, batch_metadata in tqdm(dataloader, desc=f"Processing {cohort_name}"):
            # Move images to device
            batch_images = batch_images.to(device)

            # Extract 2048-dim features (model already has _fc removed)
            embeddings = model(batch_images)  # [batch, 2048]

            # Move to CPU and store (detach to avoid saving computation graph)
            all_embeddings.append(embeddings.detach().cpu())
            all_metadata.extend(batch_metadata)

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Save embeddings organized by patient/study/laterality.
    # Images that failed to load were substituted with zero tensors in the
    # dataset so the batch could be stacked; their embeddings are garbage and
    # MUST NOT be saved. Skip and track them here.
    print("\n  Saving embeddings...")
    n_saved = 0
    n_skipped_failed = 0
    failed_samples: list[dict] = []
    for i, metadata in enumerate(tqdm(all_metadata, desc="Saving")):
        if metadata.get('load_failed', False):
            n_skipped_failed += 1
            failed_samples.append({
                'patient_id': metadata['patient_id'],
                'study_id': metadata['study_id'],
                'sop_instance_uid': metadata['sop_instance_uid'],
                'image_path': metadata['image_path'],
                'error': metadata.get('load_error'),
            })
            continue

        patient_id = metadata['patient_id']
        study_id = metadata['study_id']
        laterality = metadata['laterality']
        view_pos = metadata['view_position']
        sop_uid = metadata['sop_instance_uid']

        # Create patient/study directory structure
        study_dir = output_dir / str(patient_id) / str(study_id)
        study_dir.mkdir(parents=True, exist_ok=True)

        # Filename: {SOPINSTANCEUID}.pt (globally unique identifier)
        filename = f"{sop_uid}.pt"
        output_path = study_dir / filename

        # Save embedding with metadata (clone to ensure no computation graph)
        torch.save({
            'embedding': all_embeddings[i].detach().clone(),
            'patient_id': patient_id,
            'study_id': study_id,
            'laterality': laterality,
            'view_position': view_pos,
            'sop_instance_uid': sop_uid,
            'image_path': metadata['image_path']
        }, output_path)
        n_saved += 1

    print(f"\n  Saved {n_saved} embeddings to {output_dir}")
    if n_skipped_failed:
        print(f"  SKIPPED {n_skipped_failed} samples that failed to load.")
        # Persist the failure manifest so downstream steps can audit.
        failed_df = pd.DataFrame(failed_samples)
        manifest_path = output_dir / f"{cohort_name.replace(' ', '_').lower()}_failed_loads.csv"
        failed_df.to_csv(manifest_path, index=False)
        print(f"  Failure manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract Mammo-CLIP embeddings')
    parser.add_argument('--cases-csv', type=str,
                       default='/path/to/data/FM_Mammo/hoppr_data/breast_cancer/drift_analysis/matched_cases_Density_nPriors_d335_d425_mesh.csv',
                       help='CSV file with matched cases')
    parser.add_argument('--controls-csv', type=str,
                       default='/path/to/data/FM_Mammo/hoppr_data/breast_cancer/drift_analysis/matched_controls_Density_nPriors_d335_d425_mesh.csv',
                       help='CSV file with matched controls')
    parser.add_argument('--output-dir', type=str,
                       default='/path/to/scratch/drift_cohort/embeddings',
                       help='Output directory for embeddings')
    parser.add_argument('--checkpoint', type=str,
                       default='/path/to/models/mammo_clip/b5-model-best-epoch-7.tar',
                       help='Path to Mammo-CLIP checkpoint')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for inference')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip SOPs whose .pt already exists in --output-dir. '
                            'Filters the cohort before the DataLoader, so disk '
                            'load and forward pass are both skipped. Off by '
                            'default (preserves recompute-everything behavior).')
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\n" + "="*80)
    print("MAMMO-CLIP EMBEDDING EXTRACTION")
    print("="*80)

    # Load model
    model = load_mammo_clip_model(args.checkpoint, device)

    # Load cohorts from CSV files
    print("\nLoading cohorts from CSV...")

    # Biopsied cases
    biopsied_cases = pd.read_csv(args.cases_csv)
    print(f"  Biopsied cases: {len(biopsied_cases)}")

    # Screen-negative controls
    controls = pd.read_csv(args.controls_csv)
    print(f"  Screen-negative controls: {len(controls)}")

    # Validate cohorts have image paths
    biopsied_with_images = validate_cohort_has_images(biopsied_cases, 'Biopsied Cases')
    controls_with_images = validate_cohort_has_images(controls, 'Screen-Negative Controls')

    # Optionally drop SOPs already on disk so we only extract new ones.
    if args.skip_existing:
        biopsied_with_images, cases_skipped = filter_existing_embeddings(
            biopsied_with_images, output_dir, 'Biopsied Cases'
        )
        controls_with_images, ctrls_skipped = filter_existing_embeddings(
            controls_with_images, output_dir, 'Screen-Negative Controls'
        )
        # Persist a manifest so we can audit what was reused vs newly written.
        skipped_all = (
            [{**r, "_origin": "case"} for r in cases_skipped]
            + [{**r, "_origin": "control"} for r in ctrls_skipped]
        )
        if skipped_all:
            manifest_path = output_dir / "skip_existing_manifest.csv"
            pd.DataFrame(skipped_all).to_csv(manifest_path, index=False)
            print(f"  Skip manifest ({len(skipped_all)} reused SOPs): {manifest_path}")

    # Extract embeddings for biopsied cases (save to common directory)
    process_cohort(
        biopsied_with_images,
        model,
        device,
        output_dir,
        cohort_name='Biopsied Cases',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Extract embeddings for controls (save to same directory)
    process_cohort(
        controls_with_images,
        model,
        device,
        output_dir,
        cohort_name='Screen-Negative Controls',
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print("\n" + "="*80)
    print("EMBEDDING EXTRACTION COMPLETE")
    print("="*80)
    print(f"\nEmbeddings saved to: {output_dir}")
    print("\nNext step: Compute drift velocity (04_analyze_drift.py)")


if __name__ == '__main__':
    main()
