# Poultry Disease Detection via Bioacoustic Analysis

> An end-to-end pipeline for early disease detection in commercial 
> poultry farms using audio analysis, built during my internship 
> at [Kuppismart Soltuions PVT LTD].

## Problem Statement

Commercial poultry farms in South India lack access to regular 
veterinary monitoring. Early signs of respiratory disease 
(Newcastle, IB, CRD) often manifest acoustically before visible 
symptoms appear. This project explores whether continuous audio 
monitoring can flag at-risk pens for early intervention.

## Dataset

- **Public benchmark**: Mendeley poultry sound dataset 
  (healthy/unhealthy/noise, used for transfer learning)
- **Proprietary farm data**: 5TB of continuous 10-second audio 
  clips across multiple pens, collected over 6+ months
- **Labeled subset**: 535 clips, multi-label annotated by 3 
  independent annotators

## Approach

### 1. Baseline Model (Transfer Learning)
Trained ResNet18 on Mendeley dataset → 95.9% accuracy
Establishes feasibility of CNN-based spectrogram classification.

### 2. Data Quality Analysis
Found significant challenges with real-world farm data:
- Severe class imbalance (96 flagged clips vs 1400+ normal)
- Low inter-annotator agreement (5/96 clips had 2+ agreement)
- No vaccination/mortality records — vaccination reactions 
  can acoustically mimic disease

### 3. Anomaly Detection Pipeline
Built unsupervised Isolation Forest pipeline to rank clips by 
acoustic deviation from baseline, producing a prioritized list 
for veterinary review.

### 4. Temporal Episode Analysis
Distinguished likely disease episodes from vaccination reactions 
using duration-based heuristics:
- Short anomaly streaks (1-3 days) → likely vaccination reaction
- Sustained anomaly streaks (5+ days) → likely disease episode

Identified a 12-day sustained anomaly cluster in Pen 2-a 
(Sep 20 - Oct 1, 2025), consistent with a real disease episode.

## Results

| Stage | Metric |
|---|---|
| Mendeley baseline | 95.9% accuracy |
| Farm data inter-annotator agreement | 5/96 consensus |

## Tech Stack

- **Audio processing**: librosa
- **Deep Learning**: PyTorch, torchvision (ResNet18)
- **Anomaly Detection**: scikit-learn (Isolation Forest)
- **Cloud**: AWS S3, SageMaker (for 5TB processing)
- **Data**: pandas, openpyxl

## Project Structure
