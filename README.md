# Sample identification: Fine-tuned sampleid model for better music sample localization

## Abstract
Temporal audio alignment and sample identification face a core challenge when processing heavily repeating or looping sounds, often resulting in destructive intra-class repulsion during contrastive representation learning. In this repository, we introduce SINCERE-Audio, an advanced temporal audio localization framework built upon the Sony SampleID backbone. To solve the alignment paradox, we design a novel dual-contrastive learning objective comprising a Dense Cross-Batch InfoNCE loss for precise temporal localization and a SINCERE (SupCon) periodicity alignment loss to preserve acoustic semantics among identical loops. Furthermore, we redefine the reference input paradigm by optimizing the network to process noisy contextual audio containing multiple stems rather than clean, isolated samples. By preserving a high-dimensional feature manifold and leveraging PyTorch-native GPU digital signal processing, this framework achieves sub-frame temporal precision and reaches a state-of-the-art 72.3 mAP on the evaluation benchmark.

## Architectural and Pipeline Additions
This implementation introduces several critical modifications and additions to the baseline SampleID architecture:

* **Dual-Contrastive Loss Framework:** The pipeline optimizes two independent objectives simultaneously. `DenseCrossBatchInfoNCE` penalizes timing errors to create highly localized temporal probability distributions, while `SINCERELoss` ensures identical repeating beats are not mathematically repelled in the embedding space.
* **Acoustic Front-End & Manifold Preservation:** The system utilizes `GeMPooledSampleID` with Generalized Mean (GeM) Pooling (`GeMPoolFreq`). To prevent representational bottlenecks, the 2048-dimensional manifold is strictly maintained throughout the pipeline. The intermediate Temporal Projection layer (Conv1d 2048 to 2048) has been explicitly removed.
* **Parameter Efficient Fine-Tuning:** Weight-Decomposed Low-Rank Adaptation (DoRA) is applied and explicitly scaled to a rank of 128 to prevent 1x1 convolution bottlenecks across the deep embedding spaces.
* **Noisy Contextual References:** The model evaluates the "Reference" input as noisy contextual audio with multiple stems rather than a clean, isolated sample, improving robustness in complex mixtures.
* **GPU-Native DSP Pipeline:** CPU-bound audio processing is replaced with `TorchTimeStretcher`, a PyTorch-native Phase Vocoder that executes Fast Fourier Transforms (FFT) directly on the GPU. This is combined with Ideal Ratio Mask (IRM) application for spectral ducking and time-stretching without breaking the auto-grad graph.
* **Data Curriculum:** The `InfalliblePairSampler` handles target observability gating via strict SNR thresholds (inflection point of -15.0 dB) and scales continuous loops within a strict 15-second maximum contextual length.

## Setup
Clone the repository and install the required dependencies. The pipeline relies on the base SampleID weights, PyTorch Lightning, and Hydra.

```bash
git clone https://github.com/your-username/sincere-audio.git
cd sincere-audio
pip install -r requirements.txt
```

*Note: Ensure the original SampleID checkpoint is available in your environment for the `GeMPooledSampleID` module to load the base encoder.*

## Training
This repository utilizes PyTorch Lightning for the training loop and Hydra for configuration management.

To initiate the training pipeline with the default cyclic curriculum settings:

```bash
python train.py
```

**Curriculum Phases:**
The training process automatically progresses through three distinct curriculum phases defined in `config.yaml`:
* **Epochs 0-15:** SNR Stabilization (Distractor probability = 0).
* **Epochs 16-50:** Distractor introduction and progressive SNR dropping.
* **Epochs 51-100:** Scatter loops ramp up to maximum probability.

You can override parameters via the Hydra command line interface:

```bash
python train.py training.batch_size=32 model.dora_rank=64
```

## Usage and Code Organization
* **`lightning_module.py`**: Contains the `SampleDetectorLit` class and implements the dual-contrastive loss logic.
* **`model.py`**: Defines the `GeMPooledSampleID` architecture, the `GeMPoolFreq` layer, and the DoRA injections.
* **`dsp_core.py`**: Implements the `TorchTimeStretcher` Phase Vocoder and Ideal Ratio Mask generation.
* **`dataset.py` & `datatypes.py`**: Contains the `InfalliblePairSampler` and strict typing contracts (`AudioSample`, `AudioBatch`) for the multi-processed data loading.
* **`audio_utils.py` & `augmentations.py`**: Manages resampling to the standard 16kHz target, RMS-based SNR target alignment, and dynamic audio augmentations.

## Performances
SINCERE-Audio establishes a new performance baseline by significantly improving temporal mapping accuracy.

| Model | mAP | HR@1 | HR@10 |
| :--- | :--- | :--- | :--- |
| Bhattacharjee et al. (2025) | 0.442 | 0.155 | 0.191 |
| Baseline SampleID (Riou et al.) | 0.603 | 0.587 | 0.733 |
| **SINCERE-Audio (Ours)** | **0.723** | **-** | **-** |
