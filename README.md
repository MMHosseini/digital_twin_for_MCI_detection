# Digital Twins of Aging Minds

## Modeling and Predicting Cognitive Decline with Multimodal AI

This repository contains the code and processing pipeline for the research project **“Digital Twins of Aging Minds: Modeling and Predicting Cognitive Decline with Multimodal AI.”**

The project investigates personalized multimodal digital twins for longitudinal cognitive modeling. It combines conversational **text**, **speech/acoustic information**, participant metadata, personalized language models, and a conditional variational autoencoder (cVAE) to model participant-specific behavior and estimate cognitive status.

The current study uses longitudinal conversational data from the **I-CONECT** study and focuses on a proof-of-concept cohort with dense repeated observations per participant.

---

## Overview

The proposed framework contains three main components:

1. **Linguistic Digital Twin**
   - Builds a personalized conversational model for each participant.
   - Uses supervised fine-tuning (SFT) followed by Direct Preference Optimization (DPO).
   - Incorporates participant metadata and conversational context.
   - Models participant-specific linguistic and stylometric behavior.

2. **Acoustic Digital Twin**
   - Extracts acoustic descriptors from participant speech.
   - Uses prosodic features, speaker representations, and contextual speech embeddings.
   - Trains an acoustic mimicking model that predicts participant-specific acoustic representations from linguistic representations.

3. **Cognitive Prediction Model**
   - Uses a conditional variational autoencoder (cVAE).
   - Combines linguistic and acoustic representations.
   - Reconstructs multimodal participant behavior.
   - Estimates Montreal Cognitive Assessment (MoCA) scores.

The overall objective is to evaluate whether a personalized multimodal digital twin can preserve participant-specific conversational behavior while capturing information associated with longitudinal cognitive status.

---

## Pipeline

The repository implements the following workflow.

### 1. Data Preparation

- Split training and validation sets
- Merge transcripts
- Extract MCI/cognitive information
- Correct transcript ordering
- Prepare longitudinal participant data
- Extract audio from video
- Prepare metadata and conversation descriptors

### 2. Linguistic Digital Twin

- Create JSONL files for supervised fine-tuning
- Fine-tune the language model using SFT
- Create preference data for DPO
- Fine-tune the personalized model using DPO
- Prepare retrieval-augmented generation (RAG) data
- Run personalized language-model inference

### 3. Linguistic Feature Extraction

Generated and real participant responses are represented using:

- Text embeddings
- Sentiment descriptors
- Conversation/topic descriptors
- Stylometric information
- Participant-specific linguistic representations

These features are also used for participant/author identification experiments.

### 4. Acoustic Feature Extraction

The speech-processing pipeline extracts:

- Prosodic features
- x-vector speaker embeddings
- WavLM-based speech representations
- Additional temporal and spectral speech descriptors

The acoustic features characterize participant-specific speaking behavior and provide complementary information to the linguistic modality.

### 5. Acoustic Digital Twin

The acoustic digital twin learns a mapping from linguistic representations to participant-specific acoustic feature representations.

The workflow includes:

- Training the acoustic mimicking model
- Validating the acoustic mimicking model
- Running acoustic-model inference
- Evaluating participant identity preservation from generated acoustic representations

The acoustic model predicts representations in feature space rather than synthesizing raw speech waveforms.

### 6. Multimodal Cognitive Modeling

A conditional variational autoencoder (cVAE) is trained using multimodal representations.

The model is designed to:

- Encode linguistic and acoustic information into a shared latent representation
- Reconstruct multimodal participant representations
- Estimate MoCA scores
- Support comparison between real participant data and digital-twin representations

### 7. Evaluation

The current pipeline includes evaluation of:

- Linguistic participant identification
- Acoustic participant identification
- Linguistic reconstruction
- Acoustic reconstruction
- Multimodal reconstruction
- MoCA-score prediction
- Real participant vs. digital-twin consistency
- Multimodal vs. unimodal performance
- Audio/transcript feature relationships
- Training and validation losses

---

## Implemented Tasks

The research pipeline currently contains the following tasks:

```python
tasks = {
    "split_train_validation_sets": 0,
    "merge_transcripts": 0,
    "extract_mci_info": 0,
    "correct_transcripts_order": 0,
    "create_jsonl_for_supervised_fine_tuning": 0,
    "create_jsonl_for_direct_preference_optimization": 0,
    "supervised_fine_tune_language_model": 0,
    "dpo_fine_tune_language_model": 0,
    "prepare_rag_data": 0,
    "inference_language_model": 0,
    "extract_text_embeddings": 0,
    "extract_text_sentiments": 0,
    "extract_title_descriptor": 0,
    "text_based_author_identification": 0,
    "extract_audio_from_video": 0,
    "extract_audio_prosody": 0,
    "extract_audio_xvector": 0,
    "extract_audio_wavlm": 0,
    "train_audio_mimicker": 0,
    "validate_audio_mimicker": 0,
    "inference_audio_model": 0,
    "audio_based_author_identification": 0,
    "train_cvae_model": 0,
    "validate_cvae_model": 0,
    "report_train_and_validation_loss": 0,
    "check_audio_transcript_feature_correlation": 0,
    "plot_for_paper": 1,
}
```

> **Note:** The meaning of the numeric task flags depends on the task runner used in the repository. Update this section if `0` and `1` correspond to disabled/enabled, completed/pending, or another convention.

---

## Data Split

The study uses a **participant-specific temporal split** rather than a random sample-level split.

For each participant:

- Conversations are ordered chronologically.
- The earliest portion of the longitudinal conversations is used for model development/training.
- Later conversations are reserved for validation.
- Validation therefore evaluates performance on future, previously unseen interactions from the same participant.

This design is intended to better reflect the longitudinal digital-twin setting and reduce leakage between earlier and later conversational observations.

---

## Text Processing

The text pipeline includes:

- Transcript preprocessing and ordering
- Question-response extraction
- Metadata integration
- SFT dataset construction
- DPO preference dataset construction
- Personalized language-model inference
- Sentence/text embedding extraction
- Sentiment extraction
- Topic/title descriptor extraction
- Participant-identification experiments

The linguistic digital twin is designed to preserve participant-specific semantic, stylistic, and conversational characteristics.

---

## Language-Model Personalization

The linguistic digital twin uses a two-stage personalization strategy:

### Supervised Fine-Tuning

The model is first fine-tuned on participant-specific conversational examples consisting of:

- System instructions
- Participant metadata
- Moderator questions
- Real participant responses

### Direct Preference Optimization

DPO is then used to further encourage participant-consistent responses by contrasting preferred participant responses with behaviorally dissimilar alternatives.

This stage is intended to improve participant-specific behavioral alignment beyond supervised fine-tuning alone.

---

## Acoustic Processing

The acoustic pipeline extracts complementary speech information that may not be fully represented by text.

Current acoustic representations include:

### Prosody

Features related to speech behavior such as:

- Pitch
- Energy
- Speaking tempo
- Pauses
- MFCC-related descriptors
- Spectral characteristics
- Statistical descriptors

### x-vectors

Speaker-related embeddings are extracted to characterize participant-specific vocal properties.

### WavLM

Contextual speech representations are extracted using WavLM-based models to capture rich information from speech.

---

## Acoustic Mimicking Model

The acoustic digital twin predicts acoustic feature representations from linguistic representations.

Conceptually:

```text
Generated / Real Text
        |
        v
Linguistic Representation
        |
        v
Acoustic Mimicking Model
        |
        v
Predicted Acoustic Representation
```

The output is an acoustic representation in feature space and should not be interpreted as direct voice synthesis or voice cloning.

---

## Multimodal cVAE

The multimodal cognitive model combines text and acoustic representations using a conditional variational autoencoder.

Conceptually:

```text
Text Features --------\
                       \
                        > Multimodal Representation --> cVAE --> Reconstructed Features
                       /                                  \
Audio Features -------/                                    --> MoCA Prediction

Metadata / Context ---------------------------------------> Conditioning
```

The shared latent representation is used for both multimodal reconstruction and cognitive-score estimation.

---

## Experimental Comparisons

The project evaluates multiple conditions, including:

- Real participant responses
- Personalized digital-twin responses
- Non-personalized / non-fine-tuned language-model responses
- Linguistic-only representations
- Acoustic-only representations
- Multimodal representations

These comparisons are used to study:

1. Participant-specific identity preservation
2. Representation reconstruction quality
3. Benefits of multimodal fusion
4. Cognitive-score estimation
5. Similarity between real participants and their personalized digital twins

---

## Running the Pipeline

The exact execution command depends on the main entry point in this repository.

The codebase uses task-based execution. Enable only the task or tasks that should be run and verify that all prerequisite outputs are available before starting downstream stages.

A typical execution order is:

```text
Data preprocessing
        ↓
Train/validation split
        ↓
SFT data generation
        ↓
SFT personalization
        ↓
DPO data generation
        ↓
DPO personalization
        ↓
Language-model inference
        ↓
Text feature extraction
        ↓
Audio extraction
        ↓
Acoustic feature extraction
        ↓
Acoustic mimicker training
        ↓
Acoustic inference
        ↓
Multimodal feature construction
        ↓
cVAE training and validation
        ↓
Evaluation
        ↓
Paper figures
```

Because some stages depend on outputs from previous stages, running tasks out of order may result in missing intermediate files.

---

## Reproducibility Notes

Before publicly releasing the repository, consider documenting the following items:

- Python version
- Package versions
- Hardware used for local model training
- API model/version used for language-model personalization
- Random seeds
- Learning rates
- Batch sizes
- Number of epochs
- DPO hyperparameters
- cVAE architecture and latent dimensionality
- Loss weights
- Feature dimensionalities
- PCA settings
- Training/validation split details
- Exact directory structure expected by each task

A `requirements.txt`, `environment.yml`, or `pyproject.toml` file is strongly recommended.

---

## Dataset and Privacy

This research uses data from the **I-CONECT** study.

The underlying participant recordings and associated data may be subject to privacy, ethics, institutional, and data-use restrictions. **Do not upload restricted participant data, identifiable recordings, protected health information, API credentials, or other sensitive material to the public repository.**

The public repository should contain only code, documentation, configuration templates, and outputs that are permitted for public release.

Users who wish to reproduce the experiments should obtain access to the underlying dataset through the appropriate authorized process.

---

## Ethics

The research manuscript reports the use of de-identified I-CONECT data collected under the applicable institutional review and informed-consent procedures.

The proposed digital twins are intended for research into personalized longitudinal cognitive modeling. They should not be interpreted as standalone clinical diagnostic systems.

---

## Important Interpretation Note

The current study is a **proof-of-concept personalized longitudinal study**. Its purpose is to investigate whether dense repeated observations can support participant-specific multimodal digital twins.

The study prioritizes **longitudinal depth within individuals** over population-level cohort size. Results should therefore be interpreted as evidence of methodological feasibility rather than as population-level clinical validation.

---

## Citation

If you use this repository or build upon this work, please cite the associated manuscript.

```bibtex
@article{hosseini_digital_twins_aging_minds,
  title   = {Digital Twins of Aging Minds: Modeling and Predicting Cognitive Decline with Multimodal AI},
  author  = {Hosseini, Mohammad Mehdi and Dodge, Hiroko H. and Mahoor, Mohammad H.},
  note    = {Manuscript},
}
```

Update the BibTeX entry with the journal, year, volume, pages, DOI, and publication status when the paper is published.

---

## Acknowledgments

This work uses data from the I-CONECT study. The manuscript acknowledges support from the National Institute on Aging and the Knoebel Institute for Healthy Aging.

Please refer to the associated manuscript for the complete funding and acknowledgment statements.

---

## License

Add the appropriate software license before publicly releasing this repository.

Common options include:

- MIT License
- BSD 3-Clause License
- Apache License 2.0

The software license does **not** automatically grant permission to redistribute the underlying research dataset.

---

## Contact

For questions about the code or research, please contact the corresponding author through the contact information provided in the associated manuscript.