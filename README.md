# NER Robustness vs. Text Formality

## Research Question
**"To what extent does the formality level of a sentence, as determined by lexical and syntactic features, correlate with the F1-score degradation of a BERT-based NER model?"**

This project investigates how linguistic variation—specifically text formality—impacts the performance and robustness of Named Entity Recognition (NER) systems across diverse web corpora.

---

## Environment Setup

### Option 1: Conda (Recommended)
```bash
# Create the environment
conda create -n nlpdl_project python=3.12 -y

# Activate the environment
conda activate nlpdl_project

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Standard Python (venv)
```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

---

## How to Run

### 1. Quick Start: View Evaluation Results
If you want to see the final results without re-running the entire training process, use the comprehensive evaluation script. This will analyze the existing predictions and generate performance metrics across different formality bins.

```bash
python comprehensive_eval.py
```

### 2. Full Pipeline
To run the entire pipeline from scratch (data processing, classifier training, and NER training, use the master script. 

**Note:** Running the full pipeline is computationally expensive. It is highly recommended to use a machine with a dedicated GPU (**CUDA**) or Apple Silicon (**MPS**). On a standard CPU, training will take a significant amount of time. (Automatically downloads a pre-compiled version of PyTorch that bundles the necessary CUDA
  runtime libraries)

```bash
python run_pipeline.py
```

---

## Project Structure

- `src/preprocessing/`: Scripts for building the formality dataset from EWT.
- `src/classification/`: Training and inference for the formality classifier.
- `src/features/`: Extraction of lexical and linguistic features.
- `src/ner/`: BERT-based NER training and evaluation logic.
- `data/`: Contains processed IOB2 files and the formality dataset.
- `outputs/`: Stores model predictions and result summaries.

---

## Computational Requirements
- **Inference/Evaluation:** Can be run on a CPU in a few minutes.
- **Training (Full Pipeline):** 
    - **Recommended:** NVIDIA GPU (CUDA) or Apple Silicon (**MPS**).
    - **Reference Hardware:** Tested on an **NVIDIA RTX 3060 (12GB VRAM)**.
    - **Performance:** ~2 mins per epoch (Total: ~6 mins for 3 epochs).
    - **CPU Estimate:** 10+ hours for full training.
