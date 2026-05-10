# Fine-tuning NLLB for Spanish-to-Asturian Machine Translation with different data sizes and validation methods

Fine-tuning of [NLLB-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M) for Spanish→Asturian (`spa_Latn` → `ast_Latn`) machine translation using Low-Rank Adaptation (LoRA). Experiments were conducted in two stages: a prototyping phase on Google Colab with subsampled data, and full-scale training on a server with 42GB of GPU memory.

## Data

| Split | Corpus | Colab | Server |
|-------|--------|-------|--------|
| Train | [ES-AST Parallel Corpus (AINA)](https://huggingface.co/datasets/projecte-aina/ES-AST_Parallel_Corpus) | 5,000 | 704,378 |
| Dev   | [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) | 500 | 997 |
| Test  | [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) | 500 | 1,012 |

## Models

Trained LoRA adapters (server experiments) are available on HuggingFace Hub:

| Model | Validation | HuggingFace |
|-------|-----------|-------------|
| FT (loss val)   | Loss   | [asturian-nllb-lora-loss](https://huggingface.co/ikergf/asturian-nllb-lora-loss) |
| FT (BLEU val)   | BLEU   | [asturian-nllb-lora-bleu](https://huggingface.co/ikergf/asturian-nllb-lora-bleu) |
| FT (chrF++ val) | chrF++ | [asturian-nllb-lora-chrfpp](https://huggingface.co/ikergf/asturian-nllb-lora-chrfpp) |

## Results

Colab evaluation on FLORES+ `devtest` (SacreBLEU):

| Model | BLEU | chrF++ |
|-------|------|--------|
| Baseline (NLLB) | 15.32 | 42.17 |
| FT (loss val)   | 12.80 | 37.45 |

Server evaluation on FLORES+ `devtest` (SacreBLEU, COMET, BLEURT):

| Model | BLEU | chrF++ | COMET | BLEURT |
|-------|------|--------|-------|--------|
| Baseline (NLLB)     | 13.88          | 40.80          | 65.97          | 44.71          |
| FT (loss val)       | **16.36**      | 45.80          | 67.28          | 43.34          |
| **FT (BLEU val)**   | 16.32          | **46.09**      | **68.27**      | **44.98**      |
| FT (chrF++ val)     | 16.26          | 45.48          | 66.95          | 42.70          |

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Configuration

Edit `scripts/config.py` with your local paths and set your HuggingFace token:

```bash
export HF_TOKEN=your_token
```

### Colab

Open `es-ast-MT.ipynb` in Google Colab for the prototyping experiments with subsampled data.

### Server Pipeline

```bash
# 1. Prepare data
python scripts/prepare_data.py

# 2. Train
python scripts/train.py           # chrF++ validation
python scripts/train_bleu.py      # BLEU validation
python scripts/train_loss.py      # loss validation

# 3. Inference
python scripts/inference_baseline.py
python scripts/inference_ft.py

# 4. Evaluate
python scripts/evaluate_baseline.py
python scripts/evaluate_ft.py
```

### SLURM

```bash
sbatch slurm/train.sh
sbatch slurm/inference_baseline.sh
sbatch slurm/inference_ft.sh
sbatch slurm/evaluate_baseline.sh
sbatch slurm/evaluate_ft.sh
```

## Repository Structure

```
├── es-ast-MT.ipynb          # Colab notebook (prototyping experiments)
├── extended_abstract.pdf    # Extended abstract
├── scripts/                 # Training, inference and evaluation scripts
├── slurm/                   # SLURM job submission scripts
├── predictions/             # Model predictions on FLORES+ devtest
└── requirements.txt
```

## Citation

```bibtex
@misc{gutierrez2026asturian,
  title={Fine-tuning NLLB for Spanish-to-Asturian Machine Translation with different data sizes and validation methods},
  author={Gutierrez Fandiño, Iker},
  year={2026},
  institution={University of the Basque Country (EHU)}
}
```
