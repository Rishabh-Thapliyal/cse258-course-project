# LLM-based Recommender System

A collaborative filtering-based recommendation system that uses a full-finetuned LLM backbone to learn user and item embeddings for sequential recommendation.

Inspired by **CLLM4Rec** (Collaborative Large Language Models for Recommender Systems) and related work.

## 🎯 Overview

This system implements a novel approach to sequential recommendation:
- **Stage A**: Pretrains user/item embeddings using collaborative filtering losses with a full-finetuned LLM backbone
- Uses embeddings as soft tokens injected into the LLM
- Efficient dot-product scoring for next-item prediction

### Key Features

- ✅ **Scalable Architecture**: Uses separate embedding tables instead of expanding LLM vocabulary
- ✅ **Soft Token Injection**: Projects user/item embeddings into LLM hidden space
- ✅ **Collaborative Filtering**: Trains embeddings using interaction sequences
- ✅ **BPR Loss**: Optional Bayesian Personalized Ranking loss for better ranking
- ✅ **Dot-Product Scoring**: Efficient next-item prediction without expanding output vocab
- ✅ **Flexible LLM Backbone**: Supports GPT-2, Qwen, and other transformer models

## 📊 Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────┐
│  User Embeddings (U x d_rec)  ──→  Projection (d_model) │
│  Item Embeddings (I x d_rec)  ──→  Projection (d_model) │
└─────────────────────────────────────────────────────────┘
                         ↓
        ┌────────────────────────────────┐
        │   Soft Token Injection into    │
        │   Pretrained LLM (full-finetuning)      │
        │   Sequence: [user, item_1, ..., item_T] │
        └────────────────────────────────┘
                         ↓
        ┌────────────────────────────────┐
        │   LLM Output (B x d_model)     │
        │           ↓                    │
        │   Projection (d_rec)           │
        │           ↓                    │
        │   Dot Product with Item Embeds │
        │           ↓                    │
        │   Scores over all Items        │
        └────────────────────────────────┘
```

### Mathematical Formulation

**Collaborative Loss (Cross-Entropy):**
```
L_ce = -∑∑ log p_θ(i_{u,t} | u, i_{u,<t})
```

Where:
- `u` is the user embedding (projected to LLM space)
- `i_{u,<t}` is the sequence of items before time `t`
- `i_{u,t}` is the target next item

**BPR Loss (Optional):**
The system uses two complementary BPR losses:

1. **Context-Item BPR:**
```
L_bpr_context = -∑ log σ(score(h_context, i^+) - score(h_context, i^-))
```
Where:
- `h_context` is the LLM hidden state (projected to embedding space) representing the sequence context
- `i^+` is a positive (interacted) item
- `i^-` is a negative (non-interacted) item

2. **User-Item BPR:**
```
L_bpr_user = -∑ log σ(score(u, i^+) - score(u, i^-))
```
Where:
- `u` is the user embedding
- `i^+` is a positive (interacted) item
- `i^-` is a negative (non-interacted) item

The combined BPR loss is: `L_bpr = L_bpr_context + L_bpr_user`

**Regularization:**
```
L_reg = λ_c * (||E_user||² + ||E_item||²)
```

## 🚀 Quick Start

### 1. Installation

```bash
cd /space/mcdonald-syn01/1/projects/jsawant/llm_recommender
pip install -r requirements.txt
```

### 2. Data Preprocessing

```bash
python scripts/preprocess_data.py \
    --config config/config.yaml \
    --output_dir ./data/processed
```

This will:
- Load user sequences from the dataset
- Create user/item ID mappings
- Split into train/val/test sets (80/10/10)
- Save processed data and metadata

### 3. Training

```bash
python scripts/train_stage_a.py \
    --config config/config.yaml \
    --data_dir ./data/processed \
    --output_dir ./checkpoints/stage_a
```

This trains:
- Collaborative embeddings (learned from user-item interactions)
- Embeddings are projected to LLM hidden space and used as soft tokens
- LLM processes sequences: `[user_embed, item_1_embed, ..., item_T_embed]`
- Predicts next item using dot-product scoring

### 4. Evaluation

```bash
python scripts/predict.py \
    --checkpoint ./checkpoints/stage_a/best_model \
    --data_dir ./data/processed \
    --output_file ./test_metrics.json
```

Evaluates on test data using SASRec-style sampled evaluation (1 positive + 100 negatives).

## 📁 Project Structure

```
llm_recommender/
├── config/
│   └── config.yaml              # Main configuration file
├── data/
│   ├── __init__.py
│   ├── dataset.py               # PyTorch Dataset & DataLoader
│   └── preprocessing.py         # Data preprocessing utilities
├── models/
│   ├── __init__.py
│   ├── embeddings.py            # Collaborative embeddings
│   └── stage_a_model.py         # Stage A: Embedding pretraining
├── trainers/
│   ├── __init__.py
│   └── stage_a_trainer.py       # Trainer for Stage A
├── utils/
│   ├── __init__.py
│   ├── metrics.py               # Evaluation metrics
│   └── config.py                # Config utilities
├── scripts/
│   ├── preprocess_data.py       # Data preprocessing script
│   ├── train_stage_a.py         # Stage A training script
│   └── predict.py               # Evaluation/prediction script
├── requirements.txt
└── README.md
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

### Key Parameters:

```yaml
model:
  base_llm: "Qwen/Qwen2-0.5B"      # Base LLM (GPT-2, Qwen, etc.)
  embedding_dim: 128               # User/item embedding dimension
  freeze_llm_stage_a: false         # Freeze LLM in Stage A (false = full fine-tuning, true = frozen)
  random_init_stage_a_llm: false    # Random init LLM (vs pretrained)
  
embeddings:
  lambda_c: 0.001                   # Collaborative embedding regularization

stage_a:
  epochs: 10
  batch_size: 32
  learning_rate: 5e-4
  weight_decay: 0.001
  max_grad_norm: 1.0
  loss_weights:
    collaborative: 1.0             # Cross-entropy loss weight
    cf_bpr: 1.0                    # BPR loss weight
    regularization: 0.0            # Regularization loss weight
  collaborative:
    negative_samples: 30           # Negatives for BPR loss
    use_bpr_loss: true             # Enable BPR loss
```

### Supported LLM Models

- **GPT-2**: `"gpt2"`, `"gpt2-medium"`, `"gpt2-large"`, `"gpt2-xl"`
- **Qwen**: `"Qwen/Qwen2-0.5B"`, `"Qwen/Qwen2-1.5B"`, `"Qwen/Qwen2-7B"`

## 📈 Evaluation Metrics

The system evaluates using SASRec-style sampled evaluation:
- **Hit@K**: Whether the target item appears in top-K predictions (K=1, 5, 10, 20)
- **NDCG@K**: Normalized Discounted Cumulative Gain at K

Evaluation protocol:
- For each user, sample 100 negative items
- Build candidate set: [1 positive + 100 negatives] = 101 items
- Rank candidates and compute metrics

## 🔬 Design Decisions

### Why Not Expand Vocabulary?

❌ **Bad Approach**: Adding every user/item as LLM tokens
- Explodes vocabulary size (|V| + |U| + |I|)
- Slow, memory-intensive
- Doesn't generalize to new users/items

✅ **Our Approach**: Separate embedding tables + projection
- Fixed LLM vocabulary
- Scalable to millions of users/items
- Embeddings projected to LLM hidden space as soft tokens

### Why Collaborative Filtering Only?

- **Simplicity**: Focus on interaction patterns without text content
- **Efficiency**: Faster training without text tokenization
- **Effectiveness**: Collaborative signals are often sufficient for recommendation
- **Scalability**: Works with or without item metadata

### Training Strategy

1. **Autoregressive Training**: Predict each next item in sequence
   - Input: `[user, item_1, ..., item_T]`
   - Targets: `[item_2, ..., item_{T+1}]`
   - Uses cross-entropy loss over all positions

2. **BPR Loss**: Optional ranking loss with two components
   - **Context-item BPR**: Uses LLM hidden states (projected to embedding space) as context representation, then scores context vs positive/negative items
   - **User-item BPR**: Direct user-item interaction, scores user embedding vs positive/negative items
   - Both losses are combined to learn better ranking from both sequential context and user preferences

3. **LLM Fine-tuning**: LLM can be frozen or fully fine-tuned
   - When frozen: Only embeddings and projections are trained (saves memory and compute)
   - When fine-tuned: LLM adapts to the recommendation task while learning embeddings
   - Leverages pretrained sequential modeling capabilities

## 📚 References

- [CLLM4Rec: Collaborative Large Language Models for Recommender Systems](https://arxiv.org/abs/2311.01343)
- [SASRec: Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)
- [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations](https://arxiv.org/abs/1904.06690)


## 🤝 Contributing

This is a research project. Feel free to:
- Open issues for bugs or questions
- Submit PRs for improvements
- Adapt for your own datasets




**Dataset**: Kindle Store reviews from Amazon  
**Scale**: ~100K users, ~349K items, ~8M interactions
