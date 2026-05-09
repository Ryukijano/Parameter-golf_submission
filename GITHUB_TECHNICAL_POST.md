# Building a 16MB U-Net Transformer for OpenAI's Parameter Golf

## The Challenge

OpenAI's [Parameter Golf Challenge](https://openai.com/index/parameter-golf/?utm_source=chatgpt.com) asks participants to train the best possible language model under two brutal constraints:

* ≤16MB total artifact size
* ≤10 minutes training time on 8×H100s

Instead of scaling models larger, the challenge forces a completely different mindset:

> architecture efficiency, compression-aware optimization, and systems-level design.

## My Direction

I explored a compression-aware U-Net Transformer architecture combining:

* U-Net-style skip connections
* Grouped Query Attention (8Q / 4KV)
* SmearGate context injection
* BigramHash embeddings
* INT6 quantization + zstd compression
* Muon optimizer + orthogonal initialization
* FlashAttention-3 experimentation on Hopper GPUs

The core idea was:

> improve information flow and compressibility simultaneously under extreme parameter budgets.

## Architecture-Compression Co-Design

One thing I found fascinating about Parameter Golf is that:

* optimization,
* quantization,
* architecture,
* and evaluation

all become tightly coupled.

For example:

* Muon affects weight geometry
* which affects quantization stability
* which affects final BPB after compression.

Under these constraints, training a language model starts to feel more like:

> designing a compression system than scaling a transformer.

## Key Innovations

### U-Net Transformer Structure
The U-Net architecture with skip connections allows decoder layers to receive early encoder activations, improving gradient flow and enabling more efficient compression of intermediate representations.

### SmearGate for Context Injection
SmearGate provides token-level context injection, allowing the model to maintain contextual information while staying within tight parameter budgets.

### BigramHash Embeddings
BigramHash embeddings capture pair-wise token relationships efficiently, reducing the embedding parameter count while preserving contextual information.

### Quantization-Aware Design
INT6 quantization combined with zstd-22 compression was designed from the ground up, with architecture choices made specifically to maintain numerical stability under aggressive quantization.

### Systems-Level Optimization
The exploration included lower-level CUDA/CUTLASS paths for fused kernels and Hopper-specific optimizations, although numerical stability (especially around Newton-Schulz/Muon iterations) remains an active area of investigation.

## Experimental Results

**Important:** These results are experimental projections and have not been validated on official infrastructure.

* **Baseline GPT-2:** 1.3066 BPB
* **This exploration:** Projected toward ~1.09 BPB range
* **Projected improvement:** ~14% reduction vs baseline

The technique impact analysis shows the following projected improvements:

* 11 layers + MLP 3×: -0.08 BPB
* INT6 + zstd-22: -0.06 BPB
* Sliding window eval: -0.034 BPB
* SmearGate + OrthoInit: -0.017 BPB
* BigramHashEmbedding: -0.012 BPB

## Technical Approach

### Architecture Design
The U-Net Transformer uses 11 layers (6 encoder + 5 decoder) with learned skip connections. This design improves information flow while maintaining compressibility.

### Attention Optimization
Grouped Query Attention (GQA) with 8 query heads and 4 key-value heads reduces the parameter count while maintaining representational capacity.

### Training Dynamics
The Muon optimizer with orthogonal initialization provides better training stability under extreme quantization constraints.

### Compression Pipeline
The final compression pipeline uses INT6 quantization followed by zstd-22 compression, with architecture choices made specifically to maintain stability through this aggressive compression.

## Ongoing Research

The challenge has been one of the most interesting optimization problems I've worked on recently because it forces model design, compression, training dynamics, and systems engineering into a single objective.

I'm continuing to explore:

* recurrence mechanisms for better compression
* compression-aware optimization techniques
* partial RoPE for efficiency
* quantization-aware training methods
* Hopper-specific fused kernels

## Repository

[https://github.com/Ryukijano/Parameter-golf_submission](https://github.com/Ryukijano/Parameter-golf_submission)

## Acknowledgments

This work was supported by OpenAI's $500 compute grant for Parameter Golf exploration.

---

*Note: All BPB figures mentioned are experimental projections based on local testing and have not been validated on official Parameter Golf infrastructure.*
