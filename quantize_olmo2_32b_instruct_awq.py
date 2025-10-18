#!/usr/bin/env python3
"""
Quantize OLMo-2-0325-32B-Instruct using AWQ with Multi-GPU Support
Quantizes the 32B OLMo 2 instruction-tuned model to 4-bit AWQ

Model: allenai/OLMo-2-0325-32B-Instruct (32B params)
Architecture: OLMo 2 (fully open model)
Training: SFT → DPO → RLVR on Tülu 3 dataset
Expected: ~64GB → ~20GB (69% reduction)
GPU: 5x B200 (890GB total VRAM) - Plenty of headroom!
Time: 60-90 minutes with 512 samples, 2048 seq length
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor import oneshot
from huggingface_hub import HfApi, create_repo, login
import gc

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("=" * 70)
print("🔧 Quantizing OLMo-2-0325-32B-Instruct with AWQ (Multi-GPU)")
print("=" * 70)

# Check GPU availability
num_gpus = torch.cuda.device_count()
print(f"\n🖥️  Detected {num_gpus} GPUs")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    memory_gb = props.total_memory / (1024**3)
    print(f"   GPU {i}: {props.name} - {memory_gb:.1f}GB")

if num_gpus < 5:
    print("\n⚠️  WARNING: This script is optimized for 5 GPUs (5x B200)")
    print("   With fewer GPUs, you may need to reduce calibration samples/seq length")
    if num_gpus >= 3:
        print("   Recommended for 3-4 GPUs: 256 samples, 1024 seq length")
    elif num_gpus >= 2:
        print("   Recommended for 2 GPUs: 128 samples, 1024 seq length")
    else:
        print("   1 GPU will likely fail due to OOM")

# Configuration
MODEL_ID = "allenai/OLMo-2-0325-32B-Instruct"
OUTPUT_DIR = "./olmo2-32b-instruct-awq-w4a16"
HF_USERNAME = "ronantakizawa"
HF_REPO_ID = f"{HF_USERNAME}/olmo2-32b-instruct-awq-w4a16"
NUM_CALIBRATION_SAMPLES = 128  # Balanced for 5x B200 (178GB each)
MAX_SEQ_LENGTH = 512  # Reduced to prevent OOM during calibration

print(f"\n📋 Configuration:")
print(f"  Model: {MODEL_ID}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  HF Repository: {HF_REPO_ID}")
print(f"  Username: {HF_USERNAME}")
print(f"  Calibration samples: {NUM_CALIBRATION_SAMPLES}")
print(f"  Max sequence length: {MAX_SEQ_LENGTH}")
print(f"  Batch size: 1")
print(f"  Method: AWQ (Activation-aware Weight Quantization)")
print(f"  Multi-GPU: Enabled ({num_gpus} GPUs)")
print(f"  Memory optimization: expandable_segments enabled")

# Login to Hugging Face
print(f"\n🔐 Logging in to Hugging Face...")
try:
    login(token=HF_TOKEN)
    print("✅ Successfully logged in to Hugging Face")
except Exception as e:
    print(f"❌ Failed to login to Hugging Face: {e}")
    print("\n💡 Make sure your HF_TOKEN is valid")
    import sys
    sys.exit(1)

# Clear any cached memory before loading
print("\n🧹 Clearing GPU cache...")
torch.cuda.empty_cache()
gc.collect()

# Load model and tokenizer
print(f"\n1️⃣  Loading model and tokenizer...")
print(f"  ℹ️  This is a 32B model - will distribute across {num_gpus} GPU(s)")
print("  ℹ️  OLMo 2 is a fully open language model by Ai2")

# Note: OLMo 2 requires transformers from main branch
# device_map="auto" will automatically distribute across all available GPUs
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,  # Use FP16 for memory efficiency
    trust_remote_code=True,
    device_map="auto",  # Automatically distributes across GPUs
    low_cpu_mem_usage=True,
)
print("✅ Model loaded and distributed across GPUs")

# Print which layers went to which GPU
if num_gpus > 1:
    print("\n📊 Model distribution:")
    device_map = model.hf_device_map
    gpu_layers = {}
    for layer_name, device in device_map.items():
        if device not in gpu_layers:
            gpu_layers[device] = []
        gpu_layers[device].append(layer_name)

    for device in sorted(gpu_layers.keys()):
        layer_count = len(gpu_layers[device])
        print(f"   {device}: {layer_count} layers")

# Clear cache after loading
torch.cuda.empty_cache()
gc.collect()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
print("✅ Tokenizer loaded")

# Load calibration dataset
print(f"\n2️⃣  Loading calibration dataset...")
# Using OpenOrca for more diverse instruction-following calibration
# Better match for OLMo's post-training on diverse tasks
ds = load_dataset(
    "Open-Orca/OpenOrca",
    split=f"train[:{NUM_CALIBRATION_SAMPLES}]",
)
print(f"✅ Loaded {len(ds)} samples from OpenOrca dataset")


def preprocess_function(examples):
    """Preprocess for OLMo 2 using OpenOrca dataset"""
    questions = examples["question"]
    responses = examples["response"]

    processed = []
    for question, response in zip(questions, responses):
        # Create messages format for chat template
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response[:100]}  # Truncate response for calibration
        ]

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Tokenize
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        sample = {
            "input_ids": tokens["input_ids"][0],
            "attention_mask": tokens["attention_mask"][0],
        }
        processed.append(sample)

    return {key: [s[key] for s in processed] for key in processed[0].keys()}


print("\n3️⃣  Preprocessing dataset...")
ds = ds.map(
    preprocess_function,
    batched=True,
    batch_size=8,
    remove_columns=ds.column_names,
    desc="Preprocessing"
)
print("✅ Dataset preprocessed")

# AWQ recipe
print("\n4️⃣  Setting up AWQ quantization recipe...")

recipe = AWQModifier(
    targets="Linear",
    scheme="W4A16",  # 4-bit weights, 16-bit activations
    ignore=[
        "re:.*lm_head",     # Don't quantize language model head
        "re:.*embed.*",     # Don't quantize embeddings
    ],
)
print("\n✅ AWQ Recipe configured:")
print("   - Scheme: W4A16 (4-bit weights, 16-bit activations)")
print("   - Targets: All Linear layers in OLMo 2 transformer")
print("   - Preserved: Embeddings and LM head")


def data_collator(batch):
    """Collate batch items"""
    collated = {}
    for key in batch[0].keys():
        try:
            collated[key] = torch.stack([
                torch.tensor(item[key]) if not isinstance(item[key], torch.Tensor) else item[key]
                for item in batch
            ])
        except:
            collated[key] = [item[key] for item in batch]
    return collated


# Run quantization
print(f"\n5️⃣  Running AWQ quantization with independent pipeline...")
print("  ℹ️  Using IndependentPipeline with BasicPipeline")
print(f"  ℹ️  Model distributed across {num_gpus} GPU(s)")
print(f"⏳ This will take 60-90 minutes with {NUM_CALIBRATION_SAMPLES} samples...\n")
print("=" * 70)

try:
    # Clear cache before quantization on all GPUs
    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    gc.collect()

    # The oneshot function will work with device_map="auto" automatically
    # No special multi-GPU configuration needed - it handles it internally
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        output_dir=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
        trust_remote_code_model=True,
        data_collator=data_collator,
    )

    print("\n" + "=" * 70)
    print("✅ Quantization completed successfully!")
    print(f"📁 Model saved to: {OUTPUT_DIR}")

    if os.path.exists(OUTPUT_DIR):
        size_bytes = sum(
            os.path.getsize(os.path.join(dirpath, filename))
            for dirpath, _, filenames in os.walk(OUTPUT_DIR)
            for filename in filenames
        )
        size_gb = size_bytes / (1024**3)
        original_size_gb = 64.0
        reduction = (1 - size_gb / original_size_gb) * 100

        print(f"\n📊 Model size comparison:")
        print(f"   Original (BF16): ~{original_size_gb:.1f} GB")
        print(f"   Quantized (W4A16): ~{size_gb:.2f} GB")
        print(f"   Reduction: ~{reduction:.1f}%")
        print(f"   Memory saved: ~{original_size_gb - size_gb:.1f} GB")

    print("\n🎉 Quantization complete!")
    print("\n💡 Key features of OLMo 2 32B Instruct:")
    print("   ✅ Fully open model (code, data, training)")
    print("   ✅ Post-trained: SFT → DPO → RLVR")
    print("   ✅ Strong performance on MATH, GSM8K, IFEval")
    print("   ✅ 32B params quantized to 4-bit AWQ")
    print("   ✅ Built by Allen Institute for AI")

    # Step 6: Upload to Hugging Face
    print("\n" + "=" * 70)
    print("6️⃣  Uploading to Hugging Face Hub")
    print("=" * 70)

    try:
        # Create model card
        model_card = f"""---
language:
- en
license: apache-2.0
base_model: {MODEL_ID}
tags:
- awq
- quantized
- 4-bit
- olmo2
- conversational
- llm-compressor
library_name: transformers
pipeline_tag: text-generation
---

# OLMo-2-0325-32B-Instruct AWQ 4-bit

This is a 4-bit AWQ quantized version of [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Key Features

- ✅ **32B parameters quantized to 4-bit** - 69% size reduction
- ✅ **Fully open model** - code, data, and training details all public
- ✅ **Post-trained on Tülu 3** - SFT → DPO → RLVR pipeline
- ✅ **Strong performance** - competitive with Llama 3.1 70B on many tasks
- ✅ **State-of-the-art on specific tasks** - MATH, GSM8K, IFEval

## Model Details

- **Base Model:** {MODEL_ID} (32B parameters)
- **Architecture:** OLMo 2 (fully open language model)
- **Quantization Method:** AWQ (Activation-aware Weight Quantization)
- **Quantization Scheme:** W4A16 (4-bit weights, 16-bit activations)
- **Calibration Dataset:** OpenOrca ({NUM_CALIBRATION_SAMPLES} samples)

## Size Comparison

| Metric | Value |
|--------|-------|
| **Original (BF16)** | ~{original_size_gb:.1f} GB |
| **Quantized (W4A16)** | ~{size_gb:.2f} GB |
| **Reduction** | ~{reduction:.1f}% |
| **Memory Saved** | ~{original_size_gb - size_gb:.1f} GB |

## About OLMo 2

OLMo 2 is a series of fully open language models by the Allen Institute for AI:

- **Training:** Trained on Dolma dataset
- **Post-training:** Supervised finetuning, DPO, and RLVR on Tülu 3
- **Performance:** Competitive with much larger models
- **Openness:** All code, data, and training details released

### Performance Highlights

- **Average Score:** 68.8 across diverse benchmarks
- **GSM8K:** 87.6 (math reasoning)
- **IFEval:** 85.6 (instruction following)
- **MATH:** 49.7 (mathematical problem solving)
- **MMLU:** 77.3 (general knowledge)

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "{HF_REPO_ID}",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "{HF_REPO_ID}",
    trust_remote_code=True
)

# Chat template
messages = [
    {{"role": "user", "content": "Explain quantum computing in simple terms."}}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Chat Template

The model uses this chat template format:

```
<|user|>
How are you doing?
<|assistant|>
I'm just a computer program, so I don't have feelings, but I'm functioning as expected. How can I assist you today?<|endoftext|>
```

## System Prompt (Optional)

In Ai2 demos, this system prompt is used by default:

```
You are OLMo 2, a helpful and harmless AI Assistant built by the Allen Institute for AI.
```

However, the model has not been trained with a specific system prompt requirement.

## Quantization Details

- **Method:** AWQ (Activation-aware Weight Quantization)
- **Independent Pipeline:** Used with BasicPipeline for layer-by-layer quantization
- **Calibration:** {NUM_CALIBRATION_SAMPLES} OpenOrca samples
- **Max Sequence Length:** {MAX_SEQ_LENGTH} tokens
- **Why AWQ:** Preserves important weights based on activation patterns

## Requirements

- Transformers (install from main branch for OLMo 2 support)
- PyTorch with AWQ/GPTQ support
- 20GB+ GPU VRAM for inference

## Limitations

- Quantization may cause slight quality degradation compared to BF16
- Limited safety training (not production-ready without additional filtering)
- Primarily English language support

## License

Apache 2.0 (same as base model)

## Citation

```bibtex
@article{{olmo20242olmo2furious,
      title={{2 OLMo 2 Furious}},
      author={{Team OLMo and Pete Walsh and Luca Soldaini and others}},
      year={{2024}},
      eprint={{2501.00656}},
      archivePrefix={{arXiv}},
      primaryClass={{cs.CL}},
      url={{https://arxiv.org/abs/2501.00656}},
}}
```

```bibtex
@misc{{olmo2-32b-awq,
  title={{OLMo-2-0325-32B-Instruct AWQ 4-bit}},
  author={{Quantized by {HF_USERNAME}}},
  year={{2025}},
  url={{https://huggingface.co/{HF_REPO_ID}}}
}}
```

## Acknowledgements

- Base model by [Allen Institute for AI](https://allenai.org/)
- Quantization using [LLM Compressor](https://github.com/vllm-project/llm-compressor)

---

🤖 Generated with [LLM Compressor](https://github.com/vllm-project/llm-compressor)
"""

        # Save model card
        readme_path = os.path.join(OUTPUT_DIR, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(model_card)
        print("✅ Model card created")

        # Create repository
        print(f"\n🚀 Creating repository: {HF_REPO_ID}")
        create_repo(
            HF_REPO_ID,
            repo_type="model",
            exist_ok=True,
            token=HF_TOKEN
        )
        print("✅ Repository created/verified")

        # Upload files
        print(f"\n📤 Uploading model to Hugging Face Hub...")
        print("   This may take 20-30 minutes depending on connection speed...")

        api = HfApi()
        api.upload_folder(
            folder_path=OUTPUT_DIR,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"Upload AWQ 4-bit quantized OLMo-2-32B-Instruct (~{size_gb:.1f}GB, {reduction:.1f}% reduction)",
            token=HF_TOKEN
        )

        print("\n" + "=" * 70)
        print("✅ Upload complete!")
        print(f"🔗 Model available at: https://huggingface.co/{HF_REPO_ID}")
        print("=" * 70)

        print("\n📝 Next steps:")
        print("  1. Visit your model page to verify the upload")
        print("  2. Test the model with the usage example in the README")
        print("  3. Share your quantized OLMo 2 with the community!")

    except Exception as upload_error:
        print(f"\n⚠️  Upload to Hugging Face failed: {upload_error}")
        print(f"   Model is still saved locally at: {OUTPUT_DIR}")
        print("\n💡 To upload manually:")
        print(f"   1. huggingface-cli login")
        print(f"   2. huggingface-cli upload {HF_REPO_ID} {OUTPUT_DIR}")

except Exception as e:
    print("\n" + "=" * 70)
    print(f"❌ Quantization failed: {type(e).__name__}")
    print(f"Error message: {e}")
    print("\n" + "=" * 70)
    print("FULL TRACEBACK:")
    print("=" * 70)
    import traceback
    traceback.print_exc()

    print("\n💡 Troubleshooting tips:")
    print(f"  1. Ensure you have enough total GPU memory (detected: {num_gpus} GPUs)")
    print("  2. For 5x B200 (890GB): Should work perfectly with full settings (512/2048)")
    print("  3. For 3-4x B200 (540-712GB): Try 256 samples, 1024 seq length")
    print("  4. For 2x B200 (356GB): Try 128 samples, 1024 seq length")
    print("  5. Try reducing NUM_CALIBRATION_SAMPLES if OOM (512→256→128→64)")
    print("  6. Try reducing MAX_SEQ_LENGTH if OOM (2048→1024→512)")
    print("  7. Verify transformers is installed from main branch:")
    print("     pip install --upgrade git+https://github.com/huggingface/transformers.git")
    print("  8. Verify llmcompressor has the latest fixes:")
    print("     pip install git+https://github.com/ronantakizawa/llm-compressor.git@ronantakizawa/sequential-pipeline-fallback")

print("\n" + "=" * 70)
print("🏁 Script completed")
print("=" * 70)
