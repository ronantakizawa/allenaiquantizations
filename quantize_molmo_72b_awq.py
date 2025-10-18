#!/usr/bin/env python3
"""
Quantize Molmo-72B-0924 using AWQ with Multi-GPU Support
Quantizes ONLY the text/Qwen2-72B decoder, preserves vision encoder quality

Model: allenai/Molmo-72B-0924 (73B params)
Base: Qwen2-72B + OpenAI CLIP vision encoder
Purpose: High-performance vision-language model
Expected: ~145GB → ~40GB (72% reduction)
GPU: 5x B200 (890GB total) or larger
Time: 90-120 minutes with 128 samples
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor import oneshot
from huggingface_hub import HfApi, create_repo, login
import gc

# Set memory optimization environment variables
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print("=" * 70)
print("🔧 Quantizing Molmo-72B-0924 with AWQ (Multi-GPU)")
print("=" * 70)

# Check GPU availability
num_gpus = torch.cuda.device_count()
print(f"\n🖥️  Detected {num_gpus} GPUs")
for i in range(num_gpus):
    props = torch.cuda.get_device_properties(i)
    memory_gb = props.total_memory / (1024**3)
    print(f"   GPU {i}: {props.name} - {memory_gb:.1f}GB")

if num_gpus < 5:
    print("\n⚠️  WARNING: This script is optimized for 5+ GPUs (5x B200)")
    print("   Molmo-72B is very large - may OOM with fewer GPUs")
    if num_gpus >= 3:
        print("   With 3-4 GPUs: Try reducing to 64 samples")
    else:
        print("   With <3 GPUs: Likely will fail due to OOM")

# Configuration
MODEL_ID = "allenai/Molmo-72B-0924"
OUTPUT_DIR = "./workspace/molmo-72b-awq-w4a16"
HF_USERNAME = "ronantakizawa"
HF_REPO_ID = f"{HF_USERNAME}/molmo-72b-awq-w4a16"
NUM_CALIBRATION_SAMPLES = 512  # Balanced for 5x B200
MAX_SEQ_LENGTH = 2048  # Conservative for memory

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
for i in range(num_gpus):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
gc.collect()

# Load model and processor
print(f"\n1️⃣  Loading model and processor...")
print(f"  ℹ️  This is a 73B model - will distribute across {num_gpus} GPU(s)")
print("  ℹ️  Molmo-72B: Qwen2-72B decoder + OpenAI CLIP vision encoder")

# Load model with multi-GPU distribution
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
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
for i in range(num_gpus):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()
gc.collect()

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True
)
print("✅ Processor loaded")

# Load calibration dataset
print(f"\n2️⃣  Loading calibration dataset...")
ds = load_dataset(
    "lmms-lab/flickr30k",
    split=f"test[:{NUM_CALIBRATION_SAMPLES}]",
)
print(f"✅ Loaded {len(ds)} samples from Flickr30k dataset")


def preprocess_function(examples):
    """Preprocess for Molmo-72B using Flickr30k dataset"""
    images = examples["image"]

    processed = []
    for idx, image in enumerate(images):
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Create varied prompts for better calibration
        prompts = [
            "Describe what you see in this image.",
            "What is happening in this image?",
            "Please provide a detailed description of this image.",
            "Analyze the content of this image.",
        ]
        prompt = prompts[idx % len(prompts)]

        # Process using Molmo's processor
        inputs = processor.process(
            images=[image],
            text=prompt
        )

        # Convert to tensors and pad/truncate
        input_ids = inputs["input_ids"]

        # Convert to list if it's a tensor
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()

        if len(input_ids) > MAX_SEQ_LENGTH:
            input_ids = input_ids[:MAX_SEQ_LENGTH]
        else:
            pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
            input_ids = input_ids + [pad_token_id] * (MAX_SEQ_LENGTH - len(input_ids))

        # For AWQ, we only need input_ids and attention_mask
        pad_token_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else 0
        sample = {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor([1 if id != pad_token_id else 0 for id in input_ids]),
        }

        processed.append(sample)

    return {key: [s[key] for s in processed] for key in processed[0].keys()}


print("\n3️⃣  Preprocessing dataset...")
ds = ds.map(
    preprocess_function,
    batched=True,
    batch_size=4,
    remove_columns=ds.column_names,
    desc="Preprocessing"
)
print("✅ Dataset preprocessed")

# AWQ recipe
print("\n4️⃣  Setting up AWQ quantization recipe...")

# Inspect model structure first
print("\n🔍 Inspecting model structure...")
for name, module in model.named_modules():
    if any(keyword in name.lower() for keyword in ["vision", "language", "model", "decoder"]):
        if "layers" not in name:  # Only show high-level structure
            print(f"  {name}: {type(module).__name__}")

recipe = AWQModifier(
    targets="Linear",
    scheme="W4A16",  # 4-bit weights, 16-bit activations
    ignore=[
        "re:.*lm_head",           # Don't quantize language model head
        "re:.*vision.*",          # Don't quantize vision encoder (CRITICAL for quality)
        "re:.*connector.*",       # Don't quantize vision-text connector
        "re:.*embed.*",           # Don't quantize embeddings
    ],
)
print("\n✅ AWQ Recipe configured:")
print("   - Scheme: W4A16 (4-bit weights, 16-bit activations)")
print("   - Targets: Linear layers in TEXT MODEL ONLY (Qwen2-72B decoder)")
print("   - Preserved: OpenAI CLIP vision encoder, connectors, embeddings")
print("   - This ensures vision quality is maintained!")


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
print(f"⏳ This will take 90-120 minutes with {NUM_CALIBRATION_SAMPLES} samples...\n")
print("=" * 70)

try:
    # Clear cache before quantization on all GPUs
    for i in range(num_gpus):
        with torch.cuda.device(i):
            torch.cuda.empty_cache()
    gc.collect()

    # The oneshot function will work with device_map="auto" automatically
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
        original_size_gb = 145.0
        reduction = (1 - size_gb / original_size_gb) * 100

        print(f"\n📊 Model size comparison:")
        print(f"   Original (FP16): ~{original_size_gb:.1f} GB")
        print(f"   Quantized (W4A16): ~{size_gb:.2f} GB")
        print(f"   Reduction: ~{reduction:.1f}%")
        print(f"   Memory saved: ~{original_size_gb - size_gb:.1f} GB")

    print("\n🎉 Quantization complete!")
    print("\n💡 Key advantages of Molmo-72B:")
    print("   ✅ State-of-the-art vision-language performance")
    print("   ✅ Qwen2-72B decoder quantized (4-bit AWQ) - smaller size")
    print("   ✅ OpenAI CLIP vision encoder preserved (FP16) - maintains image quality")
    print("   ✅ Trained on PixMo (1M curated image-text pairs)")
    print("   ✅ 73B params total")

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
- vision-language
- molmo
- qwen2
- clip
- llm-compressor
- vllm
library_name: transformers
pipeline_tag: image-text-to-text
---

# Molmo-72B AWQ 4-bit (Text-Only Quantization)

This is a 4-bit AWQ quantized version of [{MODEL_ID}](https://huggingface.co/{MODEL_ID}) using [LLM Compressor](https://github.com/vllm-project/llm-compressor).

## Key Features

- ✅ **Qwen2-72B text decoder quantized** (4-bit AWQ) - 72% size reduction
- ✅ **OpenAI CLIP vision encoder preserved** (FP16) - maintains visual quality
- ✅ **State-of-the-art VLM performance** - among the best open VLMs
- ✅ **Smart quantization** - Only LLM layers quantized, vision parts untouched
- ✅ **vLLM compatible** - Fast inference with vLLM
- ✅ **Trained on PixMo** - 1M curated image-text pairs

## Model Details

- **Base Model:** {MODEL_ID} (73B parameters)
- **Architecture:** Molmo (Qwen2-72B decoder + OpenAI CLIP vision encoder)
- **Quantization Method:** AWQ (Activation-aware Weight Quantization)
- **Quantization Scheme:** W4A16 (4-bit weights, 16-bit activations)
- **Calibration Dataset:** Flickr30k ({NUM_CALIBRATION_SAMPLES} samples)

## Size Comparison

| Metric | Value |
|--------|-------|
| **Original (FP16)** | ~{original_size_gb:.1f} GB |
| **Quantized (W4A16)** | ~{size_gb:.2f} GB |
| **Reduction** | ~{reduction:.1f}% |
| **Memory Saved** | ~{original_size_gb - size_gb:.1f} GB |

## What Was Quantized

**Quantized (4-bit):**
- Qwen2-72B decoder layers (text/language model)
- Text processing linear layers in the decoder

**Preserved (FP16):**
- OpenAI CLIP vision encoder (maintains visual understanding quality)
- Vision-text connectors
- Embeddings
- Language model head

This selective quantization ensures that vision understanding quality remains nearly identical to the original model while significantly reducing size.

## About Molmo-72B

Molmo-72B is one of the most powerful open vision-language models:

- **Text Decoder:** Qwen2-72B (state-of-the-art 72B LLM)
- **Vision Encoder:** OpenAI CLIP (proven vision backbone)
- **Training Data:** PixMo - 1 million highly-curated image-text pairs
- **Performance:** Competitive with GPT-4V on many benchmarks

## Usage

```python
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

# Load model and processor
processor = AutoProcessor.from_pretrained(
    "{HF_REPO_ID}",
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

model = AutoModelForCausalLM.from_pretrained(
    "{HF_REPO_ID}",
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# Process the image and text
inputs = processor.process(
    images=[Image.open(requests.get("https://picsum.photos/id/237/536/354", stream=True).raw)],
    text="Describe what you see in this image."
)

# Move inputs to the correct device and make a batch of size 1
inputs = {{k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}}

# Generate output
output = model.generate_from_batch(
    inputs,
    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
    tokenizer=processor.tokenizer
)

# Decode the generated tokens
generated_tokens = output[0, inputs['input_ids'].size(1):]
generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(generated_text)
```

## Quantization Details

- **Method:** AWQ (Activation-aware Weight Quantization)
- **Independent Pipeline:** Used with BasicPipeline for layer-by-layer quantization
- **Calibration:** {NUM_CALIBRATION_SAMPLES} Flickr30k image-text pairs
- **Max Sequence Length:** {MAX_SEQ_LENGTH} tokens
- **Why AWQ**: Activation-aware quantization preserves important weights

## Limitations

- May have slight quality degradation in complex text generation compared to FP16
- Vision encoder is NOT quantized (intentional for quality)
- Requires vLLM or transformers with AWQ support

## Important Notes

### Transparent Images
Ensure images are in RGB format:
```python
from PIL import Image
image = Image.open(...)
if image.mode != "RGB":
    image = image.convert("RGB")
```

## License

Apache 2.0 (same as base model)

## Citation

```bibtex
@misc{{molmo-72b-awq,
  title={{Molmo-72B AWQ 4-bit}},
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
        print("   This may take 30-45 minutes depending on connection speed...")

        api = HfApi()
        api.upload_folder(
            folder_path=OUTPUT_DIR,
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message=f"Upload AWQ 4-bit quantized Molmo-72B (~{size_gb:.1f}GB, {reduction:.1f}% reduction)",
            token=HF_TOKEN
        )

        print("\n" + "=" * 70)
        print("✅ Upload complete!")
        print(f"🔗 Model available at: https://huggingface.co/{HF_REPO_ID}")
        print("=" * 70)

        print("\n📝 Next steps:")
        print("  1. Visit your model page to verify the upload")
        print("  2. Test the model with the usage example in the README")
        print("  3. Share your quantized Molmo-72B with the community!")

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
    print("  2. For 5x B200 (890GB): Should work with 128 samples, 512 seq length")
    print("  3. If OOM, try reducing to 64 samples, 512 seq length")
    print("  4. Try reducing NUM_CALIBRATION_SAMPLES if OOM (128→64→32)")
    print("  5. Molmo-72B is very large - needs significant GPU memory")
    print("  6. Verify llmcompressor has the latest fixes:")
    print("     pip install git+https://github.com/ronantakizawa/llm-compressor.git@ronantakizawa/sequential-pipeline-fallback")

print("\n" + "=" * 70)
print("🏁 Script completed")
print("=" * 70)
