#!/usr/bin/env python3
"""
Quantization Quality Evaluation Script

Compares base models with their AWQ-quantized versions across multiple metrics:
1. Perplexity (lower is better)
2. KL Divergence (lower is better - measures distribution shift)
3. Benchmark Performance (MMLU for language models, VQA for vision models)

Models to evaluate:
- allenai/Molmo-72B-0924 vs ronantakizawa/molmo-72b-awq
- allenai/OLMo-2-0325-32B-Instruct vs ronantakizawa/olmo2-32b-instruct-awq
- allenai/MolmoAct-7B-D-0812 vs ronantakizawa/molmoact-7b-d-awq
- allenai/Molmo-7B-D-0924 vs ronantakizawa/molmo-7b-d-awq
"""

import gc
import os
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
)
from torch.nn import functional as F
from tqdm import tqdm
from huggingface_hub import login
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model pairs: (base_model, quantized_model, model_type)
MODEL_PAIRS = [
    {
        "name": "Molmo-72B",
        "base": "allenai/Molmo-72B-0924",
        "quantized": "ronantakizawa/molmo-72b-awq",
        "type": "vlm",  # Vision-Language Model
    },
    {
        "name": "OLMo-2-32B-Instruct",
        "base": "allenai/OLMo-2-0325-32B-Instruct",
        "quantized": "ronantakizawa/olmo2-32b-instruct-awq",
        "type": "llm",  # Language Model
    },
    {
        "name": "MolmoAct-7B-D",
        "base": "allenai/MolmoAct-7B-D-0812",
        "quantized": "ronantakizawa/molmoact-7b-d-awq",
        "type": "vlm",
    },
    {
        "name": "Molmo-7B-D",
        "base": "allenai/Molmo-7B-D-0924",
        "quantized": "ronantakizawa/molmo-7b-d-awq",
        "type": "vlm",
    },
]

# Evaluation settings
PERPLEXITY_SAMPLES = 100  # Number of samples for perplexity evaluation
KL_DIVERGENCE_SAMPLES = 50  # Number of samples for KL divergence
BENCHMARK_SAMPLES = 100  # Number of samples for benchmark evaluation
MAX_SEQ_LENGTH = 512
OUTPUT_DIR = "/workspace/quantization_evaluation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# HUGGING FACE LOGIN
# ============================================================================

print("🔐 Logging into Hugging Face...")
login(token=hf_token)
print("✅ Logged in successfully\n")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clear_memory():
    """Clear GPU and CPU memory"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()

def load_model_and_tokenizer(model_id, model_type):
    """Load model and tokenizer/processor based on type"""
    print(f"   Loading from {model_id}...")

    if model_type == "vlm":
        # Vision-Language Model - use processor
        processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        return model, processor, "processor"
    else:
        # Language Model - use tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        return model, tokenizer, "tokenizer"

# ============================================================================
# PERPLEXITY EVALUATION
# ============================================================================

def calculate_perplexity(model, tokenizer, dataset, num_samples, model_type):
    """
    Calculate perplexity on a dataset.
    Perplexity measures how well the model predicts the next token.
    Lower perplexity = better model.
    """
    print(f"      Calculating perplexity on {num_samples} samples...")

    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="      Perplexity"):
            sample = dataset[i]

            # Tokenize
            if model_type == "vlm":
                # For VLMs, use text-only for perplexity
                text = sample.get("text", sample.get("question", ""))
            else:
                text = sample.get("text", sample.get("question", ""))

            if not text or len(text) < 10:
                continue

            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=MAX_SEQ_LENGTH,
                truncation=True,
                padding=False,
            )

            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Calculate loss
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss

            # Accumulate
            total_loss += loss.item() * inputs["input_ids"].size(1)
            total_tokens += inputs["input_ids"].size(1)

    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity

# ============================================================================
# KL DIVERGENCE EVALUATION
# ============================================================================

def calculate_kl_divergence(base_model, quant_model, tokenizer, dataset, num_samples, model_type):
    """
    Calculate KL divergence between base and quantized model outputs.
    KL divergence measures how much the quantized model's probability distribution
    differs from the base model.
    Lower KL divergence = quantized model is closer to base model.
    """
    print(f"      Calculating KL divergence on {num_samples} samples...")

    base_model.eval()
    quant_model.eval()

    total_kl = 0
    count = 0

    with torch.no_grad():
        for i in tqdm(range(min(num_samples, len(dataset))), desc="      KL Divergence"):
            sample = dataset[i]

            # Get text
            if model_type == "vlm":
                text = sample.get("text", sample.get("question", ""))
            else:
                text = sample.get("text", sample.get("question", ""))

            if not text or len(text) < 10:
                continue

            inputs = tokenizer(
                text,
                return_tensors="pt",
                max_length=MAX_SEQ_LENGTH // 2,  # Use shorter sequences for KL
                truncation=True,
                padding=False,
            )

            # Move to same device as base model
            inputs = {k: v.to(base_model.device) for k, v in inputs.items()}

            # Get logits from both models
            base_outputs = base_model(**inputs)
            base_logits = base_outputs.logits

            # Move inputs to quantized model device
            inputs = {k: v.to(quant_model.device) for k, v in inputs.items()}
            quant_outputs = quant_model(**inputs)
            quant_logits = quant_outputs.logits

            # Move quant_logits to same device as base_logits for comparison
            quant_logits = quant_logits.to(base_logits.device)

            # Calculate KL divergence
            # KL(P||Q) where P is base model, Q is quantized model
            base_probs = F.softmax(base_logits, dim=-1)
            quant_log_probs = F.log_softmax(quant_logits, dim=-1)

            kl_div = F.kl_div(
                quant_log_probs,
                base_probs,
                reduction='batchmean',
                log_target=False
            )

            total_kl += kl_div.item()
            count += 1

    avg_kl = total_kl / count if count > 0 else float('inf')
    return avg_kl

# ============================================================================
# BENCHMARK EVALUATION (MMLU for LLMs)
# ============================================================================

def evaluate_mmlu(model, tokenizer, num_samples):
    """
    Evaluate on MMLU (Massive Multitask Language Understanding) benchmark.
    Tests general knowledge and reasoning across 57 subjects.
    """
    print(f"      Evaluating on MMLU ({num_samples} samples)...")

    # Load MMLU dataset
    dataset = load_dataset("cais/mmlu", "all", split="test")
    dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in tqdm(dataset, desc="      MMLU"):
            question = sample["question"]
            choices = sample["choices"]
            answer = sample["answer"]  # Index of correct answer (0-3)

            # Format as multiple choice
            prompt = f"Question: {question}\n\nChoices:\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\n\nAnswer:"

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LENGTH)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

            # Get predicted token
            predicted_token = outputs[0, -1]
            predicted_text = tokenizer.decode(predicted_token).strip().upper()

            # Map answer index to letter
            answer_letter = ["A", "B", "C", "D"][answer]

            if predicted_text == answer_letter:
                correct += 1
            total += 1

    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy

# ============================================================================
# VQA EVALUATION (for Vision-Language Models)
# ============================================================================

def evaluate_vqa(model, processor, num_samples):
    """
    Evaluate on VQAv2 (Visual Question Answering) benchmark.
    For vision-language models only.
    """
    print(f"      Evaluating on VQAv2 ({num_samples} samples)...")

    try:
        # Load VQA dataset (using a simplified version)
        dataset = load_dataset("HuggingFaceM4/VQAv2", split="validation")
        dataset = dataset.shuffle(seed=42).select(range(min(num_samples, len(dataset))))
    except:
        print("      ⚠️  VQA dataset not available, skipping...")
        return None

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for sample in tqdm(dataset, desc="      VQA"):
            try:
                image = sample["image"]
                question = sample["question"]
                answers = sample.get("answers", [])

                if not answers:
                    continue

                # Convert image to RGB
                if image.mode != "RGB":
                    image = image.convert("RGB")

                # Process
                inputs = processor.process(
                    images=[image],
                    text=question,
                )
                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                         for k, v in inputs.items()}

                # Generate answer
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )

                # Decode
                predicted = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted = predicted.lower().strip()

                # Check if prediction matches any of the ground truth answers
                ground_truth = [a["answer"].lower().strip() for a in answers]
                if any(predicted in gt or gt in predicted for gt in ground_truth):
                    correct += 1

                total += 1

            except Exception as e:
                print(f"      Error processing sample: {e}")
                continue

    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy

# ============================================================================
# MAIN EVALUATION LOOP
# ============================================================================

def evaluate_model_pair(pair_config):
    """Evaluate a single model pair"""

    print(f"\n{'='*70}")
    print(f"📊 EVALUATING: {pair_config['name']}")
    print(f"{'='*70}\n")

    results = {
        "model_name": pair_config["name"],
        "base_model": pair_config["base"],
        "quantized_model": pair_config["quantized"],
        "model_type": pair_config["type"],
    }

    # Load dataset for perplexity and KL divergence
    print("📚 Loading evaluation dataset...")
    if pair_config["type"] == "llm":
        # Use OpenOrca for language models
        eval_dataset = load_dataset(
            "Open-Orca/OpenOrca",
            split=f"train[:{max(PERPLEXITY_SAMPLES, KL_DIVERGENCE_SAMPLES)}]"
        )
    else:
        # Use text from a general dataset for VLMs
        eval_dataset = load_dataset(
            "Open-Orca/OpenOrca",
            split=f"train[:{max(PERPLEXITY_SAMPLES, KL_DIVERGENCE_SAMPLES)}]"
        )

    print(f"✅ Loaded {len(eval_dataset)} samples\n")

    # ========================================================================
    # EVALUATE BASE MODEL
    # ========================================================================

    print(f"🔵 Evaluating BASE model: {pair_config['base']}")
    base_model, base_tokenizer, tokenizer_type = load_model_and_tokenizer(
        pair_config["base"],
        pair_config["type"]
    )

    # Perplexity
    base_perplexity = calculate_perplexity(
        base_model,
        base_tokenizer if tokenizer_type == "tokenizer" else base_tokenizer.tokenizer,
        eval_dataset,
        PERPLEXITY_SAMPLES,
        pair_config["type"]
    )
    results["base_perplexity"] = base_perplexity
    print(f"      ✅ Perplexity: {base_perplexity:.2f}\n")

    # Benchmark
    if pair_config["type"] == "llm":
        base_benchmark = evaluate_mmlu(
            base_model,
            base_tokenizer,
            BENCHMARK_SAMPLES
        )
        results["base_mmlu_accuracy"] = base_benchmark
        print(f"      ✅ MMLU Accuracy: {base_benchmark:.2f}%\n")
    else:
        base_benchmark = evaluate_vqa(
            base_model,
            base_tokenizer,
            BENCHMARK_SAMPLES
        )
        if base_benchmark is not None:
            results["base_vqa_accuracy"] = base_benchmark
            print(f"      ✅ VQA Accuracy: {base_benchmark:.2f}%\n")

    # ========================================================================
    # EVALUATE QUANTIZED MODEL
    # ========================================================================

    print(f"🟢 Evaluating QUANTIZED model: {pair_config['quantized']}")
    quant_model, quant_tokenizer, _ = load_model_and_tokenizer(
        pair_config["quantized"],
        pair_config["type"]
    )

    # Perplexity
    quant_perplexity = calculate_perplexity(
        quant_model,
        quant_tokenizer if tokenizer_type == "tokenizer" else quant_tokenizer.tokenizer,
        eval_dataset,
        PERPLEXITY_SAMPLES,
        pair_config["type"]
    )
    results["quantized_perplexity"] = quant_perplexity
    print(f"      ✅ Perplexity: {quant_perplexity:.2f}\n")

    # Benchmark
    if pair_config["type"] == "llm":
        quant_benchmark = evaluate_mmlu(
            quant_model,
            quant_tokenizer,
            BENCHMARK_SAMPLES
        )
        results["quantized_mmlu_accuracy"] = quant_benchmark
        print(f"      ✅ MMLU Accuracy: {quant_benchmark:.2f}%\n")
    else:
        quant_benchmark = evaluate_vqa(
            quant_model,
            quant_tokenizer,
            BENCHMARK_SAMPLES
        )
        if quant_benchmark is not None:
            results["quantized_vqa_accuracy"] = quant_benchmark
            print(f"      ✅ VQA Accuracy: {quant_benchmark:.2f}%\n")

    # ========================================================================
    # CALCULATE KL DIVERGENCE
    # ========================================================================

    print(f"📏 Calculating KL Divergence between models...")
    kl_divergence = calculate_kl_divergence(
        base_model,
        quant_model,
        base_tokenizer if tokenizer_type == "tokenizer" else base_tokenizer.tokenizer,
        eval_dataset,
        KL_DIVERGENCE_SAMPLES,
        pair_config["type"]
    )
    results["kl_divergence"] = kl_divergence
    print(f"      ✅ KL Divergence: {kl_divergence:.4f}\n")

    # ========================================================================
    # CALCULATE DEGRADATION
    # ========================================================================

    perplexity_degradation = ((quant_perplexity - base_perplexity) / base_perplexity) * 100
    results["perplexity_degradation_pct"] = perplexity_degradation

    if pair_config["type"] == "llm":
        benchmark_degradation = quant_benchmark - base_benchmark
        results["mmlu_degradation_pct"] = benchmark_degradation
    else:
        if base_benchmark is not None and quant_benchmark is not None:
            benchmark_degradation = quant_benchmark - base_benchmark
            results["vqa_degradation_pct"] = benchmark_degradation

    # Clean up
    del base_model, quant_model
    clear_memory()

    return results

# ============================================================================
# RUN ALL EVALUATIONS
# ============================================================================

print(f"\n{'='*70}")
print("🚀 STARTING QUANTIZATION QUALITY EVALUATION")
print(f"{'='*70}\n")
print(f"Models to evaluate: {len(MODEL_PAIRS)}")
print(f"Perplexity samples: {PERPLEXITY_SAMPLES}")
print(f"KL divergence samples: {KL_DIVERGENCE_SAMPLES}")
print(f"Benchmark samples: {BENCHMARK_SAMPLES}")
print(f"Results will be saved to: {OUTPUT_DIR}\n")

all_results = []

for pair in MODEL_PAIRS:
    try:
        results = evaluate_model_pair(pair)
        all_results.append(results)

        # Save individual results
        output_file = os.path.join(
            OUTPUT_DIR,
            f"{pair['name'].replace(' ', '_').lower()}_results.json"
        )
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"💾 Saved results to: {output_file}\n")

    except Exception as e:
        print(f"\n❌ Error evaluating {pair['name']}: {e}")
        import traceback
        traceback.print_exc()
        continue

# ============================================================================
# GENERATE SUMMARY REPORT
# ============================================================================

print(f"\n{'='*70}")
print("📊 GENERATING SUMMARY REPORT")
print(f"{'='*70}\n")

summary_file = os.path.join(OUTPUT_DIR, "summary_report.json")
with open(summary_file, "w") as f:
    json.dump(all_results, f, indent=2)

print(f"💾 Summary saved to: {summary_file}\n")

# Print summary table
print(f"\n{'='*70}")
print("📈 RESULTS SUMMARY")
print(f"{'='*70}\n")

for result in all_results:
    print(f"Model: {result['model_name']}")
    print(f"  Perplexity:")
    print(f"    Base:      {result.get('base_perplexity', 'N/A'):.2f}")
    print(f"    Quantized: {result.get('quantized_perplexity', 'N/A'):.2f}")
    print(f"    Degradation: {result.get('perplexity_degradation_pct', 'N/A'):.2f}%")
    print(f"  KL Divergence: {result.get('kl_divergence', 'N/A'):.4f}")

    if result["model_type"] == "llm":
        print(f"  MMLU Accuracy:")
        print(f"    Base:      {result.get('base_mmlu_accuracy', 'N/A'):.2f}%")
        print(f"    Quantized: {result.get('quantized_mmlu_accuracy', 'N/A'):.2f}%")
        print(f"    Difference: {result.get('mmlu_degradation_pct', 'N/A'):.2f}%")
    else:
        if 'base_vqa_accuracy' in result:
            print(f"  VQA Accuracy:")
            print(f"    Base:      {result.get('base_vqa_accuracy', 'N/A'):.2f}%")
            print(f"    Quantized: {result.get('quantized_vqa_accuracy', 'N/A'):.2f}%")
            print(f"    Difference: {result.get('vqa_degradation_pct', 'N/A'):.2f}%")
    print()

print(f"{'='*70}")
print("✅ EVALUATION COMPLETE!")
print(f"{'='*70}\n")
