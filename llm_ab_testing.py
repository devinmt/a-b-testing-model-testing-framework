import wandb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
from datasets import load_dataset
import time
from typing import Dict, List, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    model_id: str
    quantization: str  # '4bit', '8bit', 'none'
    batch_size: int
    max_length: int

@dataclass
class ModelMetrics:
    latency_ms: float
    memory_mb: float
    quality_score: float
    cost_estimate: float

class LLMEvaluator:
    def __init__(self, project_name: str = "llm-ab-testing"):
        """Initialize WandB and setup evaluation environment."""
        wandb.init(project=project_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_dataset = load_dataset("truthful_qa", "multiple_choice", split="validation[:100]")
        
    def load_model(self, config: ModelConfig) -> tuple:
        """Load model with specified configuration."""
        kwargs = {}
        if config.quantization == "4bit":
            kwargs = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.float16
            }
        elif config.quantization == "8bit":
            kwargs = {"load_in_8bit": True}
            
        model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            **kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        return model, tokenizer

    def measure_memory(self) -> float:
        """Measure current GPU memory usage."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
        return 0.0

    def evaluate_quality(self, pipeline, samples: List[str]) -> float:
        """Evaluate model quality on test dataset."""
        correct = 0
        total = len(samples)
        
        for sample in samples:
            prompt = f"Question: {sample['question']}\nChoices: {sample['mc1_targets']['choices']}\nAnswer:"
            response = pipeline(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
            
            # Simple accuracy check - could be enhanced with more sophisticated metrics
            if any(correct_answer.lower() in response.lower() 
                  for correct_answer in sample['mc1_targets']['correct_answers']):
                correct += 1
                
        return correct / total

    def estimate_cost(self, total_tokens: int, model_name: str) -> float:
        """Estimate inference cost based on token count and model size."""
        # Simplified cost estimation - could be enhanced with more precise calculations
        base_cost_per_1k_tokens = 0.0001  # Example rate
        size_multiplier = 1.5 if "large" in model_name.lower() else 1.0
        return (total_tokens / 1000) * base_cost_per_1k_tokens * size_multiplier

    def run_evaluation(self, config: ModelConfig) -> ModelMetrics:
        """Run complete evaluation for a model configuration."""
        # Start tracking run with config
        wandb.config.update(vars(config))
        
        # Load model and measure initial metrics
        start_time = time.time()
        model, tokenizer = self.load_model(config)
        pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=self.device)
        
        # Measure latency
        sample_texts = self.test_dataset["question"][:10]  # Take small sample for latency test
        latency_start = time.time()
        for text in sample_texts:
            _ = pipeline(text, max_length=config.max_length)
        avg_latency = (time.time() - latency_start) * 1000 / len(sample_texts)  # Convert to ms
        
        # Measure memory usage
        memory_usage = self.measure_memory()
        
        # Evaluate quality
        quality_score = self.evaluate_quality(pipeline, self.test_dataset[:50])
        
        # Estimate cost
        total_tokens = sum(len(tokenizer.encode(q)) for q in sample_texts)
        cost_estimate = self.estimate_cost(total_tokens, config.name)
        
        # Log metrics to WandB
        metrics = ModelMetrics(
            latency_ms=avg_latency,
            memory_mb=memory_usage,
            quality_score=quality_score,
            cost_estimate=cost_estimate
        )
        
        wandb.log({
            "latency_ms": metrics.latency_ms,
            "memory_mb": metrics.memory_mb,
            "quality_score": metrics.quality_score,
            "cost_estimate": metrics.cost_estimate
        })
        
        return metrics

def main():
    # Define configurations to test
    configs = [
        ModelConfig(
            name="Phi-2 4bit",
            model_id="microsoft/phi-2",
            quantization="4bit",
            batch_size=1,
            max_length=128
        ),
        ModelConfig(
            name="RWKV 8bit",
            model_id="RWKV/rwkv-4-pile-169m",
            quantization="8bit",
            batch_size=1,
            max_length=128
        )
    ]
    
    evaluator = LLMEvaluator()
    
    # Run evaluations
    results = {}
    for config in configs:
        print(f"Evaluating {config.name}...")
        metrics = evaluator.run_evaluation(config)
        results[config.name] = metrics
        
        print(f"Results for {config.name}:")
        print(f"Latency: {metrics.latency_ms:.2f}ms")
        print(f"Memory Usage: {metrics.memory_mb:.2f}MB")
        print(f"Quality Score: {metrics.quality_score:.2%}")
        print(f"Estimated Cost: ${metrics.cost_estimate:.4f}")
        print("-" * 50)
    
    wandb.finish()

if __name__ == "__main__":
    main()