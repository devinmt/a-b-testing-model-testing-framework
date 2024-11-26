import wandb  # Add this import at the top
from llm_ab_testing import LLMEvaluator, ModelConfig

def run_sample_evaluation():
    try:
        # Initialize evaluator
        evaluator = LLMEvaluator(project_name="llm-testing-demo")
        
        # Test single small model first
        test_config = ModelConfig(
            name="Phi-2 4bit",
            model_id="microsoft/phi-2",
            quantization="4bit",
            batch_size=1,
            max_length=128
        )
        
        print(f"Starting evaluation of {test_config.name}...")
        metrics = evaluator.run_evaluation(test_config)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Latency: {metrics.latency_ms:.2f} ms")
        print(f"Memory Usage: {metrics.memory_mb:.2f} MB")
        print(f"Quality Score: {metrics.quality_score:.2%}")
        print(f"Cost Estimate: ${metrics.cost_estimate:.6f}")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
    finally:
        wandb.finish()

if __name__ == "__main__":
    run_sample_evaluation()