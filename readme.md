# LLM A/B Testing Framework

A lightweight framework for evaluating and comparing different LLM models using metrics like latency, memory usage, quality, and cost estimation. Results are tracked using Weights & Biases and can be visualized in Tableau.

## Features

- Model evaluation across multiple metrics:
  - Inference latency
  - Memory usage
  - Quality score (using TruthfulQA dataset)
  - Cost estimation
- Support for model quantization (4-bit, 8-bit)
- Integration with Weights & Biases for experiment tracking
- CSV export for Tableau visualization
- Support for multiple LLM models (Phi-2, RWKV, etc.)

## Setup

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Weights & Biases account

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/llm-ab-testing.git
cd llm-ab-testing
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

### Requirements
```
wandb==0.18.7
torch==2.1.2
transformers==4.36.2
datasets==2.15.0
bitsandbytes==0.41.1
accelerate>=0.26.0
```

## Usage

### Basic Usage

1. Login to Weights & Biases:
```bash
wandb login
```

2. Run sample evaluation:
```bash
python test.py
```

### Custom Model Evaluation

Modify test.py to evaluate different models:

```python
test_config = ModelConfig(
    name="Your-Model-Name",
    model_id="your-model-id",
    quantization="4bit",  # or "8bit" or "none"
    batch_size=1,
    max_length=128
)
```

## Project Structure

```
llm-ab-testing/
├── requirements.txt
├── README.md
├── test.py                # Sample usage script
├── llm_evaluator.py       # Main evaluation code
└── wandb/                 # Weights & Biases logs
```

## Metrics Explained

1. **Latency (ms)**
   - Average inference time per request
   - Lower is better

2. **Memory Usage (MB)**
   - Peak GPU memory consumption
   - Important for resource planning

3. **Quality Score**
   - Based on TruthfulQA benchmark
   - Higher is better (max 1.0)

4. **Cost Estimate**
   - Estimated inference cost per 1000 tokens
   - Based on model size and token count

## Visualization in Tableau

1. Export results:
   - Results are automatically saved to CSV
   - Located in the project directory

2. Import to Tableau:
   - Open Tableau Public
   - Connect to CSV file
   - Create visualizations:
     * Model comparison dashboard
     * Performance metrics over time
     * Resource utilization analysis

## Sample Results

Example output format:
```
Starting evaluation of Phi-2 4bit...
Evaluation Results:
Latency: 245.67 ms
Memory Usage: 2048.45 MB
Quality Score: 78.50%
Cost Estimate: $0.000135
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## Limitations

- Requires CUDA-compatible GPU for optimal performance
- Limited to models available through Hugging Face
- Cost estimates are approximations

## Troubleshooting

Common issues and solutions:

1. CUDA Out of Memory:
   - Reduce batch size
   - Use model quantization
   - Try smaller models

2. Import Errors:
   - Ensure all requirements are installed
   - Check Python version compatibility

3. Wandb Connection Issues:
   - Verify internet connection
   - Check wandb login status
   - Run `wandb login --relogin`

## Future Improvements

- [ ] Add support for more metrics
- [ ] Implement prompt templating
- [ ] Add custom dataset support
- [ ] Enhance cost estimation accuracy
- [ ] Add support for model ensembles

## License

MIT License

## Acknowledgments

- Hugging Face Transformers
- Weights & Biases
- TruthfulQA Dataset

## Contact

- Your Name
- Email: your.email@example.com
- Project Link: https://github.com/yourusername/llm-ab-testing