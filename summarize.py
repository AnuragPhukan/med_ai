import torch
from transformers import pipeline

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_index = torch.cuda.current_device()
    print("Using CUDA for acceleration")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    device_index = 0  # MPS uses index 0
    print("Using MPS for acceleration")
else:
    device = torch.device("cpu")
    device_index = -1  # CPU
    print("Using CPU (no GPU acceleration available)")

# Model ID for a summarization model
model_id = "facebook/bart-large-cnn"  

# Initialize the pipeline
pipe = pipeline(
    "summarization",
    model=model_id,
    device=device_index
)

def summarize_text(text):
    """
    Summarizes the input text using a language model.
    """
    if not text:
        return "No valid text to summarize."

    # Truncate the text if necessary
    text = text[:3000]  # Adjust as per model's max input length

    # Generate the summary
    try:
        output = pipe(
            text,
            max_length=250,  
            min_length=80,   
            do_sample=False  # Use greedy decoding for deterministic output
        )
        summary = output[0]["summary_text"].strip()
        return summary
    except Exception as e:
        return f"An error occurred while generating the summary: {e}"

def extract_statistical_data(text):
    """
    Extracts statistical data from the input text using regex.
    """
    import re

    if not text:
        return None

    # Enhanced regex patterns to capture various statistical reporting formats
    patterns = [
        r"(?:effect size|odds ratio|hazard ratio|relative risk)[^\d]*([\d\.]+)[^\d]*\(?95% CI[:]?[\s]?([\d\.\-]+)[^\d]+([\d\.\-]+)\)?",
        r"(?:mean difference)[^\d]*([\d\.\-]+)[^\d]*\(?95% CI[:]?[\s]?([\d\.\-]+)[^\d]+([\d\.\-]+)\)?"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                effect_size = float(match.group(1))
                ci_lower = float(match.group(2))
                ci_upper = float(match.group(3))
                # Ensure ci_lower is less than ci_upper
                ci_lower, ci_upper = min(ci_lower, ci_upper), max(ci_lower, ci_upper)
                return {
                    'effect_size': effect_size,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper
                }
            except ValueError:
                continue  # If conversion fails, try the next pattern
    return None

if __name__ == "__main__":
    # Test the summarization function
    test_text = """
    The rapid technological developments of the past decade and the changes in echocardiographic practice brought about by these developments have resulted in the need for updated recommendations to the previously published guidelines for cardiac chamber quantification, which was the goal of the joint writing group assembled by the American Society of Echocardiography and the European Association of Cardiovascular Imaging. This document provides updated normal values for all four cardiac chambers, including three-dimensional echocardiography and myocardial deformation, when possible, on the basis of considerably larger numbers of normal subjects, compiled from multiple databases. In addition, this document attempts to eliminate several minor discrepancies that existed between previously published guidelines.
    """
    summary = summarize_text(test_text)
    print("Summary:")
    print(summary)
