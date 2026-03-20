# OpNote-AE: Automated Detection of Intraoperative Adverse Events from Spine Operative Notes

## Overview

This repository contains code for automated detection of intraoperative adverse events from spine surgery operative reports using a large language model (LLM).

The system identifies:

- Incidental (unplanned) durotomy  
- Non-durotomy intraoperative adverse events  

This code accompanies the manuscript:

> **“GPT-4-Assisted Detection of Intraoperative Adverse Events from Spine Surgery Operative Reports: A Comparison with Human Reviewers”**

---

## Requirements

Python 3.9 or higher

Install dependencies:

'''bash
pip install -r requirements.txt
'''

---

## API Access

This project requires access to the Azure OpenAI API.

Set environment variables before running:

'''bash
export AZURE_OPENAI_ENDPOINT=YOUR_ENDPOINT
export AZURE_OPENAI_API_KEY=YOUR_API_KEY
export AZURE_OPENAI_API_VERSION=2024-02-01
'''

---

## Data

Real operative notes cannot be shared due to patient privacy restrictions.

A synthetic example dataset is provided:
example_data/synthetic_oper_notes.csv

Users may substitute their own dataset in the same format.

---

## Running the Pipeline

From the repository root:

'''bash
python src/detect_adverse_events.py \
  --input_csv example_data/synthetic_oper_notes.csv \
  --output_csv results/sample_predictions.csv \
  --prompt_name adverse_event
'''

---

## Output

The pipeline produces a CSV file containing:

- Predicted adverse event classification
- Additional model response details

---

## Prompts

Prompt templates are located in the prompts/ directory and can be modified for different classification tasks.

---

## License

MIT License
