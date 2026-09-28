# Customer Feedback Analysis & AI Report Generator

An end-to-end Python project that combines **Pandas** for statistical data processing and **Hugging Face LLMs** to automatically analyze customer feedback, extract key negative insights, and generate structured executive Markdown reports.

---

##  Features
- **Data Processing:** Loads customer feedback datasets (`.csv`), calculates statistical metrics (Average Rating, Positive/Negative/Neutral counts), and filters the most frequent negative comments.
- **AI-Powered Analysis:** Uses advanced Large Language Models (`google/gemma-3-1b-it` via Hugging Face Transformers) to perform qualitative analysis and generate professional executive summaries.
- **Automated Reporting:** Exports the structured analytical report directly into a clean Markdown (`.md`) file.

---

## Tech Stack
- **Python**
- **Pandas** (Data manipulation & statistics)
- **Hugging Face Transformers** (LLM pipeline & Chat Templates)

---

##  Project Structure
```text
 customer-feedback-analysis-llm
 ┣  customer_feedback.csv          # Raw customer feedback dataset
 ┣  analysis_script.py             # Main Python script
 ┣  Executive_Feedback_Report.md   # Generated AI analytical report
 ┗  README.md                      # Project documentation
