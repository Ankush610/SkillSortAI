# 🤖 SkillSort AI — Intelligent Resume Analyzer & Ranker

SkillSort AI is a Gradio-based intelligent resume ranking system powered by the **LLaMA 3.1 Storm 8B** model. It extracts, analyzes, and ranks resumes using AI, comparing each candidate’s skills and experience against custom job requirements. Get visual insights, sorted results, and ATS-style evaluations—all in one sleek app.

## 🚀 Features

- ✅ **Extract resume content** from `.pdf`, `.docx`, or `.txt`
- 🧠 **LLM-powered analysis** using LLaMA-3.1-Storm-8B via Hugging Face
- 📊 **Score candidates** on:
  - Skills match
  - Experience relevance
  - Job stability (Behavior Index)
- 🔢 **Generate Overall Score** (out of 100) based on sub-scores
- 📈 **Visualize data** with interactive plots using Matplotlib & Seaborn
- 💾 **Export to Excel** for further analysis
- 🖥️ **Gradio UI** for seamless interaction

## 🔧 Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/SkillSort-AI.git
cd SkillSort-AI
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required libraries**

```bash
pip install torch transformers accelerate gradio python-docx pymupdf python-dotenv huggingface_hub matplotlib seaborn pandas openpyxl
```

4. **Add your Hugging Face token**

Create a `.env` file in the root directory with the following:

```
HUGGINGFACE_TOKEN=your_hf_token_here
```

5. **Run the application**

```bash
python app.py
```

## 📊 Scoring Criteria

| Metric             | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| **Skills Score**   | Compatibility between resume skills and job requirements (0-10)            |
| **Experience Score** | Match between candidate’s experience and job’s requirements (0-10)        |
| **Behavior Index** | Stability of employment based on tenure (0-10)                             |
| **Overall Score**  | Aggregated score scaled to 100                                              |

## 📁 Supported Resume Formats

- PDF (`.pdf`)
- Word (`.docx`)
- Plain text (`.txt`)

## 🛡️ Tech Stack

- 🧠 LLM: `LLaMA-3.1-Storm-8B` (via HuggingFace)
- 🐍 Python
- 🎨 Gradio for UI
- 📊 Pandas, Seaborn, Matplotlib for analysis
- 🔐 Dotenv, Accelerate for HuggingFace setup

---
<p align="center">
  <img src="./resumeAI.gif" alt="Animated Coding GIF" width="1000"/>
</p>

