# EDA Workflow

An AI-powered exploratory data analysis workflow that performs consistent, first-pass analysis of datasets using LangChain and LangGraph. The workflow runs a fixed set of analysis tools, uses an LLM to extract observations after each step, and synthesizes findings into a summary with actionable recommendations.

## How It Works

The workflow follows a sequential process:
1. **Analyze**: Runs a fixed set of predefined analysis tools on the dataset
2. **Observe**: After each tool, the LLM extracts concise observations from the results
3. **Synthesize**: Once all tools have run, the LLM summarizes findings and provides actionable recommendations

This approach combines deterministic pandas-based analysis with LLM-powered interpretation.

## Setup

### Prerequisites

- **Python 3.10 or 3.11**
- **Poetry** (dependency manager)
- **OpenAI API Key**

### Installation Steps

1. **Install Poetry** (if not already installed):
   
   **Windows (PowerShell)**:
   ```powershell
   (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
   ```
   
   **macOS/Linux**:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   
   After installation, restart your terminal. If `poetry` command is not found:
   - **Windows**: Add `%APPDATA%\Python\Scripts` to your system PATH
   - **macOS/Linux**: Add `export PATH="$HOME/.local/bin:$PATH"` to your `~/.bashrc` or `~/.zshrc`

2. **Install dependencies**:
   ```bash
   poetry install
   ```
   
   This will install all dependencies with the exact versions specified in `poetry.lock`, ensuring consistency across all environments.

3. **Set up your OpenAI API key**:
   
   **Windows**:
   ```powershell
   copy .env.example .env
   ```
   
   **macOS/Linux**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=sk-your-key-here
   ```

### Multiple Python Versions?

If you have multiple Python versions installed and want to use a specific one:

```bash
# Tell Poetry which Python to use
poetry env use python3.11  # or python3.10

# Then install dependencies
poetry install
```

Poetry will create a virtual environment with your chosen Python version.

## Usage

### Python API

```python
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from eda_workflow.eda_workflow import EDAWorkflow

load_dotenv()

# Initialize the workflow with an LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
workflow = EDAWorkflow(model=llm)

# Run analysis on a dataset
workflow.invoke_workflow("data/cafe_sales.csv")

# Retrieve results
summary = workflow.get_summary()              # str
recommendations = workflow.get_recommendations()  # list[str]
observations = workflow.get_observations()    # dict[str, list[str]]
results = workflow.get_results()              # dict
```

### Running the Example

```bash
poetry run python example_usage.py
```

This runs a full analysis on the sample dataset and prints the results for each step.

## Analysis Steps

The workflow currently runs the following analysis sequence:

1. **Dataset Profile**
   - Captures dataset shape, column names, data types, numeric summaries, and categorical summaries.

2. **Missingness Analysis**
   - Calculates missing counts and missing percentages.
   - Flags columns with high missingness.

3. **Aggregate Analysis**
   - Evaluates group-level differences across valid categorical columns and numeric metrics.
   - Stores full aggregate summaries in workflow state while showing a compact terminal summary.
   - Stores both full aggregate summaries and compact high-level aggregate deviations to support synthesized LLM-derived insights.
4. **Relationship Analysis**
   - Calculates numeric correlations.
   - Filters weak correlations.
   - Separates strongest positive and negative relationships.

After each step, the LLM extracts concise observations. The final synthesis combines those observations into a summary and recommendations.

## Output

Running `example_usage.py` will:

- print each analysis step
- print LLM-generated observations after each step
- generate a final summary and recommendations
- save a LangGraph diagram as `graph.png`

## Example Output

### Workflow Graph

![Workflow Graph](images/graph.png)


### Example Workflow Analysis

![Workflow Output](images/workflow_analysis_output.png)

## Future Improvements

- Add semantic filtering for likely "identifier columns" (example customer_id) before aggregate and correlation analysis.
- Improve filtering of low-information, sparse, and identifier-like features.
- Improve detection of structurally expected vs analytically meaningful relationships.
- Extend aggregate analysis with more robust outlier handling and statistical summaries.
- Improve domain-aware interpretation of EDA findings.

Tested on additional real-world datasets including housing data, which identified the need for additional improvements around feature filtering and relationship interpretation.

## Project Structure

```
eda-workflow/
├── data/
│   └── cafe_sales.csv             # Sample dataset
├── eda_workflow/
│   ├── __init__.py
│   ├── eda_workflow.py            # Main workflow class and graph
│   └── prompts/                   # LLM prompt templates
│       ├── extract_observations_system.txt
│       ├── extract_observations_human.txt
│       ├── synthesize_findings_system.txt
│       └── synthesize_findings_human.txt
├── .env.example                   # Environment variable template
├── example_usage.py               # Example script
├── pyproject.toml                 # Dependencies configuration
├── poetry.lock                    # Locked dependency versions
└── README.md
```

**Important**: The `poetry.lock` file is committed to ensure all users get identical, tested dependency versions.