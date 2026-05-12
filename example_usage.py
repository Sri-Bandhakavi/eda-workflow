"""
Example usage of the EDA Workflow with OpenAI.

This demonstrates how to:
1. Initialize the OpenAI model
2. Create an EDA workflow
3. Run analysis on a dataset
4. Retrieve results

Requires: OPENAI_API_KEY in .env file or environment variable
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from eda_workflow.eda_workflow import EDAWorkflow

# Load environment variables from .env file
load_dotenv()

# Path to sample dataset
data_path = os.path.join("data", "cafe_sales.csv")

# Initialize OpenAI model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# Create EDA workflow with the model
workflow = EDAWorkflow(model=llm)

# Save a visual diagram of the graph
workflow._compiled_graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
print("Graph diagram saved to graph.png\n")

# Run analysis on the dataset
print("Running EDA analysis...\n")
workflow.invoke_workflow(data_path)

# Retrieve results
summary = workflow.get_summary()
recommendations = workflow.get_recommendations()
observations = workflow.get_observations()
results = workflow.get_results()

# Display results sequentially: tool result → observations → next tool
analysis_steps = [
    ("profile_dataset", "Dataset Profile"),
    ("analyze_missingness", "Missingness Analysis"),
    ("compute_aggregates", "Aggregates Analysis"),
    ("analyze_relationships", "Relationships Analysis"),
]

for step_key, step_title in analysis_steps:
    print("=" * 60)
    print(f"{step_title.upper()}")
    print("=" * 60)
    
    # Show results
    if step_key in results:
        step_results = results[step_key]
        # show compact aggregate summary only
        if step_key == "compute_aggregates":
            display_summary = step_results.get("display_summary", step_results)
            for key, value in display_summary.items():
                print(f"{key}: {value}")
        # default printing behavior for all other nodes
        else:
            for key, value in step_results.items():
                if isinstance(value, dict) and len(str(value)) > 200:
                    print(f"{key}: {type(value).__name__} with {len(value)} items")
                else:
                    print(f"{key}: {value}")
        
    # Show observations
    print(f"\nObservations:")
    if step_key in observations and observations[step_key]:
        for obs in observations[step_key]:
            print(f"  • {obs}")
    else:
        print("  (No observations)")
    print()

# Final synthesis
print("=" * 60)
print("FINAL SYNTHESIS")
print("=" * 60)
print(f"\nSummary:\n{summary if summary else '(Not implemented yet)'}")
print("\nRecommendations:")
if recommendations:
    for rec in recommendations:
        print(f"  • {rec}")
else:
    print("  (Not implemented yet)")
