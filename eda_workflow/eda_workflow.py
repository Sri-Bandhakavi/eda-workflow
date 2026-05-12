import logging
import os
from typing import Optional, TypedDict

import pandas as pd
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

logger = logging.getLogger(__name__)
WORKFLOW_NAME = "eda_workflow"
LOG_PATH = os.path.join(os.getcwd(), "logs/")
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")


def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join(PROMPTS_DIR, filename)
    with open(prompt_path, "r") as f:
        return f.read()


class EDAWorkflow:
    """
    Exploratory Data Analysis workflow that performs consistent, first-pass analysis of datasets.
    
    Uses a fixed set of predefined analysis tools to produce structured, tabular outputs.
    Operates sequentially and deterministically through baseline EDA steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Attributes
    ----------
    response : dict or None
        Stores the full response after invoke_workflow() is called.
    """
    
    def __init__(
        self,
        model=None,
        log=False,
        log_path=None,
        checkpointer: Optional[object] = None
    ):
        self.model = model
        self.log = log
        self.log_path = log_path
        self.checkpointer = checkpointer
        self.response = None
        self._compiled_graph = make_eda_baseline_workflow(
            model=model,
            log=log,
            log_path=log_path,
            checkpointer=checkpointer
        )
    
    def invoke_workflow(self, filepath: str, **kwargs):
        """
        Run EDA analysis on the provided dataset.
        
        Parameters
        ----------
        filepath : str
            Path to the dataset file.
        **kwargs
            Additional arguments passed to the underlying graph invoke method.
        
        Returns
        -------
        None
            Results are stored in self.response and accessed via getter methods.
        """
        df = pd.read_csv(filepath)
        
        response = self._compiled_graph.invoke({
            "dataframe": df.to_dict(),
            "results": {},
            "observations": {},
            "current_step": "",
            "summary": "",
            "recommendations": [],
        }, **kwargs)
        
        self.response = response
        return None
    
    def get_summary(self):
        """Retrieves the analysis summary."""
        if self.response:
            return self.response.get("summary")
    
    def get_recommendations(self):
        """Retrieves the recommendations."""
        if self.response:
            return self.response.get("recommendations")
    
    def get_results(self):
        """Retrieves the full analysis results."""
        if self.response:
            return self.response.get("results")
    
    def get_observations(self):
        """Retrieves all observations from analysis steps."""
        if self.response:
            return self.response.get("observations")


def make_eda_baseline_workflow(
    model=None,
    log=False,
    log_path=None,
    checkpointer: Optional[object] = None
):
    """
    Factory function that creates a compiled LangGraph workflow for baseline EDA.
    
    Performs automated first-pass analysis with fixed analysis steps.
    
    Parameters
    ----------
    model : LLM, optional
        Language model for synthesizing findings.
    log : bool, default=False
        Whether to save analysis results to a file.
    log_path : str, optional
        Directory for log files.
    checkpointer : Checkpointer, optional
        LangGraph checkpointer for saving workflow state.
    
    Returns
    -------
    CompiledStateGraph
        Compiled LangGraph workflow ready to process EDA requests.
    """
    if log:
        if log_path is None:
            log_path = LOG_PATH
        if not os.path.exists(log_path):
            os.makedirs(log_path)
    
    class EDAState(TypedDict):
        dataframe: dict
        results: dict
        observations: dict[str, list[str]]
        current_step: str
        summary: str
        recommendations: list[str]
    
    def profile_dataset_node(state: EDAState):
        """Generate dataset profile with basic statistics."""
        logger.info("Profiling dataset")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        
        profile = {
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_summary": (
                df[numeric_cols].describe().to_dict() if numeric_cols else {}
            ),
            "categorical_summary": {
                col: df[col].value_counts().head(10).to_dict()
                for col in categorical_cols
            },
        }
        
        results["profile_dataset"] = profile
        
        return {
            "current_step": "profile_dataset",
            "results": results,
        }
    
    def analyze_missingness_node(state: EDAState):
        """Analyze missing values in the dataset."""
        logger.info("Analyzing missingness")
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        
        missing_count = df.isnull().sum().to_dict()
        missing_pct = (
            (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        )
        
        high_missing = {col: pct for col, pct in missing_pct.items() if pct > 20}
        
        missingness = {
            "total_rows": len(df),
            "missing_count": missing_count,
            "missing_percentage": missing_pct,
            "high_missing_columns": high_missing,
            "complete_rows": int(df.dropna().shape[0]),
            "complete_rows_pct": (
                round(df.dropna().shape[0] / len(df) * 100, 2)
                if len(df) > 0 else 0
            ),
        }
        
        results["analyze_missingness"] = missingness
        
        return {
            "current_step": "analyze_missingness",
            "results": results,
        }
    

    def compute_aggregates_node(state: EDAState):
        """Compute selected group-level aggregates and compare against overall dataset
           averages while filtering very small groups and high-cardinality categories."""

        # organize outputs into three layers:
        # display_summary: compact human-readable summary of aggregate analysis
        # segment_summaries: full aggregate calculations for downstream LLM analysis
        # aggregate_outliers: simplified summary of the top aggregate deviations
       
        logger.info("Computing aggregates")
        
        # recreate dataframe + retrieve existing workflow results
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})
        profile = results.get("profile_dataset", {})
        missingness = results.get("analyze_missingness", {})

        # retrieve candidate numeric + categorical columns
        numeric_cols = profile.get("numeric_columns", [])
        categorical_cols = profile.get("categorical_columns", [])

        # retrieve columns with high missingness
        high_missing_cols = set(
            missingness.get("high_missing_columns", {}).keys()
        )

        # exclude unreliable columns with high missingness
        numeric_cols = [
            col for col in numeric_cols
            if col not in high_missing_cols
        ]

        categorical_cols = [
            col for col in categorical_cols
            if col not in high_missing_cols
        ]

        # set simple heuristic thresholds
        max_unique_groups = 10
        min_group_pct = 0.05
        max_aggregate_outliers = 2

        # store compact display output separately from full analytical results
        aggregate_results = {
            "display_summary": {
                "categorical_columns_analyzed": [],
                "numeric_columns_analyzed": numeric_cols,
                "comparison_metric": "percent_difference_from_overall_mean",
            },
            "segment_summaries": {},
            "aggregate_outliers": [],
        }
        

        # basic guardrail: skip if no usable numeric or categorical columns
        if not numeric_cols or not categorical_cols:

            results["compute_aggregates"] = aggregate_results

            return {
                "current_step": "compute_aggregates",
                "results": results,
            }

        # temporary container for candidate outliers across all groups and metrics
        all_outliers = []

        # loop through candidate grouping columns
        for group_col in categorical_cols:

            # count unique groups for the current categorical column
            n_unique = df[group_col].nunique(dropna=True)

            # skip high-cardinality or single-value categories
            if n_unique < 2 or n_unique > max_unique_groups:
                continue

            # calculate group sizes
            group_counts = df[group_col].value_counts(dropna=True)

            # calculate % contribution of each group to the full dataset
            group_pct = group_counts / len(df)

            # keep only groups representing at least the minimum % threshold
            valid_groups = (
                group_pct[group_pct >= min_group_pct]
                .index
                .tolist()
            )

            # skip if fewer than 2 meaningful groups remain
            if len(valid_groups) < 2:
                continue

            # record that this categorical column was actually analyzed
            aggregate_results["display_summary"][
                "categorical_columns_analyzed"
                ].append(group_col)

            # filter dataframe to only meaningful groups
            df_valid = df[df[group_col].isin(valid_groups)]

            # calculate overall averages for numeric columns
            overall_means = df_valid[numeric_cols].mean()

            # calculate group-level averages
            group_means = (
                df_valid
                .groupby(group_col)[numeric_cols]
                .mean()
            )

            # compare each group mean against the overall average
            pct_diff_from_overall = (
                ((group_means - overall_means) / overall_means) * 100
            )
            # store full aggregate calculations for downstream LLM analysis
            aggregate_results["segment_summaries"][group_col] = {

                "group_counts": (
                    group_counts.loc[valid_groups].to_dict()
                ),

                "group_pct": (
                    (group_pct.loc[valid_groups] * 100)
                    .round(2)
                    .to_dict()
                ),

                "group_means": (
                    group_means.round(2).to_dict()
                ),

                "pct_diff_from_overall": (
                    pct_diff_from_overall.round(2).to_dict()
                ),
            }

            # collect group-level outliers for each numeric metric
            for numeric_col in numeric_cols:

                metric_diffs = pct_diff_from_overall[numeric_col].dropna()

                # skip cases where the overall mean is zero or comparison is invalid
                if metric_diffs.empty:
                    continue

                for group_value, percent_difference in metric_diffs.items():

                    all_outliers.append({
                        "group": group_value,
                        "metric": numeric_col,
                        "percent_difference": round(float(percent_difference), 2),
                    })

        # keep only the strongest aggregate outliers overall
        aggregate_results["aggregate_outliers"] = sorted(
            all_outliers,
            key=lambda item: abs(item["percent_difference"]),
            reverse=True
        )[:max_aggregate_outliers]

        # store aggregate results in shared workflow state
        results["compute_aggregates"] = aggregate_results

        return {
            "current_step": "compute_aggregates",
            "results": results,
        }
    
    def analyze_relationships_node(state: EDAState):
        """Analyze meaningful numeric correlations while filtering weak relationships."""
        logger.info("Analyzing relationships")

        # import helper for generating unique column pairs
        from itertools import combinations

        # recreate dataframe + retrieve existing workflow results
        df = pd.DataFrame.from_dict(state.get("dataframe"))
        results = state.get("results", {})

        # reuse outputs from earlier workflow steps
        profile = results.get("profile_dataset", {})
        missingness = results.get("analyze_missingness", {})

        # retrieve candidate numeric columns
        numeric_cols = profile.get("numeric_columns", [])

        # retrieve columns with high missingness
        high_missing_cols = set(
            missingness.get("high_missing_columns", {}).keys()
        )

        # exclude unreliable numeric columns with high missingness
        numeric_cols = [
            col for col in numeric_cols
            if col not in high_missing_cols
        ]

        # container for relationship outputs
        relationship_results = {}

        # basic guardrail: skip if fewer than 2 usable numeric columns
        if len(numeric_cols) < 2:

            results["analyze_relationships"] = relationship_results

            return {
                "current_step": "analyze_relationships",
                "results": results,
            }

        # exclude numeric columns with no meaningful variation
        numeric_cols = [
            col for col in numeric_cols
            if df[col].nunique(dropna=True) > 1
        ]

        # basic guardrail: skip if fewer than 2 numeric columns remain
        if len(numeric_cols) < 2:

            results["analyze_relationships"] = relationship_results

            return {
                "current_step": "analyze_relationships",
                "results": results,
            }

        # set simple heuristic thresholds
        min_correlation = 0.30
        max_relationships = 5

        # compute pairwise correlations between numeric columns
        corr_matrix = df[numeric_cols].corr()

        # collect meaningful correlation pairs
        relationships = []

        # loop through each unique pair of numeric columns
        for col_1, col_2 in combinations(numeric_cols, 2):

            corr_value = corr_matrix.loc[col_1, col_2]

            # skip missing/undefined correlations
            if pd.isna(corr_value):
                continue

            # skip weak relationships
            if abs(corr_value) < min_correlation:
                continue

            # store the relationship in a readable format
            relationships.append({
                "relationship": f"{col_1} vs {col_2}",
                "correlation": round(float(corr_value), 3),
                "direction": "positive" if corr_value > 0 else "negative",
            })

        # sort relationships by strongest absolute correlation
        relationships = sorted(
            relationships,
            key=lambda rel: abs(rel["correlation"]),
            reverse=True
        )

        # separate strongest positive relationships
        positive_relationships = [
            rel for rel in relationships
            if rel["direction"] == "positive"
        ][:max_relationships]

        # separate strongest negative relationships
        negative_relationships = [
            rel for rel in relationships
            if rel["direction"] == "negative"
        ][:max_relationships]

        # store concise relationship outputs
        relationship_results = {
            "numeric_columns_analyzed": numeric_cols,
            "min_correlation_threshold": min_correlation,
            "strongest_positive_relationships": positive_relationships,
            "strongest_negative_relationships": negative_relationships,
        }

        # store relationship results in shared workflow state
        results["analyze_relationships"] = relationship_results

        return {
            "current_step": "analyze_relationships",
            "results": results,
        }
    
    def extract_observations_node(state: EDAState):
        """Extract observations from the latest analysis results using LLM."""
        logger.info("Extracting observations")
        
            # read the current state, add guardrails for when the model is not provided or the current step is not in the results
        current_step = state.get("current_step", "")
        results = state.get("results", {})
        observations = state.get("observations", {})
        
        if model is None or not current_step or current_step not in results:
            return {"observations": observations} #guardrail 
        
        step_results = results.get(current_step, {})
        
            # define the expected structure of the LLM response
        class ObservationOutput(BaseModel):
            observations: list[str] = Field(description="1-2 concise, actionable observations")
        
            # load the system + human prompts used for observation extraction
        observation_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("extract_observations_system.txt")),
            ("human", load_prompt("extract_observations_human.txt")),
        ])
        
            # build the prompt -> LLM -> structured output pipeline
        chain = observation_prompt | model.with_structured_output(ObservationOutput)
            # send the current analysis step + its results to the LLM
        response = chain.invoke({
            "step_name": current_step.replace("_", " ").title(),
            "results": str(step_results)
        })
        
        observations[current_step] = response.observations

         # store the LLM-generated observations under the current workflow step
        return {
            "observations": observations,
        }
    
    def synthesize_findings_node(state: EDAState):
        """Synthesize accumulated findings into summary and recommendations."""
        logger.info("Synthesizing findings")
        
        observations = state.get("observations", {}) #Gets all accumulated observations from state
        
        if model is None:
            return {
                "summary": "No LLM provided for synthesis",
                "recommendations": [],
            }
        
        class SynthesisOutput(BaseModel):
            summary: str = Field(description="A concise 2-3 sentence summary of key findings")
            recommendations: list[str] = Field(description="3-5 actionable recommendations")

            # combine observations from all analysis steps into one formatted text block for final LLM synthesis
        all_observations = []
        for step_name, step_obs in observations.items():
            all_observations.append(f"\n{step_name.replace('_', ' ').title()}:")
            for obs in step_obs:
                all_observations.append(f"  - {obs}")
        
        observations_text = "\n".join(all_observations)
            # load the system + human prompts used for findings synthesis
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", load_prompt("synthesize_findings_system.txt")),
            ("human", load_prompt("synthesize_findings_human.txt")),
        ])
            # build the prompt -> LLM -> structured output pipeline
        chain = synthesis_prompt | model.with_structured_output(SynthesisOutput)
            # send the accumulated observations to the LLM for synthesis
        response = chain.invoke({"observations": observations_text})
        
        return {
            "summary": response.summary,
            "recommendations": response.recommendations,
        }
    
    # create a LangGraph workflow that passes EDAState between workflow nodes
    workflow = StateGraph(EDAState)
    
    workflow.add_node("profile_dataset", profile_dataset_node)
    workflow.add_node("extract_observations_1", extract_observations_node)
    workflow.add_node("analyze_missingness", analyze_missingness_node)
    workflow.add_node("extract_observations_2", extract_observations_node)
    workflow.add_node("compute_aggregates", compute_aggregates_node)
    workflow.add_node("extract_observations_3", extract_observations_node)
    workflow.add_node("analyze_relationships", analyze_relationships_node)
    workflow.add_node("extract_observations_4", extract_observations_node)
    workflow.add_node("synthesize_findings", synthesize_findings_node)
    
    workflow.set_entry_point("profile_dataset")
    
    workflow.add_edge("profile_dataset", "extract_observations_1")
    workflow.add_edge("extract_observations_1", "analyze_missingness")
    workflow.add_edge("analyze_missingness", "extract_observations_2")
    workflow.add_edge("extract_observations_2", "compute_aggregates")
    workflow.add_edge("compute_aggregates", "extract_observations_3")
    workflow.add_edge("extract_observations_3", "analyze_relationships")
    workflow.add_edge("analyze_relationships", "extract_observations_4")
    workflow.add_edge("extract_observations_4", "synthesize_findings")
    workflow.add_edge("synthesize_findings", END)
    
    app = workflow.compile(checkpointer=checkpointer, name=WORKFLOW_NAME)
    
    return app
