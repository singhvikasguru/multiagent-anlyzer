"""
CSV Analysis Agent with Code Generation and Execution Capabilities
Uses OpenAI Agents SDK to analyze CSV data and generate executable Python code
"""

import os
import openai
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from agents import Agent, function_tool, Runner
from pydantic import BaseModel
import nest_asyncio
import asyncio
import sys
from io import StringIO
import traceback
import uuid
from datetime import datetime

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()
# os.environ["OPENAI_API_KEY"] = "..."

# NOTE: SESSION_ID will be generated per analysis run (not globally)
# This ensures each question gets unique images in separate folders



def initialize_openai():
    """Initialize OpenAI client with API key"""
    # Set your API key here or from environment variable
    api_key = os.environ.get("OPENAI_API_KEY", "your-api-key-here")
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key
    client = openai.OpenAI()
    return client

# Initialize client
client = initialize_openai()

# Define Pydantic models for structured outputs
class CodeGenerationOutput(BaseModel):
    """Structured output for generated code"""
    code: str
    explanation: str
    required_libraries: List[str]
    expected_output_description: str

class CodeExecutionResult(BaseModel):
    """Result from code execution"""
    success: bool
    output: str
    error: Optional[str] = None

class ClarificationOutput(BaseModel):
    """Output from clarification agent"""
    need_clarification: str  # "yes" or "no"
    clarification_question: str
    modified_question: str

class TaskPlanOutput(BaseModel):
    """Output from task planner agent"""
    is_complex: str  # "yes" or "no"
    tasks: List[str]  # List of tasks to execute

# Custom tools for the agent
@function_tool
async def read_csv_file(file_path: str) -> str:
    """
    Read a CSV file and return basic information about it.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        JSON string with CSV information including columns, shape, and sample data, use this to determine if question can be answered
    """
    print(f"🔧 TOOL: read_csv_file({file_path})")
    try:
        df = pd.read_csv(file_path)
        
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "head": df.head().to_dict(),
            "describe": df.describe().to_dict() if len(df.select_dtypes(include='number').columns) > 0 else {},
            "null_counts": df.isnull().sum().to_dict(),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
        print(f"✅ CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return json.dumps(info, indent=2)
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return json.dumps({"error": str(e)})

@function_tool
async def execute_python_code(code: str, csv_file_path: Optional[str] = None) -> str:
    """
    Execute Python code in a controlled environment and return the output.
    
    Args:
        code: Python code to execute
        csv_file_path: Optional path to CSV file to make available as 'df' variable
        
    Returns:
        JSON string with execution results including stdout, stderr, and any errors
    """
    print(f"🔧 TOOL: execute_python_code (code length: {len(code)} chars)")
    
    # Capture stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = StringIO()
    redirected_error = StringIO()
    sys.stdout = redirected_output
    sys.stderr = redirected_error
    
    result = {
        "success": False,
        "output": "",
        "error": None,
        "variables": {}
    }
    
    try:
        # Prepare execution namespace
        exec_globals = {
            'pd': pd,
            '__builtins__': __builtins__,
        }
        
        # Load CSV if provided
        if csv_file_path:
            exec_globals['df'] = pd.read_csv(csv_file_path)
            exec_globals['csv_path'] = csv_file_path
            print(f"📊 Loaded CSV into execution environment: {csv_file_path}")
        
        # Execute the code
        exec(code, exec_globals)
        
        # Capture any variables created
        result["variables"] = {
            k: str(v) for k, v in exec_globals.items() 
            if not k.startswith('_') and k not in ['pd', 'df', 'csv_path']
        }
        
        result["success"] = True
        result["output"] = redirected_output.getvalue()
        
        if redirected_error.getvalue():
            result["stderr"] = redirected_error.getvalue()
            
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        result["traceback"] = traceback.format_exc()
    
    finally:
        # Restore stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    print(f"✅ Code execution {'successful' if result['success'] else 'failed'}")
    if result.get('error'):
        print(f"❌ Error: {result['error']}")
    
    return json.dumps(result, indent=2)

@function_tool
async def validate_python_code(code: str) -> str:
    """
    Validate Python code for syntax errors without executing it.
    
    Args:
        code: Python code to validate
        
    Returns:
        JSON string with validation results
    """
    print(f"🔧 TOOL: validate_python_code")
    result = {
        "valid": False,
        "error": None
    }
    try:
        compile(code, '<string>', 'exec')
        result["valid"] = True
        print("✅ Code validation passed")
    except SyntaxError as e:
        result["error"] = f"Syntax Error at line {e.lineno}: {e.msg}"
        print(f"❌ Syntax error: {result['error']}")
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        print(f"❌ Validation error: {result['error']}")
    
    return json.dumps(result, indent=2)

@function_tool
async def save_code_to_file(code: str, file_path: str) -> str:
    """
    Save generated code to a Python file.
    
    Args:
        code: Python code to save
        file_path: Path where to save the file
        
    Returns:
        JSON string with save operation results
    """
    print(f"🔧 TOOL: save_code_to_file({file_path})")
    try:
        with open(file_path, 'w') as f:
            f.write(code)
        print(f"✅ Code saved to {file_path}")
        return json.dumps({
            "success": True,
            "message": f"Code saved to {file_path}",
            "file_path": file_path
        })
    except Exception as e:
        print(f"❌ Save error: {str(e)}")
        return json.dumps({
            "success": False,
            "error": str(e)
        })

@function_tool
async def list_saved_images(session_id: str) -> str:
    """
    List all images saved in the session's image directory.
    
    Args:
        session_id: The session ID to look for images
        
    Returns:
        JSON string with list of image file paths
    """
    try:
        images_path = f"images/{session_id}"
        if not os.path.exists(images_path):
            return json.dumps({"images": [], "message": "No images directory found"})
        
        image_files = [f for f in os.listdir(images_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        image_paths = [os.path.join(images_path, f) for f in image_files]
        
        return json.dumps({
            "session_id": session_id,
            "images": image_paths,
            "count": len(image_paths)
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

# Define the CSV Analysis Agent
def create_csv_analysis_agent(session_id: str):
    """Create CSV analysis agent with session-specific instructions"""
    return Agent(
        name="CSV_Analysis_Code_Generator",
        instructions=f"""You are an expert data analyst and Python programmer specializing in CSV data analysis.

Your primary responsibilities are:
1. Understand the user's analysis requirements for CSV data
2. Examine the CSV file structure using the read_csv_file tool
3. Generate clean, executable Python code that performs the requested analysis, do not generate more than 6 plots/graphs/images
4. Ensure the code follows best practices and includes proper error handling
5. Validate the generated code before presenting it
6. Optionally execute the code to verify it works correctly

When generating code:
- Always start by loading the CSV file with pandas: df = pd.read_csv('path_to_file.csv')
- Include necessary imports at the top
- Add comments explaining each step
- Use descriptive variable names
- Include error handling where appropriate
- Ensure the code produces clear output (prints, plots, etc.)
- IMPORTANT: For visualizations, save ALL plots to 'images/{session_id}/plot_name.png' with descriptive names
- Example: plt.savefig('images/{session_id}/salary_by_dept.png')
- Make sure to validate the code using validate_python_code
- Make sure to save the code using save_code_to_file in a file code.txt

Code generation guidelines:
- Use pandas for data manipulation
- Use matplotlib or seaborn for visualizations
- Use numpy for numerical operations
- Keep code modular and readable
- Include docstrings for any functions you create

Always validate your code before presenting it to the user.""",
        model="o3",
        tools=[
            read_csv_file,
            execute_python_code,
            validate_python_code,
            save_code_to_file
        ]
    )

# NOTE: csv_analysis_agent will be created per function call with unique session_id
# This is done inside analysis_agent_orch_runner_streamlit() and example functions


# Define a code execution specialist agent
code_executor_agent = Agent(
    name="Code_Executor",
    instructions="""You are a code execution specialist. Your job is to:
1. Execute Python code safely
2. Interpret the results
3. Debug any errors that occur
4. Suggest fixes if code fails, update code to make sure it runs with minila code changes

When executing code:
- Always check for potential errors first
- Provide clear interpretation of results
- If execution fails, explain why and suggest corrections""",
    model="o3",
    tools=[
        execute_python_code,
        validate_python_code
    ]
)

# Define clarification agent
clarification_agent = Agent(
    name="Clarification_Agent",
    instructions="""You are a clarification specialist. Your job is to:

IMPORTANT: Call read_csv_file tool ONLY ONCE at the start.

Your job is to:
1. Call read_csv_file tool once to see CSV columns
2. Analyze if the user's question is vague or ambiguous
3. If vague (like "some analysis", "visualize data", etc.), set need_clarification="yes" and ask what specific columns/analysis they want
4. If clear, set need_clarification="no" and use the question as-is


Output format (always include all fields):
- need_clarification: "yes" or "no"
- clarification_question: the question to ask user (empty string if no clarification needed)
- modified_question: In case no clarification is needed, this should be final question to be asked to csv_analysis_agent""",
    model="o3",
    tools=[read_csv_file],
    output_type=ClarificationOutput
)

# Updated Task Planner Agent - Less Aggressive Task Breaking

task_planner_agent = Agent(
    name="Task_Planner_Agent",
    instructions="""You are a task planning specialist for data analysis using OpenAI O3.

CRITICAL: The analysis agent uses OpenAI O3 and can handle sophisticated, multi-step analysis in a single task.

Your job is to:
1. Call read_csv_file tool ONCE at the start to understand CSV structure
2. Determine if the question needs MULTIPLE DISTINCT ANALYSIS AREAS (not just multiple steps)
3. Only break into subtasks if the question covers FUNDAMENTALLY DIFFERENT analysis types

WHEN TO KEEP AS SINGLE TASK (is_complex: "no"):
- Any question that can be answered with one cohesive analysis
- Questions asking for statistics + visualization of the same data
- Trend analysis with multiple charts of the same theme
- Comparisons within the same domain (e.g., departments, time periods)
- Complex calculations that are related (correlations, regressions, statistical summaries)
- Questions like "analyze X vs Y across Z" → ONE comprehensive analysis
- Questions asking for "trends", "patterns", "insights" → ONE analysis
- Questions asking for "breakdown by department/category" → ONE analysis

WHEN TO BREAK INTO MULTIPLE TASKS (is_complex: "yes"):
- Questions covering COMPLETELY DIFFERENT data domains (e.g., "analyze sales AND analyze employee performance AND analyze inventory")
- Questions explicitly asking for separate, unrelated analyses as distinct reports
- Questions asking for fundamentally different analysis types that belong in separate reports

The O3 analyzer can handle in ONE TASK:
- Multiple related calculations
- Several visualizations of the same analysis theme
- Complex statistical analysis with multiple charts
- Data cleaning + analysis + visualization
- Comparisons across multiple dimensions
- Correlation analysis with scatter plots and heatmaps
- Trend analysis with multiple time-based visualizations
- Department/category breakdowns with summary statistics
- Outlier detection and analysis (including analysis with/without outliers)
- Multi-step analysis workflows (O3 handles the sequencing)

Examples of SINGLE TASK (is_complex: "no"):
- "Analyze salary vs experience across departments" → ONE comprehensive analysis
- "Show department performance trends with statistics and charts" → ONE cohesive analysis  
- "Calculate correlations and create visualizations for all numeric variables" → ONE related analysis
- "Compare performance ratings by department with detailed breakdown and charts" → ONE analysis domain
- "Create vendor-wise trends showing totals over time" → ONE trend analysis
- "Analyze employee hiring patterns by department and year" → ONE hiring analysis
- "Show financial performance with key metrics and visualizations" → ONE financial analysis

Examples of MULTIPLE TASKS (is_complex: "yes"):
- "Analyze employee data AND separately analyze financial performance AND separately analyze inventory levels" → 3 completely different domains
- "Give me three separate reports: hiring analysis, salary analysis, and performance analysis" → Explicitly separate reports
- "Create a sales report AND create an HR report AND create a financial report" → Distinct business reports

BIAS TOWARD SINGLE TASK: When in doubt, keep it as one task. O3 can handle complexity.

Focus each task on ANALYSIS OUTCOMES, not implementation steps:
- GOOD: "Comprehensive vendor-wise trend analysis showing totals over time with statistical insights and visualizations"
- BAD: "Read data, identify date columns, group by vendor, calculate totals, create charts" (O3 handles all steps automatically)

Output format:
- is_complex: "yes" only if TRULY multiple distinct analysis domains requiring separate reports
- tasks: List of analysis outcomes (strongly prefer single comprehensive task unless clearly separate domains)""",
    model="o3",
    tools=[read_csv_file],
    output_type=TaskPlanOutput
)

# Enhanced Executive Summary Agent with Increased Token Limit

summarizer_agent = Agent(
    name="Executive_Summary_Agent",
    instructions="""You are an executive reporting specialist that creates comprehensive, data-driven summaries.

Your job is to:
1. Review all task results from the analysis in detail
2. Use list_saved_images tool to see what visualizations were created
3. Create a comprehensive, executive-style summary with extensive data tables and insights

REQUIREMENTS:
- Lead with KEY NUMBERS and DATA
- Present findings in DETAILED TABLE FORMAT wherever possible
- Provide COMPREHENSIVE analysis, not abbreviated summaries
- Focus on FACTS, METRICS, and ACTIONABLE INSIGHTS with supporting details
- Reference all visualizations created with detailed descriptions
- Include statistical significance and confidence where applicable
- Provide context and interpretation for all metrics

ENHANCED FORMAT:
## Executive Summary

**Analysis Completed:** [Detailed description of analysis scope and dataset]

### Key Metrics Overview
| Metric | Value | Benchmark/Target | Variance | Significance |
|--------|-------|------------------|----------|--------------|
| [metric] | [value] | [target] | [+/-] | [high/medium/low] |

### Detailed Statistical Analysis
[Comprehensive breakdown of all statistical findings]

| Category | Count | Percentage | Average | Std Dev | Min | Max | Key Insights |
|----------|-------|------------|---------|---------|-----|-----|--------------|

### Comparative Analysis
[Detailed comparisons across dimensions]

| Dimension A | Dimension B | Metric | Value A | Value B | Difference | % Change | Interpretation |
|-------------|-------------|--------|---------|---------|------------|----------|----------------|

### Trend Analysis (if applicable)
| Time Period | Metric | Value | Change from Previous | % Change | Trend Direction |
|-------------|--------|-------|---------------------|----------|-----------------|

### Correlation & Relationship Analysis
| Variable 1 | Variable 2 | Correlation Coefficient | R-squared | Relationship Strength | Business Implication |
|------------|------------|-------------------------|-----------|----------------------|---------------------|

### Outlier & Exception Analysis
| Outlier Type | Count | Criteria | Impact | Recommended Action |
|--------------|-------|----------|--------|-------------------|

### Visualizations Generated
- `filename.png` - [Detailed chart type]: [Comprehensive description of what it shows, key patterns, and insights]
- `filename.png` - [Detailed chart type]: [Comprehensive description with statistical highlights]

### Data Quality Assessment
| Quality Metric | Status | Issues Found | Impact on Analysis |
|----------------|--------|--------------|-------------------|

### Detailed Findings & Interpretation
1. **[Finding Category 1]**
   - Key metric: [value with context]
   - Statistical significance: [details]
   - Business impact: [explanation]
   - Supporting data: [table/details]

2. **[Finding Category 2]**
   - Key metric: [value with context]
   - Comparison: [benchmarks/targets]
   - Trend analysis: [direction and magnitude]
   - Implications: [detailed explanation]

### Risk Factors & Considerations
| Risk Factor | Probability | Impact Level | Mitigation Strategy |
|-------------|-------------|--------------|-------------------|

### Actionable Recommendations
1. **[Priority Level - High/Medium/Low]**: [Specific recommendation]
   - Expected impact: [quantified benefit]
   - Implementation timeline: [timeframe]
   - Resource requirements: [details]
   - Success metrics: [how to measure]

2. **[Priority Level]**: [Specific recommendation]
   - Data supporting this: [specific metrics]
   - Potential challenges: [obstacles]
   - Alternative approaches: [options]

### Bottom Line Executive Summary
- **Primary Finding**: [Most critical insight with supporting data]
- **Financial Impact**: [Quantified business impact if applicable]
- **Immediate Actions**: [Top 2-3 urgent recommendations with timelines]
- **Long-term Strategy**: [Strategic implications and recommendations]
- **Key Performance Indicators to Monitor**: [Specific metrics to track]

### Appendix: Technical Details
- Dataset characteristics: [size, timeframe, completeness]
- Statistical methods used: [analysis techniques applied]
- Confidence levels: [statistical confidence in findings]
- Limitations: [data limitations and caveats]

CRITICAL: Provide comprehensive analysis with extensive detail, statistical depth, and actionable business insights. Do not abbreviate or summarize briefly. Use the full token capacity to deliver thorough, professional executive reporting.""",
    model="o3", 
    tools=[list_saved_images]
)

class CSVAnalysisRunner:
    """Runner class to orchestrate CSV analysis tasks"""
    
    def __init__(self, csv_file_path: str):
        """
        Initialize the runner with a CSV file path
        
        Args:
            csv_file_path: Path to the CSV file to analyze
        """
        self.csv_file_path = csv_file_path
        
    def analyze(self, analysis_request: str, execute_code: bool = False, session_id: str = None) -> Dict[str, Any]:
        """
        Analyze CSV based on user request
        
        Args:
            analysis_request: User's analysis requirements
            execute_code: Whether to execute the generated code
            session_id: Session ID for saving plots
            
        Returns:
            Dictionary containing generated code and results
        """
        # First, read the CSV to get column information
        import pandas as pd
        try:
            df = pd.read_csv(self.csv_file_path)
            csv_info = f"""
                        CSV Columns: {df.columns.tolist()}
                        Data Types: {df.dtypes.to_dict()}
                        Shape: {df.shape}
                        Sample Data (first 3 rows):
                        {df.head(3).to_string()}
                        """
        except Exception as e:
            csv_info = f"Error reading CSV: {e}"
        
        # Construct the full prompt with CSV information
        full_prompt = f"""
        CSV File Path: {self.csv_file_path}
        
        {csv_info}
        
        Analysis Request: {analysis_request}
        
        Please:
        1. Generate Python code that performs the requested analysis using the EXACT column names shown above
        2. Make sure to use df = pd.read_csv('{self.csv_file_path}') to load the data
        3. Use only the columns that exist in this CSV
        4. ALways Validate the code using validate_python_code tool
        5. Format the code cleanly with proper comments
        6. Explain what the code does
        
        Make sure the code is ready to run and will work with the CSV file at: {self.csv_file_path}
        """
        
        # Run the agent
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        print("="*100)
        print("Input to csv_analysis_agent", full_prompt)
        print("="*100)
        
        # Use session-specific agent (session_id is required)
        if not session_id:
            raise ValueError("session_id is required for analyze()")
        agent_to_use = create_csv_analysis_agent(session_id)
        result = Runner.run_sync(agent_to_use, input=full_prompt)
        
        response = {
            "request": analysis_request,
            "csv_file": self.csv_file_path,
            "agent_response": result.final_output if hasattr(result, "final_output") else str(result)
        }
        
        # Optionally execute the code
        if execute_code:
            execution_prompt = f"""
            Execute this analysis on the CSV file at {self.csv_file_path}:
            {result.final_output if hasattr(result, "final_output") else str(result)}
            
            Use the execute_python_code tool with the CSV file path.
            """
            print("#"*100)
            print("Input to code_executor_agent", execution_prompt)
            print("#"*100)
            exec_result = Runner.run_sync(code_executor_agent, input=execution_prompt)
            response["execution_result"] = exec_result.final_output if hasattr(exec_result, "final_output") else str(exec_result)
        
        return response
    
    def generate_code_only(self, analysis_request: str, save_to_file: Optional[str] = None) -> str:
        """
        Generate Python code without execution
        
        Args:
            analysis_request: User's analysis requirements
            save_to_file: Optional file path to save the generated code
            
        Returns:
            Generated Python code as string
        """
        # First, read the CSV to get column information
        import pandas as pd
        try:
            df = pd.read_csv(self.csv_file_path)
            csv_info = f"""
CSV File: {self.csv_file_path}
Columns: {df.columns.tolist()}
Data Types: {df.dtypes.to_dict()}
Shape: {df.shape[0]} rows, {df.shape[1]} columns

Sample Data (first 3 rows):
{df.head(3).to_string()}
"""
        except Exception as e:
            csv_info = f"Error reading CSV: {e}"
        
        full_prompt = f"""
        {csv_info}
        
        Analysis Request: {analysis_request}
        
        Generate clean, executable Python code that:
        1. Loads the CSV from: {self.csv_file_path}
        2. Uses the EXACT column names shown above
        3. Performs the requested analysis
        4. Produces clear output
        
        Provide ONLY the Python code, properly formatted and commented.
        Make it ready to copy and run.
        """
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = Runner.run_sync(csv_analysis_agent, input=full_prompt)
        code = result.final_output if hasattr(result, "final_output") else str(result)
        
        # Save to file if requested
        if save_to_file:
            try:
                with open(save_to_file, 'w') as f:
                    f.write(code)
                print(f"Code saved to: {save_to_file}")
            except Exception as e:
                print(f"Error saving code: {e}")
        
        return code

def example_generate_and_execute(question):
    """Example: Generate code and execute it"""
    # Generate unique SESSION_ID for this example run
    SESSION_ID = str(uuid.uuid4())[:8]
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Generate and Execute Code")
    print(f"Session ID: {SESSION_ID}")
    print("=" * 80)
    
    # Create CSV analysis agent with this session's ID
    csv_analysis_agent = create_csv_analysis_agent(SESSION_ID)
    
    runner = CSVAnalysisRunner("data.csv")
    
    # Step 1: Check if clarification is needed
    clarification_prompt = f"""
    CSV file path: {runner.csv_file_path}
    User question: {question}
    
    Use read_csv_file tool to examine the CSV structure.
    Decide if clarification is needed for this question.
    """
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    print("\n🤔 Checking if clarification needed...")
    clarification_result = Runner.run_sync(clarification_agent, input=clarification_prompt)
    clarification_output = clarification_result.final_output_as(ClarificationOutput)
    
    print(f"\nNeed clarification: {clarification_output.need_clarification}")

def analysis_agent_orch_runner_streamlit(question, clarification_answer=None, conversation_state=None):
    """
    Streamlit-compatible version that handles clarification through UI
    
    Args:
        question: User's question
        clarification_answer: Answer to clarification question (if any)
        conversation_state: State from previous clarification rounds
    
    Returns:
        Dictionary with result and clarification info
    """
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║         CSV ANALYSIS AGENT WITH CODE GENERATION                    ║
    ║         Powered by OpenAI Agents SDK                               ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    
    # Generate unique SESSION_ID for this analysis
    # Reuse existing session_id if this is a clarification follow-up
    print(f"\n🔍 DEBUG: conversation_state = {conversation_state}")
    if conversation_state and "session_id" in conversation_state:
        SESSION_ID = conversation_state["session_id"]
        print(f"🔄 REUSING existing session: {SESSION_ID}")
        print(f"   Reason: This is a clarification follow-up")
    else:
        SESSION_ID = str(uuid.uuid4())[:8]
        print(f"🆕 NEW session created: {SESSION_ID}")
        print(f"   Reason: {'conversation_state is None' if not conversation_state else 'No session_id in conversation_state'}")
    
    # SYSTEM - START
    print("="*80)
    print("🚀 SYSTEM: Multi-Agent Analysis Starting")
    print(f"   Session ID: {SESSION_ID}")
    print(f"   Question: {question}")
    print("="*80)
    
    try:
        # Create CSV analysis agent with this session's ID for image folder organization
        csv_analysis_agent = create_csv_analysis_agent(SESSION_ID)
        
        runner = CSVAnalysisRunner("data.csv")
        
        # If we have conversation state, use it to continue from where we left off
        if conversation_state:
            final_question = conversation_state.get("final_question", question)
            if clarification_answer:
                # Add the clarification answer to the question
                clarification_q = conversation_state.get("clarification_question", "")
                final_question = f"{final_question}. Clarification asked: '{clarification_q}'. User clarification: {clarification_answer}"
        else:
            final_question = question
        
        # Step 1: Check if clarification is needed (first time or after answer)
        clarification_prompt = f"""
        CSV file path: {runner.csv_file_path}
        User question: {final_question}
        
        Use read_csv_file tool to examine the CSV structure.
        Decide if clarification is needed for this question.
        """
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # CLARIFICATION AGENT - START
        print("\n" + "="*80)
        print("🤔 Checking if clarification needed...")
        print("🤖 AGENT: Clarification Agent - Starting")
        print(f"   Status: RUNNING")
        print("="*80)
        
        clarification_result = Runner.run_sync(clarification_agent, input=clarification_prompt)
        clarification_output = clarification_result.final_output_as(ClarificationOutput)
        
        # CLARIFICATION AGENT - COMPLETED
        print("\n" + "="*80)
        print("🤖 AGENT: Clarification Agent - Completed")
        print(f"   Status: COMPLETED")
        print(f"   Need clarification: {clarification_output.need_clarification}")
        print(f"   Modified question: {clarification_output.modified_question}")
        print("="*80)
        
        # If clarification is still needed, return clarification request
        if clarification_output.need_clarification.lower() == "yes" and not clarification_answer:
            return {
                "status": "needs_clarification",
                "clarification_question": clarification_output.clarification_question,
                "conversation_state": {
                    "final_question": final_question,
                    "clarification_question": clarification_output.clarification_question,
                    "original_question": question,
                    "session_id": SESSION_ID  # Preserve session_id for image folder consistency
                }
            }
        
        # Use the final modified question from clarification agent
        final_question = clarification_output.modified_question
        print(f"\n✅ Final question: {final_question}")
        
        # Continue with task planning and execution
        task_plan_prompt = f"""
        CSV file path: {runner.csv_file_path}
        User question: {final_question}
        
        Use read_csv_file tool to examine the CSV structure.
        
        Based on the user's question and CSV data, determine if this requires:
        1. Simple analysis (1 task) - basic statistics, simple visualizations, single analysis type
        2. Complex analysis (multiple tasks) - multiple analysis types, complex visualizations, comparisons
        
        IMPORTANT: You must ALWAYS provide at least one task. Even if the question is simple, 
        create one comprehensive task that addresses the user's request.
        
        For the question "{final_question}", create specific, actionable tasks that will fully answer the user's question.
        Each task should be clear and executable.
        
        Return "yes" for is_complex if multiple distinct analysis steps are needed, "no" if one comprehensive task can handle it.
        """
        
        # TASK PLANNER AGENT - START
        print("\n" + "="*80)
        print("📋 Planning tasks...")
        print("🤖 AGENT: Task Planner Agent - Starting")
        print(f"   Status: RUNNING")
        print("="*80)
        
        task_plan_result = Runner.run_sync(task_planner_agent, input=task_plan_prompt, max_turns=3)
        task_plan_output = task_plan_result.final_output_as(TaskPlanOutput)
        
        # TASK PLANNER AGENT - COMPLETED
        print("\n" + "="*80)
        print("🤖 AGENT: Task Planner Agent - Completed")
        print(f"   Status: COMPLETED")
        print(f"   Is complex: {task_plan_output.is_complex}")
        print(f"   Number of tasks: {len(task_plan_output.tasks)}")
        for i, task in enumerate(task_plan_output.tasks, 1):
            print(f"   Task {i}: {task}")
        print("="*80)
        
        # Execute tasks
        all_results = []
        for i, task in enumerate(task_plan_output.tasks, 1):
            # CSV ANALYSIS AGENT - START (for this task)
            print(f"\n{'='*80}")
            print(f"🚀 Executing Task {i}/{len(task_plan_output.tasks)}")
            print(f"🤖 AGENT: CSV Analysis Agent - Starting Task {i}")
            print(f"   Status: RUNNING")
            print(f"   Task: {task}")
            print(f"{'='*80}")
            
            result = runner.analyze(
                task,
                execute_code=True,
                session_id=SESSION_ID
            )
            all_results.append(result)
            
            # CSV ANALYSIS AGENT - COMPLETED (for this task)
            print(f"\n{'='*80}")
            print(f"🤖 AGENT: CSV Analysis Agent - Task {i} Completed")
            print(f"   Status: COMPLETED")
            print(f"{'='*80}")
            
            print(f"\nTask {i} - Generated Code:")
            print(result["agent_response"])
            print(f"\nTask {i} - Execution Result:")
            print(result.get("execution_result", "No execution performed"))
        
        # Generate summary
        task_summaries = []
        for i, (task, result) in enumerate(zip(task_plan_output.tasks, all_results), 1):
            task_summaries.append(f"Task {i}: {task}\nResult: {result.get('execution_result', 'No output')}")
        
        summary_prompt = f"""
        You are an executive summary agent. Create a comprehensive executive summary that directly answers the user's question.
        
        Original User Question: {question}
        Final Question After Clarification: {final_question}
        Session ID: {SESSION_ID}
        
        Task Results:
        {chr(10).join(task_summaries)}
        
        Use list_saved_images tool to see what visualizations were created.
        
        Create an EXECUTIVE SUMMARY following this structure:
        
        ## Executive Summary
        
        ### Question Answered
        [Restate the user's question clearly]
        
        ### Key Findings
        [Present the most important data insights first - numbers, percentages, trends]
        
        ### Data Analysis Results
        [Detailed findings from the analysis with specific metrics]
        
        ### Visualizations Created
        [List and describe each chart/plot created and what it shows]
        
        ### Recommendations
        [Based on the analysis, what actions or next steps would you recommend]
        
        ### Technical Details
        [Brief mention of methodology, data sources, any limitations]
        
        Make this summary actionable and executive-ready. Lead with the most important insights that directly answer the user's question.
        """
        
        # SUMMARY AGENT - START
        print(f"\n{'='*80}")
        print("📝 Creating Summary...")
        print("🤖 AGENT: Summary Agent - Starting")
        print(f"   Status: RUNNING")
        print(f"{'='*80}")
        
        summary_result = Runner.run_sync(summarizer_agent, input=summary_prompt, max_turns=3)
        final_summary = summary_result.final_output if hasattr(summary_result, "final_output") else str(summary_result)
        
        # SUMMARY AGENT - COMPLETED
        print("\n" + "="*80)
        print("🤖 AGENT: Summary Agent - Completed")
        print(f"   Status: COMPLETED")
        print("="*80)
        
        # SYSTEM - COMPLETE
        print("\n" + "="*80)
        print("✅ SYSTEM: Analysis Completed Successfully")
        print(f"   Session ID: {SESSION_ID}")
        print(f"   Total tasks: {len(task_plan_output.tasks)}")
        print("="*80)
        
        print("\n" + "="*80)
        print("📊 FINAL SUMMARY")
        print("="*80)
        print(final_summary)
        print("="*80)
        
        return {
            "status": "completed",
            "question": question,
            "final_question": final_question,
            "tasks": task_plan_output.tasks,
            "results": all_results,
            "summary": final_summary,
            "session_id": SESSION_ID
        }
        
    except Exception as e:
        error_msg = f"Error in analysis: {str(e)}\n{traceback.format_exc()}"
        
        # SYSTEM - ERROR
        print("\n" + "="*80)
        print("❌ SYSTEM: Analysis Failed")
        print(f"   Session ID: {SESSION_ID}")
        print(f"   Error: {str(e)}")
        print("="*80)
        print(error_msg)
        
        return {
            "status": "error",
            "error": error_msg
        }



def analysis_agent_orch_runner(question):
    """
    Main execution block - uncomment the example you want to run
    """
    
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║         CSV ANALYSIS AGENT WITH CODE GENERATION                    ║
    ║         Powered by OpenAI Agents SDK                               ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)
    # Example 2: Generate and execute code
    result = example_generate_and_execute(question)
    
    # Example 3: Generate code only and save to file
    # example_generate_code_only()
    
    # Interactive mode - best for testing
#     interactive_mode()
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return result

# analysis_agent_orch_runner("Give me some analysis, give visual representation")