# Multi-Agent System Architecture Documentation

## Overview

The Multi-Agent CSV Analysis System is a sophisticated orchestration framework powered by OpenAI's Agents SDK and O3 model. It employs four specialized agents working in sequence to transform natural language questions into comprehensive data analysis with code generation, execution, and executive summaries.

## Architecture Flow

```
User Question → [Clarification Agent] → [Task Planner Agent] → [CSV Analysis Agent] → [Summary Agent] → Executive Report
```

## Agent Specifications

### 🤔 Clarification Agent

**Purpose**: Analyzes incoming questions for clarity and determines if additional information is needed before proceeding.

**Model**: OpenAI O3  
**Name**: `Clarification_Agent`

#### Tools Available:
- `read_csv_file`: Examines CSV structure to understand available columns and data

#### Responsibilities:
1. **CSV Structure Analysis**: Calls `read_csv_file` once to understand available data
2. **Question Assessment**: Analyzes if user questions are vague or ambiguous
3. **Clarification Decision**: Determines if additional specificity is needed
4. **Context Integration**: Processes previous analysis summaries and questions for follow-ups

#### Input Processing:
- **User Question**: Original natural language query
- **Previous Context**: When available, includes previous question and summary
- **CSV File Path**: Location of data to be analyzed

#### Output Structure (ClarificationOutput):
```python
{
    "need_clarification": "yes" | "no",
    "clarification_question": "Question to ask user (empty if not needed)",
    "modified_question": "Final processed question for analysis"
}
```

#### Decision Logic:
- **Triggers Clarification**: Vague requests like "some analysis", "visualize data"
- **Passes Through**: Specific requests with clear columns/analysis types
- **Context-Aware**: Uses previous analysis context to better understand follow-ups

#### Context Tracking for Follow-up Questions:

**Context Integration Mechanism**:
When a follow-up question is detected (based on conversation state), the clarification agent receives enriched context:

```python
# Context detection and integration
previous_context = ""
if conversation_state and conversation_state.get("previous_summary") and conversation_state.get("previous_question"):
    previous_context = f"""
    
    PREVIOUS CONTEXT:
    Previous question: {conversation_state.get("previous_question")}
    Previous analysis summary: {conversation_state.get("previous_summary")}
    """

# Enhanced clarification prompt with context
clarification_prompt = f"""
CSV file path: {runner.csv_file_path}
User question: {final_question}{previous_context}

Use read_csv_file tool to examine the CSV structure.
Decide if clarification is needed for this question.
"""
```

**Context-Aware Processing Examples**:

1. **Previous Analysis Reference**:
   ```
   Previous: "Analyze sales by region"
   Follow-up: "Show me the top 3 only"
   
   Clarification Agent receives:
   - Current question: "Show me the top 3 only"  
   - Previous question: "Analyze sales by region"
   - Previous summary: "Sales analysis showed West: $500K, East: $400K, South: $300K..."
   
   Decision: No clarification needed (understands "top 3" refers to regions from previous analysis)
   ```

2. **Ambiguous Follow-up with Context**:
   ```
   Previous: "Employee salary analysis by department"
   Follow-up: "Can you break it down further?"
   
   Clarification Agent receives previous context and asks:
   "I see you want to break down the salary analysis further. Would you like to see:
   - Breakdown by job level within each department?
   - Breakdown by years of experience?
   - Breakdown by performance rating?"
   ```

3. **Context-Enhanced Understanding**:
   ```
   Previous: "Customer satisfaction trends over time"
   Follow-up: "What about the outliers we found?"
   
   Without context: Would need clarification on "what outliers?"
   With context: Understands outliers refer to satisfaction score anomalies from previous analysis
   ```

**Context Flow in Streamlit UI**:
```python
# In streamlit_enhanced_visibility.py
if st.session_state.analysis_results:
    last_result = st.session_state.analysis_results[-1]
    if last_result.get("status") == "completed":
        # Extract previous question from chat history
        last_user_question = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                last_user_question = msg["content"]
                break
        
        # Create conversation state with previous context
        conversation_state = {
            "previous_question": last_result.get("question", last_user_question),
            "previous_summary": last_result.get("summary", "")
        }
```

**Benefits of Context Tracking**:
- **Reduced Clarification Requests**: Fewer interruptions for obvious follow-ups
- **Improved Understanding**: Better interpretation of ambiguous references
- **Conversation Continuity**: Maintains analysis thread across multiple questions
- **Enhanced User Experience**: More natural, conversational interactions
- **Smart Reference Resolution**: Resolves "it", "that", "those" references automatically

---

### 📋 Task Planner Agent

**Purpose**: Intelligently breaks down analysis requests into executable tasks, optimized for O3's sophisticated reasoning capabilities.

**Model**: OpenAI O3  
**Name**: `Task_Planner_Agent`

#### Tools Available:
- `read_csv_file`: Examines CSV structure for task planning

#### Responsibilities:
1. **Complexity Assessment**: Determines if analysis requires single or multiple tasks
2. **Task Decomposition**: Creates specific, actionable analysis tasks
3. **O3 Optimization**: Leverages O3's multi-step reasoning capabilities
4. **Scope Management**: Prevents over-fragmentation of related analysis

#### Decision Framework:

**Single Task (is_complex: "no")**:
- Cohesive analysis within same domain
- Statistics + visualization of same data
- Trend analysis with multiple related charts
- Comparisons within same business area
- Complex but related calculations (correlations, regressions)
- Department/category breakdowns
- Multi-step workflows that O3 can handle

**Multiple Tasks (is_complex: "yes")**:
- Completely different data domains
- Explicitly requested separate reports
- Fundamentally different analysis types requiring distinct reports

#### Output Structure (TaskPlanOutput):
```python
{
    "is_complex": "yes" | "no",
    "tasks": ["List of specific analysis tasks"]
}
```

#### Examples:

**Single Task Examples**:
- "Analyze salary vs experience across departments" → Comprehensive analysis
- "Show vendor trends with statistics and charts" → Cohesive trend analysis
- "Calculate correlations for all numeric variables" → Related statistical analysis

**Multiple Task Examples**:
- "Analyze sales AND employee performance AND inventory" → 3 distinct domains
- "Create separate reports: hiring, salary, performance" → Explicitly separate

---

### 🚀 CSV Analysis Agent (Dynamic)

**Purpose**: Expert data analyst that generates, validates, and executes Python code for comprehensive data analysis.

**Model**: OpenAI O3  
**Name**: `CSV_Analysis_Code_Generator`  
**Note**: Dynamically created per session with unique session ID

#### Tools Available:
- `read_csv_file`: CSV examination and structure analysis
- `execute_python_code`: Controlled Python execution environment  
- `validate_python_code`: Syntax validation before execution
- `save_code_to_file`: Persistence of generated analysis code

#### Responsibilities:
1. **Data Understanding**: Examines CSV structure and content
2. **Code Generation**: Creates clean, executable Python analysis code
3. **Code Validation**: Ensures syntax correctness before execution
4. **Safe Execution**: Runs code in controlled environment
5. **Visualization Management**: Creates and saves charts to session-specific folders
6. **Error Handling**: Robust error management and debugging

#### Code Generation Guidelines:
- **Data Loading**: Always starts with `df = pd.read_csv('path')`
- **Library Management**: Imports pandas, matplotlib, seaborn, numpy
- **Documentation**: Comprehensive comments explaining each step
- **Error Handling**: Includes appropriate try-catch blocks
- **Visualization**: Saves all plots to `images/{session_id}/descriptive_name.png`
- **Output Limits**: Maximum 6 plots/visualizations per analysis
- **Best Practices**: Descriptive variables, modular code, docstrings

#### Session Management:
- **Unique Session ID**: Each analysis gets isolated image folder
- **Image Organization**: `images/{session_id}/plot_name.png`
- **Code Persistence**: Saves analysis code to `code.txt`

#### Execution Environment:
```python
exec_globals = {
    'pd': pandas,
    'df': loaded_dataframe,
    'csv_path': file_path,
    '__builtins__': standard_python_builtins
}
```

#### Output Tracking:
- **Execution Results**: stdout, stderr, created variables
- **Success Status**: Boolean success/failure indicator
- **Error Details**: Comprehensive error messages and tracebacks

---

### 📝 Summary Agent (Executive Summary Agent)

**Purpose**: Creates comprehensive, executive-ready summaries that synthesize all analysis results into actionable insights.

**Model**: OpenAI O3  
**Name**: `Executive_Summary_Agent`

#### Tools Available:
- `list_saved_images`: Inventories all visualizations created during analysis

#### Responsibilities:
1. **Results Synthesis**: Reviews all task execution results
2. **Visualization Cataloging**: Lists and describes all created charts
3. **Executive Reporting**: Creates structured, professional summaries
4. **Data-Driven Insights**: Focuses on metrics, trends, and actionable findings
5. **Statistical Context**: Provides significance and confidence interpretations

#### Summary Structure:
```markdown
## Executive Summary

### Question Answered
[Clear restatement of user's question]

### Key Findings
[Most important data insights with numbers and percentages]

### Data Analysis Results
[Detailed findings with specific metrics]

### Visualizations Created
[Description of each chart and its insights]

### Recommendations
[Actionable next steps based on analysis]

### Technical Details
[Methodology, data sources, limitations]
```

#### Enhanced Features:
- **Detailed Tables**: Comprehensive data presentation
- **Statistical Significance**: Confidence levels and benchmarks
- **Comparative Analysis**: Cross-category insights
- **Visual References**: Detailed description of all charts created
- **Executive Focus**: Business-ready language and actionable insights

---

## Context Tracking System

### Overview
The system implements sophisticated context awareness that allows follow-up questions to reference previous analysis results without requiring users to repeat information or provide full context each time.

### Context Propagation Flow

```
Previous Analysis → Conversation State → Enhanced Clarification Prompt → Improved Understanding
```

### Implementation Details

#### 1. Context Capture (Streamlit UI Layer)
```python
# When new analysis request is made (non-clarification)
if st.session_state.analysis_results:
    # Get the most recent completed analysis
    last_result = st.session_state.analysis_results[-1]
    if last_result.get("status") == "completed":
        # Extract the last user question from chat history
        last_user_question = None
        for msg in reversed(st.session_state.messages):
            if msg["role"] == "user":
                last_user_question = msg["content"]
                break
        
        # Create conversation state with previous context
        conversation_state = {
            "previous_question": last_result.get("question", last_user_question),
            "previous_summary": last_result.get("summary", "")
        }
```

#### 2. Context Integration (Orchestration Layer)
```python
# In analysis_agent_orch_runner_streamlit()
previous_context = ""
if conversation_state and conversation_state.get("previous_summary") and conversation_state.get("previous_question"):
    previous_context = f"""
    
    PREVIOUS CONTEXT:
    Previous question: {conversation_state.get("previous_question")}
    Previous analysis summary: {conversation_state.get("previous_summary")}
    """

# Enhanced prompt for clarification agent
clarification_prompt = f"""
CSV file path: {runner.csv_file_path}
User question: {final_question}{previous_context}

Use read_csv_file tool to examine the CSV structure.
Decide if clarification is needed for this question.
"""
```

#### 3. Context-Aware Decision Making (Clarification Agent)
The clarification agent can now make informed decisions about follow-up questions by considering:
- **Previous Question Scope**: What was analyzed before
- **Previous Results Summary**: Key findings and insights from last analysis
- **Current Question**: The new request in context of previous work
- **Data Structure**: CSV columns and data available

### Context-Aware Scenarios

#### Scenario 1: Reference Resolution
```
Previous Analysis: "Analyze employee salaries by department"
Previous Summary: "IT Dept: avg $85K, Sales: avg $65K, HR: avg $55K. IT has highest variation."

Follow-up Question: "Show me the distribution for the highest paying department"

Context Processing:
- Clarification Agent understands "highest paying department" = IT Department
- No clarification needed, proceeds directly to analysis
- Modified question: "Show salary distribution for IT Department"
```

#### Scenario 2: Implicit Reference Understanding
```
Previous Analysis: "Customer satisfaction trends by month"
Previous Summary: "Satisfaction declined from 4.2 to 3.8 from Jan to Dec. Notable dips in March (3.5) and September (3.6)."

Follow-up Question: "What caused those dips?"

Context Processing:
- Understands "those dips" refers to March and September satisfaction drops
- May ask for clarification on what additional data is available to analyze causes
- Enhanced question: "Analyze factors contributing to satisfaction dips in March and September"
```

#### Scenario 3: Build-on-Previous Analysis
```
Previous Analysis: "Sales performance by region and product category"
Previous Summary: "West region leads in Electronics ($2M), East in Clothing ($1.5M). Electronics growing 15% YoY."

Follow-up Question: "Can you break this down by quarter?"

Context Processing:
- Understands "this" refers to sales performance analysis
- No clarification needed for scope
- Modified question: "Break down sales performance by region and product category quarterly"
```

### Context State Management

#### Conversation State Structure:
```python
conversation_state = {
    # Session management
    "session_id": "unique_session_identifier",
    
    # Clarification flow
    "final_question": "processed_question_after_clarification",
    "clarification_question": "question_asked_to_user",
    "original_question": "user_raw_input",
    
    # Context tracking (NEW)
    "previous_question": "last_analysis_question",
    "previous_summary": "last_analysis_executive_summary"
}
```

#### Context Lifecycle:
1. **First Question**: No previous context, normal processing
2. **Analysis Completion**: Results stored with summary and question
3. **Follow-up Detection**: Previous context extracted and attached
4. **Context Integration**: Clarification agent receives enriched prompt
5. **Improved Processing**: Better understanding leads to fewer clarifications

### Benefits & Impact

#### User Experience Improvements:
- **Conversational Flow**: Natural follow-up questions without repetition
- **Reduced Friction**: Fewer clarification interruptions
- **Context Continuity**: Maintains analysis thread across questions
- **Reference Resolution**: Automatic understanding of "it", "that", "those"

#### System Intelligence:
- **Smarter Clarifications**: Only asks when truly needed
- **Enhanced Understanding**: Leverages previous analysis insights
- **Efficient Processing**: Reduces redundant CSV examinations
- **Better Task Planning**: Can build incrementally on previous work

### Technical Implementation Notes

#### Context Passing Chain:
```
Streamlit UI → capture_agent_execution_with_visibility() → 
analysis_agent_orch_runner_streamlit() → Clarification Agent
```

#### Safety Mechanisms:
- **Context Validation**: Checks for valid previous_summary and previous_question
- **Graceful Degradation**: Falls back to normal processing if context unavailable
- **Memory Management**: Limits context to most recent analysis to prevent bloat
- **Error Handling**: Context parsing errors don't break the main analysis flow

This context tracking system transforms the multi-agent framework from a stateless question-answer system into an intelligent, conversational analysis assistant that maintains awareness of previous work and can build incrementally on past insights.

---

## Tools & Utilities

### Core Function Tools

#### 🔧 read_csv_file(file_path: str)
**Purpose**: Comprehensive CSV examination and metadata extraction

**Returns**:
```json
{
    "shape": [rows, columns],
    "columns": ["list", "of", "columns"],
    "dtypes": {"column": "data_type"},
    "head": {"sample_data": "first_5_rows"},
    "describe": {"statistical_summary": "for_numeric_columns"},
    "null_counts": {"column": "missing_count"},
    "memory_usage": "X.XX MB"
}
```

**Usage**: Called by all agents to understand data structure before processing

---

#### ⚙️ execute_python_code(code: str, csv_file_path: Optional[str])
**Purpose**: Controlled execution of generated Python analysis code

**Features**:
- **Isolated Environment**: Captures stdout/stderr safely
- **Variable Tracking**: Monitors created variables and outputs
- **DataFrame Loading**: Automatically loads CSV as 'df' variable
- **Error Management**: Comprehensive exception handling and tracebacks

**Returns**:
```json
{
    "success": boolean,
    "output": "stdout_content",
    "error": "error_message_if_failed",
    "variables": {"created_variables": "their_values"},
    "stderr": "stderr_content_if_any"
}
```

---

#### ✅ validate_python_code(code: str)
**Purpose**: Syntax validation without execution

**Features**:
- **Compile Check**: Uses Python's compile() for syntax validation
- **Error Details**: Line numbers and specific syntax error messages
- **Pre-execution Safety**: Prevents runtime errors from bad syntax

**Returns**:
```json
{
    "valid": boolean,
    "error": "syntax_error_details_if_invalid"
}
```

---

#### 💾 save_code_to_file(code: str, file_path: str)
**Purpose**: Persists generated analysis code for reuse and audit

**Features**:
- **Code Persistence**: Saves all generated analysis code
- **File Management**: Handles file writing with error management
- **Audit Trail**: Maintains record of all generated analysis code

---

#### 🖼️ list_saved_images(session_id: str)
**Purpose**: Inventories all visualizations created during analysis session

**Returns**:
```json
{
    "session_id": "unique_session_identifier",
    "images": ["list/of/image/paths"],
    "count": "number_of_images_created"
}
```

**Usage**: Used by Summary Agent to catalog and describe visualizations

---

## Orchestration Logic

### Primary Orchestration Flow

The main orchestration is handled by `analysis_agent_orch_runner_streamlit()`:

#### Phase 1: Session Management
```python
# Session ID generation or reuse
if conversation_state and "session_id" in conversation_state:
    SESSION_ID = conversation_state["session_id"]  # Reuse for clarifications
else:
    SESSION_ID = str(uuid.uuid4())[:8]  # Generate new session
```

#### Phase 2: Context Integration
```python
# Previous context for follow-up questions
if conversation_state and conversation_state.get("previous_summary"):
    previous_context = f"""
    Previous question: {conversation_state.get("previous_question")}
    Previous analysis summary: {conversation_state.get("previous_summary")}
    """
```

#### Phase 3: Clarification Processing
```python
# Agent execution with context
clarification_result = Runner.run_sync(clarification_agent, input=clarification_prompt)

# Decision branching
if clarification_output.need_clarification == "yes" and not clarification_answer:
    return {"status": "needs_clarification", ...}
```

#### Phase 4: Task Planning
```python
# Task decomposition
task_plan_result = Runner.run_sync(task_planner_agent, input=task_plan_prompt)

# Task execution loop
for task in task_plan_output.tasks:
    result = runner.analyze(task, execute_code=True, session_id=SESSION_ID)
    all_results.append(result)
```

#### Phase 5: Summary Generation
```python
# Results synthesis
summary_result = Runner.run_sync(summarizer_agent, input=summary_prompt)
final_summary = summary_result.final_output
```

### State Management

#### Conversation State Structure:
```python
{
    "final_question": "processed_question",
    "clarification_question": "asked_clarification",
    "original_question": "user_input",
    "session_id": "unique_session_id",
    "previous_question": "last_question",  # New: Context awareness
    "previous_summary": "last_analysis_summary"  # New: Context awareness
}
```

#### Status Return Types:
- `"needs_clarification"`: Waiting for user clarification
- `"completed"`: Analysis successfully finished
- `"error"`: Analysis failed with error details

### Error Handling Strategy

1. **Tool-Level**: Each function tool includes comprehensive error handling
2. **Agent-Level**: Agents handle tool failures gracefully
3. **Orchestration-Level**: Main function catches and reports system errors
4. **Session-Level**: Session state preserved across error recovery

### Performance Optimizations

1. **Session Isolation**: Unique image folders prevent cross-contamination
2. **Context Reuse**: Previous analysis context reduces redundant processing
3. **O3 Leveraging**: Single-task preference utilizes O3's multi-step capabilities
4. **Tool Efficiency**: Minimal CSV reads, cached data structures
5. **Memory Management**: Controlled execution environments prevent memory leaks

---

## Configuration & Customization

### Model Configuration
- **All Agents**: Use OpenAI O3 model for enhanced reasoning
- **Max Turns**: Configurable per agent (default: 3 for complex agents)
- **Timeouts**: Managed by OpenAI Agents SDK

### Output Customization
- **Image Limits**: Max 6 visualizations per analysis (configurable)
- **Code Persistence**: All generated code saved to `code.txt`
- **Session Management**: Unique folders per analysis session

### Integration Points
- **Streamlit UI**: Real-time progress monitoring and interaction
- **File System**: Organized output structure with session isolation
- **API Management**: Centralized OpenAI API key management

---

## Usage Examples

### Simple Analysis Request:
```
Input: "Show me sales trends by month"
→ Clarification: No (specific request)
→ Task Planning: Single task (cohesive trend analysis)
→ Execution: Generate time-series code with visualizations
→ Summary: Executive report with trend insights
```

### Complex Analysis Request:
```
Input: "Analyze employee data and create financial report"
→ Clarification: No (clear but complex)
→ Task Planning: Multiple tasks (different domains)
→ Execution: Separate employee and financial analysis
→ Summary: Combined executive summary
```

### Follow-up Question:
```
Previous: "Sales analysis by region" 
Follow-up: "Show me the top 3 regions only"
→ Clarification: Receives previous context
→ Task Planning: Single task with context awareness
→ Execution: Filtered analysis building on previous work
→ Summary: Focused report with previous context
```

This architecture ensures reliable, scalable, and context-aware data analysis with comprehensive audit trails and executive-ready outputs.
