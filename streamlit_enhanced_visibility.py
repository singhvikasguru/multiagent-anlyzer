#!/usr/bin/env python3
"""
Multi-Agent CSV Analysis Chat Interface - ENHANCED VISIBILITY VERSION
Features: Real-time agent monitoring in main chat, live progress bars, status cards, streaming logs
"""

import streamlit as st
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
import traceback
import os
import time
import threading
from datetime import datetime
import glob

# Import your existing code
try:
    import sys
    import os
    
    # Add current directory to path if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    from multiagent import analysis_agent_orch_runner_streamlit
    AGENT_SYSTEM_AVAILABLE = True
    IMPORT_ERROR = None
    
except ImportError as e:
    AGENT_SYSTEM_AVAILABLE = False
    IMPORT_ERROR = f"Import Error: {str(e)}"
except Exception as e:
    AGENT_SYSTEM_AVAILABLE = False
    IMPORT_ERROR = f"General Error: {str(e)}"

# Page configuration
st.set_page_config(
    page_title="Multi-Agent CSV Analysis Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visibility
st.markdown("""
<style>
    /* Agent status cards */
    .agent-card {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
        background-color: #f0f2f6;
    }
    .agent-card.running {
        border-left-color: #ffa500;
        background-color: #fff3e0;
    }
    .agent-card.completed {
        border-left-color: #28a745;
        background-color: #e8f5e9;
    }
    .agent-card.error {
        border-left-color: #dc3545;
        background-color: #ffebee;
    }
    .agent-card.waiting {
        border-left-color: #17a2b8;
        background-color: #e0f7fa;
    }
    
    /* Progress indicator */
    .progress-container {
        margin: 20px 0;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 2px solid #dee2e6;
    }
    
    /* Live log container */
    .live-log {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
        font-size: 12px;
        max-height: 300px;
        overflow-y: auto;
        margin: 10px 0;
    }
    
    /* Agent timeline */
    .agent-timeline {
        position: relative;
        padding-left: 30px;
        margin: 20px 0;
    }
    .timeline-item {
        position: relative;
        padding-bottom: 20px;
    }
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -22px;
        top: 5px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background-color: #007bff;
    }
    .timeline-item.completed::before {
        background-color: #28a745;
    }
    .timeline-item.running::before {
        background-color: #ffa500;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent_activities' not in st.session_state:
        st.session_state.agent_activities = []
    if 'current_agents' not in st.session_state:
        st.session_state.current_agents = {}  # Track current status of each agent
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = []
    if 'waiting_for_clarification' not in st.session_state:
        st.session_state.waiting_for_clarification = False
    if 'clarification_data' not in st.session_state:
        st.session_state.clarification_data = None
    if 'current_analysis_status' not in st.session_state:
        st.session_state.current_analysis_status = "idle"
    if 'progress_logs' not in st.session_state:
        st.session_state.progress_logs = []
    if 'live_logs' not in st.session_state:
        st.session_state.live_logs = []
    if 'progress_percentage' not in st.session_state:
        st.session_state.progress_percentage = 0
    if 'current_task_description' not in st.session_state:
        st.session_state.current_task_description = ""
    if 'analysis_start_time' not in st.session_state:
        st.session_state.analysis_start_time = None

def add_agent_activity(agent_name, status, details, log_content="", update_progress=True):
    """Add agent activity with enhanced visibility tracking"""
    activity = {
        'agent': agent_name,
        'status': status,
        'details': details,
        'log_content': log_content,
        'timestamp': datetime.now().strftime("%H:%M:%S"),
        'full_timestamp': datetime.now(),
        'activity_id': f"{agent_name}_{int(datetime.now().timestamp() * 1000)}"
    }
    
    # Update current agents dictionary for quick status lookup
    st.session_state.current_agents[agent_name] = {
        'status': status,
        'details': details,
        'timestamp': activity['timestamp'],
        'log_content': log_content
    }
    
    # Add to activity log
    st.session_state.agent_activities.append(activity)
    
    # Add to live logs for streaming display
    log_message = f"[{activity['timestamp']}] {agent_name} - {status.upper()}: {details}"
    st.session_state.live_logs.append(log_message)
    
    # Keep only last 50 live logs to prevent memory issues
    if len(st.session_state.live_logs) > 50:
        st.session_state.live_logs = st.session_state.live_logs[-50:]
    
    # Update progress if requested
    if update_progress:
        update_progress_tracking(agent_name, status)

def update_progress_tracking(agent_name, status):
    """Update overall progress percentage based on agent completion"""
    # ACTUAL EXECUTION ORDER (verified from Untitled2.py code):
    # 1. Clarification Agent (checks if clarification needed)
    # 2. Task Planner Agent (breaks down into tasks)
    # 3. CSV Analysis Agent (analyzes and executes - may run multiple times)
    # 4. Summary Agent (creates final summary)
    # NOTE: There's NO separate "Execution Agent" - it's part of CSV Analysis
    
    # Progress targets aligned with actual execution flow
    agent_progress_targets = {
        "🤔 Clarification Agent": 15,      # Clarification done → 15%
        "📋 Task Planner Agent": 30,       # Planning done → 30%
        "🚀 CSV Analysis Agent": 75,       # Analysis/Execution done → 75%
        "⚙️ Execution Agent": 75,          # Same as CSV Analysis (not separate)
        "📝 Summary Agent": 100            # Summary done → 100%
    }
    
    running_targets = {
        "🤔 Clarification Agent": 5,       # Clarification starts → 5%
        "📋 Task Planner Agent": 20,       # Planning starts → 20%
        "🚀 CSV Analysis Agent": 40,       # Analysis starts → 40%
        "⚙️ Execution Agent": 60,          # Execution starts → 60%
        "📝 Summary Agent": 85             # Summary starts → 85%
    }
    
    current_progress = st.session_state.progress_percentage
    
    if status == "completed":
        target = agent_progress_targets.get(agent_name, current_progress)
        # Only move forward, never backward
        new_progress = max(current_progress, target)
        
        # Enhanced logging for debugging
        print(f"📊 PROGRESS UPDATE: {agent_name} COMPLETED")
        print(f"   Current: {current_progress}% → Target: {target}% → New: {new_progress}%")
        
        # Debug warning if progress would decrease
        if new_progress < current_progress:
            print(f"⚠️  WARNING: Progress blocked from decreasing!")
        
        st.session_state.progress_percentage = new_progress
        
    elif status == "running":
        target = running_targets.get(agent_name, current_progress)
        # Only move forward, never backward
        new_progress = max(current_progress, target)
        
        # Enhanced logging for debugging
        print(f"📊 PROGRESS UPDATE: {agent_name} RUNNING")
        print(f"   Current: {current_progress}% → Target: {target}% → New: {new_progress}%")
        
        # Debug warning if progress would decrease
        if new_progress < current_progress:
            print(f"⚠️  WARNING: Progress blocked from decreasing!")
        
        st.session_state.progress_percentage = new_progress

def display_live_progress_monitor():
    """Display real-time progress monitor in the main chat area"""
    if st.session_state.current_analysis_status == "running":
        st.markdown("### 🔄 Analysis in Progress")
        
        # Progress bar
        progress_col1, progress_col2 = st.columns([3, 1])
        with progress_col1:
            st.progress(st.session_state.progress_percentage / 100)
        with progress_col2:
            st.metric("Progress", f"{st.session_state.progress_percentage}%")
        
        # Current task description
        if st.session_state.current_task_description:
            st.info(f"**Current Task:** {st.session_state.current_task_description}")
        
        # Elapsed time
        if st.session_state.analysis_start_time:
            elapsed = (datetime.now() - st.session_state.analysis_start_time).total_seconds()
            st.caption(f"⏱️ Elapsed time: {elapsed:.1f}s")
        
        st.markdown("---")

def display_agent_status_cards():
    """Display status cards for each agent in the main chat area"""
    if not st.session_state.current_agents:
        return
    
    st.markdown("### 🤖 Agent Status")
    
    # Create columns for agent cards
    agents = list(st.session_state.current_agents.keys())
    
    for agent_name, agent_data in st.session_state.current_agents.items():
        status = agent_data['status']
        details = agent_data['details']
        timestamp = agent_data['timestamp']
        
        # Status emoji mapping
        status_emoji = {
            "running": "🟡",
            "completed": "✅",
            "error": "❌",
            "waiting": "⏸️",
            "idle": "⚪"
        }
        
        emoji = status_emoji.get(status, "⚪")
        
        # Create status card
        card_class = f"agent-card {status}"
        st.markdown(f"""
        <div class="{card_class}">
            <h4>{emoji} {agent_name}</h4>
            <p><strong>Status:</strong> {status.upper()}</p>
            <p><strong>Details:</strong> {details}</p>
            <p><small>Last update: {timestamp}</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show logs for running agents
        if status == "running" and agent_data.get('log_content'):
            with st.expander(f"📋 View {agent_name} Logs", expanded=False):
                st.code(agent_data['log_content'], language="text")

def display_agent_timeline():
    """Display agent execution timeline"""
    if not st.session_state.agent_activities:
        return
    
    st.markdown("### 📊 Execution Timeline")
    
    # Group activities by agent
    agent_groups = {}
    for activity in st.session_state.agent_activities:
        agent = activity['agent']
        if agent not in agent_groups:
            agent_groups[agent] = []
        agent_groups[agent].append(activity)
    
    # Display timeline
    for agent, activities in agent_groups.items():
        latest_activity = activities[-1]
        status_class = latest_activity['status']
        
        st.markdown(f"""
        <div class="timeline-item {status_class}">
            <strong>{agent}</strong> - {latest_activity['status'].upper()}<br>
            <small>{latest_activity['timestamp']}: {latest_activity['details']}</small>
        </div>
        """, unsafe_allow_html=True)

def display_live_logs():
    """Display live streaming logs"""
    if not st.session_state.live_logs:
        return
    
    st.markdown("### 📜 Live Logs")
    
    # Display logs in a code block with auto-scroll
    log_text = "\n".join(st.session_state.live_logs[-20:])  # Show last 20 logs
    # Replace newlines outside f-string to avoid backslash in f-string error
    log_text_html = log_text.replace('\n', '<br>')
    st.markdown(f"""
    <div class="live-log">
        {log_text_html}
    </div>
    """, unsafe_allow_html=True)

def capture_agent_execution_with_visibility(question, clarification_answer=None, conversation_state=None):
    """Enhanced execution with prominent visibility"""
    if not AGENT_SYSTEM_AVAILABLE:
        return None, f"Error: Could not import agent system - {IMPORT_ERROR}", True
    
    # Initialize progress tracking
    st.session_state.current_analysis_status = "running"
    st.session_state.progress_percentage = 0
    st.session_state.analysis_start_time = datetime.now()
    st.session_state.current_task_description = "Starting analysis..."
    st.session_state.live_logs = []
    st.session_state.current_agents = {}
    
    # Create placeholder for live updates
    status_placeholder = st.empty()
    
    class VisibilityLogCapture:
        def __init__(self, status_placeholder):
            self.buffer = io.StringIO()
            self.original_stdout = sys.stdout
            self.status_placeholder = status_placeholder
            self.current_agent = None
            self.agent_logs = []
            
        def write(self, text):
            self.buffer.write(text)
            self.original_stdout.write(text)
            self.original_stdout.flush()
            
            if not text.strip():
                return
            
            # Detect agent transitions
            # These triggers match the enhanced logging in Untitled2_ENHANCED.py
            agent_triggers = {
                "AGENT: Clarification Agent - Starting": "🤔 Clarification Agent",
                "AGENT: Task Planner Agent - Starting": "📋 Task Planner Agent",
                "AGENT: CSV Analysis Agent - Starting": "🚀 CSV Analysis Agent",
                "AGENT: Summary Agent - Starting": "📝 Summary Agent"
            }
            
            for trigger, agent_name in agent_triggers.items():
                if trigger in text:
                    # Complete previous agent
                    if self.current_agent:
                        add_agent_activity(
                            self.current_agent,
                            "completed",
                            "Work completed",
                            log_content="\n".join(self.agent_logs)
                        )
                    
                    # Start new agent
                    self.current_agent = agent_name
                    self.agent_logs = []
                    
                    # Set task description
                    task_desc = {
                        "🤔 Clarification Agent": "Analyzing question clarity...",
                        "📋 Task Planner Agent": "Breaking down into tasks...",
                        "🚀 CSV Analysis Agent": "Analyzing CSV data...",
                        "⚙️ Execution Agent": "Executing analysis code...",
                        "📝 Summary Agent": "Generating summary..."
                    }.get(agent_name, "Processing...")
                    
                    st.session_state.current_task_description = task_desc
                    
                    add_agent_activity(
                        agent_name,
                        "running",
                        task_desc
                    )
                    
                    # Update UI in real-time
                    with self.status_placeholder.container():
                        display_live_progress_monitor()
                        display_agent_status_cards()
                    
                    break
            
            # Capture tool calls
            if "🔧 TOOL:" in text:
                log_entry = text.strip()
                self.agent_logs.append(log_entry)
                add_agent_activity(
                    self.current_agent or "🔧 System",
                    "running",
                    log_entry,
                    update_progress=False
                )
                
                # Update UI
                with self.status_placeholder.container():
                    display_live_progress_monitor()
                    display_agent_status_cards()
            
            # Capture regular logs
            elif self.current_agent and text.strip():
                self.agent_logs.append(text.strip())
                
        def flush(self):
            self.original_stdout.flush()
        
        def getvalue(self):
            return self.buffer.getvalue()
    
    # Capture execution with visibility
    log_capture = VisibilityLogCapture(status_placeholder)
    
    try:
        sys.stdout = log_capture
        
        if clarification_answer:
            result = analysis_agent_orch_runner_streamlit(
                question=question,
                clarification_answer=clarification_answer,
                conversation_state=conversation_state
            )
        else:
            result = analysis_agent_orch_runner_streamlit(question)
        
        sys.stdout = log_capture.original_stdout
        
        # Mark all agents as completed
        for agent_name in st.session_state.current_agents.keys():
            if st.session_state.current_agents[agent_name]['status'] == 'running':
                add_agent_activity(agent_name, "completed", "Analysis complete")
        
        st.session_state.current_analysis_status = "completed"
        st.session_state.progress_percentage = 100
        
        # Final UI update
        with status_placeholder.container():
            st.success("✅ Analysis completed successfully!")
            display_agent_timeline()
        
        return result, log_capture.getvalue(), False
        
    except Exception as e:
        sys.stdout = log_capture.original_stdout
        error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
        
        st.session_state.current_analysis_status = "error"
        
        # Mark current agent as error
        if log_capture.current_agent:
            add_agent_activity(log_capture.current_agent, "error", str(e))
        
        with status_placeholder.container():
            st.error(f"❌ Analysis failed: {str(e)}")
            display_agent_timeline()
        
        return None, error_msg, True

def display_chat_messages():
    """Display chat messages with enhanced result visualization"""
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display analysis results inline
            if message["role"] == "assistant" and "analysis_result" in message:
                result = message["analysis_result"]
                
                # Show generated images
                if "generated_images" in result and result["generated_images"]:
                    st.markdown("#### 📊 Generated Visualizations")
                    for img_path in result["generated_images"]:
                        if os.path.exists(img_path):
                            # Use use_column_width for compatibility with older Streamlit
                            try:
                                st.image(img_path, use_column_width=True)
                            except TypeError:
                                # Fallback for very old versions
                                st.image(img_path)
                            st.caption(f"📁 {os.path.basename(img_path)}")
                
                # Try to find images in the images folder if not in result
                if "generated_images" not in result or not result["generated_images"]:
                    # Search for recently generated images
                    image_patterns = [
                        "images/**/*.png",
                        "images/**/*.jpg",
                        "*.png",
                        "*.jpg"
                    ]
                    found_images = []
                    for pattern in image_patterns:
                        found_images.extend(glob.glob(pattern, recursive=True))
                    
                    # Show images generated in last 60 seconds
                    if found_images:
                        import time
                        recent_images = [
                            img for img in found_images 
                            if os.path.exists(img) and (time.time() - os.path.getmtime(img)) < 60
                        ]
                        if recent_images:
                            st.markdown("#### 📊 Generated Visualizations")
                            for img_path in recent_images[:5]:  # Show max 5 images
                                # Use use_column_width for compatibility
                                try:
                                    st.image(img_path, use_column_width=True)
                                except TypeError:
                                    st.image(img_path)
                                st.caption(f"📁 {img_path}")
                
                # Show generated code
                if "generated_code" in result and result["generated_code"]:
                    with st.expander("💻 View Generated Code", expanded=False):
                        st.code(result["generated_code"], language="python")

def handle_user_input(prompt):
    """Handle user input with enhanced visibility"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # IMPORTANT: Reset progress immediately for new analysis
    # This prevents showing old progress from previous analysis
    if not st.session_state.waiting_for_clarification:
        st.session_state.progress_percentage = 0
        st.session_state.current_analysis_status = "starting"
        st.session_state.current_agents = {}
    
    # Process based on state
    if st.session_state.waiting_for_clarification:
        # Handle clarification answer
        clarification_data = st.session_state.clarification_data
        
        st.session_state.waiting_for_clarification = False
        
        # Execute with clarification
        result, logs, has_error = capture_agent_execution_with_visibility(
            question=clarification_data["original_question"],
            clarification_answer=prompt,
            conversation_state=clarification_data.get("conversation_state")
        )
        
        if result and result.get("status") == "completed":
            response_content = f"✅ **Analysis Complete!**\n\n{result.get('summary', '')}"
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_content,
                "analysis_result": result
            })
            st.session_state.analysis_results.append(result)
            
    else:
        # New analysis request
        result, logs, has_error = capture_agent_execution_with_visibility(prompt)
        
        if result:
            if result.get("status") == "needs_clarification":
                clarification_question = result.get("clarification_question", "Can you clarify?")
                response_content = f"🤔 **Need more information:**\n\n{clarification_question}"
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_content
                })
                
                st.session_state.waiting_for_clarification = True
                st.session_state.clarification_data = {
                    "original_question": prompt,
                    "clarification_question": clarification_question,
                    "conversation_state": result.get("conversation_state")
                }
                
            elif result.get("status") == "completed":
                response_content = f"✅ **Analysis Complete!**\n\n{result.get('summary', '')}"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_content,
                    "analysis_result": result
                })
                st.session_state.analysis_results.append(result)
                
            elif result.get("status") == "error":
                error_message = f"❌ **Analysis Error:**\n\n```\n{result.get('error', 'Unknown error')}\n```"
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

def main():
    """Main Streamlit app with ENHANCED VISIBILITY"""
    initialize_session_state()
    
    # Header
    st.title("🤖 Multi-Agent CSV Analysis System")
    st.markdown("**Enhanced Visibility Edition** - *Real-time agent monitoring with live progress tracking*")
    
    # Check if agent system is available
    if not AGENT_SYSTEM_AVAILABLE:
        st.error(f"⚠️ Agent system not available: {IMPORT_ERROR}")
        return
    
    # Main layout: Chat on left, Agent Monitor on right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 💬 Chat Interface")
        
        # Show clarification status
        if st.session_state.waiting_for_clarification:
            st.warning("🤔 **Waiting for your clarification...**")
        
        # Display chat messages
        chat_container = st.container()
        with chat_container:
            display_chat_messages()
        
        # Chat input
        st.markdown("---")
        if st.session_state.waiting_for_clarification:
            input_placeholder = "Please answer the clarification question above..."
        else:
            input_placeholder = "Ask me to analyze your CSV data..."
        
        if prompt := st.chat_input(input_placeholder):
            handle_user_input(prompt)
            st.rerun()
    
    with col2:
        st.markdown("### 🎯 Live Agent Monitor")
        
        # Always show this column for consistency
        # Show current status if running
        if st.session_state.current_analysis_status == "running":
            with st.container():
                display_live_progress_monitor()
                st.markdown("---")
                
                # Show only RUNNING and WAITING agents in cards
                st.markdown("#### 🔄 Active Agents")
                active_shown = False
                for agent_name, agent_data in st.session_state.current_agents.items():
                    if agent_data['status'] in ['running', 'waiting', 'error']:
                        active_shown = True
                        status = agent_data['status']
                        details = agent_data['details']
                        timestamp = agent_data['timestamp']
                        
                        # Status emoji mapping
                        status_emoji = {
                            "running": "🟡",
                            "error": "❌",
                            "waiting": "⏸️"
                        }
                        
                        emoji = status_emoji.get(status, "⚪")
                        
                        # Create status card
                        if status == "running":
                            st.warning(f"{emoji} **{agent_name}**\n\n{details}\n\n*{timestamp}*")
                        elif status == "error":
                            st.error(f"{emoji} **{agent_name}**\n\n{details}\n\n*{timestamp}*")
                        elif status == "waiting":
                            st.info(f"{emoji} **{agent_name}**\n\n{details}\n\n*{timestamp}*")
                        
                        # Show logs for running agents
                        if status == "running" and agent_data.get('log_content'):
                            with st.expander(f"📋 Logs", expanded=False):
                                st.code(agent_data['log_content'][-500:], language="text")  # Last 500 chars
                
                if not active_shown:
                    st.info("⏳ Initializing agents...")
                
                st.markdown("---")
                
                # Show completed agents in timeline
                st.markdown("#### ✅ Completed")
                completed_agents = [
                    (name, data) for name, data in st.session_state.current_agents.items()
                    if data['status'] == 'completed'
                ]
                if completed_agents:
                    for agent_name, agent_data in completed_agents:
                        st.success(f"✅ {agent_name}", icon="✅")
                else:
                    st.caption("_No agents completed yet_")
                
                st.markdown("---")
                display_live_logs()
                
        elif st.session_state.agent_activities:
            # After analysis completes, show summary
            st.success("✅ Last analysis completed")
            
            # Show execution timeline
            st.markdown("#### 📊 Execution Summary")
            display_agent_timeline()
            
        else:
            st.info("⏳ No active analysis")
            st.caption("Upload a CSV and ask a question to start")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 CSV Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your CSV file to analyze"
        )
        
        if uploaded_file is not None:
            try:
                with open("data.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success("✅ File uploaded!")
                
                import pandas as pd
                df = pd.read_csv("data.csv")
                st.markdown("**Preview:**")
                # Use height parameter for compatibility with older Streamlit
                st.dataframe(df.head(3))
                st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")
                
            except Exception as e:
                st.error(f"Upload failed: {str(e)}")
        
        st.markdown("---")
        
        # Controls
        if st.button("🗑️ Clear All"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Session info
        if st.session_state.analysis_results:
            st.markdown("---")
            st.subheader("📈 Session Stats")
            st.metric("Analyses", len(st.session_state.analysis_results))
            st.metric("Messages", len(st.session_state.messages))
            st.metric("Agent Actions", len(st.session_state.agent_activities))

if __name__ == "__main__":
    main()