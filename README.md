# Multi-Agent CSV Analysis System

A sophisticated CSV data analysis system powered by OpenAI's Agents SDK and O3 model, featuring intelligent clarification, task planning, code generation, and execution with enhanced real-time visibility.

## 🚀 Features

### Core Capabilities
- **Intelligent Question Analysis**: Clarification agent determines if user questions need more specificity
- **Context-Aware Follow-ups**: Remembers previous analysis and questions for better follow-up responses
- **Smart Task Planning**: Automatically breaks down complex analysis requests into manageable tasks
- **Code Generation & Execution**: Generates and executes Python code for data analysis
- **Visual Analytics**: Creates charts, graphs, and visualizations automatically
- **Real-time Monitoring**: Live agent status tracking with progress indicators

### Agent Architecture
1. **🤔 Clarification Agent**: Analyzes question clarity and asks for specifics when needed
2. **📋 Task Planner Agent**: Breaks down complex requests into structured tasks
3. **🚀 CSV Analysis Agent**: Generates and executes Python analysis code
4. **📝 Summary Agent**: Creates comprehensive executive summaries

### Enhanced Visibility Features
- Real-time agent status cards
- Live progress tracking with percentage completion
- Agent execution timeline
- Streaming logs and activity monitoring
- Session-based image organization

## 📋 Requirements

See `requirements.txt` for complete dependencies. Key requirements:
- Python 3.8+
- OpenAI API key
- Streamlit for web interface
- Pandas, Matplotlib, Seaborn for data analysis

## 🛠 Installation

1. **Clone or download the files**
   ```bash
   # Download multiagent.py and streamlit_enhanced_visibility.py
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up OpenAI API Key**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or update line 23 in `multiagent.py` with your API key.

4. **Run the application**
   ```bash
   streamlit run streamlit_enhanced_visibility.py
   ```

## 📊 Usage

### Basic Workflow
1. **Upload CSV**: Use the sidebar to upload your CSV file
2. **Ask Questions**: Type your analysis request in natural language
3. **Clarification**: Answer any clarification questions if prompted
4. **Monitor Progress**: Watch real-time agent execution in the right panel
5. **Review Results**: Get comprehensive analysis with visualizations

### Example Questions
- "Give me some analysis with visual representation"
- "Show me sales trends by region over time"
- "What are the correlations between different columns?"
- "Compare performance across departments"
- "Identify outliers and anomalies in the data"

### Follow-up Questions
The system now remembers previous analysis context:
- "Can you also show this by month?" (references previous analysis)
- "What about for the top 5 regions only?"
- "Create a detailed breakdown of the outliers we found"

## 🏗 Architecture

### File Structure
```
├── multiagent.py                    # Core agent orchestration system
├── streamlit_enhanced_visibility.py # Streamlit web interface
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── data.csv                        # Your uploaded CSV (auto-created)
├── images/                         # Generated visualizations
│   └── [session_id]/              # Session-specific images
└── code.txt                       # Generated analysis code
```

### Key Functions

#### multiagent.py
- `analysis_agent_orch_runner_streamlit()`: Main orchestration function
- `create_csv_analysis_agent()`: Creates session-specific analysis agent
- `read_csv_file()`: CSV examination tool
- `execute_python_code()`: Code execution in controlled environment
- `validate_python_code()`: Syntax validation
- `save_code_to_file()`: Code persistence
- `list_saved_images()`: Image management

#### streamlit_enhanced_visibility.py
- `capture_agent_execution_with_visibility()`: Real-time monitoring wrapper
- `add_agent_activity()`: Activity tracking
- `update_progress_tracking()`: Progress calculation
- `display_live_progress_monitor()`: Progress visualization
- `display_agent_timeline()`: Execution timeline

### Agent Models
- All agents use OpenAI's **O3 model** for enhanced reasoning capabilities
- Context-aware processing for complex multi-step analysis
- Structured outputs using Pydantic models

## 🔧 Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your-api-key-here
```

### Customizable Settings
- **Session Management**: Each analysis gets unique session ID for image organization
- **Progress Tracking**: Real-time agent status monitoring
- **Code Validation**: All generated code is syntax-checked before execution
- **Image Limits**: Maximum 6 plots/graphs per analysis to prevent clutter

## 🎯 Key Improvements (Latest Version)

### Context-Aware Follow-ups
- **Previous Context**: Clarification agent now receives previous question and summary
- **Smart Continuity**: Better understanding of follow-up questions
- **Session Memory**: Maintains context across multiple analysis rounds

### Enhanced Visibility
- **Real-time Monitoring**: Live agent status cards and progress tracking
- **Execution Timeline**: Complete view of agent workflow
- **Streaming Logs**: Live output capture and display
- **Error Handling**: Comprehensive error tracking and display

## 🚨 Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: OpenAI API key not found
   Solution: Set OPENAI_API_KEY environment variable or update line 23 in multiagent.py
   ```

2. **Import Error**
   ```
   Error: No module named 'agents'
   Solution: Install OpenAI Agents SDK: pip install openai-agents-sdk
   ```

3. **CSV Upload Issues**
   ```
   Error: File upload failed
   Solution: Ensure CSV is properly formatted and under size limits
   ```

4. **Memory Issues**
   ```
   Error: Out of memory
   Solution: Process smaller datasets or increase system memory
   ```

### Debug Mode
Set debug flags in the code for additional logging:
```python
# In multiagent.py, enable debug prints
DEBUG = True
```

## 🔒 Security Notes

- Code execution runs in controlled environment
- API keys should be kept secure
- Generated code is validated before execution
- Session isolation prevents cross-contamination

## 📈 Performance Tips

- **Large datasets**: Consider sampling for initial analysis
- **Complex queries**: Break into smaller, focused questions
- **Memory management**: Clear session state periodically
- **API limits**: Be mindful of OpenAI usage limits

## 🤝 Contributing

To extend the system:
1. Add new tools to agent definitions
2. Create custom analysis templates
3. Extend visualization capabilities
4. Add new agent types for specialized tasks

## 📄 License

This project is for educational and research purposes. Please ensure compliance with OpenAI's usage policies.

## 📞 Support

For issues and questions:
1. Check troubleshooting section above
2. Review OpenAI Agents SDK documentation
3. Verify API key and permissions
4. Test with sample CSV data first

---

**Built with OpenAI Agents SDK and O3 Model** | **Enhanced Visibility Edition**