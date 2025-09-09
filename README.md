# 🏠 Australian Real Estate AI Agent

An intelligent AI-powered real estate assistant that provides insights about Melbourne property data using advanced language models and data analytics.

## ✨ Features

- **Property Search**: Find detailed information about specific properties by address
- **Suburb Analytics**: Get comprehensive trends and statistics for any Melbourne suburb
- **Natural Language Interface**: Ask questions in plain English
- **Real-time Data**: Access to Melbourne housing dataset with 34,000+ property records
- **Conversation Memory**: Maintains context across multiple interactions
- **Full Observability**: Comprehensive tracing and monitoring with LangSmith

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- (Optional) LangSmith API key for monitoring

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/australian-real-estate-ai-agent.git
   cd australian-real-estate-ai-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your API keys:
   ```env
   # Required
   OPENAI_API_KEY=your_openai_api_key_here
   
   # Optional - for monitoring and debugging
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your_langsmith_api_key_here
   LANGCHAIN_PROJECT=Australian-Real-Estate-Agent
   ```

4. **Run the application**
   ```bash
   python -m app.main
   ```

## 💬 Usage Examples

Once running, you can ask questions like:

```
🏠 Australian Real Estate AI Agent
========================================
✅ LangSmith tracing enabled
📊 Session ID: abc123-def456-ghi789
🔗 View traces at: https://smith.langchain.com
========================================

Ask a question about Melbourne real estate (type 'exit' to quit): What's the price of 85 Turner St?

=== Agent Response ===
I found the property at 85 Turner St in Abbotsford. Here are the details:
- Address: 85 Turner St
- Suburb: Abbotsford
- Bedrooms: 2
- Property Type: House
- Price: $1,480,000
- Bathrooms: 1
- Land Size: 202 sqm
- Year Built: Not specified
==============================

Ask a question about Melbourne real estate (type 'exit' to quit): What's the median price in Abbotsford?

=== Agent Response ===
Here are the current trends for Abbotsford:
- Suburb: Abbotsford
- Median Price: $1,200,000
- Total Properties: 1,250
- Average Land Size: 180 sqm
==============================
```

## 🏗️ Architecture

This project uses a modern AI agent architecture with multiple specialized frameworks:

### Core Components

- **LangGraph**: Orchestrates the agent's decision-making workflow
- **LangChain**: Provides LLM integration and tool framework
- **Pydantic**: Ensures data validation and type safety
- **LangSmith**: Offers comprehensive monitoring and debugging
- **Pandas**: Handles data processing and analytics

### Data Flow

```
User Query → LangGraph Agent → LangChain Tools → Data Analysis → Natural Language Response
```

## 📁 Project Structure

```
australian-real-estate-ai-agent/
├── app/
│   ├── main.py          # Main application entry point
│   ├── agent.py         # LangGraph agent implementation
│   ├── models.py        # Pydantic data models
│   └── tools.py         # LangChain tools for data access
├── data/
│   └── Melbourne_housing_FULL.csv  # Property dataset (34,000+ records)
├── requirements.txt     # Python dependencies
├── LANGSMITH_SETUP.md  # Detailed LangSmith setup guide
└── README.md           # This file
```

## 🔧 Available Tools

The AI agent has access to two main tools:

### 1. Property Search (`get_property_details`)
- **Input**: Property address
- **Output**: Detailed property information including price, bedrooms, bathrooms, land size, etc.
- **Example**: "What's the price of 85 Turner St?"

### 2. Suburb Analytics (`get_suburb_trends`)
- **Input**: Suburb name
- **Output**: Median price, property count, average land size
- **Example**: "What's the median price in Abbotsford?"

## 📊 Dataset

The project uses the Melbourne Housing Dataset containing:
- **34,000+ property records**
- **Suburbs**: All Melbourne metropolitan areas
- **Property Types**: Houses, units, townhouses
- **Data Points**: Price, bedrooms, bathrooms, land size, year built, location coordinates

## 🔍 Monitoring & Debugging

### LangSmith Integration

The application includes comprehensive monitoring through LangSmith:

- **Real-time Tracing**: See exactly how the agent processes each request
- **Tool Execution**: Monitor which tools are called and their results
- **Performance Analytics**: Track response times and token usage
- **Error Monitoring**: Automatic error logging and debugging
- **Session Management**: Group related conversations

### Viewing Traces

1. Set up your LangSmith account at [smith.langchain.com](https://smith.langchain.com)
2. Configure your API key in the `.env` file
3. Run the application and interact with the agent
4. View traces in real-time on the LangSmith dashboard

## 🛠️ Development

### Running Tests

Test the data tools directly:
```bash
python -m app.tools
```

### Adding New Tools

1. Create a new function in `tools.py`
2. Add the `@tool` decorator with proper input schema
3. Register the tool in `agent.py`
4. The agent will automatically learn to use the new tool

### Customizing the Agent

- **Model**: Change the LLM model in `agent.py`
- **Tools**: Add or modify tools in `tools.py`
- **Data**: Replace or extend the dataset in `data/`
- **UI**: Modify the conversation interface in `main.py`

## 📋 Requirements

- `pydantic` - Data validation and modeling
- `langchain` - LLM framework and tool integration
- `langgraph` - Agent workflow orchestration
- `langchain_openai` - OpenAI integration
- `pandas` - Data processing and analytics
- `python-dotenv` - Environment variable management
- `langsmith` - Monitoring and debugging (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Melbourne Housing Dataset for providing comprehensive property data
- LangChain team for the excellent LLM framework
- OpenAI for providing the GPT-4 model
- LangSmith for monitoring and debugging capabilities

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/australian-real-estate-ai-agent/issues) page
2. Review the [LangSmith Setup Guide](LANGSMITH_SETUP.md)
3. Create a new issue with detailed information

---

**Built with ❤️ for the Australian real estate community**
