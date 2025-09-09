# LangSmith Integration Setup Guide

This guide will help you set up LangSmith for debugging and monitoring your Australian Real Estate AI Agent.

## What is LangSmith?

LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications. It provides:
- **Tracing**: See exactly how your agent processes requests
- **Monitoring**: Track performance and errors
- **Debugging**: Identify issues in your agent's reasoning
- **Analytics**: Understand usage patterns and costs

## Setup Instructions

### 1. Install Dependencies

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Get LangSmith API Key

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Sign up for a free account
3. Navigate to your API Keys section
4. Create a new API key
5. Copy the API key

### 3. Configure Environment Variables

Create a `.env` file in your project root:

```bash
cp .env.example .env
```

Edit the `.env` file with your actual API keys:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith Configuration for debugging and monitoring
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=Australian-Real-Estate-Agent

# Optional: Set to 'true' to enable debug mode
LANGCHAIN_DEBUG=false
```

### 4. Run the Application

```bash
python -m app.main
```

## What Gets Traced

The integration automatically traces:

### 1. **Model Calls**
- Input messages to the LLM
- LLM responses and tool calls
- Token usage and costs

### 2. **Tool Executions**
- Property search queries
- Suburb trend calculations
- Tool inputs and outputs
- Success/failure states

### 3. **Conversation Flow**
- User queries
- Agent responses
- Session management
- Error handling

## Viewing Traces

1. Go to [https://smith.langchain.com](https://smith.langchain.com)
2. Navigate to your project: "Australian-Real-Estate-Agent"
3. View traces in real-time as you interact with the agent

## Trace Types

### Chain Traces
- **real_estate_conversation**: Start of each user interaction
- **real_estate_conversation_complete**: Successful completion
- **real_estate_conversation_error**: Error handling

### LLM Traces
- **real_estate_agent_model_call**: Input to the LLM
- **real_estate_agent_model_response**: LLM output

### Tool Traces
- **get_property_details**: Property search execution
- **get_suburb_trends**: Suburb analysis execution
- **real_estate_tool_*_success**: Successful tool execution
- **real_estate_tool_*_error**: Tool execution errors

## Features

### Session Management
- Each conversation gets a unique session ID
- Track conversations across multiple interactions
- View conversation history and context

### Error Monitoring
- Automatic error logging
- Detailed error messages
- Stack trace information

### Performance Analytics
- Response times
- Token usage
- Cost tracking
- Success/failure rates

## Debugging Tips

### 1. Check LangSmith Status
The application will show LangSmith status on startup:
- ✅ LangSmith tracing enabled
- ⚠️ LangSmith tracing disabled

### 2. View Real-time Traces
- Traces appear in LangSmith dashboard within seconds
- Use filters to find specific conversations
- Click on traces for detailed information

### 3. Common Issues
- **No traces appearing**: Check your API key and environment variables
- **Permission errors**: Ensure your API key has proper permissions
- **Rate limiting**: LangSmith has rate limits for free accounts

## Advanced Configuration

### Custom Project Name
Change the project name in your `.env` file:
```env
LANGCHAIN_PROJECT=My-Custom-Project-Name
```

### Debug Mode
Enable detailed logging:
```env
LANGCHAIN_DEBUG=true
```

### Custom Endpoints
For enterprise users:
```env
LANGCHAIN_ENDPOINT=https://your-custom-endpoint.com
```

## Troubleshooting

### LangSmith Client Not Initializing
- Check that `LANGCHAIN_API_KEY` is set correctly
- Verify the API key is valid
- Check your internet connection

### Traces Not Appearing
- Ensure `LANGCHAIN_TRACING_V2=true`
- Check the project name matches in LangSmith
- Wait a few minutes for traces to appear

### Performance Issues
- LangSmith adds minimal overhead
- Disable tracing if needed by removing `LANGCHAIN_API_KEY`
- Use `LANGCHAIN_DEBUG=false` for production

## Benefits

1. **Debugging**: See exactly how your agent processes each request
2. **Optimization**: Identify bottlenecks and improve performance
3. **Monitoring**: Track usage patterns and costs
4. **Quality Assurance**: Ensure consistent behavior across different inputs
5. **Analytics**: Understand user interactions and preferences

## Next Steps

1. Set up your LangSmith account and API key
2. Configure your `.env` file
3. Run the application and interact with the agent
4. Check the LangSmith dashboard to see your traces
5. Use the insights to improve your agent's performance

For more information, visit the [LangSmith documentation](https://docs.smith.langchain.com/).
