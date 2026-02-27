# 📚 AI Terminal Pro - Wiki & Documentation 📚

Welcome to the comprehensive wiki for AI Terminal Pro. This document covers all aspects of the application, from basic usage to advanced features and development.

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Configuration](#configuration)
4. [Features & Usage](#features--usage)
5. [Architecture](#architecture)
6. [API Reference](#api-reference)
7. [Development Guide](#development-guide)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [FAQ](#faq)

---

## Overview

### What is AI Terminal Pro?

AI Terminal Pro is a privacy-first, terminal-based AI assistant that gives you complete control over your AI interactions. It supports local and self-hosted LLMs, features RAG-enabled memory, custom tooling, and much more—all while keeping your data private and secure.

### Key Principles

- **Privacy First**: No telemetry, no data collection, your data stays local
- **User Control**: Run models locally or self-hosted, no cloud lock-in
- **Extensibility**: Custom tools, MCP integration, plugin support
- **Developer-Friendly**: Built for power users and developers

### Two Implementations

AI Terminal Pro comes in two flavors:

1. **Python Version** (`ai.py`) - Production-ready, full-featured
2. **TypeScript/Bun Version** (`ai_2/`) - Modern, fast, in progress

Both share similar features and configurations, allowing you to choose based on your preferences.

---

## Installation & Setup

### Prerequisites

#### Python Version
- Python 3.8 or higher
- pip (Python package manager)
- Ollama (recommended) or HuggingFace models
- 4GB+ RAM (8GB+ recommended for local models)

#### TypeScript/Bun Version
- Bun runtime
- Node.js 18+ (optional, for compatibility)
- Ollama or HuggingFace models

### Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/ai-terminal-pro.git
cd ai-terminal-pro
```

#### 2. Install Dependencies

**Python Version (Full Installation):**
```bash
pip install -r requirements.txt
```

**Python Version (Minimal - Core Features Only):**
```bash
pip install requests beautifulsoup4 transformers torch cryptography flask flask-cors textual rich
```

**TypeScript/Bun Version:**
```bash
cd ai_2
bun install
bunx playwright install chromium
```

### Optional Features Installation

#### Text-to-Speech (Basic Mode)
```bash
pip install openai-whisper pyttsx3 sounddevice soundfile
```

#### Enhanced TTS Pro Mode
```bash
pip install faster-whisper kokoro pyaudio sentence-transformers chromadb tiktoken pyyaml
```

#### Camera/Vision (Sight Mode)
```bash
pip install opencv-python numpy
```

#### Web Browsing
```bash
pip install playwright
python -m playwright install chromium
```

#### Model Training
```bash
pip install peft datasets accelerate trl
```

### Initial Configuration

1. **Start Ollama** (if using Ollama backend):
   ```bash
   ollama serve
   ```

2. **Download a Model** (Ollama):
   ```bash
   ollama pull qwen2.5-coder:latest
   # or
   ollama pull llama3
   ```

3. **Edit config.json**:
   ```json
   {
     "first_run": false,
     "backend": "ollama",
     "model_name": "qwen2.5-coder:latest",
     "system_prompt": "You are a helpful AI assistant.",
     "enable_dangerous_commands": true,
     "max_context_window": 2048,
     "max_response_tokens": 250,
     "temperature": 0.7
   }
   ```

4. **Run the Application**:
   ```bash
   python ai.py
   ```

---

## Configuration

### Configuration File (`config.json`)

The main configuration file controls all aspects of the application:

```json
{
  "first_run": false,
  "backend": "ollama",
  "model_name": "qwen2.5-coder:latest",
  "system_prompt": "You are a helpful AI assistant. Use available tools (ACTION: ...) to solve problems.",
  "enable_dangerous_commands": true,
  "max_context_window": 2048,
  "max_response_tokens": 250,
  "temperature": 0.7,
  "cpu_threads": 4,
  "cpu_interop_threads": 2,
  "builder_threads": 1,
  "default_editor_command": ""
}
```

### Configuration Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `first_run` | boolean | Whether this is the first run | `true` |
| `backend` | string | AI backend: `"ollama"` or `"huggingface"` | `"ollama"` |
| `model_name` | string | Model identifier | Varies by backend |
| `system_prompt` | string | System prompt for the AI | Custom |
| `enable_dangerous_commands` | boolean | Allow file system commands | `false` |
| `max_context_window` | integer | Maximum context window size | `2048` |
| `max_response_tokens` | integer | Maximum response tokens | `250` |
| `temperature` | float | Model temperature (0.0-1.0) | `0.7` |
| `cpu_threads` | integer | CPU threads for PyTorch | `4` |
| `cpu_interop_threads` | integer | CPU interop threads | `2` |
| `builder_threads` | integer | Threads for app builder | `1` |
| `default_editor_command` | string | Default editor command | `""` |

### Backend Configuration

#### Ollama Backend

1. Install Ollama: https://ollama.ai
2. Start Ollama service: `ollama serve`
3. Download models: `ollama pull <model-name>`
4. Set in config: `"backend": "ollama"`

Popular models:
- `qwen2.5-coder:latest` - Coding-focused
- `llama3` - General purpose
- `mistral` - Balanced performance
- `codellama` - Code generation

#### HuggingFace Backend

1. Install transformers: `pip install transformers torch`
2. Set in config: `"backend": "huggingface"`
3. Specify model: `"model_name": "gpt2"` (or any HuggingFace model)

**Note**: HuggingFace backend requires significant RAM (8GB+ recommended).

---

## Features & Usage

### 1. Chat Interface

The chat interface is the core interaction mode:

#### Starting a Chat

1. Select "Start Chat" from the main menu
2. Choose or create a session
3. Start conversing!

#### Chat Commands

- `/help` or `/?` - Show help information
- `/new [name]` - Create a new chat session
- `/save [filename]` - Save current chat to file
- `/load [id|filename]` - Load a chat session
- `/project [name|desc]` - Create or switch project
- `/project_save [file]` - Save current project
- `/project_load [id|file]` - Load a project
- `/back` - Return to main menu
- `/camera` - Launch camera-only assistant
- `/voice` or `/tts` - Launch voice/TTS assistant
- `/vision` - Launch unified Vision + Voice assistant

#### Using Actions

Execute tools using the ACTION syntax:
```
ACTION: TOOL_NAME arg1 arg2 ...
```

Example:
```
ACTION: BROWSE https://example.com
```

### 2. RAG (Retrieval-Augmented Generation)

RAG allows the AI to access context from your documents.

#### Loading Documents

1. Select "Document Loader (RAG)" from the main menu
2. Choose a file to ingest
3. Documents are processed and stored in the vector database
4. The AI can now retrieve relevant context from these documents

#### Supported Formats

- Text files (`.txt`)
- Markdown (`.md`)
- Python files (`.py`)
- JSON files (`.json`)
- And more (via file parsing)

#### Document Management

- Documents are stored in `documents/` directory
- Vector embeddings stored in `ai_memory.sqlite`
- Project-specific documents can be organized by project

### 3. Custom Tools

Create custom tools to extend functionality.

#### Tool Types

1. **Python Tools** (`.py`)
   - Full Python functions
   - Parameters and return values
   - Access to system resources (with safety checks)

2. **JSON Tools** (`.json`)
   - Declarative tool definitions
   - Command or script execution
   - Parameter definitions

3. **YAML Tools** (`.yaml`)
   - Human-readable format
   - Command or script execution
   - Parameter definitions

#### Creating a Python Tool

Create a file in `custom_tools/`:

```python
def tool_name(param1, param2):
    """
    Tool description
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    """
    # Tool implementation
    result = f"Processed {param1} and {param2}"
    return result
```

#### Creating a JSON Tool

```json
{
  "name": "tool_name",
  "description": "Tool description",
  "parameters": {
    "param1": {
      "type": "string",
      "description": "Parameter 1 description"
    },
    "param2": {
      "type": "integer",
      "description": "Parameter 2 description"
    }
  },
  "command": "echo {param1} {param2}"
}
```

### 4. MCP Server Integration

Model Context Protocol (MCP) allows integration with external services.

#### Supported MCP Servers

- File system access
- GitHub integration
- Database connections
- API integrations
- Custom MCP servers

#### Adding an MCP Server

1. Select "MCP Server Management" from the main menu
2. Add server configuration
3. Provide server command and parameters
4. Server is automatically connected

### 5. Model Training

Train and fine-tune models on your data.

#### Training Types

1. **Fine-Tuning**
   - Full model fine-tuning
   - Requires significant resources
   - Best for domain-specific tasks

2. **LoRA (Low-Rank Adaptation)**
   - Parameter-efficient fine-tuning
   - Faster and requires less memory
   - Good for most use cases

3. **RLHF (Reinforcement Learning from Human Feedback)**
   - Train with human preferences
   - Improve model alignment
   - Advanced technique

#### Training Workflow

1. Prepare dataset in `training/data/`
2. Select "Model Training" from the main menu
3. Choose training type
4. Configure parameters
5. Start training
6. Monitor progress
7. Save checkpoints

### 6. API Server

Create RESTful API endpoints for remote access.

#### Creating an API

1. Select "API Management" from the main menu
2. Create new API
3. Configure:
   - Port number
   - CORS settings
   - IP whitelist
   - Authentication
4. Start the server

#### API Endpoints

- `POST /chat` - Chat endpoint
- `GET /health` - Health check
- `GET /info` - API information

#### API Security

- **Encryption**: Fernet-based encryption
- **Authentication**: API key support
- **IP Whitelisting**: Restrict access by IP
- **CORS**: Configurable cross-origin settings

### 7. App Builder (Multi-Agent Development)

Orchestrate multiple AI agents for development tasks.

#### Available Agents

- **SpecificationWriter**: Creates project specifications
- **Architect**: Designs system architecture
- **TechLead**: Makes technical decisions
- **Developer**: Writes code
- **CodeMonkey**: Handles routine coding tasks
- **Reviewer**: Reviews code
- **Troubleshooter**: Debugs issues
- **Debugger**: Finds and fixes bugs
- **TechnicalWriter**: Writes documentation

#### Using App Builder

1. Select "App Builder" from the main menu
2. Create or load a project
3. Specify requirements
4. Agents work together to build the project
5. Monitor progress
6. Review and iterate

### 8. Text-to-Speech (TTS)

Voice output capabilities.

#### Basic TTS Mode

- Uses pyttsx3
- System TTS engines
- Quick setup
- Command: `/tts` or `/voice`

#### Enhanced TTS Pro Mode

- Uses faster-whisper and kokoro
- Higher quality voices
- RAG integration
- Better performance

### 9. Camera/Vision (Sight Mode)

Computer vision capabilities.

#### Features

- Webcam access
- Real-time image processing
- Vision-based conversations
- Command: `/camera` or `/vision`

#### Requirements

- Webcam hardware
- OpenCV installed
- Sufficient system resources

### 10. Web Browsing

Browser automation with Playwright.

#### Capabilities

- Navigate websites
- Extract content
- Click elements
- Fill forms
- Search Google
- Multiple tabs

#### Usage

Use the `BROWSE` action:
```
ACTION: BROWSE https://example.com
```

---

## Architecture

### System Components

#### 1. ConfigManager
- Manages configuration file
- Handles first-run setup
- Backend/model selection

#### 2. MemoryManager
- RAG-enabled context memory
- Vector database management
- Session and project storage
- SQLite-based persistence

#### 3. MCPClient
- Model Context Protocol integration
- JSON-RPC 2.0 client
- Server management
- Tool integration

#### 4. ToolRegistry
- Custom tool management
- MCP server integration
- Tool execution
- Safety checks

#### 5. AIEngine
- Handles Ollama and HuggingFace backends
- Token management
- Context window handling
- Model inference

#### 6. ContextManager
- Conversation history management
- RAG context integration
- Project memory support
- Token counting

#### 7. APIServerManager
- Multiple API server instances
- Encryption support
- CORS configuration
- Authentication

#### 8. AppBuilderOrchestrator
- Multi-agent coordination
- Threaded execution
- Project management

### Data Storage

#### SQLite Databases

- `ai_memory.sqlite`: RAG vector database, sessions, messages
- `app_projects.sqlite`: App builder projects

#### Directories

- `documents/`: RAG documents
- `custom_tools/`: User-defined tools
- `training/`: Training data and models
- `api/`: API server configurations
- `apps/`: App builder projects
- `ai_sandbox/`: Sandboxed file operations
- `backups/`: Automatic backups

### Memory System

#### RAG Architecture

1. **Document Ingestion**
   - Files loaded and chunked
   - Text extracted and processed
   - Vector embeddings generated
   - Stored in vector database

2. **Context Retrieval**
   - Query embedded
   - Semantic search performed
   - Relevant chunks retrieved
   - Context injected into prompt

3. **Session Memory**
   - Conversation history stored
   - Project-specific context
   - Persistent across sessions

---

## API Reference

### Chat API

#### POST /chat

Send a chat message and receive a response.

**Request:**
```json
{
  "message": "Hello, how are you?",
  "session_id": "optional-session-id",
  "project_id": "optional-project-id"
}
```

**Response:**
```json
{
  "response": "I'm doing well, thank you!",
  "session_id": "session-id",
  "tokens_used": 150
}
```

### Health API

#### GET /health

Check API server health.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-12-12T14:00:00Z"
}
```

### Info API

#### GET /info

Get API server information.

**Response:**
```json
{
  "name": "api-name",
  "port": 5000,
  "backend": "ollama",
  "model": "qwen2.5-coder:latest"
}
```

---

## Development Guide

### Project Structure

```
ai-terminal-pro/
├── ai.py                 # Main Python application
├── config.json           # Configuration file
├── requirements.txt      # Python dependencies
├── README.md            # Main README
├── WIKI.md              # This file
├── CONTRIBUTING.md      # Contribution guidelines
├── LICENSE              # License file
├── tui/                 # Terminal UI components
│   ├── components/      # UI components
│   ├── screens/         # Screen implementations
│   ├── commands/        # Command handlers
│   └── utils/           # Utility functions
├── documents/           # RAG documents
├── custom_tools/        # User-defined tools
├── training/            # Training data and models
│   ├── data/           # Training datasets
│   ├── lora/           # LoRA models
│   ├── models/         # Fine-tuned models
│   └── reinforcement/  # RLHF models
├── api/                 # API configurations
├── apps/                # App builder projects
├── ai_sandbox/          # Sandbox directory
├── backups/             # Automatic backups
└── ai_2/                # TypeScript/Bun version
    ├── src/            # TypeScript source
    ├── package.json    # Dependencies
    └── ...
```

### Adding New Features

1. **Understand the Architecture**
   - Review existing code structure
   - Identify where your feature fits
   - Check for similar implementations

2. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Implement the Feature**
   - Follow code style guidelines
   - Add appropriate comments
   - Handle errors gracefully

4. **Test Your Changes**
   - Test with different configurations
   - Test edge cases
   - Ensure backward compatibility

5. **Update Documentation**
   - Update README if needed
   - Add to WIKI if it's a major feature
   - Update code comments

6. **Submit a Pull Request**
   - Clear description
   - Link related issues
   - Request review

### Code Style

- Follow PEP 8 (Python) or standard TypeScript conventions
- Use descriptive variable names
- Add docstrings/comments for complex logic
- Handle errors explicitly
- Keep functions focused and small

### Testing

- Test with different backends (Ollama, HuggingFace)
- Test with different model sizes
- Test edge cases and error conditions
- Test with optional dependencies missing

---

## Troubleshooting

### Common Issues

#### Issue: Model Not Loading

**Symptoms**: Error when starting chat, model not found

**Solutions**:
1. Check backend configuration in `config.json`
2. For Ollama: Ensure Ollama is running (`ollama serve`)
3. For Ollama: Verify model is downloaded (`ollama list`)
4. For HuggingFace: Check model name is correct
5. Check system resources (RAM, disk space)

#### Issue: Out of Memory

**Symptoms**: Application crashes, slow performance

**Solutions**:
1. Reduce `max_context_window` in config
2. Use smaller models
3. Close other applications
4. Use Ollama backend (more efficient)
5. Reduce `max_response_tokens`

#### Issue: Tools Not Working

**Symptoms**: ACTION commands fail, tools not found

**Solutions**:
1. Check tool files in `custom_tools/`
2. Verify tool syntax (Python/JSON/YAML)
3. Check dangerous commands setting
4. Review error messages in console
5. Test tools manually

#### Issue: RAG Not Working

**Symptoms**: Documents not being retrieved, no context

**Solutions**:
1. Verify documents are loaded (check `documents/` directory)
2. Check `ai_memory.sqlite` exists and is accessible
3. Ensure document format is supported
4. Try re-ingesting documents
5. Check vector database integrity

#### Issue: API Server Not Starting

**Symptoms**: API server fails to start, connection errors

**Solutions**:
1. Check port is not in use
2. Verify Flask is installed
3. Check firewall settings
4. Review API configuration
5. Check error logs

#### Issue: TTS/Voice Not Working

**Symptoms**: No audio output, voice commands not recognized

**Solutions**:
1. Install TTS dependencies
2. Check audio system configuration
3. Verify microphone permissions (for voice input)
4. Test with system TTS first
5. Check audio device settings

### Getting Help

1. **Check Documentation**
   - Review this WIKI
   - Check README.md
   - Review code comments

2. **Check Issues**
   - Search GitHub issues
   - Check for similar problems
   - Review closed issues

3. **Create an Issue**
   - Provide detailed description
   - Include error messages
   - Share relevant configuration
   - Describe steps to reproduce

---

## Best Practices

### Configuration

- **Start Simple**: Begin with default settings
- **Incremental Changes**: Adjust one parameter at a time
- **Backup Config**: Keep backups of working configurations
- **Document Changes**: Note what works for your use case

### RAG Usage

- **Organize Documents**: Use projects to organize related documents
- **Quality over Quantity**: Focus on relevant, high-quality documents
- **Regular Updates**: Re-ingest documents when they change
- **Project-Specific**: Use project memory for focused contexts

### Tool Development

- **Start Small**: Create simple tools first
- **Test Thoroughly**: Test tools in isolation
- **Document Well**: Clear descriptions and parameters
- **Safety First**: Be cautious with dangerous commands
- **Error Handling**: Handle errors gracefully

### Performance

- **Model Selection**: Choose appropriate model for your hardware
- **Context Management**: Keep context windows reasonable
- **Resource Monitoring**: Monitor CPU, RAM, and disk usage
- **Backend Choice**: Ollama is generally more efficient than HuggingFace

### Security

- **Dangerous Commands**: Disable if not needed
- **API Security**: Use encryption and authentication
- **IP Whitelisting**: Restrict API access
- **Sandbox Usage**: Use sandbox for untrusted operations
- **Regular Updates**: Keep dependencies updated

---

## FAQ

### General Questions

**Q: Which backend should I use?**  
A: Ollama is recommended for most users—it's faster, more efficient, and easier to set up. HuggingFace is good if you need specific models or want to run models entirely in Python.

**Q: Can I use both Python and TypeScript versions?**  
A: Yes, they can coexist. They share configurations but have separate data directories.

**Q: Is my data private?**  
A: Yes! All data stays local. No telemetry, no cloud services (unless you explicitly use them).

**Q: Can I use this offline?**  
A: Yes, with local models (Ollama or HuggingFace). No internet connection required after setup.

### Technical Questions

**Q: How much RAM do I need?**  
A: Depends on the model. Small models (7B) need 8GB+, larger models (13B+) need 16GB+. Ollama is more efficient than HuggingFace.

**Q: Can I use GPU?**  
A: Yes! Both backends support GPU. Install appropriate PyTorch version with CUDA support.

**Q: How do I add custom models?**  
A: For Ollama: Use `ollama create` with a Modelfile. For HuggingFace: Download models to local cache or specify path.

**Q: Can I export/import sessions?**  
A: Yes! Use `/save` and `/load` commands, or use project save/load functionality.

### Feature Questions

**Q: How does RAG work?**  
A: Documents are processed into vector embeddings, stored in a database. When you ask a question, relevant chunks are retrieved and added to the context.

**Q: What's the difference between sessions and projects?**  
A: Sessions are individual conversations. Projects group multiple sessions and have their own memory/context.

**Q: Can I create my own agents for App Builder?**  
A: Currently, agents are predefined. Custom agents would require code changes (contributions welcome!).

**Q: How secure is the API?**  
A: API supports encryption (Fernet), authentication (API keys), and IP whitelisting. Security depends on your configuration.

---

## Additional Resources

- **GitHub Repository**: [Link to repository]
- **Issues Tracker**: [Link to issues]
- **Contributing Guide**: [CONTRIBUTING.md](CONTRIBUTING.md)
- **License**: [LICENSE](LICENSE)

---

## Changelog & Version History

### Version Information

- **Current Version**: See repository releases
- **Python Version**: Production-ready
- **TypeScript/Bun Version**: In progress (~85% complete)

### Recent Updates

- Enhanced Terminal UI with Textual
- Improved help system
- Better error handling
- Performance optimizations
- Additional tool types
- Enhanced RAG capabilities

---

**Last Updated**: December 2025  
**Maintained By**: AI Terminal Pro Contributors  
**License**: MIT License

---

*For questions, issues, or contributions, please see [CONTRIBUTING.md](CONTRIBUTING.md) or open an issue on GitHub.*
