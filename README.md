# 🛡️ AI Terminal Pro 🛡️

A **privacy-first**, terminal-based AI assistant you actually control. Run LLMs locally via **Ollama**, **llama.cpp servers**, or through self-hosted endpoints like **Hugging Face**, no cloud lock-in, no data collection, no nonsense.

Built for **developers**,
Built for **power users**,
Built for **privacy-first professionals**.

---

## 🔑 Features 🔑


## 🔑 Core Features 🔑

### 🤖 AI & Model Support
- 💻 **Local or Self-Hosted Model Support** (Ollama, llama.cpp, Hugging Face)
- 🔄 **Multiple Backend Support** - Switch between Ollama, llama.cpp, and HuggingFace Transformers
- 🎯 **Configurable Model Parameters** - Temperature, context window, response tokens
- 🚀 **GPU & CPU Compatibility** - Optimized for both hardware configurations

### 🧠 Memory & Context
- 🧠 **RAG-Enabled Context Memory** (local vector database)
- 📁 **Session & Project Management** with persistent memory
- 🔍 **Semantic Search** - Vector embeddings for intelligent context retrieval
- 💾 **Persistent Storage** - SQLite-based memory with automatic backups
- 📚 **Document Ingestion** - Load documents, code, and text files for context

### 🛠️ Custom Tooling & Automation
- 🛠️ **Custom Tooling** (Python, JSON, YAML)
- 🔌 **MCP Server Integration** (file systems, APIs, GitHub, databases)
- ⚡ **Action System** - Execute tools with `ACTION:` syntax
- 🛡️ **Safety Controls** - Dangerous command protection with sandboxing
- 🌐 **Web Browsing** - Playwright-powered browser automation

### 🎨 User Interface
- 🎨 **Modern Terminal UI** built with Textual
- 📱 **Rich Terminal Output** - Color-coded, formatted responses
- ⌨️ **Keyboard Shortcuts** - Efficient navigation and commands
- 🎭 **Enhanced Help System** - Comprehensive command documentation
- 🎬 **Splash Screen** - Animated startup experience

### 🔐 Privacy & Security
- 🔐 **No Telemetry or Tracking** — Your data stays with you
- 🔒 **Encrypted API Access** for remote/mobile use (Fernet encryption)
- 🛡️ **Sandboxed Execution** - Safe tool execution environment
- 🔑 **IP Whitelisting** - Secure API access control
- 🔐 **Authentication Support** - API key protection

### 🌐 API & Remote Access
- 🌐 **RESTful API Server** - Flask-based API endpoints
- 📱 **Mobile/Remote Access** - Encrypted API for remote connections
- 🔒 **CORS Support** - Configurable cross-origin resource sharing
- 🏥 **Health Endpoints** - Monitoring and status checks
- 🔑 **Multiple API Instances** - Run multiple API servers simultaneously

### 🧪 Model Training & Fine-Tuning
- 🧪 **Fine-Tuning Support** - Train models on custom datasets
- 📊 **LoRA (Low-Rank Adaptation)** - Efficient parameter-efficient fine-tuning
- 🎓 **RLHF (Reinforcement Learning from Human Feedback)** - Train with human preferences
- 📁 **Dataset Management** - Organize and manage training data
- 💾 **Model Checkpoints** - Save and load training progress

### 🏗️ Development Tools
- 🏗️ **Multi-Agent App Builder** - Orchestrated development system
- 👥 **Specialized Agents** - SpecificationWriter, Architect, TechLead, Developer, Reviewer, and more
- 📝 **Project Management** - Track and manage development projects
- 🔄 **Auto-Update** - Update from GitHub with automatic backups
- 📦 **Project Export/Import** - Save and share project configurations

### 🎤 Voice & Vision (Optional)
- 🎤 **Text-to-Speech (TTS)** - Voice output with multiple engines (pyttsx3, kokoro)
- 🗣️ **Speech Recognition** - Whisper-based voice input
- 📷 **Camera Integration (Sight)** - Computer vision with OpenCV
- 🎥 **Vision Assistant** - Combined voice and vision capabilities
- 🎙️ **Enhanced TTS Pro Mode** - Professional-grade voice interaction with RAG

---
## 🚀 Quick Start 🚀

### Prerequisites
- **Python 3.8+**
- **Ollama** (recommended), **llama.cpp servers**, or HuggingFace models

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/ai-terminal-pro.git
   cd ai-terminal-pro
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

   # Or for minimal installation (core features only)
   pip install requests beautifulsoup4 transformers torch cryptography flask flask-cors textual rich
   ```

3. **Configure your backend**
   
   Edit `config.json`:
   ```json
   {
     "backend": "ollama",
     "model_name": "qwen2.5-coder:latest",
     "enable_dangerous_commands": true,
     "max_context_window": 2048,
     "temperature": 0.7
   }
   ```

4. **Run the application**
   ```bash
   python ai.py
   ```

### Optional Features Installation

**For Text-to-Speech (Basic):**
```bash
pip install openai-whisper pyttsx3 sounddevice soundfile
```

**For Enhanced TTS Pro:**
```bash
pip install faster-whisper kokoro pyaudio sentence-transformers chromadb tiktoken pyyaml
```

**For Camera/Vision (Sight mode):**
```bash
pip install opencv-python numpy
```

**For Web Browsing:**
```bash
pip install playwright
python -m playwright install chromium
```

**For Model Training:**
```bash
pip install peft datasets accelerate trl
```
---

## ⚙️ Tech Stack ⚙️

`Python` · `PyTorch` · `Transformers` · `SQLite` · `Flask` · `Textual` · `Cryptography`

---

## 🔑 To Do 🔑 (Not in order && mainly just noting idea's, want to add some? Let me know!)

- 💻 **Multi purpose terminal GUI??**  
- 🧠 **Full Automation**  
- 🛠️ **Background Agent** 
- 🔐 ~~**TTS**~~  
- 📁 ~~**Sight** camera enabled functionality~~ 
- 🌐 **Vision** image generation
- 🔌 **API Gateway** 
- 🧪 **GPU + CPU compatability** for user performance  
- 🎨 **You got any idea's??**

---
## 📖 Usage Guide 📖

### Main Menu Options

1. **Start Chat** - Interactive conversation with RAG context
2. **Document Loader (RAG)** - Add documents for context retrieval
3. **Tool Management** - Create and manage custom tools
4. **MCP Server Management** - Configure Model Context Protocol servers
5. **Model Training** - Fine-tune, LoRA, and RLHF training
6. **API Management** - Create and manage encrypted API endpoints
7. **App Builder** - Multi-agent development system
8. **Update from GitHub** - Auto-update with backups
9. **Settings** - Configure application settings
10. **Exit** - Exit the application

### Chat Commands

- `/help` or `/?` - Show available commands
- `/new [name]` - Create new chat session
- `/save [filename]` - Save current chat to file
- `/load [id|filename]` - Load chat session
- `/project [name|desc]` - Create/switch project
- `/back` - Return to main menu
- `/camera` - Launch camera-only assistant
- `/voice` or `/tts` - Launch voice/TTS assistant
- `/vision` - Launch unified Vision + Voice assistant

### Tool Execution

Use the `ACTION:` syntax to execute tools:
```
ACTION: TOOL_NAME arg1 arg2 ...
```

---

## 🎯 Key Capabilities 🎯

### RAG (Retrieval-Augmented Generation)
- **Document Ingestion**: Load PDFs, text files, code, markdown
- **Semantic Search**: Vector embeddings for intelligent context retrieval
- **Project Memory**: Project-specific context storage
- **Session History**: Persistent conversation memory

### Custom Tools
- **Python Tools**: Write custom Python functions
- **JSON/YAML Tools**: Define tools declaratively
- **MCP Tools**: Integrate Model Context Protocol servers
- **Web Tools**: Browser automation and web scraping
- **File Tools**: File system operations (with safety checks)

### Multi-Agent Development
- **App Builder**: Orchestrate multiple specialized agents
- **Agents**: SpecificationWriter, Architect, TechLead, Developer, CodeMonkey, Reviewer, Troubleshooter, Debugger, TechnicalWriter
- **Project Tracking**: Manage development projects with persistent state
- **Code Generation**: Automated code generation and review

### API Server
- **RESTful Endpoints**: `/chat`, `/health`, `/info`
- **Encryption**: Fernet-based encryption for secure communication
- **Authentication**: API key support
- **CORS**: Configurable cross-origin resource sharing
- **Multiple Instances**: Run multiple API servers on different ports
  
---
## 🔑 Roadmap 🔑

### Completed ✅
- ✅ Local model support (Ollama, llama.cpp, HuggingFace)
- ✅ RAG system with vector database
- ✅ Custom tooling (Python, JSON, YAML)
- ✅ MCP server integration
- ✅ Text-to-Speech (TTS)
- ✅ Camera/Vision (Sight) functionality
- ✅ Web browsing with Playwright
- ✅ Model training (Fine-tuning, LoRA, RLHF)
- ✅ API server with encryption
- ✅ Multi-agent App Builder
- ✅ Modern Terminal UI (Textual)
- ✅ Enhanced help system
- ✅ Session and project management

### In Progress 🔄
- 🔄 Full Textual menu system
- 🔄 Enhanced settings UI

### Planned 📋
- 📋 Multi-purpose terminal GUI
- 📋 Full automation capabilities
- 📋 Background agent system
- 📋 Vision (image generation)
- 📋 API Gateway
- 📋 GPU + CPU compatibility optimizations
---

<img width="655" height="969" alt="Ai Terminal Pro - Code reviewer concerns" src="https://github.com/user-attachments/assets/d909b4bb-f6b2-461e-861a-24858dad0799" />
<img width="662" height="389" alt="AI Terminal Pro - menu " src="https://github.com/user-attachments/assets/29a7d44a-5c89-4129-8557-a03b73c219b2" />
<img width="971" height="949" alt="AI Terminal Pro - menu in chat " src="https://github.com/user-attachments/assets/45f0ed87-332e-4524-884b-6291a4881333" />
<img width="1281" height="947" alt="AI Terminal Pro - menu in chat1 " src="https://github.com/user-attachments/assets/4189b475-e392-4503-bc49-1347951c45e9" />
<img width="1296" height="576" alt="FMJ9J61" src="https://github.com/user-attachments/assets/fbbc5ea5-451d-40b9-942a-8d0af8986478" />
<img width="1296" height="576" alt="xjGMhsH" src="https://github.com/user-attachments/assets/8f42fc81-b09b-4ed4-b081-1e5881a09d0b" />
<img width="1296" height="576" alt="xN0yKbW" src="https://github.com/user-attachments/assets/d3db41d6-d751-4f1f-a072-2aaab3e5af40" />

---
---

## 👥 Contributing 👥

We welcome contributions! Whether it's fixing bugs, improving documentation, or adding new features check out our [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

---

## 📜 License 📜

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.  

Use it. Modify it. Make it yours.

---

**Your data. Your models. Your rules.**
