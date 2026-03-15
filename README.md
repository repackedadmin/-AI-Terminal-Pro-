# 🛡️ AI Terminal Pro 🛡️

A **local-first**, terminal-based AI assistant with full **cloud flexibility**. Run LLMs privately via **Ollama**, **llama.cpp**, or **HuggingFace** — or connect to **OpenAI, Anthropic, Gemini, Groq, DeepSeek, and more**. Your choice. Your control.

Built for **developers**, **power users**, and **professionals who want options**.

---

## 📋 Table of Contents

- [Core Features](#-core-features-)
- [Tech Stack](#-tech-stack-)
- [Quick Start](#-quick-start-)
- [Backend Configuration](#-backend-configuration-)
- [Usage Guide](#-usage-guide-)
- [ACTION Reference](#-action-reference-)
- [Project Structure](#-project-structure-)
- [Optional Features](#optional-features-installation)
- [Troubleshooting](#troubleshooting)
- [Documentation](#-documentation-)
- [Contributing](#-contributing-)
- [License](#-license-)

---

## 🔑 Core Features 🔑

### 🤖 AI & Model Support
- **10 Backends** — Local and cloud from one interface
- **Local**: Ollama, llama.cpp (OpenAI-compatible), HuggingFace Transformers
- **Cloud**: OpenAI, Anthropic, Google Gemini, OpenRouter, Groq, DeepSeek, HuggingFace Cloud
- **Live Switching** — Change backend and model from Settings (menu 9 → option 4)
- **Configurable** — Temperature, context window, response tokens per backend
- **GPU & CPU** — Optimized for both

### 🔐 Privacy & Security
- **Local Mode** — Zero telemetry when using local backends
- **Encrypted Keys** — Cloud API keys encrypted at rest (Fernet, PBKDF2-derived)
- **Sandboxed Execution** — Tool execution in restricted environment
- **IP Whitelisting** — Optional API access control
- **API Authentication** — Key protection for local API server

### 🧠 Memory & Context
- **RAG** — Local vector database with semantic search
- **Session & Project Management** — Persistent memory
- **Document Ingestion** — PDFs, text, code, markdown
- **Self-Healing SQLite** — Automatic database recovery

### 🛠️ Custom Tooling & Automation
- **Custom Tools** — Python, JSON, YAML definitions
- **MCP Integration** — Model Context Protocol servers (file systems, APIs, GitHub)
- **ACTION Syntax** — `ACTION: TOOL_NAME args` for tool execution
- **Safety Controls** — Dangerous command protection (enable in Settings)
- **Web Browsing** — Playwright visible browser (auto-installs Chromium)
- **Desktop Automation** — OS-aware CMD, file ops (Windows, Linux, macOS)
- **Software Registry** — Auto-detect installed apps, launch with I-Frame output

### 🧩 Extension System
- **Extensions Directory** — `extensions/` for plugin auto-loading
- **Extension Manifests** — JSON config (`gemini-extension.json` format)
- **MCP Servers** — Extensions can expose their own MCP
- **Lifecycle** — Started and managed by `ExtensionManager`

### 🎨 User Interface
- **Rich Terminal** — Color-coded, formatted output
- **Keyboard-Driven** — Efficient text navigation
- **Help System** — `/help` for in-chat command docs
- **Splash Screen** — Animated startup
- **Configurable Editor** — VS Code, Cursor, Sublime, etc. for tools/MCP

### 🌐 API & Remote Access
- **REST Endpoints** — `/chat`, `/health`, `/info`
- **Fernet Encryption** — Secure API communication
- **Multiple Instances** — Run on different ports
- **CORS** — Configurable cross-origin

### 🧪 Model Training
- **Fine-Tuning** — Custom datasets
- **LoRA** — Parameter-efficient fine-tuning
- **RLHF** — PPO-based reinforcement learning
- **Checkpoints** — Save/load in `.claude/models/`

### 🏗️ Multi-Agent App Builder
- **Agents** — SpecificationWriter, Architect, TechLead, Developer, CodeMonkey, Reviewer, Troubleshooter, Debugger, TechnicalWriter
- **Project Tracking** — Persistent state
- **Code Generation** — Automated pipeline

### 🎤 Voice & Vision (Optional)
- **Basic TTS** — `pyttsx3` + Whisper (`/voice`, `/tts`)
- **TTS Pro** — `faster-whisper` + `kokoro` + RAG (`/tts_pro`)
- **Camera** — OpenCV (`/sight`)
- **Vision Assistant** — Combined voice + vision (`/vision`)

---

## ⚙️ Tech Stack ⚙️

`Python 3.8+` · `PyTorch` · `Transformers` · `SQLite` · `Flask` · `Rich` · `Cryptography` · `Playwright` · `OpenCV` · `Whisper` · `faster-whisper` · `kokoro` · `sentence-transformers` · `chromadb`

---

## 🚀 Quick Start 🚀

### Prerequisites
- **Python 3.8+**
- **Ollama** (recommended for local) — or a cloud API key

### Installation

```bash
# Clone
git clone https://github.com/repackedadmin/-AI-Terminal-Pro-.git
cd -AI-Terminal-Pro-

# Install dependencies
pip install -r requirements.txt

# Run
python ai.py
```

**Minimal install** (core only):
```bash
pip install requests beautifulsoup4 transformers torch cryptography flask flask-cors rich
```

**First run**: Select backend from Settings (menu 9 → option 4).

---

## ⚙️ Backend Configuration ⚙️

### Local Backends (No API Key)

| Backend | Setup |
|---------|-------|
| **Ollama** | `ollama pull qwen2.5-coder:latest` → Settings → Backend |
| **llama.cpp** | Start server (default `http://127.0.0.1:8080`) → Settings → Backend |
| **HuggingFace** | Settings → Backend → enter model ID (e.g. `mistralai/Mistral-7B-Instruct-v0.3`) |

### Cloud Backends (API Key Required)

1. **Settings** (menu 9) → **Manage Cloud API Keys** (option 5)
2. Add/Update API key for your provider
3. **Change AI Backend** (option 4) → select cloud provider
4. Choose model from list (or press Enter for default)

| Provider | Backend Option | API Key URL |
|----------|----------------|-------------|
| OpenAI | 4 | https://platform.openai.com/api-keys |
| Anthropic | 5 | https://console.anthropic.com/ |
| Google Gemini | 6 | https://ai.google.dev/ |
| OpenRouter | 7 | https://openrouter.ai/keys |
| Groq | 8 | https://console.groq.com/ |
| DeepSeek | 9 | https://platform.deepseek.com/api_keys |
| HuggingFace Cloud | 10 | https://huggingface.co/settings/tokens |

Keys are **encrypted at rest** and never logged.

### Settings Submenu (Main Menu → 9)

| # | Option | Description |
|---|--------|--------------|
| 1 | Toggle Dangerous Commands | Allow CMD/file ops outside sandbox |
| 2 | Edit System Prompt | Customize AI behavior |
| 3 | Set Default Editor | VS Code, Cursor, Sublime for tools/MCP |
| 4 | Change AI Backend | Switch between 10 backends |
| 5 | Manage Cloud API Keys | Add/remove encrypted keys |
| 6 | Back | Return to main menu |

---

## 📖 Usage Guide 📖

### Main Menu

| # | Option | Description |
|---|--------|--------------|
| 1 | Start Chat | Interactive chat with RAG context |
| 2 | Document Loader (RAG) | Add documents for context |
| 3 | Tool Management | Create/manage custom tools |
| 4 | MCP Server Management | Configure MCP servers |
| 5 | Model Training | Fine-tune, LoRA, RLHF |
| 6 | API Management | Encrypted API endpoints |
| 7 | App Builder | Multi-agent development |
| 8 | Update from GitHub | Auto-update with backups |
| 9 | Settings | Backend, keys, editor |
| 10 | Exit | Quit |

### Chat Commands

| Command | Description |
|---------|-------------|
| `/help` or `/?` | Show all commands |
| `/new [name]` | New chat session |
| `/save [filename]` | Save chat to file |
| `/load [id\|filename]` | Load chat session |
| `/project [name\|desc]` | Create/switch project |
| `/back` | Return to main menu |
| `/traverse` | Interactive directory browser |
| `/browser [subcmd]` | Playwright browser automation |
| `/camera` | Camera-only assistant |
| `/voice` or `/tts` | Basic voice/TTS |
| `/tts_pro` | Enhanced TTS Pro |
| `/vision` | Vision + Voice assistant |

### `/browser` Subcommands

| Subcommand | Example | Description |
|------------|---------|-------------|
| `open <url>` | `/browser open google.com` | Navigate to URL |
| `back` | `/browser back` | Go back |
| `forward` | `/browser forward` | Go forward |
| `refresh` | `/browser refresh` | Refresh page |
| `url` | `/browser url` | Show current URL |
| `screenshot [path]` | `/browser screenshot` | Take screenshot |
| `html` | `/browser html` | Get page HTML |
| `text` | `/browser text` | Get page text |
| `click <selector>` | `/browser click button#submit` | Click element |
| `fill <sel> <text>` | `/browser fill input#email x@y.com` | Fill input |
| `scroll <dir>` | `/browser scroll down` | Scroll up/down/top/bottom |
| `js <code>` | `/browser js document.title` | Execute JavaScript |
| `wait <ms>` | `/browser wait 2000` | Wait milliseconds |
| `pdf [path]` | `/browser pdf` | Export to PDF |
| `status` | `/browser status` | Show browser status |
| `close` | `/browser close` | Close browser |

---

## 🎯 ACTION Reference 🎯

All tools use: `ACTION: TOOL_NAME [arguments]`

### File Operations
| Action | Format | Example |
|--------|--------|---------|
| `FILE_READ` | `path` | `ACTION: FILE_READ ~/Desktop/file.txt` |
| `FILE_WRITE` | `path \| content` | `ACTION: FILE_WRITE out.txt \| Hello` |
| `FILE_EDIT` | `path \|\|\| old \|\|\| new` | `ACTION: FILE_EDIT f.py \|\|\| bug \|\|\| fix` |
| `FILE_APPEND` | `path \| content` | `ACTION: FILE_APPEND log.txt \| line` |
| `FILE_LIST` | `[path]` | `ACTION: FILE_LIST ~/Desktop` |

### Shell Commands (requires Dangerous Commands enabled)
| Action | Format | Example |
|--------|--------|---------|
| `CMD` | `command` | `ACTION: CMD ls -la` (Linux) |
| `CMD` | `command` | `ACTION: CMD dir` (Windows) |

### Web Browsing (Playwright)
| Action | Format | Example |
|--------|--------|---------|
| `BROWSE` | `url(s)` | `ACTION: BROWSE https://google.com` |
| `BROWSE_SEARCH` | `query` | `ACTION: BROWSE_SEARCH python tutorial` |
| `BROWSE_CLICK_FIRST` | — | `ACTION: BROWSE_CLICK_FIRST` |
| `BROWSE_CLICK` | `selector` | `ACTION: BROWSE_CLICK button#submit` |
| `BROWSE_TYPE` | `selector \| text` | `ACTION: BROWSE_TYPE input#q \| hello` |
| `BROWSE_PRESS` | `selector \| key` | `ACTION: BROWSE_PRESS input#q \| Enter` |
| `BROWSE_WAIT` | `milliseconds` | `ACTION: BROWSE_WAIT 2000` |

*Multiple URLs: `ACTION: BROWSE url1, url2, url3`*

### Installed Software Registry
| Action | Format | Example |
|--------|--------|---------|
| `APPS_LIST` | `[filter]` | `ACTION: APPS_LIST code` |
| `APPS_DETECT` | — | `ACTION: APPS_DETECT` |
| `LAUNCH_APP` | `name` | `ACTION: LAUNCH_APP code` |
| `LAUNCH_APP` | `name \| args` | `ACTION: LAUNCH_APP python3 \| --version` |
| `LAUNCH_APP` | `name \| --capture` | `ACTION: LAUNCH_APP python3 \| --capture` |

*`--capture` captures CLI output in I-Frame for terminal apps.*

### Desktop Automation
- **Platform Detection**: Auto-detects Windows, Linux, macOS
- **Desktop Path**: `~/Desktop/` works on all platforms
- **CMD**: Windows uses `dir`, `mkdir`, `type`, `del`; Linux uses `ls`, `mkdir`, `cat`, `rm`

---

## 📁 Project Structure 📁

```
ai_latest/
├── ai.py                 # Main application (~14k lines)
├── config.json           # User config (backend, keys, etc.)
├── requirements.txt      # Dependencies
├── README.md             # This file
├── CONTRIBUTING.md       # Contribution guidelines
├── ai_memory.sqlite      # Chat history, RAG documents
├── software_registry.sqlite  # Installed apps cache
├── app_projects.sqlite   # App Builder projects
├── api/                  # API keys, encryption
├── apps/                 # App Builder outputs
├── custom_tools/         # Custom tool definitions
├── documents/            # RAG documents
├── extensions/           # Plugin extensions
├── training/             # Fine-tuning data, models
│   ├── data/
│   ├── models/
│   ├── lora/
│   └── reinforcement/
└── tui/                  # TUI components
    ├── app_integration.py
    └── components/
```

---

## Optional Features Installation

| Feature | Command |
|---------|---------|
| **TTS Basic** | `pip install openai-whisper pyttsx3 sounddevice soundfile` |
| **TTS Pro** | `pip install faster-whisper kokoro pyaudio sentence-transformers chromadb tiktoken pyyaml` |
| **Camera/Vision** | `pip install opencv-python numpy` |
| **Web Browsing** | `pip install playwright` then `python -m playwright install chromium` |
| **Model Training** | `pip install peft datasets accelerate trl` |

*Chromium auto-installs on first BROWSE action if missing.*

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Ollama not found** | Install: https://ollama.ai — run `ollama serve` |
| **llama.cpp not responding** | Start server on port 8080; check Settings for base URL |
| **Cloud API errors** | Verify key in Settings → Manage Cloud API Keys |
| **Playwright fails** | Run `python -m playwright install chromium` |
| **Database errors** | Self-healing runs on startup; delete `ai_memory.sqlite` to reset |
| **No apps in APPS_LIST** | Run `ACTION: APPS_DETECT` to scan |
| **Dangerous Commands denied** | Settings → Toggle Dangerous Commands |

---

## 🔑 Roadmap 🔑

### Completed ✅
- Desktop automation (Windows, Linux, macOS)
- Installed software registry (auto-detect, LAUNCH_APP, I-Frame)
- Playwright web browsing (auto-install Chromium)
- 10 backends (local + cloud)
- Encrypted API keys, RAG, MCP, extensions
- TTS (basic + Pro), Camera/Vision
- Model training, App Builder, API server

### In Progress 🔄
- Full Textual TUI menu
- Enhanced settings UI
- Extension marketplace

### Planned 📋
- Background agent system
- API Gateway for multi-user
- Image generation integration
- Full automation pipeline

---

## 📸 Screenshots 📸

<img width="1296" height="576" alt="Screenshot 1" src="https://github.com/user-attachments/assets/fbbc5ea5-451d-40b9-942a-8d0af8986478" />
<img width="1296" height="576" alt="Screenshot 2" src="https://github.com/user-attachments/assets/8f42fc81-b09b-4ed4-b081-1e5881a09d0b" />
<img width="1296" height="576" alt="Screenshot 3" src="https://github.com/user-attachments/assets/d3db41d6-d751-4f1f-a072-2aaab3e5af40" />

---

## 📚 Documentation 📚

- **[CONTRIBUTING.md](CONTRIBUTING.md)** — Contribution guidelines
- **[CLAUDE.md](CLAUDE.md)** — Architecture reference for developers/AI assistants

---

## 👥 Contributing 👥

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📜 License 📜

**MIT License** — see [LICENSE](LICENSE).

Use it. Modify it. Make it yours.

---

**Local when you want privacy. Cloud when you need power. Always your choice.**
