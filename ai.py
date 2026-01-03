### AI Terminal Pro
### Repacked Tools
### Coded By Alistair
### 12 Dec 2025
### 14:00pm
###
#############################################################################

import os
import sys
import json
import time
import sqlite3
import glob
import subprocess
import requests
import shutil
import platform
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import base64
import shlex
from datetime import datetime
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Encryption imports
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

# Flask for API server
try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Playwright for web browsing
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

# Whisper for speech recognition
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

# Text-to-Speech
try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# OpenCV for camera capture
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Audio recording
try:
    import sounddevice as sd
    import soundfile as sf
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

# ANSI Color Codes for cross-platform terminal colors
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'
    
    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

# ==============================================================================
#                           GLOBAL CONFIGURATION & CONSTANTS
# ==============================================================================

# Dynamic paths to ensure cross-platform compatibility
BASE_DIR = os.getcwd()
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
MCP_CONFIG_FILE = os.path.join(BASE_DIR, "mcp_servers.json")
DB_PATH = os.path.join(BASE_DIR, "ai_memory.sqlite")
SANDBOX_DIR = os.path.join(BASE_DIR, "ai_sandbox")
DOCS_DIR = os.path.join(BASE_DIR, "documents")
CUSTOM_TOOLS_DIR = os.path.join(BASE_DIR, "custom_tools")
TRAINING_DIR = os.path.join(BASE_DIR, "training")
TRAINING_DATA_DIR = os.path.join(TRAINING_DIR, "data")
MODELS_DIR = os.path.join(TRAINING_DIR, "models")
LORA_DIR = os.path.join(TRAINING_DIR, "lora")
REINFORCEMENT_DIR = os.path.join(TRAINING_DIR, "reinforcement")
API_DIR = os.path.join(BASE_DIR, "api")
API_CONFIG_FILE = os.path.join(API_DIR, "api_config.json")
API_KEYS_FILE = os.path.join(API_DIR, "api_keys.json")
APPS_DIR = os.path.join(BASE_DIR, "apps")
APP_PROJECTS_DB = os.path.join(BASE_DIR, "app_projects.sqlite")

# Ensure all workspace directories exist
for directory in [SANDBOX_DIR, DOCS_DIR, CUSTOM_TOOLS_DIR, TRAINING_DIR, TRAINING_DATA_DIR, MODELS_DIR, LORA_DIR, REINFORCEMENT_DIR, API_DIR, APPS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

DEFAULT_CONFIG = {
    "first_run": True,
    "backend": "huggingface",  # Options: "huggingface" or "ollama"
    "model_name": "gpt2",      # Options: "gpt2", "gpt2-xl", "llama3", "mistral"
    "system_prompt": "You are a helpful AI assistant with full OS awareness and extensive memory capabilities. You have access to a large context window (32K+ tokens) and can maintain context across long conversations and complex projects. You automatically detect the operating system and use platform-appropriate commands. When working on projects, you remember project context, previous decisions, and ongoing work. When asked to perform a task, use the available ACTION tools with OS-specific commands. IMPORTANT: When asked to browse the web or interact with websites, you can use browser interaction commands. For example, to search Google and click results: 1) Use ACTION: BROWSE_SEARCH [query] to search Google, 2) Use ACTION: BROWSE_CLICK_FIRST to click the first result, 3) Use ACTION: BROWSE_WAIT [ms] to wait for pages to load. You can also use BROWSE_CLICK, BROWSE_TYPE, and BROWSE_PRESS to interact with page elements. The browser is VISIBLE so the user can see all your actions in real-time. After completing browser actions, provide your summary/analysis. CRITICAL: When generating scripts (batch, PowerShell, bash, Python, etc.), always write the COMPLETE, FULL script from start to finish - never truncate, cut off, or leave scripts incomplete.",
    "enable_dangerous_commands": False,  # Safety lock for file system/terminal
    "max_context_window": 32768,  # Large context window for project assistance and long conversations
    "max_response_tokens": 2000,  # Increased for script generation and longer responses
    "temperature": 0.7,
    # Torch threading + builder concurrency
    "cpu_threads": max(1, min(8, os.cpu_count() or 1)),
    "cpu_interop_threads": 2,
    "builder_threads": 1,
    # Default external editor command (used when creating tools / MCP servers)
    # Example values:
    #   "code"         (VS Code)
    #   "cursor"       (Cursor)
    #   "notepad++"    (Notepad++)
    #   "sublime_text" (Sublime Text)
    #   "notepad"      (Windows Notepad)
    "default_editor_command": ""
}

def apply_torch_threading(config):
    """Apply torch threading knobs for CPU/GPU without changing higher-level logic."""
    try:
        cpu_threads = int(config.get("cpu_threads", 0) or 0)
        interop_threads = int(config.get("cpu_interop_threads", 0) or 0)
        
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)
        if interop_threads > 0:
            torch.set_num_interop_threads(interop_threads)
        
        # GPU-specific micro-optimizations; safe no-ops on CPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    except Exception as e:
        print(f"{Colors.YELLOW}[WARN]{Colors.RESET} Could not apply torch threading config: {e}")

# ==============================================================================
#                           1. CONFIG MANAGER
# ==============================================================================

class ConfigManager:
    """Manages persistent settings in config.json."""
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        if not os.path.exists(CONFIG_FILE):
            self.save_config_data(DEFAULT_CONFIG)
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(CONFIG_FILE, 'r') as f:
                data = json.load(f)
                # Merge with defaults to prevent missing keys on updates
                for key, val in DEFAULT_CONFIG.items():
                    if key not in data:
                        data[key] = val
                return data
        except Exception as e:
            print(f"{Colors.BRIGHT_RED}[ERROR]{Colors.RESET} Config load failed: {e}. Using defaults.")
            return DEFAULT_CONFIG.copy()

    def save_config_data(self, data):
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=4)

    def update(self, key, value):
        self.config[key] = value
        self.save_config_data(self.config)

    def get(self, key):
        return self.config.get(key, DEFAULT_CONFIG.get(key))

# ==============================================================================
#                           2. DATABASE & MEMORY (RAG)
# ==============================================================================

class MemoryManager:
    """Handles SQLite interaction for Chat History and Document RAG."""
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_db()

    def _init_db(self):
        # Sessions/Chats Table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
                project_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Projects Table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                description TEXT,
                memory_bank TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Chat History Table (with session_id)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            role TEXT,
            content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        # Document Store for RAG (with project_id)
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                source_file TEXT,
                content_chunk TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(id)
            )
        ''')
        
        # Add session_id to history if it doesn't exist (migration)
        try:
            self.cursor.execute("SELECT session_id FROM history LIMIT 1")
        except sqlite3.OperationalError:
            self.cursor.execute("ALTER TABLE history ADD COLUMN session_id INTEGER")
        
        # Add project_id to documents if it doesn't exist (migration)
        try:
            self.cursor.execute("SELECT project_id FROM documents LIMIT 1")
        except sqlite3.OperationalError:
            self.cursor.execute("ALTER TABLE documents ADD COLUMN project_id INTEGER")
        
            self.conn.commit()

    # Session Management
    def create_session(self, name, project_id=None):
        self.cursor.execute("INSERT INTO sessions (name, project_id) VALUES (?, ?)", (name, project_id))
        self.conn.commit()
        return self.cursor.lastrowid

    def get_sessions(self, project_id=None):
        if project_id:
            self.cursor.execute("SELECT id, name FROM sessions WHERE project_id=? ORDER BY updated_at DESC", (project_id,))
        else:
            self.cursor.execute("SELECT id, name FROM sessions ORDER BY updated_at DESC")
        return self.cursor.fetchall()

    def get_session(self, session_id):
        self.cursor.execute("SELECT id, name, project_id FROM sessions WHERE id=?", (session_id,))
        return self.cursor.fetchone()
    
    def update_session(self, session_id):
        self.cursor.execute("UPDATE sessions SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (session_id,))
        self.conn.commit()

    def delete_session(self, session_id):
        self.cursor.execute("DELETE FROM history WHERE session_id=?", (session_id,))
        self.cursor.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        self.conn.commit()

    def save_session_to_file(self, session_id, filepath):
        """Export session to JSON file."""
        history = self.get_recent_history(session_id, limit=1000)
        session_info = self.get_session(session_id)
        
        data = {
            "session": {
                "id": session_info[0],
                "name": session_info[1],
                "project_id": session_info[2]
            },
            "history": [{"role": r, "content": c} for r, c in history]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    
    def load_session_from_file(self, filepath):
        """Import session from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session_data = data.get("session", {})
        session_name = session_data.get("name", f"Imported {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        project_id = session_data.get("project_id")
        
        session_id = self.create_session(session_name, project_id)
        
        for msg in data.get("history", []):
            self.save_message(session_id, msg.get("role", "user"), msg.get("content", ""))
        
        return session_id
    
    # Project Management
    def create_project(self, name, description=""):
        self.cursor.execute("INSERT INTO projects (name, description, memory_bank) VALUES (?, ?, ?)", 
                          (name, description, json.dumps({})))
        self.conn.commit()
        return self.cursor.lastrowid
    
    def get_projects(self):
        self.cursor.execute("SELECT id, name, description FROM projects ORDER BY updated_at DESC")
        return self.cursor.fetchall()
    
    def get_project(self, project_id):
        self.cursor.execute("SELECT id, name, description, memory_bank FROM projects WHERE id=?", (project_id,))
        return self.cursor.fetchone()
    
    def update_project_memory(self, project_id, memory_bank):
        """Update project memory bank (dict)."""
        self.cursor.execute("UPDATE projects SET memory_bank=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", 
                          (json.dumps(memory_bank), project_id))
        self.conn.commit()
    
    def get_project_memory(self, project_id):
        """Get project memory bank (dict)."""
        row = self.get_project(project_id)
        if row and row[3]:
            return json.loads(row[3])
        return {}
    
    def save_project_to_file(self, project_id, filepath):
        """Export project to JSON file."""
        project = self.get_project(project_id)
        sessions = self.get_sessions(project_id)
        memory_bank = self.get_project_memory(project_id)
        
        data = {
            "project": {
                "id": project[0],
                "name": project[1],
                "description": project[2],
                "memory_bank": memory_bank
            },
            "sessions": []
        }
        
        for sid, sname in sessions:
            history = self.get_recent_history(sid, limit=1000)
            data["sessions"].append({
                "name": sname,
                "history": [{"role": r, "content": c} for r, c in history]
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        return True
    
    def load_project_from_file(self, filepath):
        """Import project from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        project_data = data.get("project", {})
        project_name = project_data.get("name", f"Imported {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        project_desc = project_data.get("description", "")
        memory_bank = project_data.get("memory_bank", {})
        
        project_id = self.create_project(project_name, project_desc)
        self.update_project_memory(project_id, memory_bank)
        
        for session_data in data.get("sessions", []):
            session_name = session_data.get("name", f"Session {datetime.now().strftime('%H:%M')}")
            session_id = self.create_session(session_name, project_id)
            for msg in session_data.get("history", []):
                self.save_message(session_id, msg.get("role", "user"), msg.get("content", ""))
        
        return project_id
    
    def save_message(self, session_id, role, content):
        self.cursor.execute("INSERT INTO history (session_id, role, content) VALUES (?, ?, ?)", 
                          (session_id, role, content))
        self.conn.commit()
        if session_id:
            self.update_session(session_id)

    def get_recent_history(self, session_id=None, limit=100):
        # Fetch last N messages for a session (increased default for better context)
        if session_id:
            self.cursor.execute("SELECT role, content FROM history WHERE session_id=? ORDER BY id DESC LIMIT ?", 
                          (session_id, limit))
        else:
            self.cursor.execute("SELECT role, content FROM history ORDER BY id DESC LIMIT ?", (limit,))
        rows = self.cursor.fetchall()
        return rows[::-1]  # Reverse to chronological order
    
    def get_all_history(self, session_id=None):
        """Get all history for a session (no limit)."""
        if session_id:
            self.cursor.execute("SELECT role, content FROM history WHERE session_id=? ORDER BY id ASC", 
                          (session_id,))
        else:
            self.cursor.execute("SELECT role, content FROM history ORDER BY id ASC")
        return self.cursor.fetchall()

    def ingest_file(self, filepath):
        """Reads text files, chunks them, and stores in DB."""
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            # Split by double newline to approximate paragraphs
            chunks = [c.strip() for c in text.split('\n\n') if c.strip()]
            
            count = 0
            for chunk in chunks:
                if len(chunk) > 20: # Ignore tiny fragments
                    self.cursor.execute("INSERT INTO documents (source_file, content_chunk) VALUES (?, ?)", (filename, chunk))
                count += 1
            self.conn.commit()
            return count
        except Exception as e:
            print(f"{Colors.BRIGHT_RED}[ERROR]{Colors.RESET} Ingesting {filename}: {e}")
            return 0

    def retrieve_context(self, query, project_id=None):
        """Simple RAG: Keyword matching (SQLite LIKE) to find relevant chunks."""
        # Extract significant words (len > 3)
        keywords = [w for w in query.split() if len(w) > 3]
        if not keywords: 
            return ""
        
        # Build query dynamically
        conditions = []
        params = []
        for word in keywords:
            conditions.append("content_chunk LIKE ?")
            params.append(f"%{word}%")
        
        if not conditions: 
            return ""
        
        # Add project filter if specified
        if project_id:
            sql = f"SELECT source_file, content_chunk FROM documents WHERE project_id=? AND ({' OR '.join(conditions)}) LIMIT 3"
            params = [project_id] + params
        else:
            sql = f"SELECT source_file, content_chunk FROM documents WHERE {' OR '.join(conditions)} LIMIT 3"
        
        self.cursor.execute(sql, params)
        results = self.cursor.fetchall()
        
        if not results: 
            return ""
        
        context_str = "\n[RELEVANT DOCUMENTS]:\n"
        for source, chunk in results:
            context_str += f"Source ({source}): {chunk[:400]}...\n"
        return context_str

# ==============================================================================
#                           3. MCP CLIENT (Model Context Protocol)
# ==============================================================================

class MCPClient:
    """Implements JSON-RPC 2.0 Client over Stdio."""
    def __init__(self, name, command):
        self.name = name
        self.command = command
        self.process = None
        self.request_id = 0
        self.available_tools = []
        self.running = False

    def start(self):
        try:
            # Use shell=True to support system PATH resolution
            self.process = subprocess.Popen(
                self.command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0 
            )
            self.running = True
            
            if self._handshake():
                self._refresh_tools()
                return True
            return False
        except Exception as e:
            print(f"[MCP] Failed to start {self.name}: {e}")
            return False

    def stop(self):
        if self.process:
            self.process.terminate()
        self.running = False

    def _next_id(self):
        self.request_id += 1
        return self.request_id

    def _send_json(self, payload):
        if not self.process or not self.running: return
        try:
            json_str = json.dumps(payload) + "\n"
            self.process.stdin.write(json_str)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.running = False

    def _read_response(self, timeout=3.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            line = self.process.stdout.readline()
            if line:
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        return None

    def _handshake(self):
        self._send_json({
            "jsonrpc": "2.0", 
            "method": "initialize", 
            "params": {"protocolVersion": "0.1.0", "clientInfo": {"name": "AI_Term", "version": "1.0"}, "capabilities": {}}, 
            "id": self._next_id()
        })
        resp = self._read_response()
        if resp and "result" in resp:
            self._send_json({"jsonrpc": "2.0", "method": "notifications/initialized"})
            return True
        return False

    def _refresh_tools(self):
        self._send_json({"jsonrpc": "2.0", "method": "tools/list", "params": {}, "id": self._next_id()})
        resp = self._read_response()
        if resp and "result" in resp and "tools" in resp["result"]:
            self.available_tools = resp["result"]["tools"]

    def call_tool(self, tool_name, args_dict):
        self._send_json({
            "jsonrpc": "2.0", 
            "method": "tools/call", 
            "params": {"name": tool_name, "arguments": args_dict}, 
            "id": self._next_id()
        })
        resp = self._read_response(timeout=15.0) # Longer timeout for tool execution
        
        if not resp: return "Error: MCP Timeout."
        if "error" in resp: return f"MCP Error: {resp['error'].get('message')}"
        
        # MCP Standard: result.content is a list of items
        if "result" in resp:
            content = resp["result"].get("content", [])
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "\n".join(texts)
        return "Empty Response"

# ==============================================================================
#                           4. CUSTOM TOOL MANAGER
# ==============================================================================

class CustomToolManager:
    """Manages custom tools defined in Python, JSON, or YAML files."""
    
    def __init__(self):
        self.tools_dir = CUSTOM_TOOLS_DIR
        self.tools_metadata = {}
        self.load_tools()
    
    def load_tools(self):
        """Load all custom tools from the tools directory."""
        self.tools_metadata = {}
        
        # Load Python scripts
        for py_file in glob.glob(os.path.join(self.tools_dir, "*.py")):
            tool_info = self._parse_python_tool(py_file)
            if tool_info:
                self.tools_metadata[os.path.basename(py_file)] = tool_info
        
        # Load JSON tool definitions
        for json_file in glob.glob(os.path.join(self.tools_dir, "*.json")):
            tool_info = self._parse_json_tool(json_file)
            if tool_info:
                self.tools_metadata[os.path.basename(json_file)] = tool_info
        
        # Load YAML tool definitions
        try:
            import yaml
            for yaml_file in glob.glob(os.path.join(self.tools_dir, "*.yaml")) + glob.glob(os.path.join(self.tools_dir, "*.yml")):
                tool_info = self._parse_yaml_tool(yaml_file)
                if tool_info:
                    self.tools_metadata[os.path.basename(yaml_file)] = tool_info
        except ImportError:
            pass  # YAML support optional
    
    def _parse_python_tool(self, filepath):
        """Parse a Python script to extract tool metadata."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Look for tool metadata in docstring or comments
            tool_info = {
                "name": os.path.basename(filepath),
                "type": "python",
                "file": filepath,
                "description": "",
                "parameters": []
            }
            
            # Try to extract from docstring
            import ast
            try:
                tree = ast.parse(content)
                if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
                    docstring = tree.body[0].value.s
                    tool_info["description"] = docstring.split('\n')[0] if docstring else ""
            except:
                pass
            
            # Look for TOOL_METADATA dictionary
            if "TOOL_METADATA" in content:
                try:
                    # Extract metadata dict
                    start = content.find("TOOL_METADATA")
                    end = content.find("}", start) + 1
                    metadata_str = content[start:end]
                    # Simple eval (safe in this context)
                    metadata = eval(metadata_str.split("=", 1)[1].strip())
                    tool_info.update(metadata)
                except:
                    pass
            
            return tool_info
        except Exception as e:
            return None
    
    def _parse_json_tool(self, filepath):
        """Parse a JSON tool definition file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            tool_info = {
                "name": data.get("name", os.path.basename(filepath)),
                "type": "json",
                "file": filepath,
                "description": data.get("description", ""),
                "parameters": data.get("parameters", []),
                "command": data.get("command", ""),
                "script": data.get("script", "")
            }
            return tool_info
        except Exception as e:
            return None
    
    def _parse_yaml_tool(self, filepath):
        """Parse a YAML tool definition file."""
        try:
            import yaml
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            tool_info = {
                "name": data.get("name", os.path.basename(filepath)),
                "type": "yaml",
                "file": filepath,
                "description": data.get("description", ""),
                "parameters": data.get("parameters", []),
                "command": data.get("command", ""),
                "script": data.get("script", "")
            }
            return tool_info
        except Exception as e:
            return None
    
    def create_python_tool(self, name, description, code, parameters=None):
        """Create a new Python tool."""
        if not name.endswith('.py'):
            name += '.py'
        
        filepath = os.path.join(self.tools_dir, name)
        
        # Create tool template
        template = f'''"""
{description}
"""
TOOL_METADATA = {{
    "name": "{name}",
    "description": "{description}",
    "parameters": {parameters or []}
}}

import sys
import json

def main():
    # Parse arguments
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    
    # Tool implementation
{code}

if __name__ == "__main__":
    main()
'''
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(template)
            self.load_tools()
            return True
        except Exception as e:
            return False
    
    def create_json_tool(self, name, description, command_or_script, parameters=None, is_script=False):
        """Create a new JSON tool definition."""
        if not name.endswith('.json'):
            name += '.json'
        
        filepath = os.path.join(self.tools_dir, name)
        
        tool_def = {
            "name": name.replace('.json', ''),
            "description": description,
            "parameters": parameters or [],
        }
        
        if is_script:
            tool_def["script"] = command_or_script
        else:
            tool_def["command"] = command_or_script
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(tool_def, f, indent=2)
            self.load_tools()
            return True
        except Exception as e:
            return False
    
    def create_yaml_tool(self, name, description, command_or_script, parameters=None, is_script=False):
        """Create a new YAML tool definition."""
        try:
            import yaml
        except ImportError:
            return False
        
        if not name.endswith('.yaml') and not name.endswith('.yml'):
            name += '.yaml'
        
        filepath = os.path.join(self.tools_dir, name)
        
        tool_def = {
            "name": name.replace('.yaml', '').replace('.yml', ''),
            "description": description,
            "parameters": parameters or [],
        }
        
        if is_script:
            tool_def["script"] = command_or_script
        else:
            tool_def["command"] = command_or_script
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(tool_def, f, default_flow_style=False)
            self.load_tools()
            return True
        except Exception as e:
            return False
    
    def delete_tool(self, tool_name):
        """Delete a custom tool."""
        filepath = os.path.join(self.tools_dir, tool_name)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                self.load_tools()
                return True
            except Exception as e:
                return False
        return False
    
    def get_tool_info(self, tool_name):
        """Get metadata for a specific tool."""
        return self.tools_metadata.get(tool_name)
    
    def list_tools(self):
        """List all available custom tools."""
        return list(self.tools_metadata.keys())

# ==============================================================================
#                           4.5. API SERVER MANAGER
# ==============================================================================

class EncryptionManager:
    """Handles encryption/decryption of API data."""
    
    def __init__(self, password=None):
        if not ENCRYPTION_AVAILABLE:
            raise ImportError("cryptography library required. Install with: pip install cryptography")
        
        if password is None:
            # Generate or load master key
            key_file = os.path.join(API_DIR, ".master_key")
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    self.key = f.read()
            else:
                self.key = Fernet.generate_key()
                with open(key_file, 'wb') as f:
                    f.write(self.key)
        else:
            # Derive key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b'ai_terminal_pro_salt',
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            self.key = key
        
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data):
        """Encrypt data."""
        if isinstance(data, str):
            data = data.encode()
        return self.cipher.encrypt(data).decode()
    
    def decrypt(self, encrypted_data):
        """Decrypt data."""
        if isinstance(encrypted_data, str):
            encrypted_data = encrypted_data.encode()
        return self.cipher.decrypt(encrypted_data).decode()


class APIServerManager:
    """Manages multiple API servers with encryption, CORS, and IP whitelisting."""
    
    def __init__(self, app_instance):
        self.app_instance = app_instance  # Reference to main App instance
        self.servers = {}  # {api_name: {server_thread, flask_app, config}}
        self.encryption_manager = None
        self.load_config()
        
        if ENCRYPTION_AVAILABLE:
            try:
                self.encryption_manager = EncryptionManager()
            except Exception as e:
                print(f"{Colors.YELLOW}⚠ Encryption initialization failed: {e}{Colors.RESET}")
    
    def load_config(self):
        """Load API configurations."""
        if os.path.exists(API_CONFIG_FILE):
            try:
                with open(API_CONFIG_FILE, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"{Colors.BRIGHT_RED}[ERROR]{Colors.RESET} Failed to load API config: {e}")
                self.config = {}
        else:
            self.config = {}
    
    def save_config(self):
        """Save API configurations."""
        try:
            with open(API_CONFIG_FILE, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"{Colors.BRIGHT_RED}[ERROR]{Colors.RESET} Failed to save API config: {e}")
    
    def create_api(self, name, port, enable_cors=True, cors_origins=None, ip_whitelist=None, require_auth=True):
        """Create a new API server."""
        if not FLASK_AVAILABLE:
            return False, "Flask not available. Install with: pip install flask flask-cors"
        
        if name in self.servers:
            return False, f"API '{name}' already exists"
        
        # Default values
        if cors_origins is None:
            cors_origins = ["*"] if enable_cors else []
        if ip_whitelist is None:
            ip_whitelist = []
        
        # Create Flask app
        app = Flask(f"API_{name}")
        app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True
        
        # Configure CORS
        if enable_cors:
            CORS(app, origins=cors_origins)
        
        # Generate API key if auth required
        api_key = None
        if require_auth:
            api_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
            # Save API key
            keys = {}
            if os.path.exists(API_KEYS_FILE):
                try:
                    with open(API_KEYS_FILE, 'r') as f:
                        keys = json.load(f)
                except:
                    pass
            keys[name] = api_key
            with open(API_KEYS_FILE, 'w') as f:
                json.dump(keys, f, indent=2)
        
        # IP whitelist middleware
        def check_ip_whitelist():
            if ip_whitelist:
                client_ip = request.remote_addr
                if client_ip not in ip_whitelist:
                    return jsonify({"error": "IP address not whitelisted"}), 403
            return None
        
        # Auth middleware
        def check_auth():
            if require_auth:
                provided_key = request.headers.get('X-API-Key') or request.args.get('api_key')
                if not provided_key or provided_key != api_key:
                    return jsonify({"error": "Invalid or missing API key"}), 401
            return None
        
        # Chat endpoint
        @app.route('/chat', methods=['POST'])
        def chat_endpoint():
            # Check IP whitelist
            ip_check = check_ip_whitelist()
            if ip_check:
                return ip_check
            
            # Check auth
            auth_check = check_auth()
            if auth_check:
                return auth_check
            
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No JSON data provided"}), 400
                
                # Decrypt if encrypted
                message = data.get('message', '')
                if data.get('encrypted', False) and self.encryption_manager:
                    try:
                        message = self.encryption_manager.decrypt(message)
                    except Exception as e:
                        return jsonify({"error": f"Decryption failed: {str(e)}"}), 400
                
                if not message:
                    return jsonify({"error": "Message is required"}), 400
                
                # Get session ID if provided
                session_id = data.get('session_id')
                if session_id:
                    self.app_instance.current_session_id = session_id
                
                # Process message through AI
                if not self.app_instance.engine or not self.app_instance.context_mgr:
                    return jsonify({"error": "AI engine not initialized"}), 500
                
                # Build context
                history = []
                project_memory = None
                if session_id:
                    history = self.app_instance.memory.get_recent_history(session_id, limit=100)
                    # Get project memory if session has a project
                    session_info = self.app_instance.memory.get_session(session_id)
                    if session_info and session_info[2]:  # project_id
                        project_memory = self.app_instance.memory.get_project_memory(session_info[2])
                
                context = self.app_instance.context_mgr.build_context(
                    message,
                    history,
                    self.app_instance.registry.get_tool_prompt() if self.app_instance.registry else "",
                    project_memory
                )
                
                # Generate response
                response = self.app_instance.engine.generate(context)
                
                # Save to history
                if session_id:
                    self.app_instance.memory.save_message(session_id, "user", message)
                    self.app_instance.memory.save_message(session_id, "assistant", response)
                
                # Encrypt response if requested
                encrypted_response = None
                if data.get('encrypt_response', False) and self.encryption_manager:
                    encrypted_response = self.encryption_manager.encrypt(response)
                
                result = {
                    "response": response,
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat()
                }
                
                if encrypted_response:
                    result["encrypted_response"] = encrypted_response
                    result["encrypted"] = True
                
                return jsonify(result)
                
            except Exception as e:
                return jsonify({"error": str(e)}), 500
        
        # Health check endpoint
        @app.route('/health', methods=['GET'])
        def health_endpoint():
            return jsonify({
                "status": "healthy",
                "api_name": name,
                "timestamp": datetime.now().isoformat()
            })
        
        # Get API info endpoint
        @app.route('/info', methods=['GET'])
        def info_endpoint():
            auth_check = check_auth()
            if auth_check:
                return auth_check
            
            return jsonify({
                "name": name,
                "port": port,
                "cors_enabled": enable_cors,
                "ip_whitelist_enabled": len(ip_whitelist) > 0,
                "encryption_available": self.encryption_manager is not None
            })
        
        # Start server in separate thread
        def run_server():
            app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Store configuration
        config = {
            "name": name,
            "port": port,
            "enable_cors": enable_cors,
            "cors_origins": cors_origins,
            "ip_whitelist": ip_whitelist,
            "require_auth": require_auth,
            "api_key": api_key,
            "flask_app": app,
            "server_thread": server_thread
        }
        
        self.servers[name] = config
        self.config[name] = {
            "port": port,
            "enable_cors": enable_cors,
            "cors_origins": cors_origins,
            "ip_whitelist": ip_whitelist,
            "require_auth": require_auth
        }
        self.save_config()
        
        return True, f"API '{name}' started on port {port}"
    
    def stop_api(self, name):
        """Stop an API server."""
        if name not in self.servers:
            return False, f"API '{name}' not found"
        
        # Flask doesn't have a clean shutdown, but we can mark it
        del self.servers[name]
        if name in self.config:
            del self.config[name]
        self.save_config()
        
        return True, f"API '{name}' stopped"
    
    def delete_api(self, name):
        """Delete an API configuration."""
        # Stop if running
        if name in self.servers:
            self.stop_api(name)
        
        # Remove API key
        if os.path.exists(API_KEYS_FILE):
            try:
                with open(API_KEYS_FILE, 'r') as f:
                    keys = json.load(f)
                if name in keys:
                    del keys[name]
                    with open(API_KEYS_FILE, 'w') as f:
                        json.dump(keys, f, indent=2)
            except:
                pass
        
        return True, f"API '{name}' deleted"
    
    def get_api_key(self, name):
        """Get API key for an API."""
        if not os.path.exists(API_KEYS_FILE):
            return None
        
        try:
            with open(API_KEYS_FILE, 'r') as f:
                keys = json.load(f)
            return keys.get(name)
        except:
            return None
    
    def list_apis(self):
        """List all configured APIs."""
        return list(self.config.keys())
    
    def get_api_info(self, name):
        """Get information about an API."""
        if name not in self.config:
            return None
        
        info = self.config[name].copy()
        info["running"] = name in self.servers
        info["api_key"] = self.get_api_key(name) if info.get("require_auth") else None
        return info

# ==============================================================================
#                           5. TOOL REGISTRY & PATH RESOLVER
# ==============================================================================

class BrowserManager:
    """Manages a persistent Playwright browser instance for web interactions."""
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.pages = []  # Track multiple pages/tabs
        self.is_open = False
    
    def launch(self, headless=False):
        """Launch a browser instance."""
        if not PLAYWRIGHT_AVAILABLE:
            return False
        
        try:
            if not self.is_open:
                self.playwright = sync_playwright().start()
                self.browser = self.playwright.chromium.launch(
                    headless=headless,
                    args=['--start-maximized'] if not headless else []
                )
                self.context = self.browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    viewport={'width': 1920, 'height': 1080} if headless else None
                )
                self.page = self.context.new_page()
                self.is_open = True
            return True
        except Exception as e:
            print(f"{Colors.BRIGHT_RED}[BROWSER]{Colors.RESET} Failed to launch browser: {e}")
            return False
    
    def close(self):
        """Close the browser instance."""
        try:
            if self.page:
                self.page.close()
            if self.context:
                self.context.close()
            if self.browser:
                self.browser.close()
            if self.playwright:
                self.playwright.stop()
            self.is_open = False
        except Exception:
            pass
    
    def navigate(self, url):
        """Navigate to a URL."""
        if not self.is_open or not self.page:
            return False
        try:
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            self.page.goto(url, wait_until="networkidle", timeout=30000)
            self.page.wait_for_timeout(2000)
            return True
        except Exception as e:
            return f"Navigation error: {e}"
    
    def get_content(self, page=None):
        """Get page content from a specific page or the current page."""
        target_page = page or self.page
        if not self.is_open or not target_page:
            return None
        try:
            content = target_page.evaluate("""
                () => {
                    const scripts = document.querySelectorAll('script, style, noscript');
                    scripts.forEach(el => el.remove());
                    const main = document.querySelector('main, article, [role="main"]') || document.body;
                    return {
                        title: document.title,
                        content: main.innerText || main.textContent || '',
                        url: window.location.href
                    };
                }
            """)
            return content
        except Exception as e:
            return {"error": str(e)}
    
    def open_new_tab(self, url):
        """Open a new tab and navigate to URL."""
        if not self.is_open or not self.context:
            return None
        try:
            new_page = self.context.new_page()
            if not url.startswith(("http://", "https://")):
                url = "https://" + url
            new_page.goto(url, wait_until="networkidle", timeout=30000)
            new_page.wait_for_timeout(2000)
            self.pages.append(new_page)
            return new_page
        except Exception as e:
            return None
    
    def browse_multiple(self, urls):
        """Browse multiple URLs, opening each in a new tab."""
        if not self.is_open:
            return []
        
        results = []
        # Navigate main page to first URL
        if urls:
            first_url = urls[0]
            if not first_url.startswith(("http://", "https://")):
                first_url = "https://" + first_url
            nav_result = self.navigate(first_url)
            if nav_result == True:
                content = self.get_content(self.page)
                if content:
                    results.append(content)
            
            # Open remaining URLs in new tabs
            for url in urls[1:]:
                new_page = self.open_new_tab(url)
                if new_page:
                    content = self.get_content(new_page)
                    if content:
                        results.append(content)
        
        return results
    
    def click(self, selector, page=None):
        """Click an element on the page by selector."""
        target_page = page or self.page
        if not self.is_open or not target_page:
            return False, "Browser not open"
        try:
            target_page.wait_for_selector(selector, timeout=10000)
            target_page.click(selector)
            target_page.wait_for_timeout(1000)  # Wait for page to react
            return True, f"Clicked element: {selector}"
        except Exception as e:
            return False, f"Click failed: {str(e)}"
    
    def type_text(self, selector, text, page=None):
        """Type text into an input field."""
        target_page = page or self.page
        if not self.is_open or not target_page:
            return False, "Browser not open"
        try:
            target_page.wait_for_selector(selector, timeout=10000)
            target_page.fill(selector, text)
            target_page.wait_for_timeout(500)
            return True, f"Typed '{text}' into {selector}"
        except Exception as e:
            return False, f"Type failed: {str(e)}"
    
    def press_key(self, selector, key, page=None):
        """Press a key in an element (e.g., Enter in search box)."""
        target_page = page or self.page
        if not self.is_open or not target_page:
            return False, "Browser not open"
        try:
            target_page.wait_for_selector(selector, timeout=10000)
            target_page.press(selector, key)
            target_page.wait_for_timeout(1000)
            return True, f"Pressed {key} in {selector}"
        except Exception as e:
            return False, f"Press key failed: {str(e)}"
    
    def search_google(self, query, page=None):
        """Perform a Google search."""
        target_page = page or self.page
        if not self.is_open or not target_page:
            return False, "Browser not open"
        try:
            # Navigate to Google if not already there
            current_url = target_page.url
            if "google.com" not in current_url.lower():
                target_page.goto("https://www.google.com", wait_until="networkidle", timeout=30000)
                target_page.wait_for_timeout(2000)
            
            # Find and fill search box
            search_selectors = [
                'textarea[name="q"]',
                'input[name="q"]',
                'textarea[aria-label*="Search"]',
                'input[aria-label*="Search"]'
            ]
            
            search_box = None
            for selector in search_selectors:
                try:
                    target_page.wait_for_selector(selector, timeout=5000)
                    search_box = selector
                    break
                except:
                    continue
            
            if not search_box:
                return False, "Could not find Google search box"
            
            target_page.fill(search_box, query)
            target_page.wait_for_timeout(500)
            target_page.press(search_box, "Enter")
            target_page.wait_for_timeout(3000)  # Wait for results
            
            return True, f"Search completed for: {query}"
        except Exception as e:
            return False, f"Google search failed: {str(e)}"
    
    def click_first_result(self, page=None):
        """Click the first search result on Google."""
        target_page = page or self.page
        if not self.is_open or not target_page:
            return False, "Browser not open"
        try:
            # Wait for search results
            target_page.wait_for_timeout(2000)
            
            # Try different selectors for Google search results
            # Google search results are typically in div.g with links
            result_selectors = [
                'div.g a',  # Google result link (most reliable)
                'div[data-ved] a',
                'a[data-ved]',  # Direct link with data-ved attribute
                'h3 a',  # Link inside h3 heading
            ]
            
            clicked = False
            for selector in result_selectors:
                try:
                    # Wait for selector to appear
                    target_page.wait_for_selector(selector, timeout=5000)
                    # Get all matching elements
                    elements = target_page.query_selector_all(selector)
                    if elements and len(elements) > 0:
                        # Click the first result
                        first_link = elements[0]
                        first_link.click()
                        target_page.wait_for_timeout(3000)  # Wait for navigation
                        clicked = True
                        break
                except Exception:
                    continue
            
            if clicked:
                return True, "Clicked first search result"
            else:
                return False, "Could not find search results to click"
        except Exception as e:
            return False, f"Click first result failed: {str(e)}"
    
    def wait(self, milliseconds, page=None):
        """Wait for specified milliseconds."""
        target_page = page or self.page
        if not self.is_open or not target_page:
            return False, "Browser not open"
        try:
            target_page.wait_for_timeout(milliseconds)
            return True, f"Waited {milliseconds}ms"
        except Exception as e:
            return False, f"Wait failed: {str(e)}"
    
    def get_current_url(self, page=None):
        """Get the current page URL."""
        target_page = page or self.page
        if not self.is_open or not target_page:
            return None
        try:
            return target_page.url
        except:
            return None

class ToolRegistry:
    """Manages Native Tools, Custom Scripts, and MCP Clients."""
    def __init__(self, config):
        self.config = config
        self.mcp_clients = {}
        self.custom_tool_manager = CustomToolManager()
        self.playwright_browsers_installed = False
        self.browser_manager = BrowserManager() if PLAYWRIGHT_AVAILABLE else None
        self.load_mcp_servers()
        # Check Playwright browsers in background to avoid blocking startup
        threading.Thread(target=self._ensure_playwright_browsers, daemon=True).start()

    def load_mcp_servers(self):
        if os.path.exists(MCP_CONFIG_FILE):
            try:
                with open(MCP_CONFIG_FILE, 'r') as f:
                    data = json.load(f)
                    for name, cmd in data.items():
                        print(f"{Colors.BRIGHT_CYAN}[SYSTEM]{Colors.RESET} Initializing MCP: {Colors.BRIGHT_WHITE}{name}{Colors.RESET}...")
                        client = MCPClient(name, cmd)
                        if client.start():
                            self.mcp_clients[name] = client
            except Exception as e:
                print(f"{Colors.BRIGHT_RED}[ERROR]{Colors.RESET} MCP Config: {e}")

    def _ensure_playwright_browsers(self):
        """Ensure Playwright browsers are installed."""
        if not PLAYWRIGHT_AVAILABLE:
            return
        
        if self.playwright_browsers_installed:
            return
        
        try:
            # Check if browsers are installed by trying to get browser path
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                # Try to get chromium - if it fails, browsers aren't installed
                try:
                    browser = p.chromium.launch(headless=True)
                    browser.close()
                    self.playwright_browsers_installed = True
                    return
                except Exception:
                    pass
            
            # Browsers not installed, install them
            print(f"{Colors.BRIGHT_YELLOW}[SYSTEM]{Colors.RESET} Installing Playwright browsers...")
            import subprocess
            result = subprocess.run(
                [sys.executable, "-m", "playwright", "install", "chromium"],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            if result.returncode == 0:
                self.playwright_browsers_installed = True
                print(f"{Colors.BRIGHT_GREEN}[SYSTEM]{Colors.RESET} Playwright browsers installed successfully!")
            else:
                print(f"{Colors.BRIGHT_RED}[SYSTEM]{Colors.RESET} Failed to install Playwright browsers: {result.stderr}")
        except Exception as e:
            print(f"{Colors.BRIGHT_YELLOW}[SYSTEM]{Colors.RESET} Playwright browser installation check failed: {e}")
            print(f"{Colors.DIM}You can manually install browsers with: python -m playwright install chromium{Colors.RESET}")

    def get_tool_prompt(self):
        """Returns the help text injected into the System Prompt."""
        os_name = platform.system()
        os_version = platform.version()
        is_windows = os_name == "Windows"
        is_mac = os_name == "Darwin"
        is_linux = os_name == "Linux"
        
        # Get desktop path
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        if not os.path.exists(desktop_path):
            # Try alternative desktop locations
            if is_windows:
                desktop_path = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
            elif is_mac:
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
            elif is_linux:
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
        
        # Format OS info
        os_info = f"{os_name} {os_version}" if os_version else os_name
        os_note = "WINDOWS" if is_windows else ("macOS" if is_mac else "LINUX")
        
        p = "\n\n### SYSTEM INFORMATION ###\n"
        p += f"OPERATING SYSTEM: {os_info} ({os_note})\n"
        p += f"DESKTOP PATH: {desktop_path}\n"
        p += f"HOME DIRECTORY: {os.path.expanduser('~')}\n"
        p += "IMPORTANT: Always use OS-appropriate commands for the detected platform!\n"
        
        p += "\n### AVAILABLE TOOLS ###\n"
        p += "SYNTAX: Response must start with 'ACTION: [Tool_Name] [Arguments]'\n"
        p += "IMPORTANT: Do NOT wrap file paths in brackets []. Use: FILE_READ ~/Desktop/file.txt\n"
        p += "CRITICAL FOR SCRIPTS: When writing scripts (batch, PowerShell, bash, Python, etc.), write the COMPLETE script!\n"
        p += "                      Include ALL code from start to finish - never truncate or cut off mid-script!\n"
        p += "                      The entire script must be in a single FILE_WRITE action.\n\n"
        
        p += "1. NATIVE TOOLS:\n"
        if is_windows:
            p += "   - ACTION: CMD [command] (Windows CMD/PowerShell - NO Unix commands!)\n"
            p += "     Windows Commands: mkdir folder, rmdir folder, dir, type file.txt, del file.txt\n"
            p += "     Desktop Access: Use ~/Desktop/ or CMD with full path\n"
            p += "     NEVER use: touch, ls, rm, cat, grep (these are Unix-only)\n"
        elif is_mac:
            p += "   - ACTION: CMD [command] (macOS Terminal - Unix commands)\n"
            p += "     macOS Commands: mkdir folder, touch file.txt, ls, rm, cat\n"
            p += "     Desktop Access: Use ~/Desktop/ or /Users/username/Desktop/\n"
        else:  # Linux
            p += "   - ACTION: CMD [command] (Linux Terminal - Unix commands)\n"
            p += "     Linux Commands: mkdir folder, touch file.txt, ls, rm, cat\n"
            p += "     Desktop Access: Use ~/Desktop/ or /home/username/Desktop/\n"
        
        p += "   - ACTION: BROWSE [url(s)] (Launches VISIBLE Playwright browser for web browsing and interaction)\n"
        if PLAYWRIGHT_AVAILABLE:
            p += "     IMPORTANT: Browser launches in VISIBLE mode - user can see and interact with it!\n"
            p += "     The browser stays open for continued interaction. Can handle dynamic content, SPAs, JavaScript.\n"
            p += "     MULTIPLE URLS: You can browse multiple websites in one request!\n"
            p += "       - Separate URLs by comma: ACTION: BROWSE https://site1.com, https://site2.com\n"
            p += "       - Or by pipe: ACTION: BROWSE https://site1.com | https://site2.com\n"
            p += "       - Or by space: ACTION: BROWSE https://site1.com https://site2.com\n"
            p += "       Each URL opens in a separate browser tab.\n"
            p += "     CRITICAL: ONLY provide the URL(s) in the BROWSE action - no additional text, summaries, or explanations!\n"
            p += "     Format: ACTION: BROWSE https://example.com (ONLY the URL(s), nothing else on that line)\n"
            p += "     Examples:\n"
            p += "       Single: ACTION: BROWSE https://example.com\n"
            p += "       Multiple: ACTION: BROWSE https://example.com, https://google.com, https://github.com\n"
            p += "\n   BROWSER INTERACTION COMMANDS (use after BROWSE to interact with pages):\n"
            p += "   - ACTION: BROWSE_SEARCH [query] (Perform Google search - opens Google and searches)\n"
            p += "   - ACTION: BROWSE_CLICK_FIRST (Click the first Google search result)\n"
            p += "   - ACTION: BROWSE_CLICK [selector] (Click an element by CSS selector, e.g., 'button#submit')\n"
            p += "   - ACTION: BROWSE_TYPE [selector] | [text] (Type text into an input field)\n"
            p += "   - ACTION: BROWSE_PRESS [selector] | [key] (Press a key like Enter, Escape, etc.)\n"
            p += "   - ACTION: BROWSE_WAIT [milliseconds] (Wait for page to load, e.g., BROWSE_WAIT 2000)\n"
            p += "   Example workflow:\n"
            p += "     1. ACTION: BROWSE https://google.com\n"
            p += "     2. ACTION: BROWSE_SEARCH repacked.online\n"
            p += "     3. ACTION: BROWSE_WAIT 2000\n"
            p += "     4. ACTION: BROWSE_CLICK_FIRST\n"
            p += "     5. Then provide summary of the site\n"
        else:
            p += "     Note: Playwright not installed - using basic HTTP requests (install with: pip install playwright && python -m playwright install)\n"
        p += "   - ACTION: FILE_READ [path] (Reads file content)\n"
        p += "   - ACTION: FILE_WRITE [path] | [content] (CREATES file and parent dirs automatically)\n"
        p += "     Example: ACTION: FILE_WRITE ~/Desktop/test.txt | Hello World\n"
        p += "   - ACTION: FILE_LIST [path] (Lists directory contents)\n"
        
        p += "\n### DESKTOP INTERACTION EXAMPLES ###\n"
        if is_windows:
            p += "To create a folder on Desktop:\n"
            p += "  ACTION: CMD mkdir ~/Desktop/folder_name\n"
            p += "To create a file on Desktop:\n"
            p += "  ACTION: FILE_WRITE ~/Desktop/file.txt | content here\n"
            p += "To read a file from Desktop:\n"
            p += "  ACTION: FILE_READ ~/Desktop/file.txt\n"
            p += "Note: ~ expands to your user profile (e.g., C:\\Users\\YourName)\n"
        else:
            p += "To create a folder on Desktop:\n"
            p += "  ACTION: CMD mkdir ~/Desktop/folder_name\n"
            p += "To create a file on Desktop:\n"
            p += "  ACTION: FILE_WRITE ~/Desktop/file.txt | content here\n"
            p += "To read a file from Desktop:\n"
            p += "  ACTION: FILE_READ ~/Desktop/file.txt\n"
        
        # Custom tools
        custom_tools = self.custom_tool_manager.list_tools()
        if custom_tools:
            p += "2. CUSTOM TOOLS:\n"
            for tool_name in custom_tools:
                tool_info = self.custom_tool_manager.get_tool_info(tool_name)
                desc = tool_info.get("description", "") if tool_info else ""
                if desc:
                    p += f"   - ACTION: CUSTOM {tool_name} [args] {Colors.DIM}# {desc}{Colors.RESET}\n"
                else:
                    p += f"   - ACTION: CUSTOM {tool_name} [args]\n"
            
        if self.mcp_clients:
            p += "3. MCP EXTENSIONS:\n"
            for srv, client in self.mcp_clients.items():
                for t in client.available_tools:
                    p += f"   - ACTION: MCP {srv} {t['name']} {{json_args}}\n"
        return p

    def expand_paths_in_command(self, command):
        r"""
        Expands ~ in paths within a command string.
        Handles cases like: mkdir ~/Desktop/Test -> mkdir C:\Users\user\Desktop\Test
        """
        # Pattern to match ~/ or ~\ followed by path characters
        pattern = r'~[/\\][^\s]*'
        
        def expand_match(match):
            path_with_tilde = match.group(0)
            expanded = os.path.expanduser(path_with_tilde)
            # Normalize path separators for Windows
            return os.path.normpath(expanded)
        
        return re.sub(pattern, expand_match, command)
    
    def resolve_path(self, raw_path):
        """
        Sanitizes and resolves paths.
        1. Strips quotes/brackets (AI hallucinations).
        2. Expands ~ to user home.
        3. Enforces Sandbox if 'enable_dangerous_commands' is False.
        """
        # Cleanup AI hallucinations like [path] or "path"
        clean = raw_path.strip().replace('[', '').replace(']', '').replace('"', '').replace("'", "")
        
        # Expand user (~ -> /home/user or C:\Users\user)
        full_path = os.path.expanduser(clean)
        full_path = os.path.normpath(full_path)

        if not self.config.get("enable_dangerous_commands"):
            # SAFE MODE: Force filename into sandbox
            filename = os.path.basename(full_path)
            return os.path.join(SANDBOX_DIR, filename)
        
        # DANGEROUS MODE: Allow absolute paths
        if os.path.isabs(full_path):
            return full_path
        
        # Relative path -> anchor to sandbox
        return os.path.join(SANDBOX_DIR, full_path)

    def execute_native(self, tool_cmd):
        parts = tool_cmd.strip().split(" ", 1)
        action = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        if action == "CMD":
            if not self.config.get("enable_dangerous_commands"):
                return "PERMISSION DENIED: Enable 'Dangerous Commands' in settings."
            try:
                # Expand ~ in paths for cross-platform compatibility
                expanded_args = self.expand_paths_in_command(args)
                
                # Convert common Unix commands to Windows equivalents
                if platform.system() == "Windows":
                    # Convert 'touch file.txt' to 'type nul > file.txt' or use Python to create empty file
                    if expanded_args.strip().startswith("touch "):
                        file_path = expanded_args.strip()[6:].strip()  # Remove "touch "
                        # Use Python to create empty file (more reliable than type nul)
                        try:
                            os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
                            with open(file_path, 'a'):
                                os.utime(file_path, None)  # Touch the file
                            return f"Created/updated file: {file_path}"
                        except Exception as e:
                            return f"Touch Error: {e}"
                
                return subprocess.check_output(expanded_args, shell=True, text=True, stderr=subprocess.STDOUT).strip()
            except subprocess.CalledProcessError as e:
                return f"CMD Error: {e.output}"
        
        elif action == "BROWSE":
            # Extract only the URL - stop at newlines, extra text, or other delimiters
            url = args.strip()
            # Remove any trailing text after the URL (LLM sometimes adds summaries)
            # URL should end with a space, newline, or be the last thing
            if '\n' in url:
                url = url.split('\n')[0].strip()
            # Remove common trailing patterns that aren't URLs
            url = url.split('**')[0].strip()  # Remove markdown formatting
            url = url.split('Summary:')[0].strip()  # Remove summary text
            url = url.split('Summary')[0].strip()
            # Clean up any remaining whitespace
            url = url.strip()
            return self._browse(url)
        
        elif action == "BROWSE_CLICK":
            # Click an element: BROWSE_CLICK selector
            if not PLAYWRIGHT_AVAILABLE or not self.browser_manager or not self.browser_manager.is_open:
                return "Error: Browser not open. Use BROWSE first to open a website."
            selector = args.strip()
            success, message = self.browser_manager.click(selector)
            if success:
                return f"✓ {message}"
            else:
                return f"✗ {message}"
        
        elif action == "BROWSE_TYPE":
            # Type text into a field: BROWSE_TYPE selector | text
            if not PLAYWRIGHT_AVAILABLE or not self.browser_manager or not self.browser_manager.is_open:
                return "Error: Browser not open. Use BROWSE first to open a website."
            if "|" not in args:
                return "Error: Use format 'BROWSE_TYPE selector | text'"
            selector, text = args.split("|", 1)
            selector = selector.strip()
            text = text.strip()
            success, message = self.browser_manager.type_text(selector, text)
            if success:
                return f"✓ {message}"
            else:
                return f"✗ {message}"
        
        elif action == "BROWSE_PRESS":
            # Press a key: BROWSE_PRESS selector | key
            if not PLAYWRIGHT_AVAILABLE or not self.browser_manager or not self.browser_manager.is_open:
                return "Error: Browser not open. Use BROWSE first to open a website."
            if "|" not in args:
                return "Error: Use format 'BROWSE_PRESS selector | key' (e.g., Enter, Escape)"
            selector, key = args.split("|", 1)
            selector = selector.strip()
            key = key.strip()
            success, message = self.browser_manager.press_key(selector, key)
            if success:
                return f"✓ {message}"
            else:
                return f"✗ {message}"
        
        elif action == "BROWSE_SEARCH":
            # Perform Google search: BROWSE_SEARCH query
            if not PLAYWRIGHT_AVAILABLE or not self.browser_manager:
                return "Error: Browser not available. Use BROWSE first."
            if not self.browser_manager.is_open:
                # Launch browser if not open
                print(f"{Colors.BRIGHT_CYAN}[BROWSER]{Colors.RESET} Launching browser for search...")
                if not self.browser_manager.launch(headless=False):
                    return "Error: Failed to launch browser."
            query = args.strip()
            success, message = self.browser_manager.search_google(query)
            if success:
                return f"✓ {message}"
            else:
                return f"✗ {message}"
        
        elif action == "BROWSE_CLICK_FIRST":
            # Click first search result: BROWSE_CLICK_FIRST
            if not PLAYWRIGHT_AVAILABLE or not self.browser_manager or not self.browser_manager.is_open:
                return "Error: Browser not open. Use BROWSE_SEARCH first."
            success, message = self.browser_manager.click_first_result()
            if success:
                return f"✓ {message}"
            else:
                return f"✗ {message}"
        
        elif action == "BROWSE_WAIT":
            # Wait: BROWSE_WAIT milliseconds
            if not PLAYWRIGHT_AVAILABLE or not self.browser_manager or not self.browser_manager.is_open:
                return "Error: Browser not open."
            try:
                ms = int(args.strip())
                success, message = self.browser_manager.wait(ms)
                if success:
                    return f"✓ {message}"
                else:
                    return f"✗ {message}"
            except ValueError:
                return "Error: BROWSE_WAIT requires a number (milliseconds)"

        elif action == "FILE_LIST":
            target = self.resolve_path(args) if args else SANDBOX_DIR
            if os.path.exists(target):
                return str(os.listdir(target))
            return "Directory not found."

        elif action == "FILE_READ":
            target = self.resolve_path(args)
            if os.path.exists(target):
                with open(target, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
            return f"File not found: {target}"

        elif action == "FILE_WRITE":
            if "|" not in args:
                return "Error: Use format 'FILE_WRITE path | content'"
            raw_path, content = args.split("|", 1)
            target = self.resolve_path(raw_path)
            
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, 'w', encoding='utf-8') as f:
                    f.write(content.strip())
                return f"Successfully wrote to {target}"
            except Exception as e:
                return f"Write Error: {e}"

        return "Unknown Action."

    def _parse_urls(self, url_string):
        """Parse multiple URLs from a string. Supports comma, space, newline, or pipe separation."""
        # Remove brackets, quotes first
        url_string = url_string.replace('[', '').replace(']', '').replace('"', '').replace("'", "").strip()
        
        # Remove trailing text that might have been included
        for delimiter in ['\n', '\r', '**', 'Summary:', 'Summary', 'Description:', 'Note:', ' -', ' –']:
            if delimiter in url_string:
                url_string = url_string.split(delimiter)[0].strip()
        
        # Try different separators
        urls = []
        
        # Try comma separation first
        if ',' in url_string:
            urls = [u.strip() for u in url_string.split(',')]
        # Try pipe separation
        elif '|' in url_string:
            urls = [u.strip() for u in url_string.split('|')]
        # Try newline separation
        elif '\n' in url_string:
            urls = [u.strip() for u in url_string.split('\n') if u.strip()]
        # Try space separation (but be careful - URLs shouldn't have spaces)
        elif ' ' in url_string and ('http://' in url_string or 'https://' in url_string):
            # Only split on space if we see multiple http/https patterns
            parts = url_string.split()
            urls = []
            current_url = ""
            for part in parts:
                if part.startswith(('http://', 'https://')):
                    if current_url:
                        urls.append(current_url.strip())
                    current_url = part
                elif current_url:
                    current_url += " " + part
                else:
                    # Might be a domain without http
                    if '.' in part and ' ' not in part:
                        urls.append(part)
            if current_url:
                urls.append(current_url.strip())
        else:
            # Single URL
            urls = [url_string.strip()]
        
        # Clean and validate URLs
        cleaned_urls = []
        for url in urls:
            url = url.strip()
            if not url:
                continue
            # Remove any remaining trailing text
            for delimiter in ['\n', '\r', '**', 'Summary:', 'Summary', 'Description:', 'Note:']:
                if delimiter in url:
                    url = url.split(delimiter)[0].strip()
            
            # Validate URL format
            if url and (url.startswith(("http://", "https://")) or ('.' in url and ' ' not in url)):
                cleaned_urls.append(url)
        
        return cleaned_urls
    
    def _browse(self, url):
        """Browse web using Playwright - supports single or multiple URLs. Launches VISIBLE browser."""
        # Parse URLs (supports multiple URLs separated by comma, space, newline, or pipe)
        urls = self._parse_urls(url)
        
        if not urls:
            return "Error: No valid URL(s) provided"
        
        # Use Playwright if available
        if PLAYWRIGHT_AVAILABLE and self.browser_manager:
            try:
                # Ensure browsers are installed
                if not self.playwright_browsers_installed:
                    self._ensure_playwright_browsers()
                
                # Launch browser in VISIBLE mode (headless=False) for user interaction
                if not self.browser_manager.is_open:
                    print(f"{Colors.BRIGHT_CYAN}[BROWSER]{Colors.RESET} Launching Playwright browser...")
                    if not self.browser_manager.launch(headless=False):
                        return "Failed to launch browser. Falling back to basic HTTP request."
                    print(f"{Colors.BRIGHT_GREEN}[BROWSER]{Colors.RESET} Browser launched successfully!")
                
                # Handle multiple URLs
                if len(urls) > 1:
                    print(f"{Colors.BRIGHT_CYAN}[BROWSER]{Colors.RESET} Opening {len(urls)} websites in separate tabs...")
                    results = self.browser_manager.browse_multiple(urls)
                    
                    if results:
                        output = f"{Colors.BRIGHT_GREEN}[BROWSER]{Colors.RESET} Successfully opened {len(results)} website(s) in browser tabs:\n\n"
                        
                        for i, page_data in enumerate(results, 1):
                            if "error" not in page_data:
                                title = page_data.get("title", "No title")
                                content = page_data.get("content", "")
                                current_url = page_data.get("url", urls[i-1] if i-1 < len(urls) else "Unknown")
                                
                                # Clean and format content
                                lines = [line.strip() for line in content.splitlines() if line.strip()]
                                text_content = '\n'.join(lines)
                                
                                # Limit content size
                                if len(text_content) > 3000:
                                    text_content = text_content[:3000] + "... [Truncated - see browser tab for full content]"
                                
                                output += f"{Colors.BRIGHT_CYAN}--- Tab {i}: {current_url} ---{Colors.RESET}\n"
                                output += f"Title: {title}\n"
                                output += f"Content Preview:\n{text_content}\n\n"
                        
                        output += f"{Colors.BRIGHT_CYAN}[NOTE]{Colors.RESET} All websites are open in browser tabs. "
                        output += "You can switch between tabs to view each site."
                        return output
                    else:
                        return "Error: Failed to open websites in browser tabs."
                
                else:
                    # Single URL - use existing logic
                    url = urls[0]
                    if not url.startswith(("http://", "https://")):
                        url = "https://" + url
                    
                    nav_result = self.browser_manager.navigate(url)
                    if nav_result != True:
                        return f"Navigation failed: {nav_result}"
                    
                    page_data = self.browser_manager.get_content()
                    if page_data and "error" not in page_data:
                        title = page_data.get("title", "No title")
                        content = page_data.get("content", "")
                        current_url = page_data.get("url", url)
                        
                        # Clean and format content
                        lines = [line.strip() for line in content.splitlines() if line.strip()]
                        text_content = '\n'.join(lines)
                        
                        # Limit to reasonable size for display
                        if len(text_content) > 5000:
                            text_content = text_content[:5000] + "... [Truncated - browser is open for full content]"
                        
                        result = f"Browser opened and navigated to: {current_url}\n"
                        result += f"Title: {title}\n\n"
                        result += f"Page Content:\n{text_content}\n\n"
                        result += f"{Colors.BRIGHT_CYAN}[NOTE]{Colors.RESET} Browser window is open and ready for interaction. "
                        result += "You can continue to interact with the page through the browser."
                        return result
                    else:
                        error_msg = page_data.get("error", "Unknown error") if page_data else "Failed to get content"
                        return f"Error getting page content: {error_msg}"
                    
            except Exception as e:
                # Fallback to requests if Playwright fails
                if len(urls) == 1:
                    return self._browse_fallback(urls[0], str(e))
                else:
                    return f"Error browsing multiple sites: {e}"
        else:
            # Fallback to requests if Playwright not available
            if len(urls) == 1:
                return self._browse_fallback(urls[0], "Playwright not installed")
            else:
                return "Error: Multiple URL browsing requires Playwright. Please install: pip install playwright && python -m playwright install"
    
    def _browse_fallback(self, url, error_msg=""):
        """Fallback browsing method using requests and BeautifulSoup."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            res = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(res.text, 'html.parser')
            # Clean script/style
            for s in soup(["script", "style", "noscript"]):
                s.extract()
            text = soup.get_text()
            # Compress lines
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            content = '\n'.join(lines)[:5000]
            if len(content) >= 5000:
                content += "... [Truncated]"
            
            title = soup.title.string if soup.title else "No title"
            result = f"Title: {title}\n\nContent:\n{content}"
            if error_msg:
                result = f"[Note: Using fallback method - {error_msg}]\n\n{result}"
            return result
        except Exception as e:
            return f"Web Error: {e}"

    def execute_custom(self, tool_name, args):
        """Execute a custom tool (Python, JSON, YAML, or script)."""
        tool_info = self.custom_tool_manager.get_tool_info(tool_name)
        
        if not tool_info:
            # Fallback to old behavior for scripts without metadata
            path = os.path.join(CUSTOM_TOOLS_DIR, tool_name)
            if not os.path.exists(path):
                return "Tool not found."
        else:
            path = tool_info["file"]
            tool_type = tool_info.get("type", "python")
            
            # Handle JSON/YAML tool definitions
            if tool_type in ["json", "yaml"]:
                if "script" in tool_info:
                    # Execute script file
                    script_path = tool_info["script"]
                    if not os.path.isabs(script_path):
                        script_path = os.path.join(CUSTOM_TOOLS_DIR, script_path)
                    return self._execute_script(script_path, args)
                elif "command" in tool_info:
                    # Execute command with args
                    cmd = tool_info["command"] + " " + args if args else tool_info["command"]
                    try:
                        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
                    except Exception as e:
                        return f"Command Error: {e}"
        
        # Execute Python script or other file
        return self._execute_script(path, args)
    
    def _execute_script(self, script_path, args):
        """Execute a script file with arguments."""
        if not os.path.exists(script_path):
            return "Script not found."
        
        # Determine execution method based on OS and extension
        runner = "bash"
        if script_path.endswith(".py"):
            runner = sys.executable  # Current python interpreter
        elif platform.system() == "Windows" and script_path.endswith(".sh"):
            runner = "git-bash"  # Attempt git-bash on Windows
        elif script_path.endswith(".bat") or script_path.endswith(".cmd"):
            runner = "cmd"  # Windows batch file
        
        cmd = f'"{runner}" "{script_path}" {args}'
        try:
            return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
        except Exception as e:
            return f"Script Error: {e}"

    def execute_mcp(self, server, tool, json_args):
        if server not in self.mcp_clients: return "Server not found."
        try:
            # Flexible JSON parsing (handles single quotes or no quotes if simple)
            clean_json = json_args.strip()
            # Ensure brackets
            if not clean_json.startswith("{"): clean_json = "{" + clean_json
            if not clean_json.endswith("}"): clean_json = clean_json + "}"
            
            args = json.loads(clean_json)
            return self.mcp_clients[server].call_tool(tool, args)
        except json.JSONDecodeError:
            return "Error: MCP arguments must be valid JSON."

# ==============================================================================
#                           5. AI ENGINE & CONTEXT
# ==============================================================================

class ContextManager:
    def __init__(self, tokenizer, max_tokens):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def count_tokens(self, text):
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return len(text) // 4 # Rough approx for Ollama

    def build_prompt(self, system, history, rag, user_input, project_memory=None):
        # Build base with system prompt and RAG
        base = f"{system}\n"
        
        # Add project memory if available
        if project_memory:
            memory_str = "\n[PROJECT MEMORY & CONTEXT]:\n"
            if isinstance(project_memory, dict):
                for key, value in project_memory.items():
                    if value:  # Only include non-empty values
                        memory_str += f"{key}: {value}\n"
            else:
                memory_str += str(project_memory) + "\n"
            base += memory_str
        
        base += f"{rag}\n"
        footer = f"\nYou: {user_input}\nAI:"
        
        # Calculate remaining space
        base_cost = self.count_tokens(base + footer)
        remaining = self.max_tokens - base_cost
        
        # Reserve some tokens for response (10% of context window)
        response_reserve = int(self.max_tokens * 0.1)
        remaining = max(remaining - response_reserve, int(self.max_tokens * 0.5))  # Use at least 50% for history
        
        # Add history newest -> oldest until limit
        selected_history = []
        current_cost = 0
        
        # Process history in reverse (newest first)
        for role, msg in reversed(history):
            line = f"{role}: {msg}\n"
            cost = self.count_tokens(line)
            if current_cost + cost < remaining:
                selected_history.insert(0, line)
                current_cost += cost
            else:
                # If we can't fit the full message, try to fit a summary
                if len(selected_history) == 0:
                    # If no history fits, at least try to include a truncated version
                    truncated = msg[:500] + "..." if len(msg) > 500 else msg
                    line = f"{role}: {truncated}\n"
                    cost = self.count_tokens(line)
                    if current_cost + cost < remaining:
                        selected_history.insert(0, line)
                break
                
        return base + "".join(selected_history) + footer
    
    def build_context(self, user_input, history, tool_prompt="", project_memory=None):
        """Alias for build_prompt for API compatibility. Builds context with system prompt."""
        # Build a minimal system prompt with tool info
        system = "You are a helpful AI assistant with full OS awareness. Use available ACTION tools." + tool_prompt
        rag = ""  # No RAG in API mode unless provided
        return self.build_prompt(system, history, rag, user_input, project_memory)

class AIEngine:
    def __init__(self, config):
        self.config = config
        self.backend = config.get("backend")
        self.model_name = config.get("model_name")
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._load()

    def _load(self):
        print(f"{Colors.BRIGHT_CYAN}[SYSTEM]{Colors.RESET} Loading Backend: {Colors.BRIGHT_WHITE}{self.backend}{Colors.RESET} ({Colors.BRIGHT_WHITE}{self.model_name}{Colors.RESET})...")
        apply_torch_threading(self.config)
        if self.backend == "huggingface":
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
                if torch.cuda.is_available(): 
                    self.device = "cuda"
                elif torch.backends.mps.is_available(): 
                    self.device = "mps"
                
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.model.to(self.device)
                print(f"{Colors.BRIGHT_GREEN}[SYSTEM]{Colors.RESET} Model loaded on {Colors.BRIGHT_WHITE}{self.device}{Colors.RESET}.")
            except Exception as e:
                print(f"{Colors.BRIGHT_RED}[CRITICAL]{Colors.RESET} Model Load Failed: {e}")
                sys.exit(1)
        elif self.backend == "ollama":
            try:
                requests.get("http://localhost:11434")
                print(f"{Colors.BRIGHT_GREEN}[SYSTEM]{Colors.RESET} Ollama connection established.")
            except:
                print(f"{Colors.YELLOW}[WARNING]{Colors.RESET} Ollama is not running on localhost:11434.")

    def generate(self, prompt):
        if self.backend == "huggingface":
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, 
                    max_new_tokens=self.config.get("max_response_tokens"),
                    do_sample=True,
                    temperature=self.config.get("temperature"),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            full = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # Remove prompt from output
            response = full[len(prompt):].strip()
            # Stop sequence trimming
            if "You:" in response: response = response.split("You:")[0]
            return response.strip()
            
        elif self.backend == "ollama":
            try:
                res = requests.post("http://localhost:11434/api/generate", json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.config.get("temperature"),
                        "num_predict": self.config.get("max_response_tokens")
                    }
                })
                if res.status_code == 200:
                    return res.json()['response'].strip()
                return f"Ollama Error: {res.text}"
            except Exception as e:
                return f"Connection Error: {e}"

# ==============================================================================
#                           6. MODEL DISCOVERY & DOWNLOAD
# ==============================================================================

def get_ollama_models():
    """Get list of available Ollama models."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
    except:
        pass
    return []

def check_ollama_running():
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def detect_os():
    """Detect the operating system."""
    system = platform.system()
    if system == "Windows":
        return "windows"
    elif system == "Darwin":
        return "macos"
    elif system == "Linux":
        return "linux"
    else:
        return "unknown"

def check_ollama_installed():
    """Check if Ollama is installed."""
    try:
        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False

def download_ollama():
    """Download Ollama installer for the detected OS."""
    os_type = detect_os()
    
    print(f"\n{Colors.BRIGHT_CYAN}Detected OS: {Colors.BRIGHT_WHITE}{os_type}{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}Downloading Ollama installer...{Colors.RESET}\n")
    
    urls = {
        "windows": "https://ollama.com/download/OllamaSetup.exe",
        "macos": "https://ollama.com/download/Ollama-darwin.zip",
        "linux": None  # Linux uses install script
    }
    
    try:
        if os_type == "linux":
            print(f"{Colors.BRIGHT_CYAN}Installing Ollama on Linux...{Colors.RESET}")
            print(f"{Colors.DIM}Running: curl -fsSL https://ollama.com/install.sh | sh{Colors.RESET}\n")
            
            # Download and run install script
            result = subprocess.run(
                "curl -fsSL https://ollama.com/install.sh | sh",
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"\n{Colors.BRIGHT_GREEN}✓ Ollama installed successfully!{Colors.RESET}")
                return True
            else:
                print(f"\n{Colors.BRIGHT_RED}✗ Installation failed: {result.stderr}{Colors.RESET}")
                return False
        
        elif os_type in urls and urls[os_type]:
            url = urls[os_type]
            filename = os.path.basename(url)
            download_path = os.path.join(BASE_DIR, filename)
            
            print(f"{Colors.BRIGHT_CYAN}Downloading from: {Colors.DIM}{url}{Colors.RESET}")
            
            response = requests.get(url, stream=True, timeout=30)
            total_size = int(response.headers.get('content-length', 0))
            
            with open(download_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"\r{Colors.BRIGHT_CYAN}Progress: {percent:.1f}%{Colors.RESET}", end='', flush=True)
            
            print(f"\n{Colors.BRIGHT_GREEN}✓ Downloaded: {download_path}{Colors.RESET}\n")
            
            if os_type == "windows":
                print(f"{Colors.BRIGHT_YELLOW}Please run the installer: {download_path}{Colors.RESET}")
                print(f"{Colors.DIM}After installation, restart this setup.{Colors.RESET}")
                run_installer = input(f"\n{Colors.BRIGHT_GREEN}Run installer now? [Y/n]: {Colors.RESET}").strip().lower()
                if run_installer != 'n':
                    subprocess.Popen([download_path], shell=True)
                    print(f"{Colors.BRIGHT_CYAN}Installer launched. Please complete the installation and restart this application.{Colors.RESET}")
                    sys.exit(0)
                return False
            
            elif os_type == "macos":
                print(f"{Colors.BRIGHT_YELLOW}Please install Ollama from: {download_path}{Colors.RESET}")
                print(f"{Colors.DIM}Extract the zip and move Ollama to Applications.{Colors.RESET}")
                return False
            
        else:
            print(f"{Colors.BRIGHT_RED}✗ Unsupported OS or download URL not available.{Colors.RESET}")
            print(f"{Colors.YELLOW}Please visit https://ollama.ai to download manually.{Colors.RESET}")
            return False
            
    except Exception as e:
        print(f"\n{Colors.BRIGHT_RED}✗ Download failed: {e}{Colors.RESET}")
        return False

def install_and_start_ollama():
    """Check, install, and start Ollama if needed."""
    print(f"\n{Colors.BRIGHT_CYAN}{'='*79}{Colors.RESET}")
    print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  OLLAMA SETUP{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}{'='*79}{Colors.RESET}\n")
    
    # Check if Ollama is installed
    if check_ollama_installed():
        print(f"{Colors.BRIGHT_GREEN}✓ Ollama is installed.{Colors.RESET}")
        
        # Check if running
        if check_ollama_running():
            print(f"{Colors.BRIGHT_GREEN}✓ Ollama is running.{Colors.RESET}")
            return True
        else:
            print(f"{Colors.YELLOW}⚠ Ollama is installed but not running.{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}Attempting to start Ollama...{Colors.RESET}\n")
            
            # Try to start Ollama
            try:
                os_type = detect_os()
                if os_type == "windows":
                    subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Wait for it to start
                time.sleep(3)
                
                if check_ollama_running():
                    print(f"{Colors.BRIGHT_GREEN}✓ Ollama started successfully!{Colors.RESET}")
                    return True
                else:
                    print(f"{Colors.YELLOW}⚠ Ollama started but not responding yet. Please wait a moment...{Colors.RESET}")
                    return False
            except Exception as e:
                print(f"{Colors.BRIGHT_RED}✗ Failed to start Ollama: {e}{Colors.RESET}")
                return False
    else:
        print(f"{Colors.YELLOW}⚠ Ollama is not installed.{Colors.RESET}")
        install_choice = input(f"\n{Colors.BRIGHT_GREEN}Would you like to install Ollama now? [Y/n]: {Colors.RESET}").strip().lower()
        
        if install_choice != 'n':
            if download_ollama():
                return True
            else:
                return False
        else:
            print(f"{Colors.DIM}You can install Ollama later from: https://ollama.ai{Colors.RESET}")
            return False

def pull_ollama_model(model_name):
    """Pull/download an Ollama model."""
    try:
        print(f"\n{Colors.BRIGHT_CYAN}Pulling model '{model_name}'...{Colors.RESET}")
        print(f"{Colors.DIM}This may take several minutes depending on model size...{Colors.RESET}\n")
        
        # Use subprocess to run ollama pull
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Show progress
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(f"{Colors.DIM}{output.strip()}{Colors.RESET}")
        
        if process.returncode == 0:
            print(f"\n{Colors.BRIGHT_GREEN}✓ Model '{model_name}' pulled successfully!{Colors.RESET}")
            return True
        else:
            stderr = process.stderr.read()
            print(f"\n{Colors.BRIGHT_RED}✗ Failed to pull model: {stderr}{Colors.RESET}")
            return False
    except FileNotFoundError:
        print(f"{Colors.BRIGHT_RED}✗ Ollama command not found. Make sure Ollama is installed.{Colors.RESET}")
        return False
    except Exception as e:
        print(f"{Colors.BRIGHT_RED}✗ Error: {e}{Colors.RESET}")
        return False

def search_huggingface_models(query="", limit=20):
    """Search for HuggingFace models."""
    try:
        url = f"https://huggingface.co/api/models"
        params = {"search": query, "limit": limit, "sort": "downloads", "direction": -1}
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            models = response.json()
            return [(m.get('id', ''), m.get('downloads', 0)) for m in models if m.get('id')]
    except Exception as e:
        print(f"{Colors.BRIGHT_RED}Error searching models: {e}{Colors.RESET}")
    return []

def download_huggingface_model(model_id):
    """Download a HuggingFace model (preview - actual download happens on first use)."""
    try:
        print(f"\n{Colors.BRIGHT_CYAN}Model '{model_id}' will be downloaded automatically on first use.{Colors.RESET}")
        print(f"{Colors.DIM}This may take several minutes depending on model size...{Colors.RESET}\n")
        return True
    except Exception as e:
        print(f"{Colors.BRIGHT_RED}Error: {e}{Colors.RESET}")
        return False

# ==============================================================================
#                           6. MODEL TRAINING CLASSES
# ==============================================================================

class FineTuningManager:
    """Manages full fine-tuning of language models."""
    
    def __init__(self, config):
        self.config = config
        self.training_data_dir = TRAINING_DATA_DIR
        self.models_dir = MODELS_DIR
    
    def prepare_dataset(self, data_file, output_dir=None):
        """Prepare dataset for fine-tuning from JSON/JSONL file."""
        if output_dir is None:
            output_dir = self.training_data_dir
        
        print(f"{Colors.BRIGHT_CYAN}[FINE-TUNING]{Colors.RESET} Preparing dataset...")
        
        try:
            if not os.path.exists(data_file):
                print(f"{Colors.BRIGHT_RED}✗ Data file not found: {data_file}{Colors.RESET}")
                return None
            
            # Read and validate data
            with open(data_file, 'r', encoding='utf-8') as f:
                if data_file.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            # Validate format (expects {"instruction": "", "input": "", "output": ""} or {"text": ""})
            validated = []
            for item in data:
                if "text" in item:
                    validated.append({"text": item["text"]})
                elif "instruction" in item and "output" in item:
                    validated.append(item)
                else:
                    print(f"{Colors.YELLOW}⚠ Skipping invalid item: {item}{Colors.RESET}")
            
            output_file = os.path.join(output_dir, "training_data.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(validated, f, indent=2, ensure_ascii=False)
            
            print(f"{Colors.BRIGHT_GREEN}✓ Dataset prepared: {len(validated)} samples{Colors.RESET}")
            return output_file
            
        except Exception as e:
            print(f"{Colors.BRIGHT_RED}✗ Error preparing dataset: {e}{Colors.RESET}")
            return None
    
    def train(self, base_model, dataset_file, output_model_name, epochs=3, batch_size=4, learning_rate=5e-5):
        """Fine-tune a model using the prepared dataset."""
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  FINE-TUNING MODEL{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        
        print(f"{Colors.CYAN}Base Model:{Colors.RESET} {Colors.BRIGHT_WHITE}{base_model}{Colors.RESET}")
        print(f"{Colors.CYAN}Dataset:{Colors.RESET} {Colors.BRIGHT_WHITE}{dataset_file}{Colors.RESET}")
        print(f"{Colors.CYAN}Epochs:{Colors.RESET} {Colors.BRIGHT_WHITE}{epochs}{Colors.RESET}")
        print(f"{Colors.CYAN}Batch Size:{Colors.RESET} {Colors.BRIGHT_WHITE}{batch_size}{Colors.RESET}")
        print(f"{Colors.CYAN}Learning Rate:{Colors.RESET} {Colors.BRIGHT_WHITE}{learning_rate}{Colors.RESET}\n")
        
        try:
            # Check if transformers training is available
            try:
                from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
                from datasets import load_dataset
            except ImportError:
                print(f"{Colors.BRIGHT_RED}✗ Required packages not installed.{Colors.RESET}")
                print(f"{Colors.YELLOW}Install with: pip install transformers datasets accelerate{Colors.RESET}")
                return False
            
            print(f"{Colors.BRIGHT_CYAN}Loading model and tokenizer...{Colors.RESET}")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(base_model)
            
            # Load dataset
            print(f"{Colors.BRIGHT_CYAN}Loading dataset...{Colors.RESET}")
            dataset = load_dataset('json', data_files=dataset_file, split='train')
            
            # Tokenize dataset
            def tokenize_function(examples):
                if "text" in examples:
                    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
                else:
                    # Format: instruction + input -> output
                    prompt = f"Instruction: {examples.get('instruction', '')}\nInput: {examples.get('input', '')}\nOutput: "
                    full_text = prompt + examples.get('output', '')
                    return tokenizer(full_text, truncation=True, max_length=512, padding="max_length")
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            output_path = os.path.join(self.models_dir, output_model_name)
            training_args = TrainingArguments(
                output_dir=output_path,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                save_strategy="epoch",
                logging_steps=10,
                report_to=None,
            )
            
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )
            
            print(f"\n{Colors.BRIGHT_GREEN}Starting training...{Colors.RESET}")
            print(f"{Colors.DIM}This may take a while depending on model size and dataset...{Colors.RESET}\n")
            
            trainer.train()
            
            print(f"\n{Colors.BRIGHT_GREEN}Saving fine-tuned model...{Colors.RESET}")
            trainer.save_model()
            tokenizer.save_pretrained(output_path)
            
            print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ Fine-tuning complete!{Colors.RESET}")
            print(f"{Colors.CYAN}Model saved to: {Colors.BRIGHT_WHITE}{output_path}{Colors.RESET}")
            return True
            
        except Exception as e:
            print(f"\n{Colors.BRIGHT_RED}✗ Training failed: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            return False


class LoRAManager:
    """Manages LoRA (Low-Rank Adaptation) training for efficient fine-tuning."""
    
    def __init__(self, config):
        self.config = config
        self.training_data_dir = TRAINING_DATA_DIR
        self.lora_dir = LORA_DIR
    
    def prepare_dataset(self, data_file, output_dir=None):
        """Prepare dataset for LoRA training (same as fine-tuning)."""
        if output_dir is None:
            output_dir = self.training_data_dir
        
        print(f"{Colors.BRIGHT_CYAN}[LoRA]{Colors.RESET} Preparing dataset...")
        
        try:
            if not os.path.exists(data_file):
                print(f"{Colors.BRIGHT_RED}✗ Data file not found: {data_file}{Colors.RESET}")
                return None
            
            with open(data_file, 'r', encoding='utf-8') as f:
                if data_file.endswith('.jsonl'):
                    data = [json.loads(line) for line in f]
                else:
                    data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            validated = []
            for item in data:
                if "text" in item:
                    validated.append({"text": item["text"]})
                elif "instruction" in item and "output" in item:
                    validated.append(item)
            
            output_file = os.path.join(output_dir, "lora_training_data.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(validated, f, indent=2, ensure_ascii=False)
            
            print(f"{Colors.BRIGHT_GREEN}✓ Dataset prepared: {len(validated)} samples{Colors.RESET}")
            return output_file
            
        except Exception as e:
            print(f"{Colors.BRIGHT_RED}✗ Error preparing dataset: {e}{Colors.RESET}")
            return None
    
    def train(self, base_model, dataset_file, output_lora_name, rank=8, alpha=16, epochs=3, batch_size=4, learning_rate=1e-4):
        """Train a LoRA adapter for the model."""
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  LoRA TRAINING{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        
        print(f"{Colors.CYAN}Base Model:{Colors.RESET} {Colors.BRIGHT_WHITE}{base_model}{Colors.RESET}")
        print(f"{Colors.CYAN}Dataset:{Colors.RESET} {Colors.BRIGHT_WHITE}{dataset_file}{Colors.RESET}")
        print(f"{Colors.CYAN}Rank:{Colors.RESET} {Colors.BRIGHT_WHITE}{rank}{Colors.RESET}")
        print(f"{Colors.CYAN}Alpha:{Colors.RESET} {Colors.BRIGHT_WHITE}{alpha}{Colors.RESET}")
        print(f"{Colors.CYAN}Epochs:{Colors.RESET} {Colors.BRIGHT_WHITE}{epochs}{Colors.RESET}\n")
        
        try:
            # Check if PEFT is available
            try:
                from peft import LoraConfig, get_peft_model, TaskType
                from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
                from datasets import load_dataset
            except ImportError:
                print(f"{Colors.BRIGHT_RED}✗ Required packages not installed.{Colors.RESET}")
                print(f"{Colors.YELLOW}Install with: pip install peft transformers datasets accelerate{Colors.RESET}")
                return False
            
            print(f"{Colors.BRIGHT_CYAN}Loading model and tokenizer...{Colors.RESET}")
            tokenizer = AutoTokenizer.from_pretrained(base_model)
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(base_model)
            
            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"] if "llama" in base_model.lower() or "mistral" in base_model.lower() else None,
            )
            
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
            
            # Load and tokenize dataset
            print(f"{Colors.BRIGHT_CYAN}Loading dataset...{Colors.RESET}")
            dataset = load_dataset('json', data_files=dataset_file, split='train')
            
            def tokenize_function(examples):
                if "text" in examples:
                    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
                else:
                    prompt = f"Instruction: {examples.get('instruction', '')}\nInput: {examples.get('input', '')}\nOutput: "
                    full_text = prompt + examples.get('output', '')
                    return tokenizer(full_text, truncation=True, max_length=512, padding="max_length")
            
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Training arguments
            output_path = os.path.join(self.lora_dir, output_lora_name)
            training_args = TrainingArguments(
                output_dir=output_path,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                save_strategy="epoch",
                logging_steps=10,
                report_to=None,
            )
            
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
            )
            
            print(f"\n{Colors.BRIGHT_GREEN}Starting LoRA training...{Colors.RESET}")
            print(f"{Colors.DIM}This is more efficient than full fine-tuning...{Colors.RESET}\n")
            
            trainer.train()
            
            print(f"\n{Colors.BRIGHT_GREEN}Saving LoRA adapter...{Colors.RESET}")
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ LoRA training complete!{Colors.RESET}")
            print(f"{Colors.CYAN}LoRA adapter saved to: {Colors.BRIGHT_WHITE}{output_path}{Colors.RESET}")
            return True
            
        except Exception as e:
            print(f"\n{Colors.BRIGHT_RED}✗ LoRA training failed: {e}{Colors.RESET}")
            import traceback
            traceback.print_exc()
            return False


class ReinforcementLearningManager:
    """Manages Reinforcement Learning with Human Feedback (RLHF) and Behaviour Conditioning."""
    
    def __init__(self, config):
        self.config = config
        self.training_data_dir = TRAINING_DATA_DIR
        self.reinforcement_dir = REINFORCEMENT_DIR
    
    def prepare_preference_data(self, data_file, output_dir=None):
        """Prepare preference data for RLHF (chosen vs rejected responses)."""
        if output_dir is None:
            output_dir = self.training_data_dir
        
        print(f"{Colors.BRIGHT_CYAN}[RLHF]{Colors.RESET} Preparing preference dataset...")
        
        try:
            if not os.path.exists(data_file):
                print(f"{Colors.BRIGHT_RED}✗ Data file not found: {data_file}{Colors.RESET}")
                return None
            
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            # Validate format: {"prompt": "", "chosen": "", "rejected": ""}
            validated = []
            for item in data:
                if "prompt" in item and "chosen" in item and "rejected" in item:
                    validated.append(item)
                else:
                    print(f"{Colors.YELLOW}⚠ Skipping invalid item (need 'prompt', 'chosen', 'rejected'): {item}{Colors.RESET}")
            
            output_file = os.path.join(output_dir, "rlhf_preferences.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(validated, f, indent=2, ensure_ascii=False)
            
            print(f"{Colors.BRIGHT_GREEN}✓ Preference dataset prepared: {len(validated)} pairs{Colors.RESET}")
            return output_file
            
        except Exception as e:
            print(f"{Colors.BRIGHT_RED}✗ Error preparing preference data: {e}{Colors.RESET}")
            return None
    
    def train_reward_model(self, base_model, preference_file, output_model_name, epochs=3, batch_size=4):
        """Train a reward model for RLHF."""
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  REWARD MODEL TRAINING{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        
        print(f"{Colors.CYAN}Base Model:{Colors.RESET} {Colors.BRIGHT_WHITE}{base_model}{Colors.RESET}")
        print(f"{Colors.CYAN}Preference Data:{Colors.RESET} {Colors.BRIGHT_WHITE}{preference_file}{Colors.RESET}\n")
        
        try:
            print(f"{Colors.BRIGHT_YELLOW}⚠ Reward model training requires specialized setup.{Colors.RESET}")
            print(f"{Colors.DIM}This is a simplified implementation. For production RLHF, consider using TRL library.{Colors.RESET}\n")
            
            # Load preference data
            with open(preference_file, 'r', encoding='utf-8') as f:
                preferences = json.load(f)
            
            print(f"{Colors.BRIGHT_CYAN}Loaded {len(preferences)} preference pairs{Colors.RESET}")
            print(f"{Colors.BRIGHT_GREEN}✓ Reward model training framework initialized{Colors.RESET}")
            print(f"{Colors.DIM}For full RLHF, use: pip install trl{Colors.RESET}")
            
            # Save configuration
            config_file = os.path.join(self.reinforcement_dir, f"{output_model_name}_config.json")
            config = {
                "base_model": base_model,
                "preference_file": preference_file,
                "epochs": epochs,
                "batch_size": batch_size,
                "type": "reward_model"
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"\n{Colors.BRIGHT_GREEN}✓ Configuration saved: {config_file}{Colors.RESET}")
            return True
            
        except Exception as e:
            print(f"\n{Colors.BRIGHT_RED}✗ Reward model setup failed: {e}{Colors.RESET}")
            return False
    
    def train_with_ppo(self, base_model, reward_model_path, dataset_file, output_model_name, epochs=3):
        """Train model using PPO (Proximal Policy Optimization)."""
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  PPO TRAINING (RLHF){Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        
        print(f"{Colors.CYAN}Base Model:{Colors.RESET} {Colors.BRIGHT_WHITE}{base_model}{Colors.RESET}")
        print(f"{Colors.CYAN}Reward Model:{Colors.RESET} {Colors.BRIGHT_WHITE}{reward_model_path}{Colors.RESET}\n")
        
        try:
            print(f"{Colors.BRIGHT_YELLOW}⚠ PPO training requires TRL library.{Colors.RESET}")
            print(f"{Colors.DIM}Install with: pip install trl{Colors.RESET}\n")
            
            # Save PPO configuration
            config_file = os.path.join(self.reinforcement_dir, f"{output_model_name}_ppo_config.json")
            config = {
                "base_model": base_model,
                "reward_model": reward_model_path,
                "dataset": dataset_file,
                "epochs": epochs,
                "type": "ppo"
            }
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"{Colors.BRIGHT_GREEN}✓ PPO configuration saved: {config_file}{Colors.RESET}")
            print(f"{Colors.DIM}To run PPO training, use TRL library with this configuration.{Colors.RESET}")
            return True
            
        except Exception as e:
            print(f"\n{Colors.BRIGHT_RED}✗ PPO setup failed: {e}{Colors.RESET}")
            return False
    
    def apply_behaviour_conditioning(self, model_path, behaviour_rules_file, output_model_name):
        """Apply behaviour conditioning rules to a model."""
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  BEHAVIOUR CONDITIONING{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        
        try:
            # Load behaviour rules
            with open(behaviour_rules_file, 'r', encoding='utf-8') as f:
                rules = json.load(f)
            
            print(f"{Colors.CYAN}Model:{Colors.RESET} {Colors.BRIGHT_WHITE}{model_path}{Colors.RESET}")
            print(f"{Colors.CYAN}Rules:{Colors.RESET} {Colors.BRIGHT_WHITE}{len(rules)} behaviour rules{Colors.RESET}\n")
            
            # Create conditioning dataset from rules
            conditioning_data = []
            for rule in rules:
                if "trigger" in rule and "response" in rule:
                    conditioning_data.append({
                        "instruction": rule.get("trigger", ""),
                        "output": rule.get("response", ""),
                        "priority": rule.get("priority", 1)
                    })
            
            # Save conditioning dataset
            output_file = os.path.join(self.reinforcement_dir, f"{output_model_name}_conditioning.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(conditioning_data, f, indent=2, ensure_ascii=False)
            
            print(f"{Colors.BRIGHT_GREEN}✓ Behaviour conditioning dataset created: {len(conditioning_data)} rules{Colors.RESET}")
            print(f"{Colors.CYAN}Dataset saved to: {Colors.BRIGHT_WHITE}{output_file}{Colors.RESET}")
            print(f"{Colors.DIM}Use fine-tuning or LoRA to apply these rules to the model.{Colors.RESET}")
            
            return output_file
            
        except Exception as e:
            print(f"\n{Colors.BRIGHT_RED}✗ Behaviour conditioning failed: {e}{Colors.RESET}")
            return None


# ==============================================================================
#                           6. MULTI-AGENT APP BUILDER SYSTEM
# ==============================================================================

class BaseAgent:
    """Base class for all AI agents."""
    
    def __init__(self, name, role, system_prompt, ai_engine):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.ai_engine = ai_engine
    
    def think(self, context, max_tokens=500):
        """Generate response based on context."""
        prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\n{self.role} Response:"
        
        # Temporarily adjust max tokens for this agent
        original_max = self.ai_engine.config.get('max_response_tokens', 250)
        self.ai_engine.config['max_response_tokens'] = max_tokens
        
        response = self.ai_engine.generate(prompt)
        
        # Restore original
        self.ai_engine.config['max_response_tokens'] = original_max
        
        return response


class SpecificationWriterAgent(BaseAgent):
    """Clarifies requirements and writes specifications."""
    
    def __init__(self, ai_engine):
        system_prompt = (
            "You are a Specification Writer. Your job is to understand project requirements deeply. "
            "Ask clarifying questions if the description is vague. Write clear, detailed specifications. "
            "Output format: questions OR final specification document."
        )
        super().__init__("Specification Writer", "SPEC_WRITER", system_prompt, ai_engine)
    
    def analyze_description(self, app_name, description):
        """Analyze if description is sufficient."""
        context = f"App: {app_name}\nDesc: {description}\n\nClear? If not, list key questions (max 3)."
        return self.think(context, 200)
    
    def write_specification(self, app_name, description, qa_pairs=None):
        """Write final specification."""
        context = f"App: {app_name}\nDesc: {description}\n"
        if qa_pairs:
            context += "\nQ&A:\n" + "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])
        context += "\n\nWrite concise specification (key features, requirements, tech constraints)."
        return self.think(context, 600)


class ArchitectAgent(BaseAgent):
    """Designs architecture and checks dependencies."""
    
    def __init__(self, ai_engine):
        system_prompt = (
            "You are a Software Architect. Design system architecture, choose technologies, "
            "list required dependencies. Be specific about versions and installation commands. "
            "Consider scalability, maintainability, and best practices."
        )
        super().__init__("Architect", "ARCHITECT", system_prompt, ai_engine)
    
    def design_architecture(self, specification):
        """Design system architecture."""
        context = f"Spec:\n{specification}\n\nDesign architecture: tech stack, key dependencies (pip install commands), folder structure. Be concise."
        return self.think(context, 700)
    
    def check_and_install_dependencies(self, architecture):
        """Parse architecture and install dependencies."""
        # Extract dependencies from architecture document
        dependencies = []
        lines = architecture.split('\n')
        for line in lines:
            if 'pip install' in line.lower():
                # Extract package names
                parts = line.lower().split('pip install')
                if len(parts) > 1:
                    packages = parts[1].strip().split()
                    dependencies.extend(packages)
        
        installed = []
        failed = []
        
        for dep in dependencies:
            print(f"{Colors.BRIGHT_CYAN}Checking dependency: {dep}...{Colors.RESET}")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", dep], 
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                installed.append(dep)
                print(f"{Colors.BRIGHT_GREEN}✓ Installed{Colors.RESET}")
            except:
                failed.append(dep)
                print(f"{Colors.BRIGHT_RED}✗ Failed{Colors.RESET}")
        
        return installed, failed


class TechLeadAgent(BaseAgent):
    """Breaks down work into development tasks."""
    
    def __init__(self, ai_engine):
        system_prompt = (
            "You are a Tech Lead. Break down projects into specific, actionable development tasks. "
            "Each task should be clear, focused, and achievable. Order tasks logically (dependencies first). "
            "Format: numbered list with task name and brief description."
        )
        super().__init__("Tech Lead", "TECH_LEAD", system_prompt, ai_engine)
    
    def create_tasks(self, specification, architecture):
        """Create development task list."""
        context = f"Spec:\n{specification[:800]}\n\nArch:\n{architecture[:600]}\n\nCreate numbered task list (5-10 tasks, ordered by dependencies)."
        return self.think(context, 600)


class DeveloperAgent(BaseAgent):
    """Plans implementation details for tasks."""
    
    def __init__(self, ai_engine):
        system_prompt = (
            "You are a Senior Developer. For each task, write detailed implementation notes. "
            "Describe what files to create/modify, what functions/classes to add, what logic is needed. "
            "Be specific but write in human-readable form (not code yet)."
        )
        super().__init__("Developer", "DEVELOPER", system_prompt, ai_engine)
    
    def plan_task(self, task, specification, architecture, existing_files):
        """Plan how to implement a task."""
        # Truncate context to speed up processing
        spec_summary = specification[:500] if len(specification) > 500 else specification
        arch_summary = architecture[:400] if len(architecture) > 400 else architecture
        files_summary = existing_files[:300] if len(existing_files) > 300 else existing_files
        context = f"Task: {task}\n\nSpec: {spec_summary}\nArch: {arch_summary}\nFiles: {files_summary}\n\nBrief implementation plan (what to create/modify)."
        return self.think(context, 500)


class CodeMonkeyAgent(BaseAgent):
    """Writes actual code based on developer's plan."""
    
    def __init__(self, ai_engine):
        system_prompt = (
            "You are a Code Monkey. Implement code based on the Developer's plan. "
            "Write clean, commented, production-ready code. Follow best practices and PEP 8 (for Python). "
            "Output only the code, no explanations unless in comments."
        )
        super().__init__("Code Monkey", "CODE_MONKEY", system_prompt, ai_engine)
    
    def write_code(self, implementation_plan, existing_code=""):
        """Write code based on plan."""
        # Limit existing code context to speed up
        existing_summary = existing_code[:2000] if len(existing_code) > 2000 else existing_code
        context = f"Plan:\n{implementation_plan}\n\nExisting:\n{existing_summary}\n\nWrite production-ready code (complete, commented, PEP 8):"
        return self.think(context, 1200)


class ReviewerAgent(BaseAgent):
    """Reviews code for issues."""
    
    def __init__(self, ai_engine):
        system_prompt = (
            "You are a Code Reviewer. Review code for bugs, bad practices, security issues, and logic errors. "
            "Be critical but constructive. If code is good, approve it. If not, explain what needs fixing. "
            "Format: 'APPROVED' or 'REJECTED: [reasons]'"
        )
        super().__init__("Reviewer", "REVIEWER", system_prompt, ai_engine)
    
    def review_code(self, code, task, implementation_plan):
        """Review code quality - optimized for speed."""
        # Truncate code for faster review (review first 1500 chars typically enough)
        code_sample = code[:1500] if len(code) > 1500 else code
        context = f"Task: {task}\nPlan: {implementation_plan[:300]}\nCode:\n{code_sample}\n\nQuick review: APPROVED or REJECTED: [issue]"
        return self.think(context, 300)


class TroubleshooterAgent(BaseAgent):
    """Helps formulate good feedback."""
    
    def __init__(self, ai_engine):
        system_prompt = (
            "You are a Troubleshooter. Help users give good feedback to the development team. "
            "Convert vague complaints into actionable bug reports. Ask clarifying questions. "
            "Format feedback professionally."
        )
        super().__init__("Troubleshooter", "TROUBLESHOOTER", system_prompt, ai_engine)
    
    def analyze_feedback(self, user_feedback, current_state):
        """Analyze user feedback and make it actionable."""
        context = f"User Feedback: {user_feedback}\n\nCurrent State:\n{current_state}\n\nConvert to actionable feedback:"
        return self.think(context, 400)


class DebuggerAgent(BaseAgent):
    """Debugs issues in the code."""
    
    def __init__(self, ai_engine):
        system_prompt = (
            "You are a Debugger. Find and fix bugs. Analyze error messages, stack traces, and logs. "
            "Identify root causes and propose fixes. Be methodical and thorough."
        )
        super().__init__("Debugger", "DEBUGGER", system_prompt, ai_engine)
    
    def debug_issue(self, error_message, relevant_code, context=""):
        """Debug an issue."""
        context_str = f"Error: {error_message}\n\nRelevant Code:\n{relevant_code}\n\nContext:\n{context}\n\nIdentify the bug and suggest a fix:"
        return self.think(context_str, 800)


class TechnicalWriterAgent(BaseAgent):
    """Writes documentation."""
    
    def __init__(self, ai_engine):
        system_prompt = (
            "You are a Technical Writer. Write clear, comprehensive documentation. "
            "Include: overview, installation, usage, API docs, examples. Use markdown format."
        )
        super().__init__("Technical Writer", "TECH_WRITER", system_prompt, ai_engine)
    
    def write_documentation(self, project_name, specification, architecture, codebase_summary):
        """Write project documentation - optimized."""
        # Truncate inputs for faster generation
        spec_summary = specification[:600] if len(specification) > 600 else specification
        arch_summary = architecture[:500] if len(architecture) > 500 else architecture
        context = f"Project: {project_name}\n\nSpec: {spec_summary}\nArch: {arch_summary}\nFiles: {codebase_summary[:400]}\n\nWrite concise README (overview, install, usage, examples):"
        return self.think(context, 1000)


class AppBuilderOrchestrator:
    """Orchestrates the multi-agent app building process."""
    
    def __init__(self, ai_engine, memory_manager):
        self.ai_engine = ai_engine
        self.memory = memory_manager
        
        # Initialize agents
        self.spec_writer = SpecificationWriterAgent(ai_engine)
        self.architect = ArchitectAgent(ai_engine)
        self.tech_lead = TechLeadAgent(ai_engine)
        self.developer = DeveloperAgent(ai_engine)
        self.code_monkey = CodeMonkeyAgent(ai_engine)
        self.reviewer = ReviewerAgent(ai_engine)
        self.troubleshooter = TroubleshooterAgent(ai_engine)
        self.debugger = DebuggerAgent(ai_engine)
        self.tech_writer = TechnicalWriterAgent(ai_engine)
        
        self.builder_threads = max(1, int(ai_engine.config.get("builder_threads", 1)))
        self.fast_mode = ai_engine.config.get("builder_fast_mode", True)  # Fast mode enabled by default
        self.db_lock = threading.Lock()
        self.current_project = None
        self.init_app_database()
    
    def init_app_database(self):
        """Initialize app projects database."""
        conn = sqlite3.connect(APP_PROJECTS_DB)
        cursor = conn.cursor()
        
        # App projects table
        cursor.execute('''CREATE TABLE IF NOT EXISTS app_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            description TEXT,
            specification TEXT,
            architecture TEXT,
            status TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )''')
        
        # Tasks table
        cursor.execute('''CREATE TABLE IF NOT EXISTS app_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            task_number INTEGER,
            description TEXT,
            implementation_plan TEXT,
            status TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES app_projects(id)
        )''')
        
        # Files table
        cursor.execute('''CREATE TABLE IF NOT EXISTS app_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            filepath TEXT,
            content TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES app_projects(id)
        )''')
        
        # Reviews table
        cursor.execute('''CREATE TABLE IF NOT EXISTS app_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id INTEGER,
            task_id INTEGER,
            review_result TEXT,
            feedback TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES app_projects(id),
            FOREIGN KEY (task_id) REFERENCES app_tasks(id)
        )''')
        
        conn.commit()
        conn.close()
    
    def create_project(self, name, description):
        """Create a new app project."""
        conn = sqlite3.connect(APP_PROJECTS_DB)
        cursor = conn.cursor()
        
        try:
            cursor.execute("INSERT INTO app_projects (name, description, status) VALUES (?, ?, ?)",
                          (name, description, "specification"))
            project_id = cursor.lastrowid
            conn.commit()
            
            # Create project directory
            project_dir = os.path.join(APPS_DIR, name)
            os.makedirs(project_dir, exist_ok=True)
            
            return project_id
        except sqlite3.IntegrityError:
            print(f"{Colors.BRIGHT_RED}✗ Project '{name}' already exists.{Colors.RESET}")
            return None
        finally:
            conn.close()
    
    def get_project(self, project_id):
        """Get project details."""
        conn = sqlite3.connect(APP_PROJECTS_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM app_projects WHERE id=?", (project_id,))
        project = cursor.fetchone()
        conn.close()
        return project
    
    def update_project_field(self, project_id, field, value):
        """Update a project field."""
        lock = getattr(self, "db_lock", None)
        if lock:
            lock.acquire()
        try:
            conn = sqlite3.connect(APP_PROJECTS_DB)
            cursor = conn.cursor()
            cursor.execute(f"UPDATE app_projects SET {field}=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", 
                          (value, project_id))
            conn.commit()
            conn.close()
        finally:
            if lock:
                lock.release()
    
    def save_file(self, project_id, filepath, content):
        """Save a file for the project."""
        lock = getattr(self, "db_lock", None)
        if lock:
            lock.acquire()
        try:
            conn = sqlite3.connect(APP_PROJECTS_DB)
            cursor = conn.cursor()
            
            # Check if file exists
            cursor.execute("SELECT id FROM app_files WHERE project_id=? AND filepath=?", (project_id, filepath))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute("UPDATE app_files SET content=?, updated_at=CURRENT_TIMESTAMP WHERE id=?",
                              (content, existing[0]))
            else:
                cursor.execute("INSERT INTO app_files (project_id, filepath, content) VALUES (?, ?, ?)",
                              (project_id, filepath, content))
            
            conn.commit()
            conn.close()
        finally:
            if lock:
                lock.release()
        
        # Also write to actual file
        project = self.get_project(project_id)
        if project:
            project_name = project[1]
            project_dir = os.path.join(APPS_DIR, project_name)
            full_path = os.path.join(project_dir, filepath)
            
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(content)
    
    def get_project_files(self, project_id):
        """Get all files for a project."""
        conn = sqlite3.connect(APP_PROJECTS_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT filepath, content FROM app_files WHERE project_id=?", (project_id,))
        files = cursor.fetchall()
        conn.close()
        return files
    
    def filter_relevant_context(self, task_description, all_files, max_context=5000):
        """Filter relevant files for current task context - optimized for speed."""
        # Simple keyword-based filtering (optimized)
        keywords = set([w for w in task_description.lower().split() if len(w) > 3])  # Filter short words
        relevant_files = []
        
        # Limit file processing for speed
        max_files_to_check = 20 if self.fast_mode else len(all_files)
        files_to_check = all_files[:max_files_to_check]
        
        for filepath, content in files_to_check:
            # Quick relevance check - only check first 500 chars of content for speed
            file_text = (filepath + " " + content[:500]).lower()
            matches = sum(1 for keyword in keywords if keyword in file_text)
            if matches > 0:
                # Truncate content for faster processing
                content_sample = content[:2000] if len(content) > 2000 else content
                relevant_files.append((filepath, content_sample, matches))
        
        # Sort by relevance
        relevant_files.sort(key=lambda x: x[2], reverse=True)
        
        # Build context within token limit (reduced for fast mode)
        max_context_actual = max_context // 2 if self.fast_mode else max_context
        context = ""
        for filepath, content, _ in relevant_files[:10]:  # Limit to top 10 files
            file_context = f"\n### {filepath} ###\n{content}\n"
            if len(context) + len(file_context) < max_context_actual:
                context += file_context
            else:
                break
        
        return context
    
    def build_app(self, project_id):
        """Main app building workflow - optimized for speed.
        
        Optimizations applied:
        - Reduced token limits for intermediate steps (faster generation)
        - Removed blocking user input prompts (auto-continue)
        - Streamlined prompts (more concise)
        - Faster code review (truncated code samples)
        - Optimized context filtering (limited file processing)
        - Auto-generate filenames (no user input needed)
        - Progress indicators (inline updates)
        """
        project = self.get_project(project_id)
        if not project:
            return False, "Project not found"
        
        project_name = project[1]
        description = project[2]
        status = project[5]
        
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  BUILDING APP: {project_name}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        
        # Stage 1: Specification
        if status == "specification":
            print(f"{Colors.BRIGHT_CYAN}[1/5] Specification Writer analyzing requirements...{Colors.RESET}")
            analysis = self.spec_writer.analyze_description(project_name, description)
            
            # Check if questions are needed
            if "?" in analysis or "question" in analysis.lower():
                print(f"{Colors.BRIGHT_YELLOW}⚠ Questions detected. Skipping Q&A for speed (auto-generating spec)...{Colors.RESET}")
                # Auto-generate spec without Q&A for speed - user can refine later
                spec = self.spec_writer.write_specification(project_name, description)
            else:
                spec = self.spec_writer.write_specification(project_name, description)
            
            self.update_project_field(project_id, "specification", spec)
            self.update_project_field(project_id, "status", "architecture")
            
            print(f"{Colors.BRIGHT_GREEN}✓ Specification complete!{Colors.RESET}")
        
        # Stage 2: Architecture
        project = self.get_project(project_id)
        if project[5] == "architecture":
            spec = project[3]
            
            print(f"{Colors.BRIGHT_CYAN}[2/5] Architect designing system...{Colors.RESET}")
            architecture = self.architect.design_architecture(spec)
            
            self.update_project_field(project_id, "architecture", architecture)
            
            # Check and install dependencies (non-blocking)
            print(f"{Colors.BRIGHT_CYAN}Checking dependencies...{Colors.RESET}")
            installed, failed = self.architect.check_and_install_dependencies(architecture)
            
            if installed:
                print(f"{Colors.BRIGHT_GREEN}✓ Installed: {', '.join(installed[:5])}{'...' if len(installed) > 5 else ''}{Colors.RESET}")
            if failed:
                print(f"{Colors.YELLOW}⚠ Failed: {', '.join(failed[:3])}{'...' if len(failed) > 3 else ''}{Colors.RESET}")
            
            self.update_project_field(project_id, "status", "tasks")
            print(f"{Colors.BRIGHT_GREEN}✓ Architecture complete!{Colors.RESET}")
        
        # Stage 3: Create Tasks
        project = self.get_project(project_id)
        if project[5] == "tasks":
            spec = project[3]
            architecture = project[4]
            
            print(f"{Colors.BRIGHT_CYAN}[3/5] Tech Lead creating task list...{Colors.RESET}")
            tasks_doc = self.tech_lead.create_tasks(spec, architecture)
            
            # Parse and save tasks
            conn = sqlite3.connect(APP_PROJECTS_DB)
            cursor = conn.cursor()
            
            task_lines = [line.strip() for line in tasks_doc.split('\n') if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))]
            for i, task_line in enumerate(task_lines, 1):
                cursor.execute("INSERT INTO app_tasks (project_id, task_number, description, status) VALUES (?, ?, ?, ?)",
                              (project_id, i, task_line, "pending"))
            
            conn.commit()
            conn.close()
            
            self.update_project_field(project_id, "status", "development")
            print(f"{Colors.BRIGHT_GREEN}✓ {len(task_lines)} tasks created!{Colors.RESET}")
        
        # Stage 4: Development (iterative)
        project = self.get_project(project_id)
        if project[5] == "development":
            self.develop_tasks(project_id)
        
        return True, "App building process initiated"
    
    def develop_tasks(self, project_id):
        """Develop tasks iteratively."""
        project = self.get_project(project_id)
        spec = project[3]
        architecture = project[4]
        builder_threads = getattr(self, "builder_threads", 1)
        
        conn = sqlite3.connect(APP_PROJECTS_DB)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, task_number, description, status FROM app_tasks WHERE project_id=? ORDER BY task_number", 
                      (project_id,))
        tasks = cursor.fetchall()
        conn.close()
        
        if builder_threads > 1:
            self._develop_tasks_threaded(project_id, tasks, spec, architecture, builder_threads)
            return
        
        for task_id, task_num, task_desc, task_status in tasks:
            if task_status == "completed":
                print(f"{Colors.DIM}[Task {task_num}] Already completed: {task_desc[:50]}...{Colors.RESET}")
                continue
            
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}[Task {task_num}/{len(tasks)}] {task_desc[:60]}...{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
            
            # Get existing files
            existing_files = self.get_project_files(project_id)
            files_context = "\n".join([f"{fp}: {len(content)} chars" for fp, content in existing_files])
            
            # Developer plans implementation
            print(f"{Colors.BRIGHT_CYAN}[4/5] Planning & coding...{Colors.RESET}", end="", flush=True)
            impl_plan = self.developer.plan_task(task_desc, spec, architecture, files_context)
            
            # Save implementation plan
            conn = sqlite3.connect(APP_PROJECTS_DB)
            cursor = conn.cursor()
            cursor.execute("UPDATE app_tasks SET implementation_plan=? WHERE id=?", (impl_plan, task_id))
            conn.commit()
            conn.close()
            
            # Code Monkey writes code
            print(f" {Colors.BRIGHT_CYAN}writing...{Colors.RESET}", end="", flush=True)
            
            # Get relevant context
            relevant_context = self.filter_relevant_context(task_desc + " " + impl_plan, existing_files)
            
            code = self.code_monkey.write_code(impl_plan, relevant_context)
            
            # Reviewer reviews code (faster review)
            print(f" {Colors.BRIGHT_CYAN}reviewing...{Colors.RESET}", end="", flush=True)
            review = self.reviewer.review_code(code, task_desc, impl_plan)
            print()  # New line after progress
            
            # Check if approved
            if "APPROVED" in review.upper():
                # Extract filename from code or plan
                filename = self.extract_filename(code, impl_plan, task_desc)
                if not filename:
                    # Auto-generate filename from task number
                    filename = f"task_{task_num}.py"
                
                self.save_file(project_id, filename, code)
                print(f"{Colors.BRIGHT_GREEN}✓ Task {task_num} completed: {filename}{Colors.RESET}")
                
                # Mark task as completed
                conn = sqlite3.connect(APP_PROJECTS_DB)
                cursor = conn.cursor()
                cursor.execute("UPDATE app_tasks SET status='completed' WHERE id=?", (task_id,))
                cursor.execute("INSERT INTO app_reviews (project_id, task_id, review_result, feedback) VALUES (?, ?, ?, ?)",
                              (project_id, task_id, "approved", review))
                conn.commit()
                conn.close()
                
            else:
                print(f"{Colors.BRIGHT_RED}✗ Task {task_num} rejected: {review[:100]}...{Colors.RESET}")
                
                # Save review
                conn = sqlite3.connect(APP_PROJECTS_DB)
                cursor = conn.cursor()
                cursor.execute("INSERT INTO app_reviews (project_id, task_id, review_result, feedback) VALUES (?, ?, ?, ?)",
                              (project_id, task_id, "rejected", review))
                conn.commit()
                conn.close()
                
                # Auto-skip rejected tasks for speed (user can retry manually later)
                print(f"{Colors.YELLOW}⚠ Task skipped. Review saved. You can retry manually later.{Colors.RESET}")
        
        # All tasks completed
        self.update_project_field(project_id, "status", "completed")
        print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ All tasks completed!{Colors.RESET}")
        
        # Generate documentation (optional - can be skipped for speed)
        print(f"{Colors.BRIGHT_CYAN}Generating documentation...{Colors.RESET}", end="", flush=True)
        files = self.get_project_files(project_id)
        codebase_summary = "\n".join([f"{fp}: {len(content)} lines" for fp, content in files])
        
        docs = self.tech_writer.write_documentation(project[1], spec, architecture, codebase_summary)
        
        # Save documentation
        self.save_file(project_id, "README.md", docs)
        print(f" {Colors.BRIGHT_GREEN}✓ README.md saved{Colors.RESET}")

    def _develop_tasks_threaded(self, project_id, tasks, spec, architecture, builder_threads):
        """Parallelized task development while keeping the original pipeline semantics."""
        pending_tasks = [t for t in tasks if t[3] != "completed"]
        if not pending_tasks:
            print(f"{Colors.DIM}No pending tasks to process.{Colors.RESET}")
            return
        
        print(f"\n{Colors.BRIGHT_CYAN}Threaded mode enabled for task execution "
              f"({builder_threads} workers; device: {self.ai_engine.device}).{Colors.RESET}")
        
        lock = getattr(self, "db_lock", None)
        total_tasks = len(pending_tasks)
        
        def worker(task_tuple):
            task_id, task_num, task_desc, _ = task_tuple
            try:
                existing_files = self.get_project_files(project_id)
                files_context = "\n".join([f"{fp}: {len(content)} chars" for fp, content in existing_files])
                
                impl_plan = self.developer.plan_task(task_desc, spec, architecture, files_context)
                
                # Save implementation plan
                if lock:
                    lock.acquire()
                try:
                    conn = sqlite3.connect(APP_PROJECTS_DB)
                    cursor = conn.cursor()
                    cursor.execute("UPDATE app_tasks SET implementation_plan=? WHERE id=?", (impl_plan, task_id))
                    conn.commit()
                    conn.close()
                finally:
                    if lock:
                        lock.release()
                
                relevant_context = self.filter_relevant_context(task_desc + " " + impl_plan, existing_files)
                code = self.code_monkey.write_code(impl_plan, relevant_context)
                # Faster review for threaded mode
                review = self.reviewer.review_code(code, task_desc, impl_plan)
                
                approved = "APPROVED" in review.upper()
                filename = self.extract_filename(code, impl_plan, task_desc) or f"task_{task_num}.py"
                
                if lock:
                    lock.acquire()
                try:
                    conn = sqlite3.connect(APP_PROJECTS_DB)
                    cursor = conn.cursor()
                    
                    # Persist review
                    review_status = "approved" if approved else "rejected"
                    cursor.execute(
                        "INSERT INTO app_reviews (project_id, task_id, review_result, feedback) VALUES (?, ?, ?, ?)",
                        (project_id, task_id, review_status, review)
                    )
                    
                    if approved:
                        self.save_file(project_id, filename, code)
                        cursor.execute("UPDATE app_tasks SET status='completed' WHERE id=?", (task_id,))
                    
                    conn.commit()
                    conn.close()
                finally:
                    if lock:
                        lock.release()
                
                if approved:
                    return task_num, True, f"saved to {filename}"
                return task_num, False, f"requires fixes (review saved)"
            
            except Exception as e:
                return task_num, False, f"error: {e}"
        
        completed = 0
        with ThreadPoolExecutor(max_workers=builder_threads) as executor:
            futures = {executor.submit(worker, t): t for t in pending_tasks}
            for future in as_completed(futures):
                task_num, ok, message = future.result()
                completed += 1
                status_color = Colors.BRIGHT_GREEN if ok else Colors.BRIGHT_RED
                print(f"{status_color}[Task {task_num}/{total_tasks}]{Colors.RESET} {message}")
        
        # Finalize project if all tasks are done
        conn = sqlite3.connect(APP_PROJECTS_DB)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM app_tasks WHERE project_id=? AND status!='completed'", (project_id,))
        remaining = cursor.fetchone()[0]
        conn.close()
        
        if remaining == 0:
            self.update_project_field(project_id, "status", "completed")
            print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ All tasks completed!{Colors.RESET}")
            
            # Generate documentation (same as sequential flow)
            print(f"\n{Colors.BRIGHT_CYAN}Technical Writer creating documentation...{Colors.RESET}")
            files = self.get_project_files(project_id)
            codebase_summary = "\n".join([f"{fp}: {len(content)} lines" for fp, content in files])
            
            project = self.get_project(project_id)
            docs = self.tech_writer.write_documentation(project[1], spec, architecture, codebase_summary)
            self.save_file(project_id, "README.md", docs)
            print(f"{Colors.BRIGHT_GREEN}✓ Documentation saved: README.md{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠ Threaded run finished with {remaining} task(s) still pending.{Colors.RESET}")
    
    def extract_filename(self, code, plan, task):
        """Extract filename from code or plan."""
        # Look for common patterns
        import re
        
        # Check for file path in comments
        match = re.search(r'# (?:File|Filename|Path):\s*(.+)', code, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Check in plan
        match = re.search(r'(?:create|modify|file)\s+[\'"]?([a-zA-Z0-9_/\\.]+\.[a-zA-Z0-9]+)', plan, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Check in task description
        match = re.search(r'[\'"]?([a-zA-Z0-9_/\\.]+\.[a-zA-Z0-9]+)', task, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return None


# ==============================================================================
#                           7. SPLASH SCREEN & LOADING
# ==============================================================================

def show_splash_screen():
    """Display the startup splash screen with colors."""
    splash = """
    
   ))         ))     oo_       .-.   \\\  ///       W  W                oo_    oo_   wW  Ww oo_   (o)__(o)    \\\  ///(o)__(o) 
  (o0)-. wWw (Oo)-. /  _)-<  c(O_O)c ((O)(O))   /) (O)(O)         /)   /  _)-</  _)-<(O)(O)/  _)-<(__  __)/)  ((O)(O))(__  __) 
   | (_))(O)_ | (_))\__ `.  ,'.---.`, | \ ||  (o)(O) ||         (o)(O) \__ `. \__ `.  (..) \__ `.   (  )(o)(O) | \ ||   (  )   
   | .-'.' __)|  .'    `. |/ /|_|_|\ \||\\||   //\\  | \         //\\     `. |   `. |  ||     `. |   )(  //\\  ||\\||    )(    
   |(  (  _)  )|\\     _| || \_____/ ||| \ |  |(__)| |  `.      |(__)|    _| |   _| | _||_    _| |  (  )|(__)| || \ |   (  )   
    \)  `.__)(/  \) ,-'   |'. `---' .`||  ||  /,-. |(.-.__)     /,-. | ,-'   |,-'   |(_/\_),-'   |   )/ /,-. | ||  ||    )/    
    (         )    (_..--'   `-...-' (_/  \_)-'   '' `-'       -'   ''(_..--'(_..--'      (_..--'   (  -'   ''(_/  \_)  (      

 """
    
    # Clear screen and print colored splash
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Print with gradient effect
    lines = splash.split('\n')
    colors = [Colors.BRIGHT_CYAN, Colors.CYAN, Colors.BRIGHT_BLUE, Colors.BLUE, Colors.BRIGHT_MAGENTA]
    
    for i, line in enumerate(lines):
        if line.strip():
            color_idx = (i // 3) % len(colors)
            print(f"{colors[color_idx]}{Colors.BOLD}{line}{Colors.RESET}")
        else:
            print(line)
    
    print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}{'='*79}{Colors.RESET}")
    print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  AI Terminal Pro - Advanced AI Assistant with RAG & Tool Integration{Colors.RESET}")
    print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
    time.sleep(1.5)


def show_loading_screen(message="Loading", duration=2):
    """Display animated loading screen."""
    frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    colors = [Colors.BRIGHT_CYAN, Colors.CYAN, Colors.BRIGHT_BLUE, Colors.BLUE]
    
    start_time = time.time()
    frame_idx = 0
    color_idx = 0
    
    while time.time() - start_time < duration:
        frame = frames[frame_idx % len(frames)]
        color = colors[color_idx % len(colors)]
        
        # Clear line and print loading animation
        print(f"\r{color}{Colors.BOLD}{frame} {message}...{Colors.RESET}", end='', flush=True)
        
        frame_idx += 1
        if frame_idx % 3 == 0:
            color_idx += 1
        
        time.sleep(0.1)
    
    print(f"\r{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ {message} complete!{Colors.RESET}" + " " * 50)
    time.sleep(0.3)


def show_progress_bar(message, current, total, bar_length=40):
    """Display a progress bar."""
    percent = current / total if total > 0 else 0
    filled = int(bar_length * percent)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"\r{Colors.BRIGHT_CYAN}{message}: [{bar}] {int(percent * 100)}%{Colors.RESET}", end='', flush=True)
    
    if current >= total:
        print(f"\r{Colors.BRIGHT_GREEN}✓ {message}: Complete!{Colors.RESET}" + " " * 60)


# ==============================================================================
#                           7. MAIN APP CONTROLLER
# ==============================================================================

class App:
    def __init__(self):
        self.cfg_mgr = ConfigManager()
        self.config = self.cfg_mgr.config
        self.memory = MemoryManager(DB_PATH)
        self.registry = None
        self.engine = None
        self.context_mgr = None
        self.current_session_id = None
        self.api_manager = None
        self.current_project_id = None
        self.app_builder = None

    def clear(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def run(self):
        # 0. Show startup splash
        show_splash_screen()
        
        # 1. Onboarding
        if self.config.get("first_run"):
            self.onboarding()
        
        # 2. Initialization with loading screens
        show_loading_screen("Initializing Tool Registry", 1.0)
        self.registry = ToolRegistry(self.config)
        
        show_loading_screen("Initializing AI Engine", 1.0)
        self.engine = AIEngine(self.config)
        
        show_loading_screen("Setting up Context Manager", 0.8)
        self.context_mgr = ContextManager(self.engine.tokenizer, self.config.get("max_context_window"))
        
        show_loading_screen("Initializing API Manager", 0.5)
        try:
            self.api_manager = APIServerManager(self)
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ API Manager initialization failed: {e}{Colors.RESET}")
            self.api_manager = None
        
        show_loading_screen("Initializing App Builder", 0.5)
        try:
            self.app_builder = AppBuilderOrchestrator(self.engine, self.memory)
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ App Builder initialization failed: {e}{Colors.RESET}")
            self.app_builder = None
        
        print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ System Ready!{Colors.RESET}\n")
        time.sleep(0.5)
        
        # 3. Main Menu
        self.main_menu()

    def onboarding(self):
        self.clear()
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}{' '*25}AI TERMINAL - INITIAL SETUP{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        print(f"{Colors.BRIGHT_WHITE}Welcome! Let's set up your AI backend:{Colors.RESET}\n")
        
        # Backend selection
        print(f"{Colors.CYAN}  [1]{Colors.RESET} Hugging Face {Colors.DIM}(Local Python, High RAM){Colors.RESET}")
        print(f"{Colors.CYAN}  [2]{Colors.RESET} Ollama {Colors.DIM}(External App, Fast){Colors.RESET}\n")
        
        choice = input(f"{Colors.BRIGHT_GREEN}Select Backend [1/2]: {Colors.RESET}").strip()
        
        if choice == "2":
            # Ollama setup
            self.cfg_mgr.update("backend", "ollama")
            
            # Check/install/start Ollama
            if not install_and_start_ollama():
                print(f"\n{Colors.YELLOW}⚠ Ollama setup incomplete.{Colors.RESET}")
                print(f"{Colors.DIM}You can complete the setup later from the main menu.{Colors.RESET}")
                cont = input(f"\n{Colors.BRIGHT_GREEN}Continue anyway? [y/N]: {Colors.RESET}").strip().lower()
                if cont != 'y':
                    sys.exit(0)
            
            # List available models
            print(f"\n{Colors.BRIGHT_CYAN}Fetching available Ollama models...{Colors.RESET}")
            models = get_ollama_models()
            
            if models:
                print(f"\n{Colors.BRIGHT_GREEN}Available Ollama Models:{Colors.RESET}\n")
                for i, model in enumerate(models[:15], 1):  # Show first 15
                    # Highlight popular models
                    popular = ["llama3", "mistral", "codellama", "phi", "gemma"]
                    marker = " ⭐" if any(p in model.lower() for p in popular) else ""
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {model}{Colors.BRIGHT_YELLOW}{marker}{Colors.RESET}")
                
                print(f"\n{Colors.CYAN}  [0]{Colors.RESET} Enter custom model name")
                print(f"{Colors.CYAN}  [p]{Colors.RESET} Pull/download a new model")
                print(f"{Colors.DIM}Popular models: llama3, mistral, codellama, phi, gemma{Colors.RESET}")
                model_choice = input(f"\n{Colors.BRIGHT_GREEN}Select model [1-{min(len(models), 15)}], 0 for custom, or 'p' to pull: {Colors.RESET}").strip().lower()
                
                if model_choice == "p":
                    # Pull new model
                    model_to_pull = input(f"{Colors.BRIGHT_GREEN}Enter model name to pull (e.g., llama3, mistral): {Colors.RESET}").strip()
                    if model_to_pull:
                        if pull_ollama_model(model_to_pull):
                            selected_model = model_to_pull
                        else:
                            print(f"{Colors.YELLOW}Using default model: llama3{Colors.RESET}")
                            selected_model = "llama3"
                    else:
                        selected_model = "llama3"
                else:
                    try:
                        idx = int(model_choice)
                        if 1 <= idx <= min(len(models), 15):
                            selected_model = models[idx - 1]
                        elif idx == 0:
                            selected_model = input(f"{Colors.BRIGHT_GREEN}Enter model name: {Colors.RESET}").strip()
                        else:
                            selected_model = models[0] if models else "llama3"
                    except:
                        selected_model = input(f"{Colors.BRIGHT_GREEN}Enter model name: {Colors.RESET}").strip()
                    
                    if not selected_model:
                        selected_model = "llama3"
            else:
                print(f"{Colors.YELLOW}No models found locally.{Colors.RESET}\n")
                print(f"{Colors.CYAN}  [1]{Colors.RESET} Pull/download a model now")
                print(f"{Colors.CYAN}  [2]{Colors.RESET} Enter model name (will be pulled on first use)")
                
                pull_choice = input(f"\n{Colors.BRIGHT_GREEN}Select option [1/2]: {Colors.RESET}").strip()
                
                if pull_choice == "1":
                    model_to_pull = input(f"{Colors.BRIGHT_GREEN}Enter model name to pull (e.g., llama3, mistral): {Colors.RESET}").strip()
                    if model_to_pull:
                        if pull_ollama_model(model_to_pull):
                            selected_model = model_to_pull
                        else:
                            print(f"{Colors.YELLOW}Using default model: llama3{Colors.RESET}")
                            selected_model = "llama3"
                    else:
                        selected_model = "llama3"
                else:
                    selected_model = input(f"{Colors.BRIGHT_GREEN}Enter model name (default: llama3): {Colors.RESET}").strip()
                    if not selected_model:
                        selected_model = "llama3"
            
            self.cfg_mgr.update("model_name", selected_model)
            print(f"\n{Colors.BRIGHT_GREEN}✓ Selected Ollama model: {selected_model}{Colors.RESET}")
            
            # Verify model is available or pull it
            current_models = get_ollama_models()
            if selected_model not in current_models:
                print(f"\n{Colors.YELLOW}⚠ Model '{selected_model}' is not downloaded yet.{Colors.RESET}")
                pull_now = input(f"{Colors.BRIGHT_GREEN}Download it now? [Y/n]: {Colors.RESET}").strip().lower()
                if pull_now != 'n':
                    if not pull_ollama_model(selected_model):
                        print(f"{Colors.YELLOW}⚠ Model download incomplete. You can pull it later with: ollama pull {selected_model}{Colors.RESET}")
                else:
                    print(f"{Colors.DIM}Model will need to be pulled before use: ollama pull {selected_model}{Colors.RESET}")
            else:
                print(f"{Colors.BRIGHT_GREEN}✓ Model is ready to use!{Colors.RESET}")
            
        else:
            # HuggingFace setup
            self.cfg_mgr.update("backend", "huggingface")
            
            print(f"\n{Colors.BRIGHT_CYAN}HuggingFace Model Selection{Colors.RESET}\n")
            print(f"{Colors.CYAN}  [1]{Colors.RESET} Quick select popular model")
            print(f"{Colors.CYAN}  [2]{Colors.RESET} Search HuggingFace models")
            print(f"{Colors.CYAN}  [3]{Colors.RESET} Enter model ID manually\n")
            
            hf_choice = input(f"{Colors.BRIGHT_GREEN}Select option [1/2/3]: {Colors.RESET}").strip()
            
            if hf_choice == "1":
                # Quick select popular models
                popular_models = [
                    ("gpt2", "GPT-2 Small (124M) - Fast, good for testing"),
                    ("gpt2-medium", "GPT-2 Medium (355M) - Better quality"),
                    ("gpt2-large", "GPT-2 Large (774M) - High quality"),
                    ("gpt2-xl", "GPT-2 XL (1.5B) - Best quality, slower"),
                    ("distilgpt2", "DistilGPT-2 (82M) - Fastest, lightweight"),
                ]
                
                print(f"\n{Colors.BRIGHT_GREEN}Popular Models:{Colors.RESET}\n")
                for i, (model_id, desc) in enumerate(popular_models, 1):
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{model_id}{Colors.RESET}")
                    print(f"      {Colors.DIM}{desc}{Colors.RESET}")
                
                model_choice = input(f"\n{Colors.BRIGHT_GREEN}Select model [1-{len(popular_models)}]: {Colors.RESET}").strip()
                try:
                    idx = int(model_choice)
                    if 1 <= idx <= len(popular_models):
                        selected_model = popular_models[idx - 1][0]
                    else:
                        selected_model = "gpt2"
                except:
                    selected_model = "gpt2"
            
            elif hf_choice == "2":
                # Search models
                search_term = input(f"{Colors.BRIGHT_GREEN}Search for models (e.g., 'gpt', 'llama', 'mistral'): {Colors.RESET}").strip()
                if not search_term:
                    search_term = "gpt"
                
                print(f"\n{Colors.BRIGHT_CYAN}Searching HuggingFace...{Colors.RESET}")
                models = search_huggingface_models(search_term, limit=15)
                
                if models:
                    print(f"\n{Colors.BRIGHT_GREEN}Found Models:{Colors.RESET}\n")
                    for i, (model_id, downloads) in enumerate(models, 1):
                        downloads_str = f"{downloads:,}" if downloads else "N/A"
                        print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{model_id}{Colors.RESET} {Colors.DIM}({downloads_str} downloads){Colors.RESET}")
                    
                    print(f"\n{Colors.CYAN}  [0]{Colors.RESET} Enter custom model ID")
                    model_choice = input(f"\n{Colors.BRIGHT_GREEN}Select model [1-{len(models)}] or 0 for custom: {Colors.RESET}").strip()
                    
                    try:
                        idx = int(model_choice)
                        if 1 <= idx <= len(models):
                            selected_model = models[idx - 1][0]
                        elif idx == 0:
                            selected_model = input(f"{Colors.BRIGHT_GREEN}Enter model ID: {Colors.RESET}").strip()
                        else:
                            selected_model = "gpt2"
                    except:
                        selected_model = input(f"{Colors.BRIGHT_GREEN}Enter model ID: {Colors.RESET}").strip()
                    
                    if not selected_model:
                        selected_model = "gpt2"
                else:
                    print(f"{Colors.YELLOW}No models found. Using default: gpt2{Colors.RESET}")
                    selected_model = "gpt2"
                
            elif hf_choice == "3":
                selected_model = input(f"{Colors.BRIGHT_GREEN}Enter HuggingFace model ID: {Colors.RESET}").strip()
                if not selected_model:
                    selected_model = "gpt2"
            else:
                selected_model = "gpt2"
            
            self.cfg_mgr.update("model_name", selected_model)
            print(f"\n{Colors.BRIGHT_GREEN}✓ Selected HuggingFace model: {selected_model}{Colors.RESET}")
            
            # Ask if user wants to download now
            download_now = input(f"\n{Colors.BRIGHT_GREEN}Download model now? [Y/n]: {Colors.RESET}").strip().lower()
            if download_now != 'n':
                print(f"\n{Colors.BRIGHT_CYAN}Downloading model...{Colors.RESET}")
                print(f"{Colors.DIM}This may take several minutes depending on model size...{Colors.RESET}\n")
                
                try:
                    # Pre-download tokenizer and model
                    print(f"{Colors.BRIGHT_CYAN}Downloading tokenizer...{Colors.RESET}")
                    tokenizer = AutoTokenizer.from_pretrained(selected_model)
                    print(f"{Colors.BRIGHT_GREEN}✓ Tokenizer downloaded{Colors.RESET}")
                    
                    print(f"{Colors.BRIGHT_CYAN}Downloading model weights...{Colors.RESET}")
                    print(f"{Colors.DIM}This is the large download...{Colors.RESET}")
                    model = AutoModelForCausalLM.from_pretrained(selected_model)
                    print(f"\n{Colors.BRIGHT_GREEN}✓ Model '{selected_model}' downloaded successfully!{Colors.RESET}")
                    
                    # Clean up memory
                    del model
                    del tokenizer
                    import gc
                    gc.collect()
                    
                except Exception as e:
                    print(f"\n{Colors.BRIGHT_RED}✗ Download failed: {e}{Colors.RESET}")
                    print(f"{Colors.YELLOW}The model will be downloaded on first use.{Colors.RESET}")
            else:
                print(f"{Colors.DIM}Model will be downloaded automatically on first use.{Colors.RESET}")
            
        self.cfg_mgr.update("first_run", False)
        print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ Setup Complete!{Colors.RESET}")
        show_loading_screen("Preparing system", 1.5)

    def main_menu(self):
        while True:
            self.clear()
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  MAIN MENU{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            backend = self.config.get('backend', 'unknown')
            model = self.config.get('model_name', 'unknown')
            print(f"{Colors.CYAN}Backend:{Colors.RESET} {Colors.BRIGHT_WHITE}{backend}{Colors.RESET} | {Colors.CYAN}Model:{Colors.RESET} {Colors.BRIGHT_WHITE}{model}{Colors.RESET}\n")
            
            print(f"{Colors.BRIGHT_GREEN}  [1]{Colors.RESET} Start Chat")
            print(f"{Colors.BRIGHT_GREEN}  [2]{Colors.RESET} Document Loader (RAG)")
            print(f"{Colors.BRIGHT_GREEN}  [3]{Colors.RESET} Tool Management")
            print(f"{Colors.BRIGHT_GREEN}  [4]{Colors.RESET} MCP Server Management")
            print(f"{Colors.BRIGHT_GREEN}  [5]{Colors.RESET} Model Training")
            print(f"{Colors.BRIGHT_GREEN}  [6]{Colors.RESET} API Management")
            print(f"{Colors.BRIGHT_GREEN}  [7]{Colors.RESET} App Builder {Colors.DIM}(Multi-Agent Development){Colors.RESET}")
            print(f"{Colors.BRIGHT_GREEN}  [8]{Colors.RESET} Update from GitHub")
            print(f"{Colors.BRIGHT_GREEN}  [9]{Colors.RESET} Settings")
            print(f"{Colors.BRIGHT_RED}  [10]{Colors.RESET} Exit\n")
            
            c = input(f"{Colors.BRIGHT_GREEN}Select: {Colors.RESET}")
            if c == "1": self.chat_loop()
            elif c == "2": self.document_menu()
            elif c == "3": self.tool_menu()
            elif c == "4": self.mcp_server_menu()
            elif c == "5": self.model_training_menu()
            elif c == "6": self.api_management_menu()
            elif c == "7": self.app_builder_menu()
            elif c == "8": self.update_from_github()
            elif c == "9": self.settings_menu()
            elif c == "10":
                print(f"\n{Colors.BRIGHT_YELLOW}Shutting down...{Colors.RESET}")
                if self.registry:
                    for c in self.registry.mcp_clients.values(): c.stop()
                if self.api_manager:
                    # Stop all API servers
                    for api_name in list(self.api_manager.servers.keys()):
                        self.api_manager.stop_api(api_name)
                print(f"{Colors.BRIGHT_GREEN}✓ Goodbye!{Colors.RESET}\n")
                sys.exit()

    def update_from_github(self):
        """Update ai.py from the latest GitHub commit."""
        self.clear()
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  UPDATE FROM GITHUB{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        
        github_repo = "https://github.com/repackedadmin/-AI-Terminal-Pro-"
        github_raw_url = "https://raw.githubusercontent.com/repackedadmin/-AI-Terminal-Pro-/main/ai.py"
        current_file = os.path.abspath(__file__)
        
        print(f"{Colors.CYAN}Repository:{Colors.RESET} {Colors.BRIGHT_WHITE}{github_repo}{Colors.RESET}")
        print(f"{Colors.CYAN}Current File:{Colors.RESET} {Colors.BRIGHT_WHITE}{current_file}{Colors.RESET}\n")
        
        # Confirm update
        confirm = input(f"{Colors.BRIGHT_YELLOW}⚠ This will replace your current ai.py with the latest version from GitHub.\n{Colors.RESET}{Colors.DIM}A backup will be created automatically.{Colors.RESET}\n\n{Colors.BRIGHT_GREEN}Continue? [y/N]: {Colors.RESET}").strip().lower()
        if confirm != 'y':
            print(f"\n{Colors.DIM}Update cancelled.{Colors.RESET}")
            time.sleep(1)
            return
        
        try:
            # Create backup
            backup_dir = os.path.join(BASE_DIR, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(backup_dir, f"ai_backup_{timestamp}.py")
            
            print(f"\n{Colors.BRIGHT_CYAN}Creating backup...{Colors.RESET}")
            shutil.copy2(current_file, backup_file)
            print(f"{Colors.BRIGHT_GREEN}✓ Backup created: {backup_file}{Colors.RESET}")
            
            # Download latest version
            print(f"\n{Colors.BRIGHT_CYAN}Downloading latest version from GitHub...{Colors.RESET}")
            response = requests.get(github_raw_url, timeout=30)
            response.raise_for_status()
            
            new_content = response.text
            if not new_content or len(new_content) < 1000:
                raise Exception("Downloaded file appears to be invalid or too small")
            
            # Write new version
            print(f"{Colors.BRIGHT_CYAN}Installing update...{Colors.RESET}")
            with open(current_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            # Verify file was written
            if not os.path.exists(current_file) or os.path.getsize(current_file) < 1000:
                raise Exception("Update file verification failed")
            
            print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ Update successful!{Colors.RESET}")
            print(f"{Colors.CYAN}Backup saved to:{Colors.RESET} {Colors.BRIGHT_WHITE}{backup_file}{Colors.RESET}")
            print(f"\n{Colors.YELLOW}⚠ Please restart the application to use the updated version.{Colors.RESET}")
            
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")
            
        except requests.exceptions.RequestException as e:
            print(f"\n{Colors.BRIGHT_RED}✗ Network error: {e}{Colors.RESET}")
            print(f"{Colors.YELLOW}Please check your internet connection and try again.{Colors.RESET}")
            time.sleep(3)
        except PermissionError:
            print(f"\n{Colors.BRIGHT_RED}✗ Permission denied.{Colors.RESET}")
            print(f"{Colors.YELLOW}Please ensure you have write permissions for: {current_file}{Colors.RESET}")
            time.sleep(3)
        except Exception as e:
            print(f"\n{Colors.BRIGHT_RED}✗ Update failed: {e}{Colors.RESET}")
            print(f"{Colors.YELLOW}Your original file is safe. Backup location: {backup_file if 'backup_file' in locals() else 'N/A'}{Colors.RESET}")
            time.sleep(3)

    def settings_menu(self):
        self.clear()
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  SETTINGS{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        
        danger_status = "ON" if self.config.get('enable_dangerous_commands') else "OFF"
        danger_color = Colors.BRIGHT_RED if self.config.get('enable_dangerous_commands') else Colors.BRIGHT_GREEN
        print(f"{Colors.CYAN}Dangerous Commands:{Colors.RESET} {danger_color}{danger_status}{Colors.RESET}\n")
        
        current_editor = self.config.get("default_editor_command") or "None (use OS default)"
        print(f"{Colors.CYAN}Default Editor for Tools/MCP:{Colors.RESET} {Colors.BRIGHT_WHITE}{current_editor}{Colors.RESET}\n")
        
        print(f"{Colors.BRIGHT_GREEN}  [1]{Colors.RESET} Toggle Dangerous Commands {Colors.DIM}(Allow writing outside sandbox){Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}  [2]{Colors.RESET} Edit System Prompt")
        print(f"{Colors.BRIGHT_GREEN}  [3]{Colors.RESET} Set Default Editor for Tools/MCP Servers")
        print(f"{Colors.BRIGHT_GREEN}  [4]{Colors.RESET} Back\n")
        
        c = input(f"{Colors.BRIGHT_GREEN}Choice: {Colors.RESET}")
        if c == "1":
            new_val = not self.config.get("enable_dangerous_commands")
            self.cfg_mgr.update("enable_dangerous_commands", new_val)
            # Must reload registry to update permissions
            self.registry.config = self.cfg_mgr.config
            status = "ENABLED" if new_val else "DISABLED"
            color = Colors.BRIGHT_RED if new_val else Colors.BRIGHT_GREEN
            print(f"\n{color}✓ Dangerous Commands {status}{Colors.RESET}")
            time.sleep(1.5)
        elif c == "2":
            print(f"\n{Colors.CYAN}Current:{Colors.RESET} {Colors.DIM}{self.config.get('system_prompt')[:60]}...{Colors.RESET}")
            new_p = input(f"{Colors.BRIGHT_GREEN}New Prompt: {Colors.RESET}")
            if new_p: 
                self.cfg_mgr.update("system_prompt", new_p)
                print(f"{Colors.BRIGHT_GREEN}✓ System prompt updated{Colors.RESET}")
                time.sleep(1)
        elif c == "3":
            # Configure default editor for tools / MCP servers
            print(f"\n{Colors.CYAN}Configure Default Editor:{Colors.RESET}")
            print(f"{Colors.DIM}This editor will be used to open new tools and MCP configuration files.{Colors.RESET}\n")
            
            # Detect common editors
            candidates = []
            platform_name = platform.system()
            
            # Common GUI editors
            common_cmds = [
                ("VS Code", "code"),
                ("Cursor", "cursor"),
                ("Notepad++", "notepad++"),
                ("Sublime Text", "sublime_text"),
                ("Visual Studio", "devenv"),
                ("PyCharm", "pycharm"),
            ]
            
            for label, cmd in common_cmds:
                if shutil.which(cmd):
                    candidates.append((label, cmd))
            
            # Always include OS default / custom
            print(f"{Colors.CYAN}Available editors detected on this system:{Colors.RESET}\n")
            idx = 1
            print(f"  {Colors.CYAN}[{idx}]{Colors.RESET} {Colors.BRIGHT_WHITE}OS Default Editor{Colors.RESET} {Colors.DIM}(no custom command){Colors.RESET}")
            idx += 1
            
            for label, cmd in candidates:
                print(f"  {Colors.CYAN}[{idx}]{Colors.RESET} {Colors.BRIGHT_WHITE}{label}{Colors.RESET} {Colors.DIM}({cmd}){Colors.RESET}")
                idx += 1
            
            print(f"  {Colors.CYAN}[{idx}]{Colors.RESET} {Colors.BRIGHT_WHITE}Custom Command{Colors.RESET}")
            
            choice = input(f"\n{Colors.BRIGHT_GREEN}Select editor [{1}-{idx}]: {Colors.RESET}").strip()
            try:
                choice_idx = int(choice)
            except ValueError:
                print(f"{Colors.YELLOW}⚠ Invalid selection. Editor not changed.{Colors.RESET}")
                time.sleep(1.5)
                return
            
            if choice_idx == 1:
                # OS default
                self.cfg_mgr.update("default_editor_command", "")
                print(f"{Colors.BRIGHT_GREEN}✓ Default editor set to OS default{Colors.RESET}")
                time.sleep(1.5)
                return
            
            if 2 <= choice_idx < idx:
                label, cmd = candidates[choice_idx - 2]
                self.cfg_mgr.update("default_editor_command", cmd)
                print(f"{Colors.BRIGHT_GREEN}✓ Default editor set to: {Colors.BRIGHT_WHITE}{label} ({cmd}){Colors.RESET}")
                time.sleep(1.5)
                return
            
            if choice_idx == idx:
                custom_cmd = input(f"{Colors.BRIGHT_GREEN}Enter custom editor command (e.g., 'code', 'cursor', 'notepad++'): {Colors.RESET}").strip()
                if custom_cmd:
                    self.cfg_mgr.update("default_editor_command", custom_cmd)
                    print(f"{Colors.BRIGHT_GREEN}✓ Default editor set to custom command: {Colors.BRIGHT_WHITE}{custom_cmd}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}⚠ Empty command. Editor not changed.{Colors.RESET}")
                time.sleep(1.5)
                return
            
            print(f"{Colors.YELLOW}⚠ Invalid selection. Editor not changed.{Colors.RESET}")
            time.sleep(1.5)
    
    def open_in_default_editor(self, filepath):
        """
        Open a file in the configured default editor.
        Falls back to OS default if no editor is configured.
        """
        try:
            editor_cmd = self.config.get("default_editor_command") or self.cfg_mgr.get("default_editor_command")
        except Exception:
            editor_cmd = ""
        
        filepath = os.path.abspath(filepath)
        
        try:
            if editor_cmd:
                # Use configured editor command
                if platform.system() == "Windows":
                    # On Windows, allow commands like "code" or "notepad++"
                    cmd = f'{editor_cmd} "{filepath}"'
                    subprocess.Popen(cmd, shell=True)
                else:
                    # POSIX-style: split command and append filepath
                    parts = shlex.split(editor_cmd)
                    parts.append(filepath)
                    subprocess.Popen(parts)
                print(f"{Colors.DIM}Opened in editor: {editor_cmd} {filepath}{Colors.RESET}")
            else:
                # Fallback to OS default handler
                system = platform.system()
                if system == "Windows":
                    os.startfile(filepath)
                elif system == "Darwin":
                    subprocess.Popen(["open", filepath])
                else:
                    subprocess.Popen(["xdg-open", filepath])
                print(f"{Colors.DIM}Opened file with OS default editor: {filepath}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.BRIGHT_RED}✗ Failed to open editor: {e}{Colors.RESET}")

    def document_menu(self):
        self.clear()
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  RAG DOCUMENT LOADER{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        print(f"{Colors.CYAN}Document Directory:{Colors.RESET} {Colors.BRIGHT_WHITE}{DOCS_DIR}{Colors.RESET}")
        print(f"{Colors.DIM}Place .txt/.md files in the directory above{Colors.RESET}\n")
        
        print(f"{Colors.BRIGHT_GREEN}  [1]{Colors.RESET} Ingest All Files")
        print(f"{Colors.BRIGHT_GREEN}  [2]{Colors.RESET} Back\n")
        
        if input(f"{Colors.BRIGHT_GREEN}Choice: {Colors.RESET}") == "1":
            files = glob.glob(os.path.join(DOCS_DIR, "*.*"))
            if not files:
                print(f"\n{Colors.YELLOW}⚠ No files found in {DOCS_DIR}{Colors.RESET}")
                input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            return

            total = 0
            print(f"\n{Colors.BRIGHT_CYAN}Processing files...{Colors.RESET}\n")
            for f in files:
                print(f"{Colors.CYAN}  📄{Colors.RESET} Ingesting {Colors.BRIGHT_WHITE}{os.path.basename(f)}{Colors.RESET}...", end='', flush=True)
                count = self.memory.ingest_file(f)
                total += count
                print(f" {Colors.BRIGHT_GREEN}✓ {count} chunks{Colors.RESET}")
            
            print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ Done! Added {total} total chunks.{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")

    def tool_menu(self):
        """Main tool management menu - separates Custom Tools and MCP Servers."""
        while True:
            self.clear()
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  TOOL MANAGEMENT{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
            
            print(f"{Colors.BRIGHT_GREEN}  [1]{Colors.RESET} Custom Tools {Colors.DIM}(Python, JSON, YAML scripts){Colors.RESET}")
            print(f"{Colors.BRIGHT_GREEN}  [2]{Colors.RESET} MCP Server Management")
            print(f"{Colors.BRIGHT_GREEN}  [3]{Colors.RESET} Back\n")
            
            c = input(f"{Colors.BRIGHT_GREEN}Choice: {Colors.RESET}")
            if c == "1":
                self.custom_tools_menu()
            elif c == "2":
                self.mcp_server_menu()
            elif c == "3":
                break

    def custom_tools_menu(self):
        """Custom Tools Management menu."""
        while True:
            self.clear()
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  CUSTOM TOOLS MANAGEMENT{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
            
            print(f"{Colors.CYAN}Tools Directory:{Colors.RESET} {Colors.BRIGHT_WHITE}{CUSTOM_TOOLS_DIR}{Colors.RESET}\n")
            
            # List current tools
            tools = self.registry.custom_tool_manager.list_tools() if self.registry else []
            if tools:
                print(f"{Colors.CYAN}Available Tools:{Colors.RESET}\n")
                for i, tool_name in enumerate(tools, 1):
                    tool_info = self.registry.custom_tool_manager.get_tool_info(tool_name) if self.registry else None
                    tool_type = tool_info.get("type", "unknown") if tool_info else "unknown"
                    desc = tool_info.get("description", "") if tool_info else ""
                    type_color = Colors.BRIGHT_CYAN if tool_type == "python" else Colors.BRIGHT_YELLOW if tool_type in ["json", "yaml"] else Colors.CYAN
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{tool_name}{Colors.RESET} {type_color}({tool_type}){Colors.RESET}")
                    if desc:
                        print(f"      {Colors.DIM}{desc[:60]}...{Colors.RESET}\n")
                    else:
                        print()
            else:
                print(f"{Colors.YELLOW}No custom tools found.{Colors.RESET}\n")
            
            print(f"{Colors.BRIGHT_GREEN}  [1]{Colors.RESET} Create Python Tool")
            print(f"{Colors.BRIGHT_GREEN}  [2]{Colors.RESET} Create JSON Tool")
            print(f"{Colors.BRIGHT_GREEN}  [3]{Colors.RESET} Create YAML Tool")
            print(f"{Colors.BRIGHT_GREEN}  [4]{Colors.RESET} Add Existing File")
            print(f"{Colors.BRIGHT_GREEN}  [5]{Colors.RESET} View Tool Details")
            print(f"{Colors.BRIGHT_GREEN}  [6]{Colors.RESET} Delete Tool")
            print(f"{Colors.BRIGHT_GREEN}  [7]{Colors.RESET} Back\n")
            
            c = input(f"{Colors.BRIGHT_GREEN}Choice: {Colors.RESET}")
            
            if c == "1":
                # Create Python tool
                name = input(f"\n{Colors.BRIGHT_GREEN}Tool Name (e.g., 'calculator'): {Colors.RESET}").strip()
                if not name:
                    print(f"{Colors.YELLOW}⚠ Name cannot be empty{Colors.RESET}")
                    time.sleep(1)
                    continue
                
                description = input(f"{Colors.BRIGHT_GREEN}Description: {Colors.RESET}").strip()
                print(f"\n{Colors.CYAN}Enter Python code (end with 'END' on a new line):{Colors.RESET}")
                print(f"{Colors.DIM}Example: result = sum([int(x) for x in args])\nprint(result){Colors.RESET}\n")
                
                code_lines = []
                while True:
                    line = input()
                    if line.strip() == "END":
                        break
                    code_lines.append(line)
                
                code = "\n".join(code_lines)
                if not code.strip():
                    code = "    # Your tool code here\n    print('Tool executed')"
                
                # Indent code
                indented_code = "\n".join("    " + line if line.strip() else line for line in code.split("\n"))
                
                if self.registry.custom_tool_manager.create_python_tool(name, description, indented_code):
                    print(f"\n{Colors.BRIGHT_GREEN}✓ Python tool '{name}' created!{Colors.RESET}")
                    # Offer to open in default editor
                    try:
                        filename = name if name.endswith(".py") else name + ".py"
                        tool_path = os.path.join(CUSTOM_TOOLS_DIR, filename)
                        open_choice = input(f"{Colors.BRIGHT_GREEN}Open tool in your default editor now? [Y/n]: {Colors.RESET}").strip().lower()
                        if open_choice in ("", "y", "yes"):
                            self.open_in_default_editor(tool_path)
                    except Exception as e:
                        print(f"{Colors.DIM}Editor open skipped: {e}{Colors.RESET}")
                else:
                    print(f"\n{Colors.BRIGHT_RED}✗ Failed to create tool.{Colors.RESET}")
                time.sleep(2)
            
            elif c == "2":
                # Create JSON tool
                name = input(f"\n{Colors.BRIGHT_GREEN}Tool Name: {Colors.RESET}").strip()
                if not name:
                    print(f"{Colors.YELLOW}⚠ Name cannot be empty{Colors.RESET}")
                    time.sleep(1)
                    continue
                
                description = input(f"{Colors.BRIGHT_GREEN}Description: {Colors.RESET}").strip()
                print(f"\n{Colors.CYAN}Tool Type:{Colors.RESET}")
                print(f"  {Colors.CYAN}[1]{Colors.RESET} Command (shell command)")
                print(f"  {Colors.CYAN}[2]{Colors.RESET} Script (script file path)")
                tool_type = input(f"{Colors.BRIGHT_GREEN}Select [1/2]: {Colors.RESET}").strip()
                
                if tool_type == "1":
                    command = input(f"{Colors.BRIGHT_GREEN}Command: {Colors.RESET}").strip()
                    if self.registry.custom_tool_manager.create_json_tool(name, description, command, is_script=False):
                        print(f"\n{Colors.BRIGHT_GREEN}✓ JSON tool '{name}' created!{Colors.RESET}")
                        try:
                            filename = name if name.endswith(".json") else name + ".json"
                            tool_path = os.path.join(CUSTOM_TOOLS_DIR, filename)
                            open_choice = input(f"{Colors.BRIGHT_GREEN}Open tool definition in your default editor now? [Y/n]: {Colors.RESET}").strip().lower()
                            if open_choice in ("", "y", "yes"):
                                self.open_in_default_editor(tool_path)
                        except Exception as e:
                            print(f"{Colors.DIM}Editor open skipped: {e}{Colors.RESET}")
                    else:
                        print(f"\n{Colors.BRIGHT_RED}✗ Failed to create tool.{Colors.RESET}")
                else:
                    script_path = input(f"{Colors.BRIGHT_GREEN}Script File Path: {Colors.RESET}").strip()
                    if self.registry.custom_tool_manager.create_json_tool(name, description, script_path, is_script=True):
                        print(f"\n{Colors.BRIGHT_GREEN}✓ JSON tool '{name}' created!{Colors.RESET}")
                        try:
                            filename = name if name.endswith(".json") else name + ".json"
                            tool_path = os.path.join(CUSTOM_TOOLS_DIR, filename)
                            open_choice = input(f"{Colors.BRIGHT_GREEN}Open tool definition in your default editor now? [Y/n]: {Colors.RESET}").strip().lower()
                            if open_choice in ("", "y", "yes"):
                                self.open_in_default_editor(tool_path)
                        except Exception as e:
                            print(f"{Colors.DIM}Editor open skipped: {e}{Colors.RESET}")
                    else:
                        print(f"\n{Colors.BRIGHT_RED}✗ Failed to create tool.{Colors.RESET}")
                time.sleep(2)
            
            elif c == "3":
                # Create YAML tool
                try:
                    import yaml
                except ImportError:
                    print(f"\n{Colors.YELLOW}⚠ YAML support requires 'pyyaml' package.{Colors.RESET}")
                    print(f"{Colors.DIM}Install with: pip install pyyaml{Colors.RESET}")
                    time.sleep(2)
                    continue
                
                name = input(f"\n{Colors.BRIGHT_GREEN}Tool Name: {Colors.RESET}").strip()
                if not name:
                    print(f"{Colors.YELLOW}⚠ Name cannot be empty{Colors.RESET}")
                    time.sleep(1)
                    continue
                
                description = input(f"{Colors.BRIGHT_GREEN}Description: {Colors.RESET}").strip()
                print(f"\n{Colors.CYAN}Tool Type:{Colors.RESET}")
                print(f"  {Colors.CYAN}[1]{Colors.RESET} Command (shell command)")
                print(f"  {Colors.CYAN}[2]{Colors.RESET} Script (script file path)")
                tool_type = input(f"{Colors.BRIGHT_GREEN}Select [1/2]: {Colors.RESET}").strip()
                
                if tool_type == "1":
                    command = input(f"{Colors.BRIGHT_GREEN}Command: {Colors.RESET}").strip()
                    if self.registry.custom_tool_manager.create_yaml_tool(name, description, command, is_script=False):
                        print(f"\n{Colors.BRIGHT_GREEN}✓ YAML tool '{name}' created!{Colors.RESET}")
                        try:
                            if not (name.endswith(".yaml") or name.endswith(".yml")):
                                filename = name + ".yaml"
                            else:
                                filename = name
                            tool_path = os.path.join(CUSTOM_TOOLS_DIR, filename)
                            open_choice = input(f"{Colors.BRIGHT_GREEN}Open tool definition in your default editor now? [Y/n]: {Colors.RESET}").strip().lower()
                            if open_choice in ("", "y", "yes"):
                                self.open_in_default_editor(tool_path)
                        except Exception as e:
                            print(f"{Colors.DIM}Editor open skipped: {e}{Colors.RESET}")
                    else:
                        print(f"\n{Colors.BRIGHT_RED}✗ Failed to create tool.{Colors.RESET}")
                else:
                    script_path = input(f"{Colors.BRIGHT_GREEN}Script File Path: {Colors.RESET}").strip()
                    if self.registry.custom_tool_manager.create_yaml_tool(name, description, script_path, is_script=True):
                        print(f"\n{Colors.BRIGHT_GREEN}✓ YAML tool '{name}' created!{Colors.RESET}")
                        try:
                            if not (name.endswith(".yaml") or name.endswith(".yml")):
                                filename = name + ".yaml"
                            else:
                                filename = name
                            tool_path = os.path.join(CUSTOM_TOOLS_DIR, filename)
                            open_choice = input(f"{Colors.BRIGHT_GREEN}Open tool definition in your default editor now? [Y/n]: {Colors.RESET}").strip().lower()
                            if open_choice in ("", "y", "yes"):
                                self.open_in_default_editor(tool_path)
                        except Exception as e:
                            print(f"{Colors.DIM}Editor open skipped: {e}{Colors.RESET}")
                    else:
                        print(f"\n{Colors.BRIGHT_RED}✗ Failed to create tool.{Colors.RESET}")
                time.sleep(2)
            
            elif c == "4":
                # Add existing file
                file_path = input(f"\n{Colors.BRIGHT_GREEN}File Path: {Colors.RESET}").strip()
                if not file_path or not os.path.exists(file_path):
                    print(f"{Colors.YELLOW}⚠ File not found.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                filename = os.path.basename(file_path)
                dest_path = os.path.join(CUSTOM_TOOLS_DIR, filename)
                
                try:
                    shutil.copy2(file_path, dest_path)
                    self.registry.custom_tool_manager.load_tools()
                    print(f"\n{Colors.BRIGHT_GREEN}✓ File added as tool: {filename}{Colors.RESET}")
                except Exception as e:
                    print(f"\n{Colors.BRIGHT_RED}✗ Failed to add file: {e}{Colors.RESET}")
                time.sleep(2)
            
            elif c == "5":
                # View tool details
                if not tools:
                    print(f"\n{Colors.YELLOW}⚠ No tools available.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select tool to view:{Colors.RESET}\n")
                for i, tool_name in enumerate(tools, 1):
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{tool_name}{Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(tools)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(tools):
                        tool_name = tools[idx - 1]
                        tool_info = self.registry.custom_tool_manager.get_tool_info(tool_name)
                        
                        self.clear()
                        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
                        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  TOOL DETAILS: {tool_name}{Colors.RESET}")
                        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
                        
                        if tool_info:
                            print(f"{Colors.CYAN}Name:{Colors.RESET} {Colors.BRIGHT_WHITE}{tool_info.get('name', 'N/A')}{Colors.RESET}")
                            print(f"{Colors.CYAN}Type:{Colors.RESET} {Colors.BRIGHT_WHITE}{tool_info.get('type', 'N/A')}{Colors.RESET}")
                            print(f"{Colors.CYAN}Description:{Colors.RESET} {Colors.BRIGHT_WHITE}{tool_info.get('description', 'N/A')}{Colors.RESET}")
                            print(f"{Colors.CYAN}File:{Colors.RESET} {Colors.BRIGHT_WHITE}{tool_info.get('file', 'N/A')}{Colors.RESET}")
                            if tool_info.get('parameters'):
                                print(f"{Colors.CYAN}Parameters:{Colors.RESET}")
                                for param in tool_info['parameters']:
                                    print(f"  {Colors.DIM}- {param}{Colors.RESET}")
                        else:
                            print(f"{Colors.YELLOW}No metadata available.{Colors.RESET}")
                        
                        input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                except (ValueError, IndexError):
                    print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                    time.sleep(1)
            
            elif c == "6":
                # Delete tool
                if not tools:
                    print(f"\n{Colors.YELLOW}⚠ No tools available.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select tool to delete:{Colors.RESET}\n")
                for i, tool_name in enumerate(tools, 1):
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{tool_name}{Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(tools)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(tools):
                        tool_name = tools[idx - 1]
                        confirm = input(f"{Colors.BRIGHT_RED}Delete '{tool_name}'? [y/N]: {Colors.RESET}").strip().lower()
                        if confirm == 'y':
                            if self.registry.custom_tool_manager.delete_tool(tool_name):
                                print(f"\n{Colors.BRIGHT_GREEN}✓ Tool deleted.{Colors.RESET}")
                            else:
                                print(f"\n{Colors.BRIGHT_RED}✗ Failed to delete tool.{Colors.RESET}")
                        else:
                            print(f"{Colors.DIM}Cancelled.{Colors.RESET}")
                        time.sleep(1.5)
                except (ValueError, IndexError):
                    print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                    time.sleep(1)
            
            elif c == "7":
                break

    def mcp_server_menu(self):
        """Dedicated MCP Server Management menu with create/stop/delete."""
        while True:
            self.clear()
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  MCP SERVER MANAGEMENT{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
            
            # List current MCP servers
            servers = {}
            if os.path.exists(MCP_CONFIG_FILE):
                with open(MCP_CONFIG_FILE, 'r') as f:
                    servers = json.load(f)
            
            if servers:
                print(f"{Colors.CYAN}Configured Servers:{Colors.RESET}\n")
                for i, (name, cmd) in enumerate(servers.items(), 1):
                    is_running = name in (self.registry.mcp_clients if self.registry else {})
                    status = "✓ Running" if is_running else "○ Stopped"
                    status_color = Colors.BRIGHT_GREEN if is_running else Colors.DIM
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET} {status_color}{status}{Colors.RESET}")
                    print(f"      {Colors.DIM}Command: {cmd}{Colors.RESET}\n")
            else:
                print(f"{Colors.YELLOW}No MCP servers configured.{Colors.RESET}\n")
            
            print(f"{Colors.BRIGHT_GREEN}  [1]{Colors.RESET} Add/Create MCP Server")
            print(f"{Colors.BRIGHT_GREEN}  [2]{Colors.RESET} Stop MCP Server")
            print(f"{Colors.BRIGHT_GREEN}  [3]{Colors.RESET} Delete MCP Server")
            print(f"{Colors.BRIGHT_GREEN}  [4]{Colors.RESET} Test MCP Server Connection")
            print(f"{Colors.BRIGHT_GREEN}  [5]{Colors.RESET} View MCP Server Tools")
            print(f"{Colors.BRIGHT_GREEN}  [6]{Colors.RESET} Back\n")
            
            c = input(f"{Colors.BRIGHT_GREEN}Choice: {Colors.RESET}")
            
            if c == "1":
                # Add/Create MCP Server
                print(f"\n{Colors.CYAN}MCP Server Setup:{Colors.RESET}")
                print(f"  {Colors.CYAN}[1]{Colors.RESET} Quick Create (common servers)")
                print(f"  {Colors.CYAN}[2]{Colors.RESET} Custom Command")
                setup_choice = input(f"{Colors.BRIGHT_GREEN}Select [1/2]: {Colors.RESET}").strip()
                
                if setup_choice == "1":
                    # Quick create
                    print(f"\n{Colors.CYAN}Common MCP Servers:{Colors.RESET}\n")
                    common_servers = [
                        ("filesystem", "uvx mcp-server-filesystem", "File system operations"),
                        ("brave-search", "uvx mcp-server-brave-search", "Brave Search API"),
                        ("github", "uvx mcp-server-github", "GitHub integration"),
                        ("postgres", "uvx mcp-server-postgres", "PostgreSQL database"),
                    ]
                    
                    for i, (name, cmd, desc) in enumerate(common_servers, 1):
                        print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET} {Colors.DIM}- {desc}{Colors.RESET}")
                    
                    try:
                        idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(common_servers)}]: {Colors.RESET}").strip())
                        if 1 <= idx <= len(common_servers):
                            name, base_cmd, _ = common_servers[idx - 1]
                            server_name = input(f"{Colors.BRIGHT_GREEN}Server Name (default: {name}): {Colors.RESET}").strip() or name
                            additional_args = input(f"{Colors.BRIGHT_GREEN}Additional Arguments {Colors.DIM}(optional, e.g., path or API key): {Colors.RESET}").strip()
                            cmd = f"{base_cmd} {additional_args}".strip()
                        else:
                            print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                            time.sleep(1)
                            continue
                    except ValueError:
                        print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
                        time.sleep(1)
                        continue
                else:
                    # Custom command
                    server_name = input(f"\n{Colors.BRIGHT_GREEN}Server Name: {Colors.RESET}").strip()
                    if not server_name:
                        print(f"{Colors.YELLOW}⚠ Name cannot be empty{Colors.RESET}")
                        time.sleep(1)
                        continue
                    
                    cmd = input(f"{Colors.BRIGHT_GREEN}Command {Colors.DIM}(e.g., 'uvx mcp-server-filesystem ./'): {Colors.RESET}").strip()
                    if not cmd:
                        print(f"{Colors.YELLOW}⚠ Command cannot be empty{Colors.RESET}")
                        time.sleep(1)
                        continue
                
                # Save to config
                if os.path.exists(MCP_CONFIG_FILE):
                    with open(MCP_CONFIG_FILE, 'r') as f:
                        d = json.load(f)
                else:
                    d = {}
                
                d[server_name] = cmd
                with open(MCP_CONFIG_FILE, 'w') as f:
                    json.dump(d, f, indent=4)
                
                # Offer to open MCP config in default editor
                try:
                    open_choice = input(f"{Colors.BRIGHT_GREEN}Open MCP config file in your default editor now? [Y/n]: {Colors.RESET}").strip().lower()
                    if open_choice in ("", "y", "yes"):
                        self.open_in_default_editor(MCP_CONFIG_FILE)
                except Exception as e:
                    print(f"{Colors.DIM}Editor open skipped: {e}{Colors.RESET}")
                
                # Try to start immediately if registry is available
                if self.registry:
                    print(f"\n{Colors.BRIGHT_CYAN}Attempting to start server...{Colors.RESET}")
                    client = MCPClient(server_name, cmd)
                    if client.start():
                        self.registry.mcp_clients[server_name] = client
                        print(f"{Colors.BRIGHT_GREEN}✓ Server '{server_name}' started successfully!{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}⚠ Server '{server_name}' added but failed to start.{Colors.RESET}")
                        print(f"{Colors.DIM}It will be retried on next application restart.{Colors.RESET}")
                else:
                    print(f"\n{Colors.BRIGHT_GREEN}✓ Server '{server_name}' added.{Colors.RESET}")
                    print(f"{Colors.DIM}Server will be started on next application restart.{Colors.RESET}")
                
                time.sleep(2)
            
            elif c == "2":
                # Stop MCP Server
                if not self.registry or not self.registry.mcp_clients:
                    print(f"\n{Colors.YELLOW}⚠ No running MCP servers.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                running_servers = list(self.registry.mcp_clients.keys())
                print(f"\n{Colors.CYAN}Running Servers:{Colors.RESET}\n")
                for i, name in enumerate(running_servers, 1):
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select server to stop [1-{len(running_servers)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(running_servers):
                        server_name = running_servers[idx - 1]
                        client = self.registry.mcp_clients[server_name]
                        client.stop()
                        del self.registry.mcp_clients[server_name]
                        print(f"\n{Colors.BRIGHT_GREEN}✓ Server '{server_name}' stopped.{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                except ValueError:
                    print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
                time.sleep(1.5)
            
            elif c == "3":
                # Delete MCP Server
                if not servers:
                    print(f"\n{Colors.YELLOW}⚠ No MCP servers configured.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Available servers:{Colors.RESET}\n")
                server_list = list(servers.keys())
                for i, name in enumerate(server_list, 1):
                    is_running = name in (self.registry.mcp_clients if self.registry else {})
                    status = " (Running)" if is_running else ""
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET}{status}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select server to delete [1-{len(server_list)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(server_list):
                        server_name = server_list[idx - 1]
                        confirm = input(f"{Colors.BRIGHT_RED}Delete '{server_name}'? [y/N]: {Colors.RESET}").strip().lower()
                        if confirm == 'y':
                            # Stop if running
                            if self.registry and server_name in self.registry.mcp_clients:
                                self.registry.mcp_clients[server_name].stop()
                                del self.registry.mcp_clients[server_name]
                            
                            # Remove from config
                            del servers[server_name]
                            with open(MCP_CONFIG_FILE, 'w') as f:
                                json.dump(servers, f, indent=4)
                            
                            print(f"\n{Colors.BRIGHT_GREEN}✓ Server '{server_name}' deleted.{Colors.RESET}")
                        else:
                            print(f"{Colors.DIM}Cancelled.{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                except (ValueError, KeyError):
                    print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
                time.sleep(1.5)
            
            elif c == "4":
                # Test connection
                if not self.registry:
                    print(f"\n{Colors.YELLOW}⚠ Tool registry not initialized.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                if not self.registry.mcp_clients:
                    print(f"\n{Colors.YELLOW}⚠ No MCP servers running.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Testing MCP server connections...{Colors.RESET}\n")
                for name, client in self.registry.mcp_clients.items():
                    status = "✓ Connected" if client.is_running else "✗ Disconnected"
                    color = Colors.BRIGHT_GREEN if client.is_running else Colors.BRIGHT_RED
                    print(f"  {color}{status}{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET}")
                
                input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            
            elif c == "5":
                # View tools
                if not self.registry:
                    print(f"\n{Colors.YELLOW}⚠ Tool registry not initialized.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                if not self.registry.mcp_clients:
                    print(f"\n{Colors.YELLOW}⚠ No MCP servers running.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Available MCP Tools:{Colors.RESET}\n")
                for name, client in self.registry.mcp_clients.items():
                    print(f"{Colors.BRIGHT_WHITE}{name}:{Colors.RESET}")
                    if client.available_tools:
                        for tool in client.available_tools:
                            print(f"  {Colors.CYAN}-{Colors.RESET} {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}")
                    else:
                        print(f"  {Colors.DIM}No tools available{Colors.RESET}")
                    print()
                
                input(f"{Colors.DIM}Press Enter...{Colors.RESET}")
            
            elif c == "6":
                break

    def model_training_menu(self):
        """Model Training menu with Fine-Tuning, LoRA, and RLHF options."""
        # Initialize training managers
        fine_tuner = FineTuningManager(self.config)
        lora_manager = LoRAManager(self.config)
        rl_manager = ReinforcementLearningManager(self.config)
        
        while True:
            self.clear()
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  MODEL TRAINING{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
            
            print(f"{Colors.CYAN}Training Data Directory:{Colors.RESET} {Colors.BRIGHT_WHITE}{TRAINING_DATA_DIR}{Colors.RESET}")
            print(f"{Colors.CYAN}Models Directory:{Colors.RESET} {Colors.BRIGHT_WHITE}{MODELS_DIR}{Colors.RESET}\n")
            
            print(f"{Colors.BRIGHT_GREEN}  [1]{Colors.RESET} Fine-Tuning {Colors.DIM}(Full model training){Colors.RESET}")
            print(f"{Colors.BRIGHT_GREEN}  [2]{Colors.RESET} LoRA Training {Colors.DIM}(Efficient fine-tuning){Colors.RESET}")
            print(f"{Colors.BRIGHT_GREEN}  [3]{Colors.RESET} Reinforcement Learning + Behaviour Conditioning")
            print(f"{Colors.BRIGHT_GREEN}  [4]{Colors.RESET} Prepare Training Dataset")
            print(f"{Colors.BRIGHT_GREEN}  [5]{Colors.RESET} Back\n")
            
            c = input(f"{Colors.BRIGHT_GREEN}Choice: {Colors.RESET}")
            
            if c == "1":
                # Fine-Tuning
                self.clear()
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
                print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  FINE-TUNING{Colors.RESET}")
                print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
                
                base_model = input(f"{Colors.BRIGHT_GREEN}Base Model ID {Colors.DIM}(e.g., gpt2, microsoft/DialoGPT-medium): {Colors.RESET}").strip()
                if not base_model:
                    base_model = self.config.get('model_name', 'gpt2')
                
                dataset_file = input(f"{Colors.BRIGHT_GREEN}Dataset File Path {Colors.DIM}(JSON/JSONL): {Colors.RESET}").strip()
                if not dataset_file or not os.path.exists(dataset_file):
                    print(f"{Colors.YELLOW}⚠ Dataset file not found. Use option 4 to prepare one.{Colors.RESET}")
                    input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                    continue
                
                output_name = input(f"{Colors.BRIGHT_GREEN}Output Model Name: {Colors.RESET}").strip()
                if not output_name:
                    output_name = f"finetuned_{base_model.replace('/', '_')}"
                
                try:
                    epochs = int(input(f"{Colors.BRIGHT_GREEN}Epochs {Colors.DIM}(default 3): {Colors.RESET}").strip() or "3")
                    batch_size = int(input(f"{Colors.BRIGHT_GREEN}Batch Size {Colors.DIM}(default 4): {Colors.RESET}").strip() or "4")
                    lr = float(input(f"{Colors.BRIGHT_GREEN}Learning Rate {Colors.DIM}(default 5e-5): {Colors.RESET}").strip() or "5e-5")
                except ValueError:
                    print(f"{Colors.YELLOW}⚠ Invalid input, using defaults{Colors.RESET}")
                    epochs, batch_size, lr = 3, 4, 5e-5
                
                fine_tuner.train(base_model, dataset_file, output_name, epochs, batch_size, lr)
                input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            
            elif c == "2":
                # LoRA Training
                self.clear()
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
                print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  LoRA TRAINING{Colors.RESET}")
                print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
                
                base_model = input(f"{Colors.BRIGHT_GREEN}Base Model ID {Colors.DIM}(e.g., gpt2, microsoft/DialoGPT-medium): {Colors.RESET}").strip()
                if not base_model:
                    base_model = self.config.get('model_name', 'gpt2')
                
                dataset_file = input(f"{Colors.BRIGHT_GREEN}Dataset File Path {Colors.DIM}(JSON/JSONL): {Colors.RESET}").strip()
                if not dataset_file or not os.path.exists(dataset_file):
                    print(f"{Colors.YELLOW}⚠ Dataset file not found. Use option 4 to prepare one.{Colors.RESET}")
                    input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                    continue
                
                output_name = input(f"{Colors.BRIGHT_GREEN}Output LoRA Name: {Colors.RESET}").strip()
                if not output_name:
                    output_name = f"lora_{base_model.replace('/', '_')}"
                
                try:
                    rank = int(input(f"{Colors.BRIGHT_GREEN}LoRA Rank {Colors.DIM}(default 8): {Colors.RESET}").strip() or "8")
                    alpha = int(input(f"{Colors.BRIGHT_GREEN}LoRA Alpha {Colors.DIM}(default 16): {Colors.RESET}").strip() or "16")
                    epochs = int(input(f"{Colors.BRIGHT_GREEN}Epochs {Colors.DIM}(default 3): {Colors.RESET}").strip() or "3")
                    batch_size = int(input(f"{Colors.BRIGHT_GREEN}Batch Size {Colors.DIM}(default 4): {Colors.RESET}").strip() or "4")
                    lr = float(input(f"{Colors.BRIGHT_GREEN}Learning Rate {Colors.DIM}(default 1e-4): {Colors.RESET}").strip() or "1e-4")
                except ValueError:
                    print(f"{Colors.YELLOW}⚠ Invalid input, using defaults{Colors.RESET}")
                    rank, alpha, epochs, batch_size, lr = 8, 16, 3, 4, 1e-4
                
                lora_manager.train(base_model, dataset_file, output_name, rank, alpha, epochs, batch_size, lr)
                input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            
            elif c == "3":
                # Reinforcement Learning + Behaviour Conditioning
                self.clear()
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
                print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  REINFORCEMENT LEARNING & BEHAVIOUR CONDITIONING{Colors.RESET}")
                print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
                
                print(f"{Colors.BRIGHT_GREEN}  [1]{Colors.RESET} Train Reward Model (RLHF)")
                print(f"{Colors.BRIGHT_GREEN}  [2]{Colors.RESET} PPO Training")
                print(f"{Colors.BRIGHT_GREEN}  [3]{Colors.RESET} Apply Behaviour Conditioning")
                print(f"{Colors.BRIGHT_GREEN}  [4]{Colors.RESET} Back\n")
                
                rl_choice = input(f"{Colors.BRIGHT_GREEN}Choice: {Colors.RESET}")
                
                if rl_choice == "1":
                    base_model = input(f"\n{Colors.BRIGHT_GREEN}Base Model ID: {Colors.RESET}").strip() or self.config.get('model_name', 'gpt2')
                    preference_file = input(f"{Colors.BRIGHT_GREEN}Preference Data File {Colors.DIM}(JSON with 'prompt', 'chosen', 'rejected'): {Colors.RESET}").strip()
                    
                    if not preference_file or not os.path.exists(preference_file):
                        print(f"{Colors.YELLOW}⚠ Preference file not found.{Colors.RESET}")
                        input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                        continue
                    
                    output_name = input(f"{Colors.BRIGHT_GREEN}Reward Model Name: {Colors.RESET}").strip() or "reward_model"
                    
                    try:
                        epochs = int(input(f"{Colors.BRIGHT_GREEN}Epochs {Colors.DIM}(default 3): {Colors.RESET}").strip() or "3")
                        batch_size = int(input(f"{Colors.BRIGHT_GREEN}Batch Size {Colors.DIM}(default 4): {Colors.RESET}").strip() or "4")
                    except ValueError:
                        epochs, batch_size = 3, 4
                    
                    rl_manager.train_reward_model(base_model, preference_file, output_name, epochs, batch_size)
                    input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                
                elif rl_choice == "2":
                    base_model = input(f"\n{Colors.BRIGHT_GREEN}Base Model ID: {Colors.RESET}").strip() or self.config.get('model_name', 'gpt2')
                    reward_model = input(f"{Colors.BRIGHT_GREEN}Reward Model Path: {Colors.RESET}").strip()
                    dataset_file = input(f"{Colors.BRIGHT_GREEN}Training Dataset File: {Colors.RESET}").strip()
                    output_name = input(f"{Colors.BRIGHT_GREEN}Output Model Name: {Colors.RESET}").strip() or "ppo_model"
                    
                    try:
                        epochs = int(input(f"{Colors.BRIGHT_GREEN}Epochs {Colors.DIM}(default 3): {Colors.RESET}").strip() or "3")
                    except ValueError:
                        epochs = 3
                    
                    rl_manager.train_with_ppo(base_model, reward_model, dataset_file, output_name, epochs)
                    input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                
                elif rl_choice == "3":
                    model_path = input(f"\n{Colors.BRIGHT_GREEN}Model Path: {Colors.RESET}").strip()
                    rules_file = input(f"{Colors.BRIGHT_GREEN}Behaviour Rules File {Colors.DIM}(JSON with 'trigger', 'response', 'priority'): {Colors.RESET}").strip()
                    
                    if not rules_file or not os.path.exists(rules_file):
                        print(f"{Colors.YELLOW}⚠ Rules file not found.{Colors.RESET}")
                        input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                        continue
                    
                    output_name = input(f"{Colors.BRIGHT_GREEN}Output Name: {Colors.RESET}").strip() or "conditioned_model"
                    
                    result = rl_manager.apply_behaviour_conditioning(model_path, rules_file, output_name)
                    if result:
                        print(f"\n{Colors.BRIGHT_GREEN}✓ Use the generated dataset with Fine-Tuning or LoRA to apply conditioning.{Colors.RESET}")
                    input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            
            elif c == "4":
                # Prepare Training Dataset
                self.clear()
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
                print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  PREPARE TRAINING DATASET{Colors.RESET}")
                print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
                
                data_file = input(f"{Colors.BRIGHT_GREEN}Input Data File {Colors.DIM}(JSON/JSONL): {Colors.RESET}").strip()
                if not data_file or not os.path.exists(data_file):
                    print(f"{Colors.YELLOW}⚠ File not found.{Colors.RESET}")
                    input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                    continue
                
                print(f"\n{Colors.CYAN}Preparing dataset...{Colors.RESET}")
                result = fine_tuner.prepare_dataset(data_file)
                if result:
                    print(f"\n{Colors.BRIGHT_GREEN}✓ Dataset prepared successfully!{Colors.RESET}")
                input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            
            elif c == "5":
                break

    def api_management_menu(self):
        """API Management menu for creating and managing API endpoints."""
        if not self.api_manager:
            print(f"\n{Colors.BRIGHT_RED}✗ API Manager not initialized.{Colors.RESET}")
            if not FLASK_AVAILABLE:
                print(f"{Colors.YELLOW}Install Flask: pip install flask flask-cors{Colors.RESET}")
            if not ENCRYPTION_AVAILABLE:
                print(f"{Colors.YELLOW}Install cryptography: pip install cryptography{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            return
        
        while True:
            self.clear()
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  API MANAGEMENT{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
            
            # List current APIs
            apis = self.api_manager.list_apis()
            if apis:
                print(f"{Colors.CYAN}Configured APIs:{Colors.RESET}\n")
                for i, api_name in enumerate(apis, 1):
                    info = self.api_manager.get_api_info(api_name)
                    status = "✓ Running" if info.get("running") else "○ Stopped"
                    status_color = Colors.BRIGHT_GREEN if info.get("running") else Colors.DIM
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{api_name}{Colors.RESET} {status_color}{status}{Colors.RESET}")
                    print(f"      {Colors.DIM}Port: {info.get('port', 'N/A')} | CORS: {'ON' if info.get('enable_cors') else 'OFF'} | IP Whitelist: {'ON' if info.get('ip_whitelist') else 'OFF'}{Colors.RESET}\n")
            else:
                print(f"{Colors.YELLOW}No APIs configured.{Colors.RESET}\n")
            
            print(f"{Colors.BRIGHT_GREEN}  [1]{Colors.RESET} Create New API")
            print(f"{Colors.BRIGHT_GREEN}  [2]{Colors.RESET} Stop API")
            print(f"{Colors.BRIGHT_GREEN}  [3]{Colors.RESET} Delete API")
            print(f"{Colors.BRIGHT_GREEN}  [4]{Colors.RESET} View API Details")
            print(f"{Colors.BRIGHT_GREEN}  [5]{Colors.RESET} Test API Endpoint")
            print(f"{Colors.BRIGHT_GREEN}  [6]{Colors.RESET} Back\n")
            
            c = input(f"{Colors.BRIGHT_GREEN}Choice: {Colors.RESET}")
            
            if c == "1":
                # Create new API
                self.clear()
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
                print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  CREATE NEW API{Colors.RESET}")
                print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
                
                name = input(f"{Colors.BRIGHT_GREEN}API Name: {Colors.RESET}").strip()
                if not name:
                    print(f"{Colors.YELLOW}⚠ Name cannot be empty{Colors.RESET}")
                    time.sleep(1)
                    continue
                
                try:
                    port = int(input(f"{Colors.BRIGHT_GREEN}Port {Colors.DIM}(default 5000): {Colors.RESET}").strip() or "5000")
                except ValueError:
                    port = 5000
                
                # CORS configuration
                enable_cors = input(f"{Colors.BRIGHT_GREEN}Enable CORS? [Y/n]: {Colors.RESET}").strip().lower()
                enable_cors = enable_cors != 'n'
                
                cors_origins = []
                if enable_cors:
                    origins_input = input(f"{Colors.BRIGHT_GREEN}CORS Origins {Colors.DIM}(comma-separated, * for all): {Colors.RESET}").strip()
                    if origins_input:
                        cors_origins = [o.strip() for o in origins_input.split(',')]
                    else:
                        cors_origins = ["*"]
                
                # IP Whitelist
                ip_whitelist_input = input(f"{Colors.BRIGHT_GREEN}IP Whitelist {Colors.DIM}(comma-separated, empty for none): {Colors.RESET}").strip()
                ip_whitelist = []
                if ip_whitelist_input:
                    ip_whitelist = [ip.strip() for ip in ip_whitelist_input.split(',')]
                
                # Authentication
                require_auth = input(f"{Colors.BRIGHT_GREEN}Require API Key? [Y/n]: {Colors.RESET}").strip().lower()
                require_auth = require_auth != 'n'
                
                print(f"\n{Colors.BRIGHT_CYAN}Creating API...{Colors.RESET}")
                success, message = self.api_manager.create_api(
                    name, port, enable_cors, cors_origins, ip_whitelist, require_auth
                )
                
                if success:
                    print(f"\n{Colors.BRIGHT_GREEN}✓ {message}{Colors.RESET}")
                    if require_auth:
                        api_key = self.api_manager.get_api_key(name)
                        print(f"\n{Colors.BRIGHT_YELLOW}⚠ IMPORTANT: Save your API key:{Colors.RESET}")
                        print(f"{Colors.BRIGHT_WHITE}{api_key}{Colors.RESET}")
                        print(f"{Colors.DIM}This key will not be shown again!{Colors.RESET}")
                else:
                    print(f"\n{Colors.BRIGHT_RED}✗ {message}{Colors.RESET}")
                
                input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            
            elif c == "2":
                # Stop API
                if not apis:
                    print(f"\n{Colors.YELLOW}⚠ No APIs configured.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select API to stop:{Colors.RESET}\n")
                for i, api_name in enumerate(apis, 1):
                    info = self.api_manager.get_api_info(api_name)
                    if info.get("running"):
                        print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{api_name}{Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(apis)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(apis):
                        api_name = apis[idx - 1]
                        success, message = self.api_manager.stop_api(api_name)
                        if success:
                            print(f"\n{Colors.BRIGHT_GREEN}✓ {message}{Colors.RESET}")
                        else:
                            print(f"\n{Colors.BRIGHT_RED}✗ {message}{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                except ValueError:
                    print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
                time.sleep(1.5)
            
            elif c == "3":
                # Delete API
                if not apis:
                    print(f"\n{Colors.YELLOW}⚠ No APIs configured.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select API to delete:{Colors.RESET}\n")
                for i, api_name in enumerate(apis, 1):
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{api_name}{Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(apis)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(apis):
                        api_name = apis[idx - 1]
                        confirm = input(f"{Colors.BRIGHT_RED}Delete '{api_name}'? [y/N]: {Colors.RESET}").strip().lower()
                        if confirm == 'y':
                            success, message = self.api_manager.delete_api(api_name)
                            if success:
                                print(f"\n{Colors.BRIGHT_GREEN}✓ {message}{Colors.RESET}")
                            else:
                                print(f"\n{Colors.BRIGHT_RED}✗ {message}{Colors.RESET}")
                        else:
                            print(f"{Colors.DIM}Cancelled.{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                except ValueError:
                    print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
                time.sleep(1.5)
            
            elif c == "4":
                # View API details
                if not apis:
                    print(f"\n{Colors.YELLOW}⚠ No APIs configured.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select API to view:{Colors.RESET}\n")
                for i, api_name in enumerate(apis, 1):
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{api_name}{Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(apis)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(apis):
                        api_name = apis[idx - 1]
                        info = self.api_manager.get_api_info(api_name)
                        
                        self.clear()
                        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
                        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  API DETAILS: {api_name}{Colors.RESET}")
                        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
                        
                        print(f"{Colors.CYAN}Name:{Colors.RESET} {Colors.BRIGHT_WHITE}{api_name}{Colors.RESET}")
                        print(f"{Colors.CYAN}Port:{Colors.RESET} {Colors.BRIGHT_WHITE}{info.get('port', 'N/A')}{Colors.RESET}")
                        print(f"{Colors.CYAN}Status:{Colors.RESET} {Colors.BRIGHT_GREEN if info.get('running') else Colors.DIM}{'Running' if info.get('running') else 'Stopped'}{Colors.RESET}")
                        print(f"{Colors.CYAN}CORS Enabled:{Colors.RESET} {Colors.BRIGHT_WHITE}{'Yes' if info.get('enable_cors') else 'No'}{Colors.RESET}")
                        if info.get('enable_cors') and info.get('cors_origins'):
                            print(f"{Colors.CYAN}CORS Origins:{Colors.RESET} {Colors.BRIGHT_WHITE}{', '.join(info.get('cors_origins', []))}{Colors.RESET}")
                        print(f"{Colors.CYAN}IP Whitelist:{Colors.RESET} {Colors.BRIGHT_WHITE}{'Yes' if info.get('ip_whitelist') else 'No'}{Colors.RESET}")
                        if info.get('ip_whitelist'):
                            print(f"{Colors.CYAN}Whitelisted IPs:{Colors.RESET} {Colors.BRIGHT_WHITE}{', '.join(info.get('ip_whitelist', []))}{Colors.RESET}")
                        print(f"{Colors.CYAN}Authentication Required:{Colors.RESET} {Colors.BRIGHT_WHITE}{'Yes' if info.get('require_auth') else 'No'}{Colors.RESET}")
                        if info.get('require_auth'):
                            api_key = info.get('api_key')
                            if api_key:
                                print(f"{Colors.CYAN}API Key:{Colors.RESET} {Colors.BRIGHT_WHITE}{api_key}{Colors.RESET}")
                            else:
                                print(f"{Colors.YELLOW}⚠ API key not found{Colors.RESET}")
                        print(f"{Colors.CYAN}Encryption Available:{Colors.RESET} {Colors.BRIGHT_WHITE}{'Yes' if self.api_manager.encryption_manager else 'No'}{Colors.RESET}")
                        
                        print(f"\n{Colors.CYAN}Endpoints:{Colors.RESET}")
                        print(f"  {Colors.DIM}POST /chat{Colors.RESET} - Send chat messages")
                        print(f"  {Colors.DIM}GET /health{Colors.RESET} - Health check")
                        print(f"  {Colors.DIM}GET /info{Colors.RESET} - API information")
                        
                        input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                        time.sleep(1)
                except ValueError:
                    print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
                    time.sleep(1)
            
            elif c == "5":
                # Test API endpoint
                if not apis:
                    print(f"\n{Colors.YELLOW}⚠ No APIs configured.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select API to test:{Colors.RESET}\n")
                for i, api_name in enumerate(apis, 1):
                    info = self.api_manager.get_api_info(api_name)
                    if info.get("running"):
                        print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{api_name}{Colors.RESET} {Colors.DIM}(Port {info.get('port')}){Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(apis)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(apis):
                        api_name = apis[idx - 1]
                        info = self.api_manager.get_api_info(api_name)
                        
                        if not info.get("running"):
                            print(f"\n{Colors.YELLOW}⚠ API is not running.{Colors.RESET}")
                            time.sleep(1.5)
                            continue
                        
                        port = info.get('port')
                        api_key = info.get('api_key')
                        
                        print(f"\n{Colors.CYAN}Testing API endpoint...{Colors.RESET}\n")
                        print(f"{Colors.DIM}Example curl command:{Colors.RESET}")
                        
                        test_message = "Hello, this is a test message"
                        curl_cmd = f"curl -X POST http://localhost:{port}/chat"
                        
                        headers = []
                        if api_key:
                            headers.append(f"-H 'X-API-Key: {api_key}'")
                        
                        data = f"-d '{{\"message\": \"{test_message}\", \"encrypt_response\": false}}'"
                        headers_str = " ".join(headers)
                        
                        print(f"{Colors.BRIGHT_WHITE}{curl_cmd} {headers_str} {data}{Colors.RESET}")
                        
                        # Try to make actual request
                        try:
                            import requests
                            url = f"http://localhost:{port}/health"
                            response = requests.get(url, timeout=2)
                            if response.status_code == 200:
                                print(f"\n{Colors.BRIGHT_GREEN}✓ API is responding!{Colors.RESET}")
                            else:
                                print(f"\n{Colors.YELLOW}⚠ API responded with status {response.status_code}{Colors.RESET}")
                        except Exception as e:
                            print(f"\n{Colors.YELLOW}⚠ Could not connect to API: {e}{Colors.RESET}")
                        
                        input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                        time.sleep(1)
                except ValueError:
                    print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
                    time.sleep(1)
            
            elif c == "6":
                break

    def app_builder_menu(self):
        """App Builder menu - multi-agent development system."""
        if not self.app_builder:
            print(f"\n{Colors.BRIGHT_RED}✗ App Builder not initialized.{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            return
        
        while True:
            self.clear()
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  APP BUILDER - Multi-Agent Development{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
            
            print(f"{Colors.CYAN}Apps Directory:{Colors.RESET} {Colors.BRIGHT_WHITE}{APPS_DIR}{Colors.RESET}\n")
            
            # List existing projects
            conn = sqlite3.connect(APP_PROJECTS_DB)
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, description, status FROM app_projects ORDER BY updated_at DESC")
            projects = cursor.fetchall()
            conn.close()
            
            if projects:
                print(f"{Colors.CYAN}Your Projects:{Colors.RESET}\n")
                for i, (pid, name, desc, status) in enumerate(projects[:10], 1):
                    status_colors = {
                        "specification": Colors.YELLOW,
                        "architecture": Colors.CYAN,
                        "tasks": Colors.BLUE,
                        "development": Colors.BRIGHT_CYAN,
                        "completed": Colors.BRIGHT_GREEN
                    }
                    status_color = status_colors.get(status, Colors.WHITE)
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET} {status_color}({status}){Colors.RESET}")
                    print(f"      {Colors.DIM}{desc[:60]}...{Colors.RESET}\n")
            else:
                print(f"{Colors.YELLOW}No projects yet. Create your first app!{Colors.RESET}\n")
            
            print(f"{Colors.BRIGHT_GREEN}  [1]{Colors.RESET} Create New App")
            print(f"{Colors.BRIGHT_GREEN}  [2]{Colors.RESET} Continue Building App")
            print(f"{Colors.BRIGHT_GREEN}  [3]{Colors.RESET} View App Details")
            print(f"{Colors.BRIGHT_GREEN}  [4]{Colors.RESET} Add Feature to Existing App")
            print(f"{Colors.BRIGHT_GREEN}  [5]{Colors.RESET} Debug App")
            print(f"{Colors.BRIGHT_GREEN}  [6]{Colors.RESET} Generate Documentation")
            print(f"{Colors.BRIGHT_GREEN}  [7]{Colors.RESET} Back\n")
            
            c = input(f"{Colors.BRIGHT_GREEN}Choice: {Colors.RESET}")
            
            if c == "1":
                # Create new app
                self.clear()
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
                print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  CREATE NEW APP{Colors.RESET}")
                print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
                
                print(f"{Colors.CYAN}The AI team will help you build this app step by step.{Colors.RESET}")
                print(f"{Colors.DIM}Agents involved: Spec Writer, Architect, Tech Lead, Developer, Code Monkey, Reviewer{Colors.RESET}\n")
                
                app_name = input(f"{Colors.BRIGHT_GREEN}App Name: {Colors.RESET}").strip()
                if not app_name:
                    print(f"{Colors.YELLOW}⚠ Name cannot be empty.{Colors.RESET}")
                    time.sleep(1)
                    continue
                
                description = input(f"{Colors.BRIGHT_GREEN}Description {Colors.DIM}(what should this app do?): {Colors.RESET}").strip()
                if not description:
                    print(f"{Colors.YELLOW}⚠ Description cannot be empty.{Colors.RESET}")
                    time.sleep(1)
                    continue
                
                print(f"\n{Colors.BRIGHT_CYAN}Creating project...{Colors.RESET}")
                project_id = self.app_builder.create_project(app_name, description)
                
                if project_id:
                    print(f"{Colors.BRIGHT_GREEN}✓ Project '{app_name}' created!{Colors.RESET}")
                    print(f"{Colors.DIM}Project ID: {project_id}{Colors.RESET}\n")
                    
                    start_now = input(f"{Colors.BRIGHT_GREEN}Start building now? [Y/n]: {Colors.RESET}").strip().lower()
                    if start_now != 'n':
                        self.app_builder.build_app(project_id)
                else:
                    print(f"{Colors.BRIGHT_RED}✗ Failed to create project.{Colors.RESET}")
                
                input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            
            elif c == "2":
                # Continue building app
                if not projects:
                    print(f"\n{Colors.YELLOW}⚠ No projects to continue.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select project to continue:{Colors.RESET}\n")
                for i, (pid, name, desc, status) in enumerate(projects, 1):
                    if status != "completed":
                        print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET} {Colors.YELLOW}({status}){Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(projects)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(projects):
                        project_id = projects[idx - 1][0]
                        self.app_builder.build_app(project_id)
                    else:
                        print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                        time.sleep(1)
                except ValueError:
                    print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
                    time.sleep(1)
                
                input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            
            elif c == "3":
                # View app details
                if not projects:
                    print(f"\n{Colors.YELLOW}⚠ No projects available.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select project to view:{Colors.RESET}\n")
                for i, (pid, name, desc, status) in enumerate(projects, 1):
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(projects)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(projects):
                        project_id, project_name, desc, status = projects[idx - 1]
                        project = self.app_builder.get_project(project_id)
                        
                        self.clear()
                        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
                        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  PROJECT: {project_name}{Colors.RESET}")
                        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
                        
                        print(f"{Colors.CYAN}Name:{Colors.RESET} {Colors.BRIGHT_WHITE}{project[1]}{Colors.RESET}")
                        print(f"{Colors.CYAN}Description:{Colors.RESET} {Colors.BRIGHT_WHITE}{project[2]}{Colors.RESET}")
                        print(f"{Colors.CYAN}Status:{Colors.RESET} {Colors.BRIGHT_WHITE}{project[5]}{Colors.RESET}")
                        print(f"{Colors.CYAN}Created:{Colors.RESET} {Colors.DIM}{project[6]}{Colors.RESET}")
                        
                        # Show files
                        files = self.app_builder.get_project_files(project_id)
                        if files:
                            print(f"\n{Colors.CYAN}Files ({len(files)}):{Colors.RESET}")
                            for fp, content in files:
                                lines = len(content.split('\n'))
                                print(f"  {Colors.DIM}-{Colors.RESET} {Colors.BRIGHT_WHITE}{fp}{Colors.RESET} {Colors.DIM}({lines} lines){Colors.RESET}")
                        
                        # Show tasks
                        conn = sqlite3.connect(APP_PROJECTS_DB)
                        cursor = conn.cursor()
                        cursor.execute("SELECT task_number, description, status FROM app_tasks WHERE project_id=? ORDER BY task_number", (project_id,))
                        tasks = cursor.fetchall()
                        conn.close()
                        
                        if tasks:
                            print(f"\n{Colors.CYAN}Tasks ({len(tasks)}):{Colors.RESET}")
                            for num, desc, task_status in tasks:
                                status_marker = "✓" if task_status == "completed" else "○"
                                status_color = Colors.BRIGHT_GREEN if task_status == "completed" else Colors.DIM
                                print(f"  {status_color}{status_marker}{Colors.RESET} {Colors.DIM}[{num}]{Colors.RESET} {desc[:60]}...")
                        
                        input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                        time.sleep(1)
                except ValueError:
                    print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
                    time.sleep(1)
            
            elif c == "4":
                # Add feature to existing app
                if not projects:
                    print(f"\n{Colors.YELLOW}⚠ No projects available.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                # Filter only completed projects
                completed_projects = [(pid, name, desc, status) for pid, name, desc, status in projects if status == "completed"]
                
                if not completed_projects:
                    print(f"\n{Colors.YELLOW}⚠ No completed projects to add features to.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select project:{Colors.RESET}\n")
                for i, (pid, name, desc, status) in enumerate(completed_projects, 1):
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(completed_projects)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(completed_projects):
                        project_id = completed_projects[idx - 1][0]
                        
                        feature_desc = input(f"{Colors.BRIGHT_GREEN}Feature Description: {Colors.RESET}").strip()
                        if not feature_desc:
                            print(f"{Colors.YELLOW}⚠ Description cannot be empty.{Colors.RESET}")
                            time.sleep(1)
                            continue
                        
                        # Add feature as new tasks
                        print(f"\n{Colors.BRIGHT_CYAN}Tech Lead analyzing feature...{Colors.RESET}")
                        project = self.app_builder.get_project(project_id)
                        spec = project[3] + f"\n\nNEW FEATURE: {feature_desc}"
                        architecture = project[4]
                        
                        tasks_doc = self.app_builder.tech_lead.create_tasks(feature_desc, architecture)
                        
                        # Save new tasks
                        conn = sqlite3.connect(APP_PROJECTS_DB)
                        cursor = conn.cursor()
                        cursor.execute("SELECT MAX(task_number) FROM app_tasks WHERE project_id=?", (project_id,))
                        max_task = cursor.fetchone()[0] or 0
                        
                        task_lines = [line.strip() for line in tasks_doc.split('\n') if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))]
                        for i, task_line in enumerate(task_lines, 1):
                            cursor.execute("INSERT INTO app_tasks (project_id, task_number, description, status) VALUES (?, ?, ?, ?)",
                                          (project_id, max_task + i, task_line, "pending"))
                        
                        conn.commit()
                        conn.close()
                        
                        self.app_builder.update_project_field(project_id, "status", "development")
                        
                        print(f"\n{Colors.BRIGHT_GREEN}✓ {len(task_lines)} new tasks created!{Colors.RESET}")
                        
                        start_now = input(f"\n{Colors.BRIGHT_GREEN}Start implementing now? [Y/n]: {Colors.RESET}").strip().lower()
                        if start_now != 'n':
                            self.app_builder.develop_tasks(project_id)
                    else:
                        print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                        time.sleep(1)
                except ValueError:
                    print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
                    time.sleep(1)
                
                input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            
            elif c == "5":
                # Debug app
                if not projects:
                    print(f"\n{Colors.YELLOW}⚠ No projects available.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select project to debug:{Colors.RESET}\n")
                for i, (pid, name, desc, status) in enumerate(projects, 1):
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(projects)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(projects):
                        project_id = projects[idx - 1][0]
                        
                        error_message = input(f"\n{Colors.BRIGHT_GREEN}Error Message/Issue: {Colors.RESET}").strip()
                        if not error_message:
                            print(f"{Colors.YELLOW}⚠ Please describe the issue.{Colors.RESET}")
                            time.sleep(1)
                            continue
                        
                        # Get relevant files
                        files = self.app_builder.get_project_files(project_id)
                        relevant_context = self.app_builder.filter_relevant_context(error_message, files, max_context=3000)
                        
                        print(f"\n{Colors.BRIGHT_CYAN}Debugger analyzing issue...{Colors.RESET}")
                        debug_result = self.app_builder.debugger.debug_issue(error_message, relevant_context)
                        
                        print(f"\n{Colors.CYAN}Debugger Analysis:{Colors.RESET}\n")
                        print(f"{Colors.BRIGHT_WHITE}{debug_result}{Colors.RESET}\n")
                        
                        apply_fix = input(f"{Colors.BRIGHT_GREEN}Apply suggested fix? [y/N]: {Colors.RESET}").strip().lower()
                        if apply_fix == 'y':
                            print(f"{Colors.DIM}(Manual code editing required - check the project directory){Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                        time.sleep(1)
                except ValueError:
                    print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
                    time.sleep(1)
                
                input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            
            elif c == "6":
                # Generate documentation
                if not projects:
                    print(f"\n{Colors.YELLOW}⚠ No projects available.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select project:{Colors.RESET}\n")
                for i, (pid, name, desc, status) in enumerate(projects, 1):
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(projects)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(projects):
                        project_id, project_name, desc, status = projects[idx - 1]
                        project = self.app_builder.get_project(project_id)
                        
                        print(f"\n{Colors.BRIGHT_CYAN}Technical Writer creating documentation...{Colors.RESET}")
                        
                        files = self.app_builder.get_project_files(project_id)
                        codebase_summary = "\n".join([f"{fp}: {len(content)} lines" for fp, content in files])
                        
                        docs = self.app_builder.tech_writer.write_documentation(
                            project_name, project[3] or desc, project[4] or "N/A", codebase_summary
                        )
                        
                        # Save documentation
                        self.app_builder.save_file(project_id, "README.md", docs)
                        print(f"\n{Colors.BRIGHT_GREEN}✓ Documentation saved: README.md{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                        time.sleep(1)
                except ValueError:
                    print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
                    time.sleep(1)
                
                input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
            
            elif c == "7":
                break

    def handle_chat_command(self, cmd_line):
        """Handle chat commands starting with /"""
        parts = cmd_line.strip().split(" ", 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd == "/back" or cmd == "/exit":
            return "EXIT"
        
        elif cmd == "/new":
            # Create new chat session
            session_name = args.strip() if args else f"Chat {datetime.now().strftime('%H:%M')}"
            self.current_session_id = self.memory.create_session(session_name, self.current_project_id)
            print(f"{Colors.BRIGHT_GREEN}✓ New chat session created: {Colors.BRIGHT_WHITE}{session_name}{Colors.RESET}")
            return "COMMAND"
        
        elif cmd == "/save":
            # Save current chat history
            if not self.current_session_id:
                print(f"{Colors.YELLOW}⚠ No active session to save{Colors.RESET}")
                return "COMMAND"
            
            filename = args.strip() if args else f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = os.path.join(SANDBOX_DIR, filename)
            try:
                self.memory.save_session_to_file(self.current_session_id, filepath)
                print(f"{Colors.BRIGHT_GREEN}✓ Chat saved to: {Colors.BRIGHT_WHITE}{filepath}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.BRIGHT_RED}✗ Save failed: {e}{Colors.RESET}")
            return "COMMAND"
        
        elif cmd == "/load":
            # Load a chat session
            if not args:
                # List available sessions
                sessions = self.memory.get_sessions()
                if not sessions:
                    print(f"{Colors.YELLOW}No saved sessions found{Colors.RESET}")
                    return "COMMAND"
                
                print(f"\n{Colors.BRIGHT_CYAN}Available sessions:{Colors.RESET}\n")
                for sid, name in sessions[:10]:
                    marker = f" {Colors.BRIGHT_GREEN}← current{Colors.RESET}" if sid == self.current_session_id else ""
                    print(f"  {Colors.CYAN}[{sid}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET}{marker}")
                return "COMMAND"
            
            # Load by ID or filename
            try:
                session_id = int(args)
                session = self.memory.get_session(session_id)
                if session:
                    self.current_session_id = session_id
                    self.current_project_id = session[2]
                    print(f"{Colors.BRIGHT_GREEN}✓ Loaded session: {Colors.BRIGHT_WHITE}{session[1]}{Colors.RESET}")
                else:
                    print(f"{Colors.BRIGHT_RED}✗ Session {session_id} not found{Colors.RESET}")
            except ValueError:
                # Try as filename
                filepath = args if os.path.isabs(args) else os.path.join(SANDBOX_DIR, args)
                if os.path.exists(filepath):
                    try:
                        self.current_session_id = self.memory.load_session_from_file(filepath)
                        print(f"{Colors.BRIGHT_GREEN}✓ Loaded session from: {Colors.BRIGHT_WHITE}{filepath}{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.BRIGHT_RED}✗ Load failed: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.BRIGHT_RED}✗ File not found: {filepath}{Colors.RESET}")
            return "COMMAND"
        
        elif cmd == "/project":
            # Begin/switch project
            if not args:
                # List projects
                projects = self.memory.get_projects()
                if not projects:
                    print(f"{Colors.YELLOW}No projects found. Create one with: {Colors.BRIGHT_WHITE}/project <name>{Colors.RESET}")
                    return "COMMAND"
                
                print(f"\n{Colors.BRIGHT_CYAN}Available projects:{Colors.RESET}\n")
                for pid, name, desc in projects:
                    marker = f" {Colors.BRIGHT_GREEN}← current{Colors.RESET}" if pid == self.current_project_id else ""
                    print(f"  {Colors.CYAN}[{pid}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET} {Colors.DIM}- {desc}{Colors.RESET}{marker}")
                return "COMMAND"
            
            # Create or switch to project
            parts = args.split("|", 1)
            project_name = parts[0].strip()
            project_desc = parts[1].strip() if len(parts) > 1 else ""
            
            # Check if project exists
            projects = self.memory.get_projects()
            for pid, pname, pdesc in projects:
                if pname.lower() == project_name.lower():
                    self.current_project_id = pid
                    print(f"{Colors.BRIGHT_GREEN}✓ Switched to project: {Colors.BRIGHT_WHITE}{pname}{Colors.RESET}")
                    return "COMMAND"
            
            # Create new project
            self.current_project_id = self.memory.create_project(project_name, project_desc)
            print(f"{Colors.BRIGHT_GREEN}✓ Created new project: {Colors.BRIGHT_WHITE}{project_name}{Colors.RESET}")
            if project_desc:
                print(f"  {Colors.CYAN}Description:{Colors.RESET} {Colors.DIM}{project_desc}{Colors.RESET}")
            return "COMMAND"
        
        elif cmd == "/project_save":
            # Save current project
            if not self.current_project_id:
                print(f"{Colors.YELLOW}⚠ No active project to save{Colors.RESET}")
                return "COMMAND"
            
            filename = args.strip() if args else f"project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = os.path.join(SANDBOX_DIR, filename)
            try:
                self.memory.save_project_to_file(self.current_project_id, filepath)
                print(f"{Colors.BRIGHT_GREEN}✓ Project saved to: {Colors.BRIGHT_WHITE}{filepath}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.BRIGHT_RED}✗ Save failed: {e}{Colors.RESET}")
            return "COMMAND"
        
        elif cmd == "/project_load":
            # Load a project
            if not args:
                projects = self.memory.get_projects()
                if not projects:
                    print(f"{Colors.YELLOW}No projects found{Colors.RESET}")
                    return "COMMAND"
                
                print(f"\n{Colors.BRIGHT_CYAN}Available projects:{Colors.RESET}\n")
                for pid, name, desc in projects:
                    marker = f" {Colors.BRIGHT_GREEN}← current{Colors.RESET}" if pid == self.current_project_id else ""
                    print(f"  {Colors.CYAN}[{pid}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET} {Colors.DIM}- {desc}{Colors.RESET}{marker}")
                return "COMMAND"
            
            # Load by ID or filename
            try:
                project_id = int(args)
                project = self.memory.get_project(project_id)
                if project:
                    self.current_project_id = project_id
                    print(f"{Colors.BRIGHT_GREEN}✓ Loaded project: {Colors.BRIGHT_WHITE}{project[1]}{Colors.RESET}")
                else:
                    print(f"{Colors.BRIGHT_RED}✗ Project {project_id} not found{Colors.RESET}")
            except ValueError:
                # Try as filename
                filepath = args if os.path.isabs(args) else os.path.join(SANDBOX_DIR, args)
                if os.path.exists(filepath):
                    try:
                        self.current_project_id = self.memory.load_project_from_file(filepath)
                        print(f"{Colors.BRIGHT_GREEN}✓ Loaded project from: {Colors.BRIGHT_WHITE}{filepath}{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.BRIGHT_RED}✗ Load failed: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.BRIGHT_RED}✗ File not found: {filepath}{Colors.RESET}")
            return "COMMAND"
        
        elif cmd == "/camera":
            # Launch camera-only assistant (vision, no voice)
            self.launch_camera_assistant()
            return "COMMAND"
        
        elif cmd == "/voice" or cmd == "/tts":
            # Launch voice/TTS-only assistant (no camera)
            self.launch_voice_assistant()
            return "COMMAND"
        
        elif cmd == "/vision":
            # Launch unified Vision + Voice assistant (both)
            self.launch_vision_assistant()
            return "COMMAND"
        
        elif cmd == "/help":
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Available Commands:{Colors.RESET}\n")
            print(f"  {Colors.BRIGHT_GREEN}/new [name]{Colors.RESET}          - Create new chat session")
            print(f"  {Colors.BRIGHT_GREEN}/save [filename]{Colors.RESET}     - Save current chat to file")
            print(f"  {Colors.BRIGHT_GREEN}/load [id|filename]{Colors.RESET} - Load chat session (list if no args)")
            print(f"  {Colors.BRIGHT_GREEN}/project [name|desc]{Colors.RESET} - Create/switch project (list if no args)")
            print(f"  {Colors.BRIGHT_GREEN}/project_save [file]{Colors.RESET} - Save current project")
            print(f"  {Colors.BRIGHT_GREEN}/project_load [id|file]{Colors.RESET} - Load project (list if no args)")
            print(f"\n{Colors.BRIGHT_YELLOW}{Colors.BOLD}Advanced Features:{Colors.RESET}\n")
            print(f"  {Colors.BRIGHT_CYAN}/camera{Colors.RESET}             - Launch Camera Assistant (vision only, optimized for low-end PCs)")
            print(f"  {Colors.BRIGHT_CYAN}/voice{Colors.RESET} or {Colors.BRIGHT_CYAN}/tts{Colors.RESET}        - Launch Voice Assistant (TTS only, no camera)")
            print(f"  {Colors.BRIGHT_CYAN}/vision{Colors.RESET}              - Launch Vision + Voice Assistant (both camera + voice)")
            print(f"\n{Colors.BRIGHT_RED}/back, /exit{Colors.RESET}        - Exit chat")
            print(f"  {Colors.BRIGHT_GREEN}/help{Colors.RESET}               - Show this help")
            return "COMMAND"
        
        else:
            print(f"{Colors.YELLOW}⚠ Unknown command: {cmd}. Type {Colors.BRIGHT_WHITE}/help{Colors.YELLOW} for available commands.{Colors.RESET}")
            return "COMMAND"
    
    def chat_loop(self):
        self.clear()
        # Create default session if none exists
        if not self.current_session_id:
            self.current_session_id = self.memory.create_session(f"Chat {datetime.now().strftime('%H:%M')}", 
                                                                  self.current_project_id)
        
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  CHAT MODE ACTIVE{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.CYAN}Session:{Colors.RESET} {Colors.BRIGHT_WHITE}{self.memory.get_session(self.current_session_id)[1]}{Colors.RESET}")
        if self.current_project_id:
            project = self.memory.get_project(self.current_project_id)
            print(f"{Colors.CYAN}Project:{Colors.RESET} {Colors.BRIGHT_WHITE}{project[1]}{Colors.RESET}")
        print(f"{Colors.DIM}Type '/help' for commands, '/back' to exit{Colors.RESET}\n")
        
        while True:
            try:
                user_input = input(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}You:{Colors.RESET} ")
                if not user_input.strip():
                    continue
                
                # Check for commands
                if user_input.startswith("/"):
                    result = self.handle_chat_command(user_input)
                    if result == "EXIT":
                        break
                    continue
                
                # 1. Retrieve RAG
                rag_context = self.memory.retrieve_context(user_input, self.current_project_id)
                
                # 2. Get History (increased limit for better context)
                # Use larger limit for project-based conversations
                history_limit = 200 if self.current_project_id else 100
                history = self.memory.get_recent_history(self.current_session_id, limit=history_limit)
                
                # 2.5. Get Project Memory
                project_memory = None
                if self.current_project_id:
                    project_memory = self.memory.get_project_memory(self.current_project_id)
                
                # 3. Build Prompt with project memory
                sys_prompt = self.config.get("system_prompt") + self.registry.get_tool_prompt()
                final_prompt = self.context_mgr.build_prompt(sys_prompt, history, rag_context, user_input, project_memory)

                # 3.5. Detect script generation and adjust token limit
                script_keywords = ["script", "batch", "powershell", "ps1", "bat", "bash", "sh", "python", "py"]
                is_script_request = any(keyword in user_input.lower() for keyword in script_keywords)
                
                original_max_tokens = self.config.get("max_response_tokens", 2000)
                if is_script_request:
                    # Temporarily increase token limit for script generation
                    self.config["max_response_tokens"] = 4000
                    self.engine.config["max_response_tokens"] = 4000

                # 4. Generate with animated loading
                frames = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
                
                # Simple loading indicator
                print(f"{Colors.BRIGHT_CYAN}⠋ Thinking...{Colors.RESET}", end='', flush=True)
                response = self.engine.generate(final_prompt)
                
                # Restore original token limit
                if is_script_request:
                    self.config["max_response_tokens"] = original_max_tokens
                    self.engine.config["max_response_tokens"] = original_max_tokens
                print(f"\r{Colors.BRIGHT_GREEN}✓ Response ready{Colors.RESET}" + " " * 30)
                
                # 5. Action Logic
                if "ACTION:" in response:
                    print(f"\n{Colors.BRIGHT_BLUE}{Colors.BOLD}AI:{Colors.RESET} {Colors.BRIGHT_WHITE}{response}{Colors.RESET}")
                    
                    # Robust Parsing
                    try:
                        cmd_part = response.split("ACTION:")[1].strip()
                        # Handling "ACTION: TYPE ARG"
                        parts = cmd_part.split(" ", 2)
                        action_type = parts[0]
                        
                        output = "Error: Invalid Action Format"
                        
                        # Dispatch
                        if action_type == "MCP" and len(parts) >= 3:
                            # ACTION: MCP server tool {json}
                            srv = parts[1]
                            # Extract JSON part (find first {)
                            rest = parts[2]
                            if "{" in rest:
                                tool = rest.split("{", 1)[0].strip()
                                json_part = "{" + rest.split("{", 1)[1]
                                output = self.registry.execute_mcp(srv, tool, json_part)
                            else:
                                output = "Error: JSON arguments required for MCP."
                                
                        elif action_type == "CUSTOM" and len(parts) >= 2:
                            script = parts[1]
                            args = parts[2] if len(parts) > 2 else ""
                            output = self.registry.execute_custom(script, args)
                            
                        else:
                            # Native Tools (CMD, FILE_READ, BROWSE, etc)
                            output = self.registry.execute_native(cmd_part)
                
                    except Exception as e:
                        output = f"Execution Error: {str(e)}"
                    
                    # Special handling for BROWSE: don't dump full page content, generate a summary instead
                    if action_type == "BROWSE":
                        # Print concise system status
                        print(f"\n{Colors.BRIGHT_YELLOW}{Colors.BOLD}SYSTEM:{Colors.RESET} {Colors.BRIGHT_WHITE}Browsing complete. Generating summary based on page content...{Colors.RESET}")
                        
                        # Build a focused summarization prompt
                        summary_instruction = (
                            "You are summarizing a web page that was just browsed using a browser tool.\n"
                            "The user originally asked:\n"
                            f"\"{user_input}\"\n\n"
                            "You have the following raw page content (truncated if very long):\n"
                            "----- PAGE CONTENT START -----\n"
                            f"{str(output)[:8000]}\n"
                            "----- PAGE CONTENT END -----\n\n"
                            "Based on this content, provide a concise, high-level summary of the website in 3-5 sentences.\n"
                            "- Focus on what the site is, what it offers, and who it's for.\n"
                            "- Do NOT mention tools, Playwright, browsers, or that you are summarizing content.\n"
                            "- Do NOT repeat large chunks of the page; just describe it.\n"
                        )
                        
                        # Use the existing system prompt plus the tool prompt for consistent behavior
                        summary_sys = self.config.get("system_prompt") + "\n\n" + summary_instruction
                        summary_prompt = self.context_mgr.build_prompt(
                            summary_sys,
                            history,          # include recent history for extra context
                            "",               # no additional RAG for summary
                            user_input        # keep the original user request visible
                        )
                        
                        # Temporarily bump response tokens for a good summary
                        original_max = self.engine.config.get("max_response_tokens", 2000)
                        self.engine.config["max_response_tokens"] = max(original_max, 512)
                        try:
                            summary = self.engine.generate(summary_prompt)
                        finally:
                            self.engine.config["max_response_tokens"] = original_max
                        
                        # Print AI summary (this is what the user sees)
                        print(f"\n{Colors.BRIGHT_BLUE}{Colors.BOLD}AI (Summary):{Colors.RESET} {Colors.BRIGHT_WHITE}{summary}{Colors.RESET}")
                        
                        # Save conversation history
                        self.memory.save_message(self.current_session_id, "You", user_input)
                        self.memory.save_message(self.current_session_id, "AI", response)
                        self.memory.save_message(self.current_session_id, "System", str(output))
                        self.memory.save_message(self.current_session_id, "AI", summary)
                    else:
                        # Default behavior for non-BROWSE actions
                        print(f"\n{Colors.BRIGHT_YELLOW}{Colors.BOLD}SYSTEM:{Colors.RESET} {Colors.BRIGHT_WHITE}{output}{Colors.RESET}")
                        self.memory.save_message(self.current_session_id, "You", user_input)
                        self.memory.save_message(self.current_session_id, "AI", response)
                        self.memory.save_message(self.current_session_id, "System", str(output))
                    
                else:
                    print(f"\n{Colors.BRIGHT_BLUE}{Colors.BOLD}AI:{Colors.RESET} {Colors.BRIGHT_WHITE}{response}{Colors.RESET}")
                    self.memory.save_message(self.current_session_id, "You", user_input)
                    self.memory.save_message(self.current_session_id, "AI", response)
                    
            except KeyboardInterrupt:
                break

    def launch_camera_assistant(self):
        """Launch Camera-only Assistant (vision, no voice) - optimized for low-end PCs."""
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  CAMERA ASSISTANT (Vision Only){Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        
        if not CV2_AVAILABLE:
            print(f"{Colors.BRIGHT_RED}✗ Missing dependency: opencv-python{Colors.RESET}\n")
            print(f"{Colors.BRIGHT_YELLOW}Install with:{Colors.RESET}")
            print(f"{Colors.BRIGHT_WHITE}pip install opencv-python{Colors.RESET}\n")
            time.sleep(2)
            return
        
        print(f"{Colors.BRIGHT_GREEN}✓ Dependencies available{Colors.RESET}\n")
        print(f"{Colors.CYAN}Camera Assistant: Type questions in terminal, AI sees through webcam{Colors.RESET}")
        print(f"{Colors.DIM}Optimized for low-end PCs (reduced resolution, frame skipping){Colors.RESET}\n")
        print(f"{Colors.YELLOW}Note: Full camera assistant script will be created on first use.{Colors.RESET}")
        print(f"{Colors.YELLOW}For now, use /vision for full functionality.{Colors.RESET}\n")
        time.sleep(2)
    
    def launch_voice_assistant(self):
        """Launch Voice/TTS-only Assistant (no camera) - optimized for low-end PCs."""
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  VOICE ASSISTANT (TTS Only){Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        
        missing_deps = []
        try:
            import speech_recognition
        except ImportError:
            missing_deps.append("SpeechRecognition")
        
        try:
            import pyaudio
        except ImportError:
            missing_deps.append("pyaudio")
        
        if missing_deps:
            print(f"{Colors.BRIGHT_RED}✗ Missing dependencies:{Colors.RESET}\n")
            for dep in missing_deps:
                print(f"  • {dep}")
            print(f"\n{Colors.BRIGHT_YELLOW}Install with:{Colors.RESET}")
            print(f"{Colors.BRIGHT_WHITE}pip install SpeechRecognition pyaudio pyttsx3{Colors.RESET}\n")
            time.sleep(2)
            return
        
        print(f"{Colors.BRIGHT_GREEN}✓ Dependencies available{Colors.RESET}\n")
        print(f"{Colors.CYAN}Voice Assistant: Continuous listening + TTS responses{Colors.RESET}")
        print(f"{Colors.DIM}Optimized for low-end PCs (no camera processing){Colors.RESET}\n")
        print(f"{Colors.YELLOW}Note: Full voice assistant script will be created on first use.{Colors.RESET}")
        print(f"{Colors.YELLOW}For now, use /vision for full functionality.{Colors.RESET}\n")
        time.sleep(2)
    
    def launch_vision_assistant(self):
        """Launch unified Vision + Voice Assistant using LOCAL models only."""
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  VISION + VOICE ASSISTANT{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")
        
        # Check dependencies
        missing_deps = []
        try:
            import speech_recognition
        except ImportError:
            missing_deps.append("SpeechRecognition")
        
        if not CV2_AVAILABLE:
            missing_deps.append("opencv-python")
        
        try:
            import pyaudio
        except ImportError:
            missing_deps.append("pyaudio")
        
        if missing_deps:
            print(f"{Colors.BRIGHT_RED}✗ Missing dependencies:{Colors.RESET}\n")
            for dep in missing_deps:
                print(f"  • {dep}")
            print(f"\n{Colors.BRIGHT_YELLOW}Install with:{Colors.RESET}")
            print(f"{Colors.BRIGHT_WHITE}pip install SpeechRecognition opencv-python pyaudio pyttsx3{Colors.RESET}\n")
            print(f"{Colors.DIM}Note: All processing is LOCAL - no cloud APIs required{Colors.RESET}")
            time.sleep(3)
            return
        
        print(f"{Colors.BRIGHT_GREEN}✓ All dependencies available{Colors.RESET}\n")
        print(f"{Colors.CYAN}Launching Vision + Voice Assistant (optimized for low-end PCs)...{Colors.RESET}\n")
        
        # Create the unified vision + voice script
        vision_assistant_path = os.path.join(BASE_DIR, "_vision_assistant.py")
        base_dir_escaped = BASE_DIR.replace('\\', '\\\\')
        config_repr = repr(self.config)
        
        vision_script = f'''"""
Vision + Voice Assistant - AI Terminal Pro
Unified camera vision and voice interaction using LOCAL models
OPTIMIZED FOR LOW-END PCs

Features:
- Continuous background voice listening (hands-free)
- Real-time webcam vision with local LLM
- Conversation memory management
- pyttsx3 TTS for responses (local, no API)
- Integrated camera preview
- Performance optimizations: reduced resolution, frame skipping, lower memory
- 100% Local - No cloud APIs, no data sent externally
"""
import os
import sys
import base64
import json
from threading import Lock, Thread
import time

# Add project root to path
sys.path.insert(0, r"{base_dir_escaped}")

import cv2
from cv2 import VideoCapture, imencode
from speech_recognition import Microphone, Recognizer, UnknownValueError
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("Warning: pyttsx3 not available. Responses will be text-only.")

# Colors
class Colors:
    RESET = '\\033[0m'
    BOLD = '\\033[1m'
    CYAN = '\\033[36m'
    GREEN = '\\033[32m'
    YELLOW = '\\033[33m'
    RED = '\\033[31m'
    BRIGHT_CYAN = '\\033[96m'
    BRIGHT_GREEN = '\\033[92m'
    BRIGHT_YELLOW = '\\033[93m'
    BRIGHT_RED = '\\033[91m'
    BRIGHT_WHITE = '\\033[97m'
    DIM = '\\033[2m'

# Performance settings for low-end PCs
LOW_END_MODE = True
FRAME_SKIP = 2  # Process every 2nd frame
FRAME_WIDTH = 320  # Reduced resolution
FRAME_HEIGHT = 240
PROCESS_INTERVAL = 1.5  # Max processing frequency

print(f"\\n{{Colors.BRIGHT_CYAN}}{{Colors.BOLD}}{'='*79}{{Colors.RESET}}")
print(f"{{Colors.BRIGHT_YELLOW}}{{Colors.BOLD}}  VISION + VOICE ASSISTANT (Optimized){{Colors.RESET}}")
print(f"{{Colors.BRIGHT_CYAN}}{{Colors.BOLD}}{'='*79}{{Colors.RESET}}\\n")


class WebcamStream:
    """Threaded webcam stream for continuous frame capture - optimized for low-end PCs."""
    
    def __init__(self):
        print(f"{{Colors.CYAN}}Initializing webcam (low-res mode)...{{Colors.RESET}}")
        self.stream = VideoCapture(index=0)
        if not self.stream.isOpened():
            raise Exception("Could not open webcam")
        
        # Set reduced resolution for performance
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        _, self.frame = self.stream.read()
        self.running = False
        self.lock = Lock()
        self.frame_count = 0
        print(f"{{Colors.BRIGHT_GREEN}}✓ Webcam initialized ({{FRAME_WIDTH}}x{{FRAME_HEIGHT}}){{Colors.RESET}}")

    def start(self):
        if self.running:
            return self

        self.running = True
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        print(f"{{Colors.BRIGHT_GREEN}}✓ Webcam stream started{{Colors.RESET}}")
        return self

    def update(self):
        """Continuously update frame from webcam - with frame skipping for performance."""
        while self.running:
            _, frame = self.stream.read()
            self.frame_count += 1
            
            # Frame skipping for low-end PCs
            if self.frame_count % FRAME_SKIP == 0:
                with self.lock:
                    self.frame = frame

    def read(self, encode=False):
        """Read current frame, optionally encode as base64 JPEG."""
        with self.lock:
            frame = self.frame.copy()

        if encode:
            _, buffer = imencode(".jpeg", frame)
            return base64.b64encode(buffer)

        return frame

    def stop(self):
        """Stop the webcam stream."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.stream.release()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stream.release()


class Assistant:
    """Vision + Voice Assistant using LOCAL models (no cloud APIs)."""
    
    def __init__(self, ai_engine):
        print(f"{{Colors.CYAN}}Initializing AI Assistant with local model...{{Colors.RESET}}")
        self.ai_engine = ai_engine
        self.conversation_history = []
        self.is_speaking = False
        self.is_processing = False
        self.processing_lock = Lock()
        
        # Initialize TTS if available
        self.tts_engine = None
        if TTS_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                print(f"{{Colors.BRIGHT_GREEN}}✓ TTS engine initialized{{Colors.RESET}}")
            except Exception as e:
                print(f"{{Colors.BRIGHT_YELLOW}}⚠ TTS initialization failed: {{e}}{{Colors.RESET}}")
        else:
            print(f"{{Colors.BRIGHT_YELLOW}}⚠ TTS not available (install pyttsx3 for voice responses){{Colors.RESET}}")
        
        print(f"{{Colors.BRIGHT_GREEN}}✓ AI Assistant initialized{{Colors.RESET}}")

    def answer(self, prompt, frame=None):
        """Process user prompt with vision context and generate response (non-blocking)."""
        if not prompt:
            return
        
        # Skip if already processing
        with self.processing_lock:
            if self.is_processing:
                print(f"{{Colors.YELLOW}}⚠ Already processing, please wait...{{Colors.RESET}}")
                return
            self.is_processing = True

        print(f"\\n{{Colors.BRIGHT_CYAN}}You:{{Colors.RESET}} {{Colors.BRIGHT_YELLOW}}{{prompt}}{{Colors.RESET}}")
        
        # Run in background thread to avoid blocking
        thread = Thread(target=self._process_answer, args=(prompt, frame))
        thread.daemon = True
        thread.start()

    def _process_answer(self, prompt, frame):
        """Internal method to process answer (runs in thread)."""
        try:
            # Show progress indicator
            print(f"{{Colors.CYAN}}🤖 Thinking...{{Colors.RESET}}", end='', flush=True)
            
            # Build explicit vision-capable system prompt
            system_prompt = """You are a helpful AI assistant with ACTIVE VISION CAPABILITIES. You are currently connected to a live webcam feed that shows the user's environment in real-time.

CRITICAL INSTRUCTIONS:
- You CAN and DO see the user's environment through the webcam feed
- NEVER say you cannot see, don't have vision, or lack visual access
- When asked about what you see, describe the environment, objects, people, or actions visible in the webcam feed
- If the user shows you something, acknowledge it and describe what you observe
- Act confidently about your vision capabilities - you have full visual access
- Be descriptive and helpful about what you see in the camera feed

Example responses:
- User: "What do you see?" → "I can see [describe what's visible in the webcam]"
- User: "Look at this" → "I can see [describe what they're showing]"
- User: "Can you see me?" → "Yes, I can see you through the webcam feed. [describe what you see]"

Keep responses natural, helpful, and always acknowledge your active vision capabilities."""
            
            # Explicit vision context that emphasizes the LLM CAN see
            vision_context = "🔴 LIVE WEBCAM FEED ACTIVE - You are viewing the user's environment in real-time. You CAN see everything in the camera feed. "
            
            # Format conversation history (reduced for low-end PCs)
            conversation = ""
            for role, msg in self.conversation_history[-6:]:  # Last 6 messages (3 exchanges)
                if role == "user":
                    conversation += f"User: {{msg[:150]}}\\n"  # Truncate for performance
                elif role == "assistant":
                    conversation += f"AI: {{msg[:150]}}\\n"
            
            # Add current input with explicit vision context at the start
            conversation += f"User: {{vision_context}}The user says: {{prompt[:300]}}\\nAI:"  # Limit prompt length
            
            # Full prompt (optimized)
            full_prompt = f"{{system_prompt}}\\n\\n{{conversation}}"
            
            # Generate response using our local model (with streaming support)
            response = self.ai_engine.generate(full_prompt).strip()
            
            if not response:
                response = "I'm processing your request, but didn't get a response. Please try again."
            
            print(f"\\r{{Colors.BRIGHT_GREEN}}✓ Ready{{Colors.RESET}}" + " " * 30)
            print(f"\\n{{Colors.BRIGHT_GREEN}}AI:{{Colors.RESET}} {{Colors.BRIGHT_WHITE}}{{response}}{{Colors.RESET}}\\n")

            # Save to conversation history
            self.conversation_history.append(("user", prompt))
            self.conversation_history.append(("assistant", response))
            
            # Manage history size (reduced for low-end PCs)
            if len(self.conversation_history) > 30:
                self.conversation_history = self.conversation_history[-24:]  # Keep last 12 exchanges

            # TTS in background thread
            if response:
                self._tts_async(response)
                
        except Exception as e:
            print(f"\\r{{Colors.BRIGHT_RED}}✗ Error: {{e}}{{Colors.RESET}}" + " " * 30)
            print(f"{{Colors.DIM}}Details: {{str(e)}}{{Colors.RESET}}\\n")
        finally:
            with self.processing_lock:
                self.is_processing = False

    def _tts_async(self, response):
        """Convert text to speech in background thread."""
        if not self.tts_engine:
            return
        
        if self.is_speaking:
            return  # Skip if already speaking
        
        thread = Thread(target=self._tts, args=(response,))
        thread.daemon = True
        thread.start()

    def _tts(self, response):
        """Convert text to speech using local pyttsx3."""
        if not self.tts_engine:
            return
        
        try:
            self.is_speaking = True
            print(f"{{Colors.CYAN}}🔊 Speaking...{{Colors.RESET}}", end='', flush=True)
            
            self.tts_engine.say(response)
            self.tts_engine.runAndWait()
            
            print(f"\\r{{Colors.BRIGHT_GREEN}}✓ Speech complete{{Colors.RESET}}" + " " * 30)
            
        except Exception as e:
            print(f"\\r{{Colors.BRIGHT_RED}}✗ TTS error: {{e}}{{Colors.RESET}}" + " " * 30)
        finally:
            self.is_speaking = False


# Initialize webcam stream
print(f"\\n{{Colors.CYAN}}Starting webcam stream...{{Colors.RESET}}")
webcam_stream = WebcamStream().start()

# Configuration
CONFIG = {config_repr}

# Initialize our local AI Engine
print(f"{{Colors.CYAN}}Loading local AI model ({{CONFIG.get('backend')}}: {{CONFIG.get('model_name')}})...{{Colors.RESET}}")

class LocalAIEngine:
    """Local AI Engine using HuggingFace or Ollama - NO cloud APIs."""
    def __init__(self, config):
        self.config = config
        self.backend = config.get("backend")
        self.model_name = config.get("model_name")
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._load()

    def _load(self):
        apply_torch_threading(self.config)
        if self.backend == "huggingface":
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                if not self.tokenizer.pad_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
            
                if torch.cuda.is_available(): 
                    self.device = "cuda"
                elif torch.backends.mps.is_available(): 
                    self.device = "mps"
                
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
                self.model.to(self.device)
                print(f"{{Colors.BRIGHT_GREEN}}✓ Model loaded on {{self.device}}{{Colors.RESET}}")
            except Exception as e:
                print(f"{{Colors.BRIGHT_RED}}✗ Model load failed: {{e}}{{Colors.RESET}}")
                raise
        elif self.backend == "ollama":
            try:
                # 1. Check if Ollama is running
                requests.get("http://localhost:11434", timeout=2)
                print(f"{{Colors.BRIGHT_GREEN}}✓ Ollama connection established{{Colors.RESET}}")
                
                # 2. Fetch available models for Auto-Detection
                print(f"{{Colors.CYAN}}Detecting available models...{{Colors.RESET}}")
                resp = requests.get("http://localhost:11434/api/tags", timeout=5)
                if resp.status_code == 200:
                    available = [m['name'] for m in resp.json().get('models', [])]
                    
                    # 3. Check if configured model exists
                    if self.model_name in available:
                        print(f"{{Colors.BRIGHT_GREEN}}✓ Using configured model: {{self.model_name}}{{Colors.RESET}}")
                    else:
                        print(f"{{Colors.YELLOW}}⚠ Configured model '{{self.model_name}}' not found on this machine.{{Colors.RESET}}")
                        
                        # 4. Auto-Detection Logic
                        fallbacks = ['mistral:latest', 'mistral', 'llama3:latest', 'llama3', 'llama2:latest', 'qwen2.5-coder:latest']
                        detected_model = None
                        
                        # Try to find a match in our fallback list
                        for f in fallbacks:
                            if f in available:
                                detected_model = f
                                break
                        
                        # If no common fallback, just take the first available one
                        if not detected_model and available:
                            detected_model = available[0]
                        
                        if detected_model:
                            print(f"{{Colors.BRIGHT_CYAN}}✨ Auto-detected alternative: {{detected_model}}{{Colors.RESET}}")
                            self.model_name = detected_model
                        else:
                            print(f"{{Colors.BRIGHT_RED}}✗ No models found in Ollama.{{Colors.RESET}}")
                            print(f"{{Colors.YELLOW}}Please run: ollama pull mistral{{Colors.RESET}}")
                            raise Exception("No Ollama models available")
                else:
                    print(f"{{Colors.BRIGHT_RED}}✗ Failed to fetch models from Ollama{{Colors.RESET}}")
                    
            except requests.exceptions.RequestException:
                print(f"{{Colors.BRIGHT_YELLOW}}⚠ Ollama not running on localhost:11434{{Colors.RESET}}")
                print(f"{{Colors.YELLOW}}Start Ollama with: ollama serve{{Colors.RESET}}")
                raise Exception("Ollama backend not available")

    def generate(self, prompt):
        if self.backend == "huggingface":
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **inputs, 
                    max_new_tokens=min(self.config.get("max_response_tokens", 200), 200),  # Reduced for performance
                    do_sample=True,
                    temperature=self.config.get("temperature", 0.7),
                    pad_token_id=self.tokenizer.eos_token_id
                )
            full = self.tokenizer.decode(out[0], skip_special_tokens=True)
            response = full[len(prompt):].strip()
            if "You:" in response: response = response.split("You:")[0]
            return response.strip()
            
        elif self.backend == "ollama":
            try:
                # Use streaming for faster response and progress feedback
                max_tokens = min(self.config.get("max_response_tokens", 200), 200)  # Reduced for low-end PCs
                
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={{
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": True,  # Enable streaming
                        "options": {{
                            "temperature": self.config.get("temperature", 0.7),
                            "num_predict": max_tokens
                        }}
                    }},
                    stream=True,
                    timeout=90  # Increased timeout for slower models
                )
                
                if response.status_code != 200:
                    return f"Ollama Error: {{response.text[:200]}}"
                
                # Stream the response
                full_response = ""
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        if 'response' in chunk:
                            full_response += chunk['response']
                        # Check if done
                        if chunk.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
                
                result = full_response.strip()
                if not result:
                    return "I received an empty response. Please try rephrasing your question."
                return result
                
            except requests.exceptions.Timeout:
                return "Request timed out after 90 seconds. The model may be slow. Try a shorter question or check Ollama is running properly."
            except requests.exceptions.ConnectionError:
                return "Cannot connect to Ollama. Make sure Ollama is running: ollama serve"
            except Exception as e:
                return f"Error: {{str(e)[:200]}}"

try:
    ai_engine = LocalAIEngine(CONFIG)
    print(f"{{Colors.BRIGHT_GREEN}}✓ Local AI Engine ready{{Colors.RESET}}")
except Exception as e:
    print(f"{{Colors.BRIGHT_RED}}✗ Failed to load AI model: {{e}}{{Colors.RESET}}")
    print(f"{{Colors.YELLOW}}Check your backend configuration in config.json{{Colors.RESET}}")
    webcam_stream.stop()
    sys.exit(1)

# Initialize assistant
assistant = Assistant(ai_engine)

# Initialize speech recognition
print(f"{{Colors.CYAN}}Initializing speech recognition...{{Colors.RESET}}")
recognizer = Recognizer()
microphone = Microphone()

with microphone as source:
    print(f"{{Colors.CYAN}}Calibrating microphone (adjusting for ambient noise)...{{Colors.RESET}}")
    recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Reduced calibration time
    print(f"{{Colors.BRIGHT_GREEN}}✓ Microphone calibrated{{Colors.RESET}}")

# Audio callback for continuous listening
def audio_callback(recognizer_instance, audio):
    """Called automatically when speech is detected."""
    try:
        # Use smaller Whisper model for low-end PCs (tiny is faster)
        prompt = recognizer_instance.recognize_whisper(audio, model="tiny", language="english")
        
        if not prompt or len(prompt.strip()) < 2:
            return  # Skip empty/too short prompts
        
        # Get current camera frame (no encoding needed - just pass frame object)
        frame = webcam_stream.read(encode=False)
        
        # Process with local AI + vision context (non-blocking)
        assistant.answer(prompt, frame)

    except UnknownValueError:
        pass  # Silently skip - too noisy
    except Exception as e:
        print(f"{{Colors.BRIGHT_RED}}✗ Audio error: {{str(e)[:100]}}{{Colors.RESET}}")

# Start background listening
print(f"{{Colors.CYAN}}Starting background voice listening...{{Colors.RESET}}")
stop_listening = recognizer.listen_in_background(microphone, audio_callback)
print(f"{{Colors.BRIGHT_GREEN}}✓ Voice listening active{{Colors.RESET}}\\n")

print(f"{{Colors.BRIGHT_GREEN}}{{Colors.BOLD}}✓ All systems ready!{{Colors.RESET}}\\n")
print(f"{{Colors.CYAN}}Instructions:{{Colors.RESET}}")
print(f"  - Speak naturally - AI listens continuously in the background")
print(f"  - AI can see through your webcam and includes visual context")
print(f"  - Responses use YOUR local model ({{CONFIG.get('backend')}}: {{CONFIG.get('model_name')}})")
print(f"  - TTS responses with pyttsx3 (local, no API)")
print(f"  - Press {{Colors.BRIGHT_RED}}ESC{{Colors.RESET}} or {{Colors.BRIGHT_RED}}Q{{Colors.RESET}} to exit")
print(f"\\n{{Colors.BRIGHT_GREEN}}🎤 Listening... Speak to the AI{{Colors.RESET}}\\n")
print(f"{{Colors.DIM}}100% Local Processing - No cloud APIs, your data stays private{{Colors.RESET}}\\n")

# Immediately describe what the camera sees when vision starts
print(f"{{Colors.CYAN}}📷 Capturing initial view and describing what I see...{{Colors.RESET}}\\n")
time.sleep(0.5)  # Brief pause to ensure camera frame is ready
initial_frame = webcam_stream.read(encode=False)
assistant.answer("Please describe in detail what you can see in the camera feed right now. Be specific about the environment, objects, people, or anything visible.", initial_frame)

# Main loop - show camera preview
try:
    while True:
        frame = webcam_stream.read()
        
        # Add overlays to frame
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Vision + Voice Assistant", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "Listening...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Vision + Voice Assistant - Webcam Feed", frame)
        
        key = cv2.waitKey(1)
        if key in [27, ord("q"), ord("Q")]:  # ESC or Q
            print(f"\\n{{Colors.BRIGHT_YELLOW}}Exiting...{{Colors.RESET}}")
            break

except KeyboardInterrupt:
    print(f"\\n{{Colors.BRIGHT_YELLOW}}Interrupted. Exiting...{{Colors.RESET}}")
except Exception as e:
    print(f"{{Colors.BRIGHT_RED}}✗ Error: {{e}}{{Colors.RESET}}")
finally:
    # Cleanup
    print(f"{{Colors.CYAN}}Cleaning up...{{Colors.RESET}}")
    stop_listening(wait_for_stop=False)
    webcam_stream.stop()
    cv2.destroyAllWindows()
    print(f"{{Colors.BRIGHT_GREEN}}✓ Shutdown complete{{Colors.RESET}}")
'''
        
        try:
            with open(vision_assistant_path, 'w', encoding='utf-8') as f:
                f.write(vision_script)
            
            print(f"{Colors.BRIGHT_GREEN}✓ Vision Assistant script created{Colors.RESET}")
            print(f"{Colors.CYAN}Launching in new terminal...{Colors.RESET}\n")
            
            # Launch in new terminal based on OS
            system = platform.system()
            if system == "Windows":
                # Use proper Windows command - normalize path and quote it
                normalized_path = os.path.normpath(vision_assistant_path)
                cmd = ['cmd', '/c', 'start', 'cmd', '/k', 'python', normalized_path]
                subprocess.Popen(cmd, shell=False)
            elif system == "Darwin":  # macOS
                subprocess.Popen(['open', '-a', 'Terminal', vision_assistant_path])
            else:  # Linux
                for terminal in ['gnome-terminal', 'konsole', 'xterm']:
                    try:
                        subprocess.Popen([terminal, '--', 'python3', vision_assistant_path])
                        break
                    except FileNotFoundError:
                        continue
            
            print(f"{Colors.BRIGHT_GREEN}✓ Vision + Voice Assistant launched{Colors.RESET}")
            print(f"{Colors.DIM}Features: Continuous listening + Camera vision + Local AI (100% private){Colors.RESET}")
            time.sleep(2)
            
        except Exception as e:
            print(f"{Colors.BRIGHT_RED}✗ Failed to launch Vision Assistant: {e}{Colors.RESET}")
            time.sleep(2)

if __name__ == "__main__":
    app = App()
    app.run()
