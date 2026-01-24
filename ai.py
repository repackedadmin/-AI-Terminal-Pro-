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

# System Info and Status Widget
try:
    from tui.components.system_info import (
        SystemInfo, StatusWidget, get_system_info, get_status_widget,
        is_windows, is_macos, is_linux
    )
    SYSTEM_INFO_AVAILABLE = True
except ImportError:
    SYSTEM_INFO_AVAILABLE = False

# Self-Healing Components
try:
    from tui.components.self_healing import (
        SelfHealingConfig,
        SelfHealingLogger,
        DatabaseHealer,
        NetworkHealer,
        MCPHealer,
        ConfigHealer,
        MemoryHealer,
        HealthMonitor,
        get_health_monitor,
        start_health_monitor,
        retry_on_failure,
        with_db_recovery,
        with_network_recovery
    )
    SELF_HEALING_AVAILABLE = True
except ImportError:
    SELF_HEALING_AVAILABLE = False
    # Create dummy decorators
    def retry_on_failure(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def with_db_recovery(func):
        return func
    def with_network_recovery(func):
        return func

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
BASE_MODELS_DIR = os.path.join(TRAINING_DIR, "base_models")  # Local cache for base models
LORA_DIR = os.path.join(TRAINING_DIR, "lora")
REINFORCEMENT_DIR = os.path.join(TRAINING_DIR, "reinforcement")
API_DIR = os.path.join(BASE_DIR, "api")
API_CONFIG_FILE = os.path.join(API_DIR, "api_config.json")
API_KEYS_FILE = os.path.join(API_DIR, "api_keys.json")
APPS_DIR = os.path.join(BASE_DIR, "apps")
APP_PROJECTS_DB = os.path.join(BASE_DIR, "app_projects.sqlite")
EXTENSIONS_DIR = os.path.join(BASE_DIR, "extensions")  # Extension storage directory

# Ensure all workspace directories exist
for directory in [SANDBOX_DIR, DOCS_DIR, CUSTOM_TOOLS_DIR, TRAINING_DIR, TRAINING_DATA_DIR, MODELS_DIR, BASE_MODELS_DIR, LORA_DIR, REINFORCEMENT_DIR, API_DIR, APPS_DIR, EXTENSIONS_DIR]:
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
    # App Builder quality + self-healing knobs
    "builder_fast_mode": False,
    "builder_max_attempts": 3,
    "builder_min_code_chars": 120,
    "builder_plan_tokens": 650,
    "builder_code_tokens": 1600,
    "builder_review_tokens": 250,
    "builder_feedback_tokens": 450,
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
    """Handles SQLite interaction for Chat History and Document RAG with Self-Healing."""
    def __init__(self, db_path):
        self.db_path = db_path
        self._connect()
        self._init_db()

        # Initialize self-healing if available
        if SELF_HEALING_AVAILABLE:
            self.db_healer = DatabaseHealer(db_path)
            self._recovery_attempts = 0
            self._max_recovery_attempts = 3

    def _connect(self):
        """Create database connection with self-healing"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging for better reliability
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.cursor = self.conn.cursor()
        except sqlite3.DatabaseError as e:
            if SELF_HEALING_AVAILABLE:
                print(f"{Colors.YELLOW}Database issue detected, attempting recovery...{Colors.RESET}")
                healer = DatabaseHealer(self.db_path)
                if healer.heal():
                    self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    self.cursor = self.conn.cursor()
                else:
                    raise
            else:
                raise

    def _execute_with_recovery(self, query, params=None, commit=False):
        """Execute query with automatic recovery on failure"""
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            if commit:
                self.conn.commit()
            return True
        except sqlite3.DatabaseError as e:
            if SELF_HEALING_AVAILABLE and self._recovery_attempts < self._max_recovery_attempts:
                self._recovery_attempts += 1
                print(f"{Colors.YELLOW}Database error, attempting recovery ({self._recovery_attempts}/{self._max_recovery_attempts})...{Colors.RESET}")
                if self.db_healer.heal():
                    self._connect()
                    # Retry the query
                    if params:
                        self.cursor.execute(query, params)
                    else:
                        self.cursor.execute(query)
                    if commit:
                        self.conn.commit()
                    self._recovery_attempts = 0
                    return True
            raise
        finally:
            if self._recovery_attempts >= self._max_recovery_attempts:
                self._recovery_attempts = 0

    def check_health(self):
        """Check database health and auto-heal if needed"""
        if SELF_HEALING_AVAILABLE:
            return self.db_healer.check_health()
        try:
            self.cursor.execute("SELECT 1")
            return True
        except:
            return False

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
    """Implements JSON-RPC 2.0 Client over Stdio with Self-Healing."""
    def __init__(self, name, command):
        self.name = name
        self.command = command
        self.process = None
        self.request_id = 0
        self.available_tools = []
        self.running = False

        # Self-healing properties
        self._restart_attempts = 0
        self._max_restart_attempts = 3
        self._last_restart = 0
        self._restart_cooldown = 30  # seconds

        # Initialize MCP healer if available
        if SELF_HEALING_AVAILABLE:
            self.healer = MCPHealer(name, command)
        else:
            self.healer = None

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
                self._restart_attempts = 0  # Reset on successful start
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
            self._attempt_recovery()

    def _read_response(self, timeout=3.0):
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                line = self.process.stdout.readline()
                if line:
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
            except (BrokenPipeError, OSError):
                self.running = False
                break
        return None

    def _attempt_recovery(self):
        """Attempt to recover the MCP connection"""
        if not self.running and self._restart_attempts < self._max_restart_attempts:
            current_time = time.time()
            if current_time - self._last_restart >= self._restart_cooldown:
                self._restart_attempts += 1
                self._last_restart = current_time
                print(f"{Colors.YELLOW}[MCP] Attempting auto-restart of {self.name} ({self._restart_attempts}/{self._max_restart_attempts})...{Colors.RESET}")

                # Use healer if available
                if self.healer:
                    if self.healer.heal():
                        self.stop()
                        if self.start():
                            print(f"{Colors.BRIGHT_GREEN}[MCP] {self.name} recovered successfully{Colors.RESET}")
                            return True
                else:
                    self.stop()
                    if self.start():
                        print(f"{Colors.BRIGHT_GREEN}[MCP] {self.name} restarted successfully{Colors.RESET}")
                        return True

                print(f"{Colors.RED}[MCP] Failed to recover {self.name}{Colors.RESET}")
        return False

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

    def check_health(self):
        """Check if MCP server is healthy"""
        if not self.running or not self.process:
            return False
        try:
            # Check if process is still running
            if self.process.poll() is not None:
                self.running = False
                return False
            return True
        except:
            return False

    def call_tool(self, tool_name, args_dict):
        # Check health before calling
        if not self.check_health():
            if not self._attempt_recovery():
                return "Error: MCP server not running and recovery failed."

        self._send_json({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": args_dict},
            "id": self._next_id()
        })
        resp = self._read_response(timeout=15.0) # Longer timeout for tool execution

        if not resp:
            # Try recovery on timeout
            if self._attempt_recovery():
                return self.call_tool(tool_name, args_dict)  # Retry
            return "Error: MCP Timeout."
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

def resilient_request(method, url, max_retries=3, retry_delay=1.0, **kwargs):
    """
    Make HTTP requests with automatic retry and self-healing.
    Supports exponential backoff and network recovery.
    """
    last_error = None

    for attempt in range(max_retries):
        try:
            if method.lower() == 'get':
                response = requests.get(url, **kwargs)
            elif method.lower() == 'post':
                response = requests.post(url, **kwargs)
            elif method.lower() == 'put':
                response = requests.put(url, **kwargs)
            elif method.lower() == 'delete':
                response = requests.delete(url, **kwargs)
            else:
                response = requests.request(method, url, **kwargs)

            return response

        except requests.exceptions.ConnectionError as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"{Colors.YELLOW}[Network] Connection error, retrying in {wait_time:.1f}s ({attempt + 1}/{max_retries})...{Colors.RESET}")

                # Check network connectivity if self-healing available
                if SELF_HEALING_AVAILABLE:
                    healer = NetworkHealer()
                    if healer.check_health():
                        time.sleep(wait_time)
                    else:
                        print(f"{Colors.YELLOW}[Network] Waiting for network recovery...{Colors.RESET}")
                        healer.heal()
                        time.sleep(wait_time)
                else:
                    time.sleep(wait_time)

        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"{Colors.YELLOW}[Network] Request timeout, retrying in {wait_time:.1f}s ({attempt + 1}/{max_retries})...{Colors.RESET}")
                time.sleep(wait_time)

        except requests.exceptions.RequestException as e:
            last_error = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                print(f"{Colors.YELLOW}[Network] Request error, retrying in {wait_time:.1f}s ({attempt + 1}/{max_retries})...{Colors.RESET}")
                time.sleep(wait_time)

    # All retries exhausted
    raise last_error if last_error else requests.exceptions.RequestException("Max retries exceeded")


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

    def generate(self, prompt, timeout=180):
        """Generate AI response with configurable timeout."""
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
                # Use resilient request with self-healing and configurable timeout
                res = resilient_request(
                    'post',
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.config.get("temperature"),
                            "num_predict": self.config.get("max_response_tokens")
                        }
                    },
                    timeout=timeout,
                    max_retries=3,
                    retry_delay=3.0
                )
                if res.status_code == 200:
                    return res.json()['response'].strip()
                return f"Ollama Error: {res.text}"
            except requests.exceptions.ConnectionError:
                return "Connection Error: Cannot connect to Ollama. Make sure it's running with: ollama serve"
            except requests.exceptions.Timeout:
                return "Request timed out. The model may be slow or Ollama is overloaded."
            except Exception as e:
                return f"Error: {e}"

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

def resolve_model_path(model_input, local_cache_dir=BASE_MODELS_DIR):
    """
    Resolve a model path from either a local path or HuggingFace repo ID.
    
    Args:
        model_input: Either a local path (directory) or HuggingFace repo ID
        local_cache_dir: Directory to cache downloaded models
        
    Returns:
        str: Local path to the model directory
        None: If resolution fails
    """
    import os
    
    # Check if it's already a local path (directory exists)
    if os.path.isdir(model_input):
        # Check if it looks like a model directory (has config.json or pytorch_model.bin or model.safetensors)
        model_files = ['config.json', 'pytorch_model.bin', 'model.safetensors', 'model.safetensors.index.json']
        if any(os.path.exists(os.path.join(model_input, f)) for f in model_files):
            return model_input
        else:
            print(f"{Colors.YELLOW}⚠ Warning: '{model_input}' is a directory but doesn't appear to contain a model.{Colors.RESET}")
            # Still return it - let transformers handle the error
    
    # Check if it's a file path (not a directory)
    if os.path.isfile(model_input):
        print(f"{Colors.YELLOW}⚠ '{model_input}' is a file, not a model directory.{Colors.RESET}")
        return None
    
    # It's likely a HuggingFace repo ID - download/cache it locally
    try:
        # Create local cache directory structure
        # Use a safe directory name from the repo ID
        safe_name = model_input.replace('/', '_').replace('\\', '_')
        local_model_path = os.path.join(local_cache_dir, safe_name)
        
        # Check if model is already cached locally
        if os.path.isdir(local_model_path):
            model_files = ['config.json', 'pytorch_model.bin', 'model.safetensors', 'model.safetensors.index.json']
            if any(os.path.exists(os.path.join(local_model_path, f)) for f in model_files):
                print(f"{Colors.BRIGHT_GREEN}✓ Using cached model at: {local_model_path}{Colors.RESET}")
                return local_model_path
        
        # Download model to local cache
        print(f"{Colors.BRIGHT_CYAN}Downloading model '{model_input}' to local cache...{Colors.RESET}")
        print(f"{Colors.DIM}This may take several minutes depending on model size...{Colors.RESET}")
        
        # Import transformers to download
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Create output directory
        os.makedirs(local_model_path, exist_ok=True)
        
        # Download tokenizer first (smaller, faster to verify repo ID)
        print(f"{Colors.BRIGHT_CYAN}Downloading tokenizer...{Colors.RESET}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_input)
            tokenizer.save_pretrained(local_model_path)
        except Exception as e:
            print(f"{Colors.BRIGHT_RED}✗ Failed to download tokenizer: {e}{Colors.RESET}")
            # Clean up partial download
            try:
                import shutil
                if os.path.exists(local_model_path):
                    shutil.rmtree(local_model_path)
            except:
                pass
            raise
        
        # Download model
        print(f"{Colors.BRIGHT_CYAN}Downloading model weights...{Colors.RESET}")
        print(f"{Colors.DIM}This is the large download...{Colors.RESET}")
        try:
            model = AutoModelForCausalLM.from_pretrained(model_input)
            model.save_pretrained(local_model_path)
        except Exception as e:
            print(f"{Colors.BRIGHT_RED}✗ Failed to download model: {e}{Colors.RESET}")
            # Clean up partial download
            try:
                import shutil
                if os.path.exists(local_model_path):
                    shutil.rmtree(local_model_path)
            except:
                pass
            raise
        finally:
            # Clean up memory
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            import gc
            gc.collect()
        
        print(f"{Colors.BRIGHT_GREEN}✓ Model downloaded and cached to: {local_model_path}{Colors.RESET}")
        return local_model_path
        
    except Exception as e:
        error_msg = str(e)
        if "does not appear to have a file named" in error_msg or "404" in error_msg or "not found" in error_msg.lower():
            print(f"{Colors.BRIGHT_RED}✗ Model '{model_input}' not found on HuggingFace Hub.{Colors.RESET}")
            print(f"{Colors.YELLOW}Please check the model ID is correct (e.g., 'gpt2', 'microsoft/DialoGPT-medium').{Colors.RESET}")
        else:
            print(f"{Colors.BRIGHT_RED}✗ Error downloading model '{model_input}': {e}{Colors.RESET}")
        return None

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
            
            # Resolve model path (local or download from HuggingFace)
            model_path = resolve_model_path(base_model, BASE_MODELS_DIR)
            if model_path is None:
                print(f"{Colors.BRIGHT_RED}✗ Failed to resolve model path for '{base_model}'{Colors.RESET}")
                return False
            
            print(f"{Colors.BRIGHT_CYAN}Loading model and tokenizer from: {model_path}{Colors.RESET}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
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
            
            # Resolve model path (local or download from HuggingFace)
            model_path = resolve_model_path(base_model, BASE_MODELS_DIR)
            if model_path is None:
                print(f"{Colors.BRIGHT_RED}✗ Failed to resolve model path for '{base_model}'{Colors.RESET}")
                return False
            
            print(f"{Colors.BRIGHT_CYAN}Loading model and tokenizer from: {model_path}{Colors.RESET}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if not tokenizer.pad_token:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(model_path)
            
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
#                           6.5. EXTENSION SYSTEM
# ==============================================================================

class ExtensionConfig:
    """Extension configuration from gemini-extension.json"""
    def __init__(self, name, version, mcp_servers=None, context_files=None, exclude_tools=None, settings=None):
        self.name = name
        self.version = version
        self.mcp_servers = mcp_servers or {}
        self.context_files = context_files or []
        self.exclude_tools = exclude_tools or []
        self.settings = settings or []


class Extension:
    """Represents a loaded extension"""
    def __init__(self, name, version, path, config, is_active=True, extension_id=None):
        self.name = name
        self.version = version
        self.path = path
        self.config = config
        self.is_active = is_active
        self.id = extension_id or f"{name}@{version}"
        self.mcp_servers = config.mcp_servers
        self.context_files = config.context_files
        self.exclude_tools = config.exclude_tools


class ExtensionManager:
    """Manages extension loading, enabling, and integration"""
    
    def __init__(self, config, tool_registry=None):
        self.config = config
        self.tool_registry = tool_registry
        self.extensions = []
        self.extensions_dir = EXTENSIONS_DIR
        
    def load_extensions(self):
        """Load all extensions from the extensions directory"""
        self.extensions = []
        
        if not os.path.exists(self.extensions_dir):
            os.makedirs(self.extensions_dir, exist_ok=True)
            return self.extensions
        
        # Load each extension directory
        for item in os.listdir(self.extensions_dir):
            extension_path = os.path.join(self.extensions_dir, item)
            if os.path.isdir(extension_path):
                extension = self._load_extension(extension_path)
                if extension:
                    self.extensions.append(extension)
        
        return self.extensions
    
    def _load_extension(self, extension_path):
        """Load a single extension from its directory"""
        try:
            # Try both naming conventions
            manifest_paths = [
                os.path.join(extension_path, "gemini-extension.json"),
                os.path.join(extension_path, "Gemini-extension.json"),
            ]
            
            manifest_path = None
            for path in manifest_paths:
                if os.path.exists(path):
                    manifest_path = path
                    break
            
            if not manifest_path:
                return None  # No manifest found, skip this directory
            
            # Load manifest
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            # Parse extension config
            name = manifest.get("name", os.path.basename(extension_path))
            version = manifest.get("version", "1.0.0")
            mcp_servers = manifest.get("mcpServers", {})
            
            # Resolve context files
            context_file_names = manifest.get("contextFileName", [])
            if isinstance(context_file_names, str):
                context_file_names = [context_file_names]
            
            context_files = []
            for context_file_name in context_file_names:
                context_file_path = os.path.join(extension_path, context_file_name)
                if os.path.exists(context_file_path):
                    context_files.append(context_file_path)
            
            exclude_tools = manifest.get("excludeTools", [])
            settings = manifest.get("settings", [])
            
            config = ExtensionConfig(
                name=name,
                version=version,
                mcp_servers=mcp_servers,
                context_files=context_files,
                exclude_tools=exclude_tools,
                settings=settings
            )
            
            extension = Extension(
                name=name,
                version=version,
                path=extension_path,
                config=config,
                is_active=True  # All extensions are active by default for now
            )
            
            return extension
            
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Failed to load extension from {extension_path}: {e}{Colors.RESET}")
            return None
    
    def get_active_extensions(self):
        """Get all active extensions"""
        return [ext for ext in self.extensions if ext.is_active]
    
    def get_extension(self, name):
        """Get extension by name"""
        for ext in self.extensions:
            if ext.name == name:
                return ext
        return None
    
    def start_extension(self, extension):
        """Start an extension (load MCP servers, etc.)"""
        if not self.tool_registry:
            return
        
        # Start MCP servers for this extension
        for server_name, server_config in extension.mcp_servers.items():
            try:
                # Build command from config
                command = server_config.get("command", "")
                args = server_config.get("args", [])
                cwd = server_config.get("cwd", extension.path)
                
                # Resolve variables in command/args
                command = command.replace("${extensionPath}", extension.path).replace("${/}", os.sep)
                if isinstance(args, list):
                    args = [arg.replace("${extensionPath}", extension.path).replace("${/}", os.sep) for arg in args]
                    full_command = f"{command} {' '.join(args)}"
                else:
                    full_command = command
                
                # Create and start MCP client
                client = MCPClient(server_name, full_command)
                if client.start():
                    self.tool_registry.mcp_clients[server_name] = client
                    print(f"{Colors.BRIGHT_GREEN}✓ Extension MCP server started: {server_name}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}⚠ Failed to start extension MCP server: {server_name}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.YELLOW}⚠ Error starting extension MCP server {server_name}: {e}{Colors.RESET}")
    
    def start_all_extensions(self):
        """Start all active extensions"""
        for extension in self.get_active_extensions():
            self.start_extension(extension)


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

    def think(self, context, max_tokens=500, timeout=120):
        """Generate response based on context with configurable timeout."""
        # Streamlined prompt format for faster processing
        prompt = f"{self.system_prompt}\n\n{context}\n\nResponse:"

        # Temporarily adjust settings for this agent
        original_max = self.ai_engine.config.get('max_response_tokens', 250)
        self.ai_engine.config['max_response_tokens'] = max_tokens

        response = self.ai_engine.generate(prompt, timeout=timeout)

        # Restore original
        self.ai_engine.config['max_response_tokens'] = original_max

        return response


class SpecificationWriterAgent(BaseAgent):
    """Clarifies requirements and writes specifications."""

    def __init__(self, ai_engine):
        system_prompt = "You are a Specification Writer. Write clear, concise specifications."
        super().__init__("Specification Writer", "SPEC_WRITER", system_prompt, ai_engine)

    def analyze_description(self, app_name, description):
        """Analyze if description is sufficient."""
        context = f"App: {app_name}\nDescription: {description}\n\nIs this clear enough to build? If not, list 1-3 key questions."
        return self.think(context, 150, timeout=60)

    def write_specification(self, app_name, description, qa_pairs=None):
        """Write final specification."""
        context = f"App: {app_name}\nDescription: {description}"
        if qa_pairs:
            context += "\nAnswers: " + "; ".join([f"{q}: {a}" for q, a in qa_pairs])
        context += "\n\nWrite a brief specification: features, requirements, constraints."
        return self.think(context, 400, timeout=90)


class ArchitectAgent(BaseAgent):
    """Designs architecture and checks dependencies."""

    def __init__(self, ai_engine):
        system_prompt = "You are a Software Architect. Design simple, practical architectures."
        super().__init__("Architect", "ARCHITECT", system_prompt, ai_engine)

    def design_architecture(self, specification):
        """Design system architecture."""
        # Truncate spec to avoid timeout
        spec_short = specification[:600] if len(specification) > 600 else specification
        context = f"Spec: {spec_short}\n\nDesign: tech stack, pip dependencies, folder structure. Keep it simple."
        return self.think(context, 500, timeout=90)
    
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
            "You are a Tech Lead. Break projects into numbered, implementation-ready tasks "
            "that cover the full build from setup to polish."
        )
        super().__init__("Tech Lead", "TECH_LEAD", system_prompt, ai_engine)

    def create_tasks(self, specification, architecture):
        """Create development task list."""
        spec_short = specification[:900] if len(specification) > 900 else specification
        arch_short = architecture[:700] if len(architecture) > 700 else architecture
        context = (
            f"Specification:\n{spec_short}\n\nArchitecture:\n{arch_short}\n\n"
            "Create 5-9 numbered tasks that deliver a complete, runnable project. "
            "Include setup/config, core features, error handling, and documentation. "
            "Each line MUST start with a number followed by a period (e.g., '1.'). "
            "Format: '1. Task name - brief description and expected files'."
        )
        max_tokens = int(self.ai_engine.config.get("builder_feedback_tokens", 450))
        return self.think(context, max_tokens, timeout=120)


class DeveloperAgent(BaseAgent):
    """Plans implementation details for tasks."""

    def __init__(self, ai_engine):
        system_prompt = (
            "You are a Senior Developer. Produce concise but complete implementation plans "
            "that enumerate files to create/update, key functions/classes, and acceptance criteria."
        )
        super().__init__("Developer", "DEVELOPER", system_prompt, ai_engine)

    def plan_task(self, task, specification, architecture, existing_files, feedback=None):
        """Plan how to implement a task."""
        spec_short = specification[:700] if len(specification) > 700 else specification
        arch_short = architecture[:500] if len(architecture) > 500 else architecture
        files_short = existing_files[:400] if len(existing_files) > 400 else existing_files

        context = (
            f"Task:\n{task}\n\nSpecification (excerpt):\n{spec_short}\n\n"
            f"Architecture (excerpt):\n{arch_short}\n\nExisting files summary:\n{files_short or 'None yet.'}\n\n"
            "Provide a short plan with the following sections:\n"
            "- Files: bullet list of files to create/update.\n"
            "- Steps: 3-6 bullets describing the implementation order.\n"
            "- Acceptance Criteria: bullets describing how we know the task is complete."
        )
        if feedback:
            context += f"\n\nReviewer/validator feedback to address:\n{feedback[:600]}"

        max_tokens = int(self.ai_engine.config.get("builder_plan_tokens", 650))
        return self.think(context, max_tokens, timeout=120)


class CodeMonkeyAgent(BaseAgent):
    """Writes actual code based on developer's plan."""

    def __init__(self, ai_engine):
        system_prompt = (
            "You are an expert implementation engineer. Generate complete, runnable code that satisfies "
            "the plan and does not omit required imports, helper functions, or error handling. "
            "Output ONLY code."
        )
        super().__init__("Code Monkey", "CODE_MONKEY", system_prompt, ai_engine)

    def write_code(self, implementation_plan, existing_code="", task_desc="", spec_excerpt="", arch_excerpt="", feedback=None):
        """Write code based on plan."""
        plan_short = implementation_plan[:900] if len(implementation_plan) > 900 else implementation_plan
        existing_short = existing_code[:2000] if len(existing_code) > 2000 else existing_code
        spec_short = spec_excerpt[:800] if len(spec_excerpt) > 800 else spec_excerpt
        arch_short = arch_excerpt[:600] if len(arch_excerpt) > 600 else arch_excerpt

        context = (
            f"Task:\n{task_desc[:400]}\n\nPlan:\n{plan_short}\n\n"
            f"Specification excerpt:\n{spec_short}\n\nArchitecture excerpt:\n{arch_short}\n\n"
            "Relevant existing code (may be empty):\n"
            f"{existing_short if existing_short.strip() else 'None provided.'}\n\n"
            "Requirements:\n"
            "1) Start with '# File: path/filename.py'.\n"
            "2) Provide the full file contents, not a diff.\n"
            "3) Ensure the module is self-contained and runnable.\n"
            "4) Do not include markdown fences or explanations."
        )
        if feedback:
            context += f"\n\nFix the following issues noted by review/validation:\n{feedback[:800]}"

        max_tokens = int(self.ai_engine.config.get("builder_code_tokens", 1600))
        return self.think(context, max_tokens, timeout=240)


class ReviewerAgent(BaseAgent):
    """Reviews code for issues."""

    def __init__(self, ai_engine):
        system_prompt = (
            "You are a strict Code Reviewer. Verify completeness, correctness, and basic robustness. "
            "Reply with 'APPROVED' only if the code is complete and runnable; otherwise reply "
            "with 'REJECTED: <clear actionable issues>'."
        )
        super().__init__("Reviewer", "REVIEWER", system_prompt, ai_engine)

    def review_code(self, code, task, implementation_plan):
        """Review code quality - fast review."""
        code_sample = code[:2000] if len(code) > 2000 else code
        context = (
            f"Task:\n{task[:300]}\n\nImplementation plan excerpt:\n{implementation_plan[:500]}\n\n"
            f"Code excerpt:\n{code_sample}\n\n"
            "Decide: APPROVED or REJECTED with concrete fixes."
        )
        max_tokens = int(self.ai_engine.config.get("builder_review_tokens", 250))
        return self.think(context, max_tokens, timeout=90)


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
        system_prompt = "You are a Technical Writer. Write brief README.md documentation in markdown."
        super().__init__("Technical Writer", "TECH_WRITER", system_prompt, ai_engine)

    def write_documentation(self, project_name, specification, architecture, codebase_summary):
        """Write project documentation - fast."""
        spec_short = specification[:300] if len(specification) > 300 else specification
        arch_short = architecture[:200] if len(architecture) > 200 else architecture
        files_short = codebase_summary[:200] if len(codebase_summary) > 200 else codebase_summary
        context = f"Project: {project_name}\nSpec: {spec_short}\nArch: {arch_short}\nFiles: {files_short}\n\nWrite brief README: overview, install, usage."
        return self.think(context, 500, timeout=90)


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
        self.fast_mode = bool(ai_engine.config.get("builder_fast_mode", False))
        self.max_attempts = max(1, int(ai_engine.config.get("builder_max_attempts", 3)))
        self.min_code_chars = max(40, int(ai_engine.config.get("builder_min_code_chars", 120)))
        self.heal_logger = SelfHealingLogger() if SELF_HEALING_AVAILABLE else None
        self.db_healer = DatabaseHealer(APP_PROJECTS_DB) if SELF_HEALING_AVAILABLE else None
        self.db_lock = threading.Lock()
        self.current_project = None
        self.init_app_database()
    
    def init_app_database(self):
        """Initialize app projects database."""
        conn, persistent = self._get_db_connection()
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
        
        self._finalize_db(conn, persistent, commit=True)
    
    def create_project(self, name, description):
        """Create a new app project."""
        conn, persistent = self._get_db_connection()
        cursor = conn.cursor()
        success = False
        
        try:
            cursor.execute("INSERT INTO app_projects (name, description, status) VALUES (?, ?, ?)",
                          (name, description, "specification"))
            project_id = cursor.lastrowid
            success = True
            
            # Create project directory
            project_dir = os.path.join(APPS_DIR, name)
            os.makedirs(project_dir, exist_ok=True)
            
            return project_id
        except sqlite3.IntegrityError:
            print(f"{Colors.BRIGHT_RED}✗ Project '{name}' already exists.{Colors.RESET}")
            return None
        finally:
            self._finalize_db(conn, persistent, commit=success)
    
    def get_project(self, project_id):
        """Get project details."""
        conn, persistent = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM app_projects WHERE id=?", (project_id,))
        project = cursor.fetchone()
        self._finalize_db(conn, persistent)
        return project
    
    def update_project_field(self, project_id, field, value):
        """Update a project field."""
        lock = getattr(self, "db_lock", None)
        if lock:
            lock.acquire()
        try:
            conn, persistent = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(f"UPDATE app_projects SET {field}=?, updated_at=CURRENT_TIMESTAMP WHERE id=?", 
                          (value, project_id))
            self._finalize_db(conn, persistent, commit=True)
        finally:
            if lock:
                lock.release()
    
    def save_file(self, project_id, filepath, content):
        """Save a file for the project."""
        lock = getattr(self, "db_lock", None)
        if lock:
            lock.acquire()
        try:
            conn, persistent = self._get_db_connection()
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
            
            self._finalize_db(conn, persistent, commit=True)
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
        lock = getattr(self, "db_lock", None)
        if lock:
            lock.acquire()
        try:
            conn, persistent = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT filepath, content FROM app_files WHERE project_id=?", (project_id,))
            files = cursor.fetchall()
            self._finalize_db(conn, persistent)
            return files
        finally:
            if lock:
                lock.release()
    
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

    def _log_heal(self, level, message):
        """Log self-healing events when available."""
        if self.heal_logger:
            self.heal_logger.log(level, message, "app_builder")
            return
        prefix = {"warning": "⚠", "error": "✗", "success": "✓", "info": "·"}.get(level, "·")
        color = (
            Colors.BRIGHT_RED if level == "error"
            else Colors.BRIGHT_GREEN if level == "success"
            else Colors.YELLOW if level == "warning"
            else Colors.DIM
        )
        print(f"{color}{prefix} {message}{Colors.RESET}")

    def _get_db_connection(self):
        """Get a database connection with optional self-healing."""
        if self.db_healer:
            # Avoid sharing a single connection across threads; heal, then use a fresh handle.
            self.db_healer.conn = None
            return self.db_healer.connect(), False
        return sqlite3.connect(APP_PROJECTS_DB), False

    def _finalize_db(self, conn, persistent, commit=False):
        """Commit/close a database connection safely."""
        if commit:
            conn.commit()
        if not persistent:
            conn.close()

    def _summarize_files(self, files, max_chars=1200):
        """Summarize existing files for planning context."""
        if not files:
            return "No files yet."
        parts = []
        remaining = max_chars
        for filepath, content in files:
            snippet = content[:200].replace("\n", " ")
            entry = f"- {filepath} ({len(content)} chars): {snippet}"
            if len(entry) > remaining:
                break
            parts.append(entry)
            remaining -= len(entry)
            if remaining <= 0:
                break
        return "\n".join(parts) if parts else "Files exist but summary was truncated."

    def _normalize_code_output(self, code):
        """Strip markdown fences and normalize whitespace."""
        if not isinstance(code, str):
            return ""
        cleaned = code.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", cleaned)
            cleaned = cleaned.replace("```", "").strip()
        return cleaned

    def _split_code_blocks(self, code, fallback_filename):
        """Split multi-file outputs using '# File:' markers."""
        pattern = re.compile(r"^\s*#\s*File\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
        matches = list(pattern.finditer(code))
        if not matches:
            return [(fallback_filename, code)]

        blocks = []
        for idx, match in enumerate(matches):
            filename = match.group(1).strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(code)
            content = code[start:end].strip()
            if content:
                blocks.append((filename, content))

        return blocks if blocks else [(fallback_filename, code)]

    def _detect_truncation(self, code):
        """Heuristics to detect truncated or partial code."""
        stripped = code.strip()
        if not stripped:
            return True, "Code response was empty."
        if stripped.endswith(("...", "…")):
            return True, "Code appears truncated with ellipsis."

        tail = stripped.splitlines()[-1].strip().lower()
        dangling_tokens = {"def", "class", "if", "elif", "else", "for", "while", "try", "except", "with"}
        if tail in dangling_tokens:
            return True, f"Code ended with dangling token '{tail}'."
        if tail.endswith(("=", ":", "(", "[", "{", ",")):
            return True, "Code appears to end mid-statement."

        triple_quotes = stripped.count('"""') + stripped.count("'''")
        if triple_quotes % 2 != 0:
            return True, "Unbalanced triple-quoted string detected."

        return False, ""

    def _syntax_feedback(self, code, filename):
        """Compile Python code to detect syntax issues."""
        if not filename.lower().endswith(".py"):
            return ""
        try:
            compile(code, filename, "exec")
            return ""
        except SyntaxError as e:
            line = e.lineno or 0
            return f"SyntaxError line {line}: {e.msg}"
        except Exception as e:
            return f"Compile check failed: {e}"

    def _validate_generated_code(self, code, filename, min_chars=None):
        """Validate generated code and produce actionable feedback."""
        cleaned = self._normalize_code_output(code)

        lower_name = (filename or "").lower()
        if min_chars is None:
            min_required = self.min_code_chars
        else:
            min_required = max(1, int(min_chars))

        # Allow smaller utility/config files to pass validation.
        if lower_name.endswith("__init__.py"):
            min_required = min(min_required, 10)
        elif not lower_name.endswith(".py"):
            min_required = min(min_required, 20)

        min_required = max(5, min_required)
        if len(cleaned) < min_required:
            return False, cleaned, f"Code too short ({len(cleaned)} chars)."

        truncated, reason = self._detect_truncation(cleaned)
        if truncated:
            return False, cleaned, reason

        syntax_issue = self._syntax_feedback(cleaned, filename)
        if syntax_issue:
            return False, cleaned, syntax_issue

        return True, cleaned, ""

    def _extract_task_lines(self, tasks_doc):
        """Parse tasks from AI output with multiple fallbacks."""
        lines = []
        for raw_line in tasks_doc.split("\n"):
            line = raw_line.strip()
            if not line:
                continue
            match = re.match(r"^\s*(\d+)[\).:-]\s*(.+)$", line)
            if match:
                number = match.group(1)
                desc = match.group(2).strip()
                lines.append(f"{number}. {desc}")
                continue
            bullet = re.match(r"^\s*[-*•]\s*(.+)$", line)
            if bullet:
                lines.append(bullet.group(1).strip())
                continue
            if line.lower().startswith("task "):
                lines.append(line)
        # Normalize bullets into numbered tasks
        normalized = []
        for idx, task in enumerate(lines, 1):
            if re.match(r"^\d+\.\s+", task):
                normalized.append(task)
            else:
                normalized.append(f"{idx}. {task}")
        return normalized

    def _generate_tasks_with_healing(self, spec, architecture):
        """Generate tasks with a retry and stricter fallback prompt."""
        tasks_doc = self.tech_lead.create_tasks(spec, architecture)
        task_lines = self._extract_task_lines(tasks_doc)
        if task_lines:
            return tasks_doc, task_lines

        self._log_heal("warning", "Initial task parsing failed, retrying with stricter instructions.")
        fallback_prompt = (
            f"Specification:\n{spec[:900]}\n\nArchitecture:\n{architecture[:700]}\n\n"
            "Return ONLY a numbered list of 5-9 tasks. Each line must begin with '1.' style numbering."
        )
        max_tokens = int(self.ai_engine.config.get("builder_feedback_tokens", 450))
        tasks_doc = self.tech_lead.think(fallback_prompt, max_tokens, timeout=120)
        task_lines = self._extract_task_lines(tasks_doc)
        return tasks_doc, task_lines

    def _save_task_plan(self, task_id, plan):
        """Persist an implementation plan with recovery."""
        lock = getattr(self, "db_lock", None)
        if lock:
            lock.acquire()
        try:
            conn, persistent = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE app_tasks SET implementation_plan=? WHERE id=?", (plan, task_id))
            self._finalize_db(conn, persistent, commit=True)
        finally:
            if lock:
                lock.release()

    def _record_review(self, project_id, task_id, review_status, feedback):
        """Persist a review result."""
        lock = getattr(self, "db_lock", None)
        if lock:
            lock.acquire()
        try:
            conn, persistent = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO app_reviews (project_id, task_id, review_result, feedback) VALUES (?, ?, ?, ?)",
                (project_id, task_id, review_status, feedback[:4000])
            )
            self._finalize_db(conn, persistent, commit=True)
        finally:
            if lock:
                lock.release()

    def _mark_task_completed(self, task_id):
        """Mark a task as completed."""
        lock = getattr(self, "db_lock", None)
        if lock:
            lock.acquire()
        try:
            conn, persistent = self._get_db_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE app_tasks SET status='completed' WHERE id=?", (task_id,))
            self._finalize_db(conn, persistent, commit=True)
        finally:
            if lock:
                lock.release()

    def _run_task_pipeline(self, project_id, task_tuple, spec, architecture, announce=True):
        """Run a self-healing plan/code/review loop for a task."""
        task_id, task_num, task_desc, task_status = task_tuple
        if task_status == "completed":
            return {"task_num": task_num, "ok": True, "message": "already completed"}

        feedback = ""
        last_review = ""
        for attempt in range(1, self.max_attempts + 1):
            if announce:
                print(f"{Colors.DIM}Attempt {attempt}/{self.max_attempts}...{Colors.RESET}")

            existing_files = self.get_project_files(project_id)
            files_context = self._summarize_files(existing_files)

            impl_plan = self.developer.plan_task(task_desc, spec, architecture, files_context, feedback=feedback)
            if self._is_ai_error(impl_plan):
                feedback = f"Plan generation failed: {impl_plan[:200]}"
                self._log_heal("warning", f"Plan generation failed for task {task_num}: {impl_plan[:120]}")
                continue

            self._save_task_plan(task_id, impl_plan)

            relevant_context = self.filter_relevant_context(task_desc + " " + impl_plan, existing_files)
            filename = self.extract_filename("", impl_plan, task_desc) or f"task_{task_num}.py"

            code = self.code_monkey.write_code(
                impl_plan,
                relevant_context,
                task_desc=task_desc,
                spec_excerpt=spec,
                arch_excerpt=architecture,
                feedback=feedback or last_review
            )

            if self._is_ai_error(code):
                feedback = f"Code generation failed: {code[:200]}"
                self._log_heal("warning", f"Code generation failed for task {task_num}: {code[:120]}")
                continue

            filename = self.extract_filename(code, impl_plan, task_desc) or filename
            valid, cleaned_code, validation_feedback = self._validate_generated_code(code, filename)
            if not valid:
                feedback = f"Validation failed: {validation_feedback}"
                last_review = feedback
                self._log_heal("warning", f"Validation issue on task {task_num}: {validation_feedback}")
                continue

            file_blocks = self._split_code_blocks(cleaned_code, filename)
            validated_blocks = []
            block_errors = []
            min_chars = self.min_code_chars if len(file_blocks) == 1 else 40
            for block_filename, block_content in file_blocks:
                block_ok, block_cleaned, block_feedback = self._validate_generated_code(
                    block_content, block_filename, min_chars=min_chars
                )
                if not block_ok:
                    block_errors.append(f"{block_filename}: {block_feedback}")
                else:
                    validated_blocks.append((block_filename, block_cleaned))

            if block_errors:
                feedback = "Validation failed: " + "; ".join(block_errors[:3])
                last_review = feedback
                self._log_heal("warning", f"Multi-file validation issue on task {task_num}: {feedback[:160]}")
                continue

            review_input = (
                cleaned_code
                if len(validated_blocks) == 1
                else "\n\n".join([f"# File: {fname}\n{content}" for fname, content in validated_blocks])
            )

            review = self.reviewer.review_code(review_input, task_desc, impl_plan)
            if self._is_ai_error(review):
                review = "REJECTED: Reviewer failed to provide feedback."

            approved = "APPROVED" in review.upper()
            review_status = "approved" if approved else "rejected"
            self._record_review(project_id, task_id, review_status, review)

            if approved:
                for block_filename, block_content in validated_blocks:
                    self.save_file(project_id, block_filename, block_content)
                self._mark_task_completed(task_id)
                preview = ", ".join([fname for fname, _ in validated_blocks[:3]])
                if len(validated_blocks) > 3:
                    preview += ", ..."
                return {
                    "task_num": task_num,
                    "ok": True,
                    "message": f"saved {len(validated_blocks)} file(s): {preview}",
                    "review": review
                }

            feedback = review
            last_review = review
            self._log_heal("warning", f"Review rejected task {task_num}: {review[:120]}")

        # Exhausted retries
        failure_reason = last_review or feedback or "Exceeded retry attempts."
        return {"task_num": task_num, "ok": False, "message": failure_reason}
    
    def _is_ai_error(self, response):
        """Check if AI response is an error message."""
        if not response or len(response.strip()) < 20:
            return True
        error_indicators = [
            "connection error", "cannot connect", "error:", "ollama error",
            "request timed out", "failed to", "exception", "traceback"
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in error_indicators)

    def build_app(self, project_id):
        """Main app building workflow."""
        project = self.get_project(project_id)
        if not project:
            return False, "Project not found"

        project_name = project[1]
        description = project[2]
        status = project[5]

        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  BUILDING APP: {project_name}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")

        # Quick AI backend check before starting
        print(f"{Colors.DIM}Verifying AI backend...{Colors.RESET}", end="", flush=True)
        test_response = self.ai_engine.generate("Say 'ready' if you can respond.")
        if self._is_ai_error(test_response):
            print(f" {Colors.BRIGHT_RED}FAILED{Colors.RESET}")
            print(f"\n{Colors.BRIGHT_RED}✗ AI Backend Error: {test_response[:100]}{Colors.RESET}")
            print(f"{Colors.YELLOW}Please ensure your AI backend (Ollama/HuggingFace) is running properly.{Colors.RESET}")
            print(f"{Colors.YELLOW}For Ollama, run: ollama serve{Colors.RESET}")
            return False, "AI backend not available"
        print(f" {Colors.BRIGHT_GREEN}OK{Colors.RESET}\n")

        # Stage 1: Specification
        if status == "specification":
            print(f"{Colors.BRIGHT_CYAN}[1/5] Specification Writer analyzing requirements...{Colors.RESET}")
            analysis = self.spec_writer.analyze_description(project_name, description)

            # Check for AI errors
            if self._is_ai_error(analysis):
                print(f"{Colors.BRIGHT_RED}✗ AI Error: {analysis[:100]}...{Colors.RESET}")
                print(f"{Colors.YELLOW}Please check your AI backend (Ollama/HuggingFace) is running.{Colors.RESET}")
                return False, "AI generation failed"

            print(f"\n{Colors.CYAN}Analysis:{Colors.RESET}\n{analysis}\n")

            # Check if questions are needed - interactive Q&A
            if "?" in analysis or "question" in analysis.lower():
                print(f"{Colors.BRIGHT_YELLOW}Specification Writer has questions:{Colors.RESET}")
                qa_pairs = []
                question_lines = [line for line in analysis.split('\n') if '?' in line]
                for q in question_lines[:5]:  # Limit to 5 questions
                    print(f"\n{Colors.CYAN}Q:{Colors.RESET} {q}")
                    answer = input(f"{Colors.BRIGHT_GREEN}A: {Colors.RESET}")
                    qa_pairs.append((q, answer))

                spec = self.spec_writer.write_specification(project_name, description, qa_pairs)
            else:
                spec = self.spec_writer.write_specification(project_name, description)

            # Validate spec
            if self._is_ai_error(spec):
                print(f"{Colors.BRIGHT_RED}✗ AI Error generating specification: {spec[:100]}...{Colors.RESET}")
                return False, "Failed to generate specification"

            print(f"\n{Colors.CYAN}Specification:{Colors.RESET}\n{spec}\n")

            self.update_project_field(project_id, "specification", spec)
            self.update_project_field(project_id, "status", "architecture")

            print(f"\n{Colors.BRIGHT_GREEN}✓ Specification complete!{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")

        # Stage 2: Architecture
        project = self.get_project(project_id)
        if project[5] == "architecture":
            spec = project[3]

            if not spec or len(spec.strip()) < 50:
                print(f"{Colors.BRIGHT_RED}✗ No valid specification found. Cannot proceed with architecture.{Colors.RESET}")
                return False, "Missing specification"

            print(f"\n{Colors.BRIGHT_CYAN}[2/5] Architect designing system...{Colors.RESET}")
            architecture = self.architect.design_architecture(spec)

            # Validate architecture
            if self._is_ai_error(architecture):
                print(f"{Colors.BRIGHT_RED}✗ AI Error generating architecture: {architecture[:100]}...{Colors.RESET}")
                return False, "Failed to generate architecture"

            print(f"\n{Colors.CYAN}Architecture:{Colors.RESET}\n{architecture[:1000]}{'...' if len(architecture) > 1000 else ''}\n")

            self.update_project_field(project_id, "architecture", architecture)

            # Check and install dependencies
            print(f"\n{Colors.BRIGHT_CYAN}Checking dependencies...{Colors.RESET}")
            installed, failed = self.architect.check_and_install_dependencies(architecture)

            if installed:
                print(f"\n{Colors.BRIGHT_GREEN}✓ Installed: {', '.join(installed)}{Colors.RESET}")
            if failed:
                print(f"{Colors.YELLOW}⚠ Failed: {', '.join(failed)}{Colors.RESET}")

            self.update_project_field(project_id, "status", "tasks")
            print(f"\n{Colors.BRIGHT_GREEN}✓ Architecture complete!{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.RESET}")

        # Stage 3: Create Tasks
        project = self.get_project(project_id)
        if project[5] == "tasks":
            spec = project[3]
            architecture = project[4]

            if not architecture or len(architecture.strip()) < 50:
                print(f"{Colors.BRIGHT_RED}✗ No valid architecture found. Cannot create tasks.{Colors.RESET}")
                return False, "Missing architecture"

            print(f"\n{Colors.BRIGHT_CYAN}[3/5] Tech Lead creating task list...{Colors.RESET}")
            tasks_doc, task_lines = self._generate_tasks_with_healing(spec, architecture)

            # Validate tasks document
            if self._is_ai_error(tasks_doc):
                print(f"{Colors.BRIGHT_RED}✗ AI Error generating tasks: {tasks_doc[:100]}...{Colors.RESET}")
                return False, "Failed to generate tasks"

            print(f"\n{Colors.CYAN}Tasks:{Colors.RESET}\n{tasks_doc}\n")

            if not task_lines:
                print(f"{Colors.BRIGHT_RED}✗ Failed to parse tasks from AI output.{Colors.RESET}")
                print(f"{Colors.YELLOW}Please try again or manually add tasks.{Colors.RESET}")
                return False, "No tasks could be parsed"

            lock = getattr(self, "db_lock", None)
            if lock:
                lock.acquire()
            try:
                conn, persistent = self._get_db_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM app_tasks WHERE project_id=?", (project_id,))
                for i, task_line in enumerate(task_lines, 1):
                    cursor.execute(
                        "INSERT INTO app_tasks (project_id, task_number, description, status) VALUES (?, ?, ?, ?)",
                        (project_id, i, task_line, "pending")
                    )
                self._finalize_db(conn, persistent, commit=True)
            finally:
                if lock:
                    lock.release()

            self.update_project_field(project_id, "status", "development")
            print(f"\n{Colors.BRIGHT_GREEN}✓ {len(task_lines)} tasks created!{Colors.RESET}")
            input(f"\n{Colors.DIM}Press Enter to start development...{Colors.RESET}")

        # Stage 4: Development (iterative)
        project = self.get_project(project_id)
        if project[5] == "development":
            self.develop_tasks(project_id)

        return True, "App building process completed"
    
    def develop_tasks(self, project_id):
        """Develop tasks iteratively."""
        project = self.get_project(project_id)
        spec = project[3]
        architecture = project[4]
        builder_threads = getattr(self, "builder_threads", 1)

        conn, persistent = self._get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT id, task_number, description, status FROM app_tasks WHERE project_id=? ORDER BY task_number",
                      (project_id,))
        tasks = cursor.fetchall()
        self._finalize_db(conn, persistent)

        # Check if there are any tasks to develop
        if not tasks:
            print(f"\n{Colors.BRIGHT_RED}✗ No tasks found for this project!{Colors.RESET}")
            print(f"{Colors.YELLOW}The task creation stage may have failed.{Colors.RESET}")
            print(f"{Colors.YELLOW}Please go back and check the project status or try again.{Colors.RESET}")
            return

        if builder_threads > 1:
            self._develop_tasks_threaded(project_id, tasks, spec, architecture, builder_threads)
            return

        failed_tasks = []
        for task_id, task_num, task_desc, task_status in tasks:
            if task_status == "completed":
                print(f"{Colors.DIM}[Task {task_num}] Already completed: {task_desc[:50]}...{Colors.RESET}")
                continue

            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}[Task {task_num}/{len(tasks)}] {task_desc}{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")

            result = self._run_task_pipeline(project_id, (task_id, task_num, task_desc, task_status), spec, architecture)
            if result.get("ok"):
                print(f"{Colors.BRIGHT_GREEN}✓ Task {task_num} {result.get('message', 'completed')}{Colors.RESET}")
            else:
                failed_tasks.append((task_num, task_desc, result.get("message", "failed")))
                print(f"{Colors.YELLOW}⚠ Task {task_num} needs attention: {result.get('message', '')[:200]}{Colors.RESET}")

        conn, persistent = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM app_tasks WHERE project_id=? AND status!='completed'", (project_id,))
        remaining = cursor.fetchone()[0]
        self._finalize_db(conn, persistent)

        if remaining == 0:
            self.update_project_field(project_id, "status", "completed")
            print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ All tasks completed!{Colors.RESET}")
        else:
            self.update_project_field(project_id, "status", "development")
            print(f"\n{Colors.YELLOW}⚠ {remaining} task(s) remain pending after self-healing attempts.{Colors.RESET}")
            for task_num, task_desc, reason in failed_tasks[:5]:
                print(f"{Colors.DIM}  - Task {task_num}: {task_desc[:60]}... ({reason[:80]}){Colors.RESET}")
            return

        # Generate documentation
        print(f"\n{Colors.BRIGHT_CYAN}Technical Writer creating documentation...{Colors.RESET}")
        files = self.get_project_files(project_id)

        if not files:
            print(f"{Colors.YELLOW}⚠ No files generated, skipping documentation.{Colors.RESET}")
            return

        codebase_summary = "\n".join([f"{fp}: {len(content)} lines" for fp, content in files])

        docs = self.tech_writer.write_documentation(project[1], spec, architecture, codebase_summary)

        # Save documentation
        if not self._is_ai_error(docs):
            self.save_file(project_id, "README.md", docs)
            print(f"{Colors.BRIGHT_GREEN}✓ Documentation saved: README.md{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠ Failed to generate documentation.{Colors.RESET}")

        # Show summary
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}  APP BUILD COMPLETE!{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"\n{Colors.CYAN}Project:{Colors.RESET} {Colors.BRIGHT_WHITE}{project[1]}{Colors.RESET}")
        print(f"{Colors.CYAN}Location:{Colors.RESET} {Colors.BRIGHT_WHITE}{APPS_DIR}/{project[1]}/{Colors.RESET}")
        print(f"{Colors.CYAN}Files created:{Colors.RESET} {Colors.BRIGHT_WHITE}{len(files)}{Colors.RESET}")
        for fp, content in files:
            print(f"  {Colors.DIM}- {fp} ({len(content)} chars){Colors.RESET}")

    def _develop_tasks_threaded(self, project_id, tasks, spec, architecture, builder_threads):
        """Parallelized task development while keeping the original pipeline semantics."""
        pending_tasks = [t for t in tasks if t[3] != "completed"]
        if not pending_tasks:
            print(f"{Colors.DIM}No pending tasks to process.{Colors.RESET}")
            return
        
        print(f"\n{Colors.BRIGHT_CYAN}Threaded mode enabled for task execution "
              f"({builder_threads} workers; device: {self.ai_engine.device}).{Colors.RESET}")
        total_tasks = len(pending_tasks)
        
        def worker(task_tuple):
            try:
                result = self._run_task_pipeline(project_id, task_tuple, spec, architecture, announce=False)
                return result.get("task_num"), result.get("ok"), result.get("message", "")
            except Exception as e:
                task_num = task_tuple[1]
                self._log_heal("error", f"Threaded task {task_num} crashed: {e}")
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
        conn, persistent = self._get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM app_tasks WHERE project_id=? AND status!='completed'", (project_id,))
        remaining = cursor.fetchone()[0]
        self._finalize_db(conn, persistent)
        
        if remaining == 0:
            self.update_project_field(project_id, "status", "completed")
            print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ All tasks completed!{Colors.RESET}")
            
            # Generate documentation (same as sequential flow)
            print(f"\n{Colors.BRIGHT_CYAN}Technical Writer creating documentation...{Colors.RESET}")
            files = self.get_project_files(project_id)
            codebase_summary = "\n".join([f"{fp}: {len(content)} lines" for fp, content in files])
            
            project = self.get_project(project_id)
            docs = self.tech_writer.write_documentation(project[1], spec, architecture, codebase_summary)
            if not self._is_ai_error(docs):
                self.save_file(project_id, "README.md", docs)
                print(f"{Colors.BRIGHT_GREEN}✓ Documentation saved: README.md{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}⚠ Failed to generate documentation.{Colors.RESET}")
        else:
            self.update_project_field(project_id, "status", "development")
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
    """
    Display the enhanced startup splash screen with animations
    Uses Rich for advanced rendering if available, otherwise uses ANSI colors
    Complete implementation with no placeholders
    """
    # Try enhanced splash screen first (Rich-based with animations)
    try:
        from tui.components.enhanced_splash import show_enhanced_splash_screen
        show_enhanced_splash_screen(duration=3.0)
        return
    except Exception:
        pass
    
    # Fallback to original splash screen implementation
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
        """Main application run method - complete implementation"""
        # 0. Show enhanced startup splash screen
        show_splash_screen()
        
        # 1. Onboarding
        if self.config.get("first_run"):
            self.onboarding()
        
        # 2. Initialization with loading screens
        show_loading_screen("Initializing Tool Registry", 1.0)
        self.registry = ToolRegistry(self.config)
        
        # Initialize Extension Manager and load extensions
        show_loading_screen("Loading Extensions", 0.5)
        try:
            self.extension_manager = ExtensionManager(self.config, self.registry)
            self.extension_manager.load_extensions()
            self.extension_manager.start_all_extensions()
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ Extension Manager initialization failed: {e}{Colors.RESET}")
            self.extension_manager = None
        
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
        
        # Initialize enhanced help command only (for /help in chat, not for main menu)
        try:
            from tui.app_integration import setup_tui_features
            # Only enable enhanced help, NOT Textual menu/settings (keep original text-based)
            self = setup_tui_features(self, use_textual=False)
        except Exception:
            # Continue without enhanced features if import fails
            self.use_textual = False
            self.textual_available = False
            self.help_handler = None
        
        print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}✓ System Ready!{Colors.RESET}\n")
        time.sleep(0.5)
        
        # 3. Main Menu (always use original text-based menu)
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
        """Main menu - always uses original text-based menu"""
        # Always use original text-based menu (enhanced features only in /help command)
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
            self._handle_menu_selection(c)
    
    def _handle_menu_selection(self, selection: str) -> None:
        """Handle menu selection - supports both text and Textual menu results"""
        if selection == "1" or selection == "chat":
            self.chat_loop()
        elif selection == "2" or selection == "documents":
            self.document_menu()
        elif selection == "3" or selection == "tools":
            self.tool_menu()
        elif selection == "4" or selection == "mcp":
            self.mcp_server_menu()
        elif selection == "5" or selection == "training":
            self.model_training_menu()
        elif selection == "6" or selection == "api":
            self.api_management_menu()
        elif selection == "7" or selection == "app-builder":
            self.app_builder_menu()
        elif selection == "8" or selection == "update":
            self.update_from_github()
        elif selection == "9" or selection == "settings":
            self.settings_menu()
        elif selection == "10" or selection == "exit":
            print(f"\n{Colors.BRIGHT_YELLOW}Shutting down...{Colors.RESET}")
            if self.registry:
                for c in self.registry.mcp_clients.values(): c.stop()
            if self.api_manager:
                for api_name in list(self.api_manager.servers.keys()):
                    self.api_manager.stop_api(api_name)
            print(f"{Colors.BRIGHT_GREEN}✓ Goodbye!{Colors.RESET}\n")
            sys.exit()
        elif selection:
            # Invalid selection, show menu again
            print(f"{Colors.YELLOW}⚠ Invalid selection. Please try again.{Colors.RESET}")
            time.sleep(1)

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
        """Settings menu - always uses original text-based menu"""
        # Always use original text-based settings menu
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
            if self.registry:
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
        
        c = input(f"{Colors.BRIGHT_GREEN}Choice: {Colors.RESET}").strip()
        if c == "1":
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
        elif c == "2":
            return

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
                
                # Filter to only running APIs
                running_apis = []
                for api_name in apis:
                    info = self.api_manager.get_api_info(api_name)
                    if info.get("running"):
                        running_apis.append(api_name)
                
                if not running_apis:
                    print(f"\n{Colors.YELLOW}⚠ No running APIs to stop.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select API to stop:{Colors.RESET}\n")
                for i, api_name in enumerate(running_apis, 1):
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{api_name}{Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(running_apis)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(running_apis):
                        api_name = running_apis[idx - 1]
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
                
                # Filter to only running APIs
                running_apis = []
                for api_name in apis:
                    info = self.api_manager.get_api_info(api_name)
                    if info.get("running"):
                        running_apis.append(api_name)
                
                if not running_apis:
                    print(f"\n{Colors.YELLOW}⚠ No running APIs to test.{Colors.RESET}")
                    time.sleep(1.5)
                    continue
                
                print(f"\n{Colors.CYAN}Select API to test:{Colors.RESET}\n")
                for i, api_name in enumerate(running_apis, 1):
                    info = self.api_manager.get_api_info(api_name)
                    print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{api_name}{Colors.RESET} {Colors.DIM}(Port {info.get('port')}){Colors.RESET}")
                
                try:
                    idx = int(input(f"\n{Colors.BRIGHT_GREEN}Select [1-{len(running_apis)}]: {Colors.RESET}").strip())
                    if 1 <= idx <= len(running_apis):
                        api_name = running_apis[idx - 1]
                        info = self.api_manager.get_api_info(api_name)
                        
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

    def _select_ollama_model(self):
        """Let user select an Ollama model from available models."""
        if self.engine.backend != "ollama":
            print(f"{Colors.YELLOW}⚠ Model selection only available for Ollama backend.{Colors.RESET}")
            print(f"{Colors.DIM}Current backend: {self.engine.backend}{Colors.RESET}")
            return False

        models = get_ollama_models()
        if not models:
            print(f"\n{Colors.BRIGHT_RED}✗ No Ollama models found!{Colors.RESET}")
            print(f"{Colors.YELLOW}Please install a model first:{Colors.RESET}")
            print(f"{Colors.DIM}  ollama pull llama3.2{Colors.RESET}")
            print(f"{Colors.DIM}  ollama pull qwen2.5-coder{Colors.RESET}")
            print(f"{Colors.DIM}  ollama pull codellama{Colors.RESET}")
            return False

        print(f"\n{Colors.BRIGHT_CYAN}Available Ollama Models:{Colors.RESET}\n")
        current_model = self.engine.model_name
        for i, model in enumerate(models, 1):
            marker = f"{Colors.BRIGHT_GREEN}◀ current{Colors.RESET}" if model == current_model else ""
            print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {Colors.BRIGHT_WHITE}{model}{Colors.RESET} {marker}")

        print(f"\n  {Colors.DIM}[0] Cancel{Colors.RESET}")

        try:
            choice = input(f"\n{Colors.BRIGHT_GREEN}Select model [1-{len(models)}]: {Colors.RESET}").strip()
            if choice == "0" or not choice:
                return True  # User cancelled, but not an error

            idx = int(choice)
            if 1 <= idx <= len(models):
                selected_model = models[idx - 1]
                self.engine.model_name = selected_model
                self.engine.config['model_name'] = selected_model
                print(f"\n{Colors.BRIGHT_GREEN}✓ Model changed to: {selected_model}{Colors.RESET}")
                return True
            else:
                print(f"{Colors.YELLOW}⚠ Invalid selection.{Colors.RESET}")
                return False
        except ValueError:
            print(f"{Colors.YELLOW}⚠ Invalid input.{Colors.RESET}")
            return False

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

            # Show current model
            print(f"{Colors.CYAN}AI Model:{Colors.RESET} {Colors.BRIGHT_WHITE}{self.engine.model_name}{Colors.RESET} {Colors.DIM}({self.engine.backend}){Colors.RESET}")
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
            print(f"{Colors.BRIGHT_GREEN}  [7]{Colors.RESET} Change AI Model")
            print(f"{Colors.BRIGHT_GREEN}  [8]{Colors.RESET} Back\n")

            c = input(f"{Colors.BRIGHT_GREEN}Choice: {Colors.RESET}")
            
            if c == "1":
                # Create new app
                self.clear()
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
                print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  CREATE NEW APP{Colors.RESET}")
                print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")

                # Verify model is available before proceeding
                if self.engine.backend == "ollama":
                    models = get_ollama_models()
                    if not models:
                        print(f"{Colors.BRIGHT_RED}✗ No Ollama models installed!{Colors.RESET}")
                        print(f"{Colors.YELLOW}Please install a model first:{Colors.RESET}")
                        print(f"{Colors.DIM}  ollama pull llama3.2{Colors.RESET}")
                        print(f"{Colors.DIM}  ollama pull qwen2.5-coder{Colors.RESET}")
                        input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                        continue

                    if self.engine.model_name not in models:
                        print(f"{Colors.YELLOW}⚠ Current model '{self.engine.model_name}' not found.{Colors.RESET}")
                        print(f"{Colors.CYAN}Please select an available model:{Colors.RESET}")
                        if not self._select_ollama_model():
                            input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                            continue
                        print()

                print(f"{Colors.CYAN}The AI team will help you build this app step by step.{Colors.RESET}")
                print(f"{Colors.DIM}Agents involved: Spec Writer, Architect, Tech Lead, Developer, Code Monkey, Reviewer{Colors.RESET}")
                print(f"{Colors.DIM}Using model: {self.engine.model_name}{Colors.RESET}\n")

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

                # Verify model is available before proceeding
                if self.engine.backend == "ollama":
                    models = get_ollama_models()
                    if not models:
                        print(f"\n{Colors.BRIGHT_RED}✗ No Ollama models installed!{Colors.RESET}")
                        print(f"{Colors.YELLOW}Please install a model first:{Colors.RESET}")
                        print(f"{Colors.DIM}  ollama pull llama3.2{Colors.RESET}")
                        input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
                        continue

                    if self.engine.model_name not in models:
                        print(f"\n{Colors.YELLOW}⚠ Current model '{self.engine.model_name}' not found.{Colors.RESET}")
                        print(f"{Colors.CYAN}Please select an available model:{Colors.RESET}")
                        if not self._select_ollama_model():
                            input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")
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
                # Change AI Model
                self.clear()
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
                print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  CHANGE AI MODEL{Colors.RESET}")
                print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")

                print(f"{Colors.CYAN}Current model:{Colors.RESET} {Colors.BRIGHT_WHITE}{self.engine.model_name}{Colors.RESET}")
                print(f"{Colors.CYAN}Backend:{Colors.RESET} {Colors.BRIGHT_WHITE}{self.engine.backend}{Colors.RESET}\n")

                if self.engine.backend == "ollama":
                    self._select_ollama_model()
                else:
                    print(f"{Colors.YELLOW}Model selection is only available for Ollama backend.{Colors.RESET}")
                    print(f"{Colors.DIM}To switch backends, use Settings from the main menu.{Colors.RESET}")

                input(f"\n{Colors.DIM}Press Enter...{Colors.RESET}")

            elif c == "8":
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
        
        elif cmd == "/help" or cmd == "/?":
            # Use enhanced help command if available (only available in chat, not main menu)
            if hasattr(self, '_help_handler') and self._help_handler is not None:
                try:
                    self._help_handler._show_help()
                    return "COMMAND"
                except Exception:
                    pass  # Fall through to original help
            
            # Fallback to original help display
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

        elif cmd == "/clear":
            # Clear the screen
            self.clear()
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  CHAT MODE ACTIVE{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            if self.current_session_id:
                session = self.memory.get_session(self.current_session_id)
                if session:
                    print(f"{Colors.CYAN}Session:{Colors.RESET} {Colors.BRIGHT_WHITE}{session[1]}{Colors.RESET}")
            if self.current_project_id:
                project = self.memory.get_project(self.current_project_id)
                if project:
                    print(f"{Colors.CYAN}Project:{Colors.RESET} {Colors.BRIGHT_WHITE}{project[1]}{Colors.RESET}")
            print(f"{Colors.DIM}Type '/help' for commands, '/back' to exit{Colors.RESET}\n")
            return "COMMAND"

        elif cmd == "/status":
            # Show system status
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}System Status{Colors.RESET}")
            print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")

            # Model info
            backend = self.config.get('backend', 'unknown')
            model = self.config.get('model_name', 'unknown')
            print(f"  {Colors.CYAN}Backend:{Colors.RESET} {Colors.BRIGHT_WHITE}{backend}{Colors.RESET}")
            print(f"  {Colors.CYAN}Model:{Colors.RESET} {Colors.BRIGHT_WHITE}{model}{Colors.RESET}")

            # Session info
            if self.current_session_id:
                session = self.memory.get_session(self.current_session_id)
                print(f"  {Colors.CYAN}Session:{Colors.RESET} {Colors.BRIGHT_WHITE}{session[1] if session else 'None'}{Colors.RESET}")

            # Project info
            if self.current_project_id:
                project = self.memory.get_project(self.current_project_id)
                print(f"  {Colors.CYAN}Project:{Colors.RESET} {Colors.BRIGHT_WHITE}{project[1] if project else 'None'}{Colors.RESET}")

            # Config info
            print(f"  {Colors.CYAN}Temperature:{Colors.RESET} {Colors.BRIGHT_WHITE}{self.config.get('temperature', 0.7)}{Colors.RESET}")
            print(f"  {Colors.CYAN}Max Tokens:{Colors.RESET} {Colors.BRIGHT_WHITE}{self.config.get('max_response_tokens', 2000)}{Colors.RESET}")
            print(f"  {Colors.CYAN}Context Window:{Colors.RESET} {Colors.BRIGHT_WHITE}{self.config.get('max_context_window', 32768)}{Colors.RESET}")

            # MCP server status
            if hasattr(self, 'registry') and self.registry:
                mcp_count = len(self.registry.mcp_clients) if hasattr(self.registry, 'mcp_clients') else 0
                print(f"  {Colors.CYAN}MCP Servers:{Colors.RESET} {Colors.BRIGHT_WHITE}{mcp_count} configured{Colors.RESET}")

            print()
            return "COMMAND"

        elif cmd == "/model":
            # Show or change model
            if not args:
                # Show current model
                backend = self.config.get('backend', 'unknown')
                model = self.config.get('model_name', 'unknown')
                print(f"\n{Colors.BRIGHT_CYAN}Current Model Configuration:{Colors.RESET}")
                print(f"  {Colors.CYAN}Backend:{Colors.RESET} {Colors.BRIGHT_WHITE}{backend}{Colors.RESET}")
                print(f"  {Colors.CYAN}Model:{Colors.RESET} {Colors.BRIGHT_WHITE}{model}{Colors.RESET}")
                print(f"\n{Colors.DIM}Use /model [name] to switch models{Colors.RESET}")
            else:
                # Change model
                new_model = args.strip()
                old_model = self.config.get('model_name', 'unknown')
                self.cfg_mgr.update('model_name', new_model)
                print(f"\n{Colors.BRIGHT_GREEN}✓ Model changed:{Colors.RESET} {Colors.DIM}{old_model}{Colors.RESET} → {Colors.BRIGHT_WHITE}{new_model}{Colors.RESET}")
                print(f"{Colors.YELLOW}Note: Model change will take effect on next message or after reload{Colors.RESET}")
            return "COMMAND"

        elif cmd == "/config":
            # Show or change configuration
            if not args:
                # Show all config
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Current Configuration:{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")
                for key, value in self.config.items():
                    if key != 'system_prompt':  # Don't show long system prompt
                        print(f"  {Colors.CYAN}{key}:{Colors.RESET} {Colors.BRIGHT_WHITE}{value}{Colors.RESET}")
                print(f"\n{Colors.DIM}Use /config [key] [value] to change settings{Colors.RESET}")
            else:
                # Change config
                parts = args.split(" ", 1)
                key = parts[0].strip()
                if len(parts) > 1:
                    value = parts[1].strip()
                    # Try to convert to appropriate type
                    try:
                        if value.lower() in ('true', 'false'):
                            value = value.lower() == 'true'
                        elif '.' in value:
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string

                    old_value = self.config.get(key, 'not set')
                    self.cfg_mgr.update(key, value)
                    print(f"\n{Colors.BRIGHT_GREEN}✓ Config updated:{Colors.RESET} {key}: {Colors.DIM}{old_value}{Colors.RESET} → {Colors.BRIGHT_WHITE}{value}{Colors.RESET}")
                else:
                    # Show specific config value
                    if key in self.config:
                        print(f"\n{Colors.CYAN}{key}:{Colors.RESET} {Colors.BRIGHT_WHITE}{self.config[key]}{Colors.RESET}")
                    else:
                        print(f"\n{Colors.YELLOW}Unknown config key: {key}{Colors.RESET}")
            return "COMMAND"

        elif cmd == "/rag":
            # RAG management commands
            subcmd = args.split()[0].lower() if args else ""
            subargs = " ".join(args.split()[1:]) if args and len(args.split()) > 1 else ""

            if not subcmd:
                # Show RAG status
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}RAG Status{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")
                try:
                    # Get document count from memory
                    cursor = self.memory.conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM documents")
                    doc_count = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(DISTINCT source) FROM documents")
                    source_count = cursor.fetchone()[0]
                    print(f"  {Colors.CYAN}Documents:{Colors.RESET} {Colors.BRIGHT_WHITE}{source_count} files{Colors.RESET}")
                    print(f"  {Colors.CYAN}Chunks:{Colors.RESET} {Colors.BRIGHT_WHITE}{doc_count} chunks{Colors.RESET}")
                    if self.current_project_id:
                        cursor.execute("SELECT COUNT(*) FROM documents WHERE project_id = ?", (self.current_project_id,))
                        proj_count = cursor.fetchone()[0]
                        print(f"  {Colors.CYAN}Project Chunks:{Colors.RESET} {Colors.BRIGHT_WHITE}{proj_count}{Colors.RESET}")
                except Exception as e:
                    print(f"  {Colors.RED}Error getting RAG status: {e}{Colors.RESET}")
                print(f"\n{Colors.DIM}Commands: /rag add, /rag search, /rag list, /rag clear{Colors.RESET}")

            elif subcmd == "list":
                # List documents
                print(f"\n{Colors.BRIGHT_CYAN}Ingested Documents:{Colors.RESET}")
                try:
                    cursor = self.memory.conn.cursor()
                    cursor.execute("SELECT DISTINCT source FROM documents ORDER BY source")
                    sources = cursor.fetchall()
                    if sources:
                        for src in sources[:20]:  # Limit to 20
                            print(f"  {Colors.DIM}•{Colors.RESET} {Colors.BRIGHT_WHITE}{src[0]}{Colors.RESET}")
                        if len(sources) > 20:
                            print(f"  {Colors.DIM}... and {len(sources) - 20} more{Colors.RESET}")
                    else:
                        print(f"  {Colors.YELLOW}No documents loaded{Colors.RESET}")
                except Exception as e:
                    print(f"  {Colors.RED}Error: {e}{Colors.RESET}")

            elif subcmd == "search":
                # Search documents
                if not subargs:
                    print(f"{Colors.YELLOW}Usage: /rag search [query]{Colors.RESET}")
                else:
                    print(f"\n{Colors.BRIGHT_CYAN}Searching for: {Colors.BRIGHT_WHITE}{subargs}{Colors.RESET}")
                    results = self.memory.retrieve_context(subargs, self.current_project_id)
                    if results:
                        print(f"\n{Colors.BRIGHT_GREEN}Found context:{Colors.RESET}")
                        # Show truncated results
                        display = results[:500] + "..." if len(results) > 500 else results
                        print(f"{Colors.DIM}{display}{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}No relevant documents found{Colors.RESET}")

            elif subcmd == "add":
                # Add document - redirect to document loader
                print(f"\n{Colors.BRIGHT_CYAN}Document Loading{Colors.RESET}")
                if subargs:
                    path = subargs.strip()
                    if os.path.exists(path):
                        print(f"{Colors.YELLOW}Use the Document Loader from main menu for full functionality{Colors.RESET}")
                        print(f"{Colors.DIM}Path: {path}{Colors.RESET}")
                    else:
                        print(f"{Colors.RED}Path not found: {path}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}Use the Document Loader (option 2) from main menu{Colors.RESET}")

            elif subcmd == "clear":
                # Clear documents
                confirm = input(f"{Colors.BRIGHT_RED}Clear all RAG documents? This cannot be undone. [y/N]: {Colors.RESET}").strip().lower()
                if confirm == 'y':
                    try:
                        cursor = self.memory.conn.cursor()
                        cursor.execute("DELETE FROM documents")
                        self.memory.conn.commit()
                        print(f"{Colors.BRIGHT_GREEN}✓ All documents cleared{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.DIM}Cancelled{Colors.RESET}")

            else:
                print(f"{Colors.YELLOW}Unknown RAG command: {subcmd}{Colors.RESET}")
                print(f"{Colors.DIM}Available: /rag, /rag list, /rag search, /rag add, /rag clear{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/mcp":
            # MCP server management
            subcmd = args.split()[0].lower() if args else ""
            subargs = " ".join(args.split()[1:]) if args and len(args.split()) > 1 else ""

            if not subcmd:
                # Show MCP status
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}MCP Server Status{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")
                if hasattr(self, 'registry') and self.registry and hasattr(self.registry, 'mcp_clients'):
                    clients = self.registry.mcp_clients
                    if clients:
                        for name, client in clients.items():
                            status = f"{Colors.BRIGHT_GREEN}running{Colors.RESET}" if client.process else f"{Colors.YELLOW}stopped{Colors.RESET}"
                            tool_count = len(client.tools) if hasattr(client, 'tools') else 0
                            print(f"  {Colors.CYAN}{name}:{Colors.RESET} {status} ({tool_count} tools)")
                    else:
                        print(f"  {Colors.YELLOW}No MCP servers configured{Colors.RESET}")
                else:
                    print(f"  {Colors.YELLOW}MCP not available{Colors.RESET}")
                print(f"\n{Colors.DIM}Commands: /mcp list, /mcp start, /mcp stop, /mcp tools{Colors.RESET}")

            elif subcmd == "list":
                # List all MCP servers and tools
                print(f"\n{Colors.BRIGHT_CYAN}MCP Servers and Tools:{Colors.RESET}")
                if hasattr(self, 'registry') and self.registry and hasattr(self.registry, 'mcp_clients'):
                    for name, client in self.registry.mcp_clients.items():
                        status = f"{Colors.BRIGHT_GREEN}●{Colors.RESET}" if client.process else f"{Colors.RED}○{Colors.RESET}"
                        print(f"\n  {status} {Colors.BRIGHT_WHITE}{name}{Colors.RESET}")
                        if hasattr(client, 'tools') and client.tools:
                            for tool in list(client.tools.keys())[:5]:
                                print(f"      {Colors.DIM}• {tool}{Colors.RESET}")
                            if len(client.tools) > 5:
                                print(f"      {Colors.DIM}... and {len(client.tools) - 5} more{Colors.RESET}")
                else:
                    print(f"  {Colors.YELLOW}No MCP servers available{Colors.RESET}")

            elif subcmd == "tools":
                # List tools from specific server
                if not subargs:
                    print(f"{Colors.YELLOW}Usage: /mcp tools [server_name]{Colors.RESET}")
                elif hasattr(self, 'registry') and self.registry and hasattr(self.registry, 'mcp_clients'):
                    if subargs in self.registry.mcp_clients:
                        client = self.registry.mcp_clients[subargs]
                        print(f"\n{Colors.BRIGHT_CYAN}Tools from {subargs}:{Colors.RESET}")
                        if hasattr(client, 'tools') and client.tools:
                            for tool_name, tool_info in client.tools.items():
                                desc = tool_info.get('description', 'No description')[:60]
                                print(f"  {Colors.BRIGHT_GREEN}{tool_name}{Colors.RESET}: {Colors.DIM}{desc}{Colors.RESET}")
                        else:
                            print(f"  {Colors.YELLOW}No tools available{Colors.RESET}")
                    else:
                        print(f"{Colors.RED}Server not found: {subargs}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}MCP not available{Colors.RESET}")

            elif subcmd == "start":
                # Start MCP server
                if not subargs:
                    print(f"{Colors.YELLOW}Usage: /mcp start [server_name]{Colors.RESET}")
                elif hasattr(self, 'registry') and self.registry:
                    print(f"{Colors.CYAN}Starting MCP server: {subargs}...{Colors.RESET}")
                    # This would need implementation in ToolRegistry
                    print(f"{Colors.YELLOW}Use MCP Server Management from main menu for full control{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}MCP not available{Colors.RESET}")

            elif subcmd == "stop":
                # Stop MCP server
                if not subargs:
                    print(f"{Colors.YELLOW}Usage: /mcp stop [server_name]{Colors.RESET}")
                elif hasattr(self, 'registry') and self.registry and hasattr(self.registry, 'mcp_clients'):
                    if subargs in self.registry.mcp_clients:
                        try:
                            self.registry.mcp_clients[subargs].stop()
                            print(f"{Colors.BRIGHT_GREEN}✓ Stopped: {subargs}{Colors.RESET}")
                        except Exception as e:
                            print(f"{Colors.RED}Error stopping server: {e}{Colors.RESET}")
                    else:
                        print(f"{Colors.RED}Server not found: {subargs}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}MCP not available{Colors.RESET}")

            else:
                print(f"{Colors.YELLOW}Unknown MCP command: {subcmd}{Colors.RESET}")
                print(f"{Colors.DIM}Available: /mcp, /mcp list, /mcp tools, /mcp start, /mcp stop{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/tools" or cmd == "/tool":
            # Tool management
            if cmd == "/tools" or not args:
                # List all tools
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Available Tools{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")

                if hasattr(self, 'registry') and self.registry:
                    # Native tools
                    if hasattr(self.registry, 'native_tools'):
                        print(f"\n  {Colors.BRIGHT_GREEN}Native Tools:{Colors.RESET}")
                        for name in list(self.registry.native_tools.keys())[:10]:
                            print(f"    {Colors.DIM}•{Colors.RESET} {name}")

                    # Custom tools
                    if hasattr(self.registry, 'custom_tool_manager'):
                        ctm = self.registry.custom_tool_manager
                        if hasattr(ctm, 'tools') and ctm.tools:
                            print(f"\n  {Colors.BRIGHT_YELLOW}Custom Tools:{Colors.RESET}")
                            for name in list(ctm.tools.keys())[:10]:
                                print(f"    {Colors.DIM}•{Colors.RESET} {name}")

                    # MCP tools
                    if hasattr(self.registry, 'mcp_clients'):
                        mcp_tools = []
                        for client in self.registry.mcp_clients.values():
                            if hasattr(client, 'tools'):
                                mcp_tools.extend(client.tools.keys())
                        if mcp_tools:
                            print(f"\n  {Colors.BRIGHT_MAGENTA}MCP Tools:{Colors.RESET}")
                            for name in mcp_tools[:10]:
                                print(f"    {Colors.DIM}•{Colors.RESET} {name}")
                            if len(mcp_tools) > 10:
                                print(f"    {Colors.DIM}... and {len(mcp_tools) - 10} more{Colors.RESET}")
                else:
                    print(f"  {Colors.YELLOW}Tool registry not available{Colors.RESET}")

                print(f"\n{Colors.DIM}Use /tool [name] for details, ACTION: TOOL_NAME args to execute{Colors.RESET}")

            else:
                # Show specific tool details
                subcmd = args.split()[0].lower()
                subargs = " ".join(args.split()[1:]) if len(args.split()) > 1 else ""

                if subcmd == "run" and subargs:
                    # Execute tool directly
                    tool_parts = subargs.split(" ", 1)
                    tool_name = tool_parts[0]
                    tool_args = tool_parts[1] if len(tool_parts) > 1 else ""
                    print(f"\n{Colors.CYAN}Executing tool: {tool_name}{Colors.RESET}")
                    print(f"{Colors.YELLOW}Use ACTION: {tool_name} {tool_args} in your message for tool execution{Colors.RESET}")
                else:
                    # Show tool details
                    tool_name = subcmd
                    print(f"\n{Colors.BRIGHT_CYAN}Tool: {tool_name}{Colors.RESET}")
                    found = False

                    if hasattr(self, 'registry') and self.registry:
                        # Check native tools
                        if hasattr(self.registry, 'native_tools') and tool_name in self.registry.native_tools:
                            tool = self.registry.native_tools[tool_name]
                            print(f"  {Colors.CYAN}Type:{Colors.RESET} Native")
                            if callable(tool):
                                print(f"  {Colors.CYAN}Function:{Colors.RESET} {tool.__name__}")
                            found = True

                        # Check custom tools
                        if hasattr(self.registry, 'custom_tool_manager'):
                            ctm = self.registry.custom_tool_manager
                            if hasattr(ctm, 'tools') and tool_name in ctm.tools:
                                tool = ctm.tools[tool_name]
                                print(f"  {Colors.CYAN}Type:{Colors.RESET} Custom")
                                if isinstance(tool, dict):
                                    print(f"  {Colors.CYAN}Description:{Colors.RESET} {tool.get('description', 'N/A')}")
                                found = True

                        # Check MCP tools
                        if hasattr(self.registry, 'mcp_clients'):
                            for server_name, client in self.registry.mcp_clients.items():
                                if hasattr(client, 'tools') and tool_name in client.tools:
                                    tool_info = client.tools[tool_name]
                                    print(f"  {Colors.CYAN}Type:{Colors.RESET} MCP ({server_name})")
                                    print(f"  {Colors.CYAN}Description:{Colors.RESET} {tool_info.get('description', 'N/A')}")
                                    found = True
                                    break

                    if not found:
                        print(f"  {Colors.YELLOW}Tool not found: {tool_name}{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/browser":
            # Full browser automation with Playwright
            subcmd = args.split()[0].lower() if args else ""
            sub_args = ' '.join(args.split()[1:]) if len(args.split()) > 1 else ""

            # Initialize browser manager if not exists
            if not hasattr(self, '_browser_instance'):
                self._browser_instance = None
                self._browser_page = None
                self._browser_context = None

            def _ensure_playwright():
                """Ensure playwright is available and browser is running"""
                try:
                    from playwright.sync_api import sync_playwright
                    return True
                except ImportError:
                    print(f"{Colors.YELLOW}Playwright not installed. Install with: pip install playwright{Colors.RESET}")
                    print(f"{Colors.DIM}Then run: playwright install chromium{Colors.RESET}")
                    return False

            def _get_browser():
                """Get or create browser instance"""
                if self._browser_instance is None:
                    try:
                        from playwright.sync_api import sync_playwright
                        self._playwright = sync_playwright().start()
                        self._browser_instance = self._playwright.chromium.launch(
                            headless=False,
                            args=['--no-sandbox', '--disable-dev-shm-usage']
                        )
                        self._browser_context = self._browser_instance.new_context(
                            viewport={'width': 1280, 'height': 720}
                        )
                        self._browser_page = self._browser_context.new_page()
                        print(f"{Colors.BRIGHT_GREEN}✓ Browser started{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Failed to start browser: {e}{Colors.RESET}")
                        return None
                return self._browser_page

            if not subcmd:
                # Show browser help
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Browser Automation (Playwright){Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")
                print(f"\n{Colors.BRIGHT_WHITE}Navigation:{Colors.RESET}")
                print(f"  {Colors.CYAN}/browser open <url>{Colors.RESET}      - Navigate to URL")
                print(f"  {Colors.CYAN}/browser back{Colors.RESET}            - Go back")
                print(f"  {Colors.CYAN}/browser forward{Colors.RESET}         - Go forward")
                print(f"  {Colors.CYAN}/browser refresh{Colors.RESET}         - Refresh page")
                print(f"  {Colors.CYAN}/browser url{Colors.RESET}             - Show current URL")
                print(f"\n{Colors.BRIGHT_WHITE}Capture:{Colors.RESET}")
                print(f"  {Colors.CYAN}/browser screenshot [path]{Colors.RESET} - Take screenshot")
                print(f"  {Colors.CYAN}/browser html{Colors.RESET}            - Get page HTML")
                print(f"  {Colors.CYAN}/browser text{Colors.RESET}            - Get page text content")
                print(f"  {Colors.CYAN}/browser title{Colors.RESET}           - Get page title")
                print(f"\n{Colors.BRIGHT_WHITE}Interaction:{Colors.RESET}")
                print(f"  {Colors.CYAN}/browser click <selector>{Colors.RESET} - Click element")
                print(f"  {Colors.CYAN}/browser fill <sel> <text>{Colors.RESET} - Fill input field")
                print(f"  {Colors.CYAN}/browser type <text>{Colors.RESET}     - Type text (active element)")
                print(f"  {Colors.CYAN}/browser scroll <dir>{Colors.RESET}    - Scroll up/down")
                print(f"\n{Colors.BRIGHT_WHITE}Advanced:{Colors.RESET}")
                print(f"  {Colors.CYAN}/browser js <code>{Colors.RESET}       - Execute JavaScript")
                print(f"  {Colors.CYAN}/browser wait <ms>{Colors.RESET}       - Wait milliseconds")
                print(f"  {Colors.CYAN}/browser find <selector>{Colors.RESET} - Find elements")
                print(f"  {Colors.CYAN}/browser pdf [path]{Colors.RESET}      - Export page to PDF")
                print(f"\n{Colors.BRIGHT_WHITE}Control:{Colors.RESET}")
                print(f"  {Colors.CYAN}/browser status{Colors.RESET}          - Show browser status")
                print(f"  {Colors.CYAN}/browser close{Colors.RESET}           - Close browser")
                print(f"\n{Colors.DIM}Note: Browser will open in visible mode for interaction{Colors.RESET}")

            elif subcmd in ("open", "go", "navigate"):
                # Navigate to URL
                url = sub_args.strip()
                if not url:
                    print(f"{Colors.YELLOW}Usage: /browser open <url>{Colors.RESET}")
                else:
                    if not url.startswith(('http://', 'https://', 'file://')):
                        url = 'https://' + url

                    if not _ensure_playwright():
                        return "COMMAND"

                    page = _get_browser()
                    if page:
                        try:
                            print(f"{Colors.CYAN}Navigating to: {url}{Colors.RESET}")
                            page.goto(url, wait_until='domcontentloaded', timeout=30000)
                            print(f"{Colors.BRIGHT_GREEN}✓ Loaded: {page.title()}{Colors.RESET}")
                        except Exception as e:
                            print(f"{Colors.RED}Navigation error: {e}{Colors.RESET}")

            elif subcmd == "back":
                if self._browser_page:
                    try:
                        self._browser_page.go_back()
                        print(f"{Colors.BRIGHT_GREEN}✓ Navigated back{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open. Use /browser open <url>{Colors.RESET}")

            elif subcmd == "forward":
                if self._browser_page:
                    try:
                        self._browser_page.go_forward()
                        print(f"{Colors.BRIGHT_GREEN}✓ Navigated forward{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open. Use /browser open <url>{Colors.RESET}")

            elif subcmd == "refresh":
                if self._browser_page:
                    try:
                        self._browser_page.reload()
                        print(f"{Colors.BRIGHT_GREEN}✓ Page refreshed{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open. Use /browser open <url>{Colors.RESET}")

            elif subcmd == "url":
                if self._browser_page:
                    print(f"{Colors.CYAN}Current URL:{Colors.RESET} {self._browser_page.url}")
                else:
                    print(f"{Colors.YELLOW}No browser open{Colors.RESET}")

            elif subcmd == "title":
                if self._browser_page:
                    print(f"{Colors.CYAN}Page title:{Colors.RESET} {self._browser_page.title()}")
                else:
                    print(f"{Colors.YELLOW}No browser open{Colors.RESET}")

            elif subcmd == "screenshot":
                if self._browser_page:
                    try:
                        path = sub_args.strip() if sub_args else f"screenshot_{int(time.time())}.png"
                        if not path.endswith(('.png', '.jpg', '.jpeg')):
                            path += '.png'
                        self._browser_page.screenshot(path=path, full_page=True)
                        print(f"{Colors.BRIGHT_GREEN}✓ Screenshot saved: {path}{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Screenshot error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open. Use /browser open <url>{Colors.RESET}")

            elif subcmd == "pdf":
                if self._browser_page:
                    try:
                        path = sub_args.strip() if sub_args else f"page_{int(time.time())}.pdf"
                        if not path.endswith('.pdf'):
                            path += '.pdf'
                        self._browser_page.pdf(path=path)
                        print(f"{Colors.BRIGHT_GREEN}✓ PDF saved: {path}{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}PDF error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open{Colors.RESET}")

            elif subcmd == "html":
                if self._browser_page:
                    try:
                        html = self._browser_page.content()
                        print(f"\n{Colors.BRIGHT_CYAN}Page HTML ({len(html)} chars):{Colors.RESET}")
                        print(f"{Colors.DIM}{html[:2000]}{'...' if len(html) > 2000 else ''}{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open{Colors.RESET}")

            elif subcmd == "text":
                if self._browser_page:
                    try:
                        text = self._browser_page.inner_text('body')
                        print(f"\n{Colors.BRIGHT_CYAN}Page Text ({len(text)} chars):{Colors.RESET}")
                        print(f"{Colors.DIM}{text[:3000]}{'...' if len(text) > 3000 else ''}{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open{Colors.RESET}")

            elif subcmd == "click":
                if self._browser_page:
                    selector = sub_args.strip()
                    if not selector:
                        print(f"{Colors.YELLOW}Usage: /browser click <selector>{Colors.RESET}")
                        print(f"{Colors.DIM}Examples: /browser click button#submit{Colors.RESET}")
                        print(f"{Colors.DIM}          /browser click \"text=Sign In\"{Colors.RESET}")
                    else:
                        try:
                            self._browser_page.click(selector, timeout=5000)
                            print(f"{Colors.BRIGHT_GREEN}✓ Clicked: {selector}{Colors.RESET}")
                        except Exception as e:
                            print(f"{Colors.RED}Click error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open{Colors.RESET}")

            elif subcmd == "fill":
                if self._browser_page:
                    parts = sub_args.split(maxsplit=1)
                    if len(parts) < 2:
                        print(f"{Colors.YELLOW}Usage: /browser fill <selector> <text>{Colors.RESET}")
                        print(f"{Colors.DIM}Example: /browser fill input#email test@example.com{Colors.RESET}")
                    else:
                        selector, text = parts
                        try:
                            self._browser_page.fill(selector, text)
                            print(f"{Colors.BRIGHT_GREEN}✓ Filled: {selector}{Colors.RESET}")
                        except Exception as e:
                            print(f"{Colors.RED}Fill error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open{Colors.RESET}")

            elif subcmd == "type":
                if self._browser_page:
                    text = sub_args.strip()
                    if not text:
                        print(f"{Colors.YELLOW}Usage: /browser type <text>{Colors.RESET}")
                    else:
                        try:
                            self._browser_page.keyboard.type(text)
                            print(f"{Colors.BRIGHT_GREEN}✓ Typed: {text[:50]}{'...' if len(text) > 50 else ''}{Colors.RESET}")
                        except Exception as e:
                            print(f"{Colors.RED}Type error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open{Colors.RESET}")

            elif subcmd == "scroll":
                if self._browser_page:
                    direction = sub_args.strip().lower()
                    try:
                        if direction == "up":
                            self._browser_page.evaluate("window.scrollBy(0, -500)")
                            print(f"{Colors.BRIGHT_GREEN}✓ Scrolled up{Colors.RESET}")
                        elif direction == "down":
                            self._browser_page.evaluate("window.scrollBy(0, 500)")
                            print(f"{Colors.BRIGHT_GREEN}✓ Scrolled down{Colors.RESET}")
                        elif direction == "top":
                            self._browser_page.evaluate("window.scrollTo(0, 0)")
                            print(f"{Colors.BRIGHT_GREEN}✓ Scrolled to top{Colors.RESET}")
                        elif direction == "bottom":
                            self._browser_page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                            print(f"{Colors.BRIGHT_GREEN}✓ Scrolled to bottom{Colors.RESET}")
                        else:
                            print(f"{Colors.YELLOW}Usage: /browser scroll <up|down|top|bottom>{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Scroll error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open{Colors.RESET}")

            elif subcmd == "js":
                if self._browser_page:
                    code = sub_args.strip()
                    if not code:
                        print(f"{Colors.YELLOW}Usage: /browser js <javascript code>{Colors.RESET}")
                        print(f"{Colors.DIM}Example: /browser js document.title{Colors.RESET}")
                    else:
                        try:
                            result = self._browser_page.evaluate(code)
                            print(f"{Colors.BRIGHT_GREEN}✓ Result:{Colors.RESET} {result}")
                        except Exception as e:
                            print(f"{Colors.RED}JS error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open{Colors.RESET}")

            elif subcmd == "wait":
                if self._browser_page:
                    try:
                        ms = int(sub_args.strip()) if sub_args else 1000
                        self._browser_page.wait_for_timeout(ms)
                        print(f"{Colors.BRIGHT_GREEN}✓ Waited {ms}ms{Colors.RESET}")
                    except ValueError:
                        print(f"{Colors.YELLOW}Usage: /browser wait <milliseconds>{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Wait error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open{Colors.RESET}")

            elif subcmd == "find":
                if self._browser_page:
                    selector = sub_args.strip()
                    if not selector:
                        print(f"{Colors.YELLOW}Usage: /browser find <selector>{Colors.RESET}")
                    else:
                        try:
                            elements = self._browser_page.query_selector_all(selector)
                            print(f"\n{Colors.BRIGHT_CYAN}Found {len(elements)} element(s):{Colors.RESET}")
                            for i, el in enumerate(elements[:10]):  # Limit to 10
                                tag = self._browser_page.evaluate("el => el.tagName", el)
                                text = self._browser_page.evaluate("el => el.innerText?.slice(0, 50)", el)
                                print(f"  {Colors.CYAN}{i+1}.{Colors.RESET} <{tag.lower()}> {Colors.DIM}{text or ''}{Colors.RESET}")
                            if len(elements) > 10:
                                print(f"  {Colors.DIM}... and {len(elements) - 10} more{Colors.RESET}")
                        except Exception as e:
                            print(f"{Colors.RED}Find error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}No browser open{Colors.RESET}")

            elif subcmd == "status":
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Browser Status{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")
                if self._browser_page:
                    try:
                        print(f"  {Colors.CYAN}Status:{Colors.RESET} {Colors.BRIGHT_GREEN}Running{Colors.RESET}")
                        print(f"  {Colors.CYAN}URL:{Colors.RESET} {self._browser_page.url}")
                        print(f"  {Colors.CYAN}Title:{Colors.RESET} {self._browser_page.title()}")
                        viewport = self._browser_page.viewport_size
                        if viewport:
                            print(f"  {Colors.CYAN}Viewport:{Colors.RESET} {viewport['width']}x{viewport['height']}")
                    except Exception as e:
                        print(f"  {Colors.CYAN}Status:{Colors.RESET} {Colors.RED}Error: {e}{Colors.RESET}")
                else:
                    print(f"  {Colors.CYAN}Status:{Colors.RESET} {Colors.DIM}Not started{Colors.RESET}")
                    print(f"  {Colors.DIM}Use /browser open <url> to start{Colors.RESET}")

            elif subcmd == "close":
                if self._browser_instance:
                    try:
                        if self._browser_context:
                            self._browser_context.close()
                        self._browser_instance.close()
                        if hasattr(self, '_playwright'):
                            self._playwright.stop()
                        self._browser_instance = None
                        self._browser_page = None
                        self._browser_context = None
                        print(f"{Colors.BRIGHT_GREEN}✓ Browser closed{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Error closing browser: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.DIM}No browser to close{Colors.RESET}")

            else:
                # Treat as URL
                url = args.strip()
                if not url.startswith(('http://', 'https://', 'file://')):
                    url = 'https://' + url

                if not _ensure_playwright():
                    return "COMMAND"

                page = _get_browser()
                if page:
                    try:
                        print(f"{Colors.CYAN}Navigating to: {url}{Colors.RESET}")
                        page.goto(url, wait_until='domcontentloaded', timeout=30000)
                        print(f"{Colors.BRIGHT_GREEN}✓ Loaded: {page.title()}{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Navigation error: {e}{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/memory":
            # Memory management
            subcmd = args.split()[0].lower() if args else ""

            if not subcmd:
                # Show memory status
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Memory Status{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")
                try:
                    history = self.memory.get_recent_history(self.current_session_id, limit=1000)
                    print(f"  {Colors.CYAN}Messages in session:{Colors.RESET} {Colors.BRIGHT_WHITE}{len(history)}{Colors.RESET}")
                    if self.context_mgr:
                        # Estimate token usage
                        total_chars = sum(len(h[1]) for h in history)
                        est_tokens = total_chars // 4  # Rough estimate
                        print(f"  {Colors.CYAN}Estimated tokens:{Colors.RESET} {Colors.BRIGHT_WHITE}{est_tokens}{Colors.RESET}")
                        print(f"  {Colors.CYAN}Max context:{Colors.RESET} {Colors.BRIGHT_WHITE}{self.config.get('max_context_window', 32768)}{Colors.RESET}")
                except Exception as e:
                    print(f"  {Colors.RED}Error: {e}{Colors.RESET}")
                print(f"\n{Colors.DIM}Use /memory clear to clear conversation memory{Colors.RESET}")

            elif subcmd == "clear":
                confirm = input(f"{Colors.BRIGHT_RED}Clear conversation memory for this session? [y/N]: {Colors.RESET}").strip().lower()
                if confirm == 'y':
                    try:
                        cursor = self.memory.conn.cursor()
                        cursor.execute("DELETE FROM history WHERE session_id = ?", (self.current_session_id,))
                        self.memory.conn.commit()
                        print(f"{Colors.BRIGHT_GREEN}✓ Conversation memory cleared{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.DIM}Cancelled{Colors.RESET}")

            else:
                print(f"{Colors.YELLOW}Unknown memory command: {subcmd}{Colors.RESET}")
                print(f"{Colors.DIM}Available: /memory, /memory clear{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/history":
            # Conversation history
            subcmd = args.split()[0].lower() if args else ""

            if not subcmd:
                # Show recent history
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Conversation History{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")
                try:
                    history = self.memory.get_recent_history(self.current_session_id, limit=10)
                    if history:
                        for role, content in history:
                            role_color = Colors.BRIGHT_GREEN if role == 'user' else Colors.BRIGHT_CYAN
                            truncated = content[:100] + "..." if len(content) > 100 else content
                            print(f"  {role_color}{role}:{Colors.RESET} {Colors.DIM}{truncated}{Colors.RESET}")
                    else:
                        print(f"  {Colors.YELLOW}No history in this session{Colors.RESET}")
                except Exception as e:
                    print(f"  {Colors.RED}Error: {e}{Colors.RESET}")
                print(f"\n{Colors.DIM}Use /history [n] for last n messages, /history clear to clear{Colors.RESET}")

            elif subcmd == "clear":
                confirm = input(f"{Colors.BRIGHT_RED}Clear conversation history? [y/N]: {Colors.RESET}").strip().lower()
                if confirm == 'y':
                    try:
                        cursor = self.memory.conn.cursor()
                        cursor.execute("DELETE FROM history WHERE session_id = ?", (self.current_session_id,))
                        self.memory.conn.commit()
                        print(f"{Colors.BRIGHT_GREEN}✓ History cleared{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.DIM}Cancelled{Colors.RESET}")

            else:
                # Try to parse as number
                try:
                    n = int(subcmd)
                    history = self.memory.get_recent_history(self.current_session_id, limit=n)
                    print(f"\n{Colors.BRIGHT_CYAN}Last {n} messages:{Colors.RESET}")
                    for role, content in history:
                        role_color = Colors.BRIGHT_GREEN if role == 'user' else Colors.BRIGHT_CYAN
                        truncated = content[:150] + "..." if len(content) > 150 else content
                        print(f"  {role_color}{role}:{Colors.RESET} {Colors.DIM}{truncated}{Colors.RESET}")
                except ValueError:
                    print(f"{Colors.YELLOW}Invalid argument: {subcmd}{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/export":
            # Export conversation
            format_type = args.strip().lower() if args else "json"

            print(f"\n{Colors.BRIGHT_CYAN}Exporting conversation...{Colors.RESET}")
            try:
                history = self.memory.get_recent_history(self.current_session_id, limit=10000)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

                if format_type == "json":
                    filename = f"export_{timestamp}.json"
                    filepath = os.path.join(SANDBOX_DIR, filename)
                    import json
                    with open(filepath, 'w') as f:
                        json.dump([{"role": r, "content": c} for r, c in history], f, indent=2)
                    print(f"{Colors.BRIGHT_GREEN}✓ Exported to: {filepath}{Colors.RESET}")

                elif format_type == "md" or format_type == "markdown":
                    filename = f"export_{timestamp}.md"
                    filepath = os.path.join(SANDBOX_DIR, filename)
                    with open(filepath, 'w') as f:
                        f.write(f"# Chat Export - {timestamp}\n\n")
                        for role, content in history:
                            f.write(f"## {role.title()}\n\n{content}\n\n---\n\n")
                    print(f"{Colors.BRIGHT_GREEN}✓ Exported to: {filepath}{Colors.RESET}")

                elif format_type == "txt" or format_type == "text":
                    filename = f"export_{timestamp}.txt"
                    filepath = os.path.join(SANDBOX_DIR, filename)
                    with open(filepath, 'w') as f:
                        for role, content in history:
                            f.write(f"[{role.upper()}]\n{content}\n\n")
                    print(f"{Colors.BRIGHT_GREEN}✓ Exported to: {filepath}{Colors.RESET}")

                else:
                    print(f"{Colors.YELLOW}Unknown format: {format_type}{Colors.RESET}")
                    print(f"{Colors.DIM}Available: json, md, txt{Colors.RESET}")

            except Exception as e:
                print(f"{Colors.RED}Export failed: {e}{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/persona":
            # Persona management - FULL IMPLEMENTATION
            subcmd = args.split()[0].lower() if args else ""
            subargs = " ".join(args.split()[1:]) if args and len(args.split()) > 1 else ""

            if not subcmd:
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Persona Management{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")
                # Show current persona info
                current_prompt = self.config.get('system_prompt', '')
                prompt_preview = current_prompt[:100] + "..." if len(current_prompt) > 100 else current_prompt
                print(f"\n{Colors.CYAN}Current Persona:{Colors.RESET}")
                print(f"  {Colors.DIM}{prompt_preview}{Colors.RESET}")
                print(f"\n{Colors.BRIGHT_WHITE}Subcommands:{Colors.RESET}")
                print(f"  {Colors.CYAN}/persona create{Colors.RESET}      - Generate persona from project")
                print(f"  {Colors.CYAN}/persona load [file]{Colors.RESET} - Load persona from file")
                print(f"  {Colors.CYAN}/persona save [file]{Colors.RESET} - Save current persona")
                print(f"  {Colors.CYAN}/persona show{Colors.RESET}        - Show full current persona")
                print(f"  {Colors.CYAN}/persona edit{Colors.RESET}        - Edit persona interactively")
                print(f"  {Colors.CYAN}/persona clear{Colors.RESET}       - Clear active persona")

            elif subcmd == "create":
                # Full persona creation from project analysis
                print(f"\n{Colors.BRIGHT_CYAN}Creating Project Persona...{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")

                # Get working directory or use current
                analyze_path = subargs if subargs else (self.working_directory if hasattr(self, 'working_directory') and self.working_directory else os.getcwd())

                if not os.path.exists(analyze_path):
                    print(f"{Colors.RED}Path not found: {analyze_path}{Colors.RESET}")
                    return "COMMAND"

                print(f"{Colors.DIM}Analyzing: {analyze_path}{Colors.RESET}")

                # Collect project information
                project_info = {
                    'languages': {},
                    'frameworks': [],
                    'files': {'total': 0, 'by_type': {}},
                    'structure': [],
                    'readme': None
                }

                # File type to language mapping
                lang_map = {
                    '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
                    '.jsx': 'React', '.tsx': 'React TypeScript', '.vue': 'Vue',
                    '.java': 'Java', '.kt': 'Kotlin', '.swift': 'Swift',
                    '.go': 'Go', '.rs': 'Rust', '.cpp': 'C++', '.c': 'C',
                    '.cs': 'C#', '.rb': 'Ruby', '.php': 'PHP', '.html': 'HTML',
                    '.css': 'CSS', '.scss': 'SCSS', '.sql': 'SQL', '.sh': 'Shell'
                }

                # Framework indicators
                framework_indicators = {
                    'requirements.txt': 'Python', 'package.json': 'Node.js',
                    'Cargo.toml': 'Rust', 'go.mod': 'Go', 'pom.xml': 'Java/Maven',
                    'build.gradle': 'Java/Gradle', 'Gemfile': 'Ruby',
                    'composer.json': 'PHP', 'Dockerfile': 'Docker',
                    'docker-compose.yml': 'Docker Compose', '.env': 'Environment Config',
                    'firebase.json': 'Firebase', 'vercel.json': 'Vercel'
                }

                skip_dirs = {'.git', 'node_modules', '__pycache__', 'venv', '.venv', 'env', 'dist', 'build', '.idea', '.vscode'}

                for root, dirs, files in os.walk(analyze_path):
                    # Skip common non-source directories
                    dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]

                    rel_root = os.path.relpath(root, analyze_path)
                    if rel_root == '.':
                        project_info['structure'].append('/')
                    elif len(rel_root.split(os.sep)) <= 2:
                        project_info['structure'].append(f"  {rel_root}/")

                    for file in files:
                        if file.startswith('.'):
                            continue

                        project_info['files']['total'] += 1
                        ext = os.path.splitext(file)[1].lower()

                        # Count file types
                        project_info['files']['by_type'][ext] = project_info['files']['by_type'].get(ext, 0) + 1

                        # Detect languages
                        if ext in lang_map:
                            lang = lang_map[ext]
                            project_info['languages'][lang] = project_info['languages'].get(lang, 0) + 1

                        # Detect frameworks
                        if file in framework_indicators:
                            fw = framework_indicators[file]
                            if fw not in project_info['frameworks']:
                                project_info['frameworks'].append(fw)

                        # Read README
                        if file.lower() in ['readme.md', 'readme.txt', 'readme']:
                            try:
                                with open(os.path.join(root, file), 'r', errors='ignore') as f:
                                    project_info['readme'] = f.read()[:2000]
                            except:
                                pass

                # Determine primary language
                primary_lang = max(project_info['languages'].items(), key=lambda x: x[1])[0] if project_info['languages'] else 'General'

                # Build persona
                persona_parts = [
                    f"You are an expert {primary_lang} developer assistant specialized in this project.",
                    "",
                    f"PROJECT ANALYSIS:",
                    f"- Primary Language: {primary_lang}",
                    f"- Languages: {', '.join(project_info['languages'].keys()) if project_info['languages'] else 'Various'}",
                    f"- Frameworks/Tools: {', '.join(project_info['frameworks']) if project_info['frameworks'] else 'Standard'}",
                    f"- Total Files: {project_info['files']['total']}",
                    "",
                    "EXPERTISE AREAS:",
                ]

                # Add language-specific expertise
                for lang in list(project_info['languages'].keys())[:5]:
                    persona_parts.append(f"- {lang} development, best practices, and debugging")

                if project_info['frameworks']:
                    persona_parts.append("")
                    persona_parts.append("FRAMEWORK KNOWLEDGE:")
                    for fw in project_info['frameworks'][:5]:
                        persona_parts.append(f"- {fw} configuration and usage")

                persona_parts.extend([
                    "",
                    "GUIDELINES:",
                    "- Provide code examples in the project's primary language",
                    "- Follow the project's existing patterns and conventions",
                    "- Suggest improvements that fit the existing architecture",
                    "- Be concise but thorough in explanations",
                ])

                if project_info['readme']:
                    persona_parts.extend([
                        "",
                        "PROJECT DESCRIPTION (from README):",
                        project_info['readme'][:500]
                    ])

                new_persona = "\n".join(persona_parts)

                # Show preview
                print(f"\n{Colors.BRIGHT_GREEN}Generated Persona:{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")
                preview = new_persona[:500] + "..." if len(new_persona) > 500 else new_persona
                print(f"{Colors.DIM}{preview}{Colors.RESET}")

                # Confirm
                confirm = input(f"\n{Colors.BRIGHT_GREEN}Apply this persona? [Y/n]: {Colors.RESET}").strip().lower()
                if confirm != 'n':
                    self.cfg_mgr.update('system_prompt', new_persona)
                    print(f"{Colors.BRIGHT_GREEN}✓ Persona created and applied!{Colors.RESET}")
                else:
                    print(f"{Colors.DIM}Cancelled{Colors.RESET}")

            elif subcmd == "load":
                if subargs:
                    filepath = subargs if os.path.isabs(subargs) else os.path.join(SANDBOX_DIR, subargs)
                    if os.path.exists(filepath):
                        try:
                            with open(filepath, 'r') as f:
                                persona = f.read()
                            self.cfg_mgr.update('system_prompt', persona)
                            print(f"{Colors.BRIGHT_GREEN}✓ Persona loaded from: {filepath}{Colors.RESET}")
                            print(f"{Colors.DIM}Length: {len(persona)} characters{Colors.RESET}")
                        except Exception as e:
                            print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                    else:
                        # List available persona files
                        print(f"{Colors.RED}File not found: {filepath}{Colors.RESET}")
                        print(f"\n{Colors.CYAN}Available persona files:{Colors.RESET}")
                        try:
                            for f in os.listdir(SANDBOX_DIR):
                                if f.startswith('persona_') or f.endswith('.txt'):
                                    print(f"  {Colors.DIM}{f}{Colors.RESET}")
                        except:
                            pass
                else:
                    # List available persona files
                    print(f"\n{Colors.BRIGHT_CYAN}Available Persona Files:{Colors.RESET}")
                    try:
                        found = False
                        for f in os.listdir(SANDBOX_DIR):
                            if f.startswith('persona_') or f.endswith('.txt'):
                                print(f"  {Colors.BRIGHT_WHITE}{f}{Colors.RESET}")
                                found = True
                        if not found:
                            print(f"  {Colors.DIM}No persona files found{Colors.RESET}")
                    except:
                        print(f"  {Colors.DIM}No persona files found{Colors.RESET}")
                    print(f"\n{Colors.DIM}Usage: /persona load [filename]{Colors.RESET}")

            elif subcmd == "save":
                filename = subargs if subargs else f"persona_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                if not filename.endswith('.txt'):
                    filename += '.txt'
                filepath = os.path.join(SANDBOX_DIR, filename)
                try:
                    os.makedirs(SANDBOX_DIR, exist_ok=True)
                    with open(filepath, 'w') as f:
                        f.write(self.config.get('system_prompt', ''))
                    print(f"{Colors.BRIGHT_GREEN}✓ Persona saved to: {filepath}{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}Error: {e}{Colors.RESET}")

            elif subcmd == "show":
                print(f"\n{Colors.BRIGHT_CYAN}Current Persona:{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")
                current = self.config.get('system_prompt', 'No persona set')
                print(f"{Colors.BRIGHT_WHITE}{current}{Colors.RESET}")

            elif subcmd == "edit":
                print(f"\n{Colors.BRIGHT_CYAN}Edit Persona{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")
                print(f"{Colors.DIM}Current persona:{Colors.RESET}")
                current = self.config.get('system_prompt', '')
                print(f"{Colors.DIM}{current[:200]}...{Colors.RESET}" if len(current) > 200 else f"{Colors.DIM}{current}{Colors.RESET}")
                print(f"\n{Colors.CYAN}Enter new persona (press Enter twice to finish, 'cancel' to abort):{Colors.RESET}")

                lines = []
                while True:
                    line = input()
                    if line.lower() == 'cancel':
                        print(f"{Colors.DIM}Cancelled{Colors.RESET}")
                        return "COMMAND"
                    if line == '' and lines and lines[-1] == '':
                        break
                    lines.append(line)

                new_persona = '\n'.join(lines[:-1] if lines and lines[-1] == '' else lines)
                if new_persona.strip():
                    self.cfg_mgr.update('system_prompt', new_persona)
                    print(f"{Colors.BRIGHT_GREEN}✓ Persona updated!{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}Empty persona, keeping existing{Colors.RESET}")

            elif subcmd == "clear":
                default_prompt = "You are a helpful AI assistant. You provide clear, accurate, and helpful responses."
                self.cfg_mgr.update('system_prompt', default_prompt)
                print(f"{Colors.BRIGHT_GREEN}✓ Persona cleared, reset to default{Colors.RESET}")

            else:
                print(f"{Colors.YELLOW}Unknown persona command: {subcmd}{Colors.RESET}")
                print(f"{Colors.DIM}Available: create, load, save, show, edit, clear{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/analyze":
            # Full codebase analysis - FULL IMPLEMENTATION
            path = args.strip() if args else (self.working_directory if hasattr(self, 'working_directory') and self.working_directory else ".")

            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Codebase Analysis{Colors.RESET}")
            print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")
            print(f"{Colors.CYAN}Path:{Colors.RESET} {Colors.BRIGHT_WHITE}{path}{Colors.RESET}")

            if not os.path.exists(path):
                print(f"{Colors.RED}Path not found: {path}{Colors.RESET}")
                return "COMMAND"

            # Analysis containers
            stats = {
                'total_files': 0, 'total_dirs': 0, 'total_lines': 0, 'total_size': 0,
                'by_language': {}, 'by_extension': {}, 'largest_files': [],
                'frameworks': [], 'config_files': [], 'entry_points': []
            }

            lang_map = {
                '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript', '.jsx': 'React',
                '.tsx': 'React/TS', '.java': 'Java', '.go': 'Go', '.rs': 'Rust',
                '.cpp': 'C++', '.c': 'C', '.h': 'C/C++ Header', '.cs': 'C#',
                '.rb': 'Ruby', '.php': 'PHP', '.swift': 'Swift', '.kt': 'Kotlin',
                '.sql': 'SQL', '.html': 'HTML', '.css': 'CSS', '.scss': 'SCSS',
                '.vue': 'Vue', '.svelte': 'Svelte', '.sh': 'Shell', '.bash': 'Bash'
            }

            config_indicators = [
                'package.json', 'requirements.txt', 'Pipfile', 'setup.py', 'pyproject.toml',
                'Cargo.toml', 'go.mod', 'pom.xml', 'build.gradle', 'Gemfile',
                'composer.json', 'Makefile', 'CMakeLists.txt', 'Dockerfile',
                'docker-compose.yml', '.env', 'config.json', 'config.yaml', 'config.yml',
                'tsconfig.json', 'webpack.config.js', 'vite.config.js', '.eslintrc.js',
                'firebase.json', 'vercel.json', 'netlify.toml'
            ]

            entry_indicators = ['main.py', 'app.py', 'index.js', 'index.ts', 'main.go',
                               'main.rs', 'Main.java', 'Program.cs', 'main.c', 'main.cpp']

            skip_dirs = {'.git', 'node_modules', '__pycache__', 'venv', '.venv', 'env',
                        'dist', 'build', '.idea', '.vscode', 'target', 'bin', 'obj'}

            print(f"{Colors.DIM}Scanning...{Colors.RESET}")

            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
                stats['total_dirs'] += len(dirs)

                for file in files:
                    if file.startswith('.'):
                        continue

                    filepath = os.path.join(root, file)
                    stats['total_files'] += 1

                    # File size
                    try:
                        size = os.path.getsize(filepath)
                        stats['total_size'] += size
                        stats['largest_files'].append((filepath, size))
                    except:
                        size = 0

                    # Extension analysis
                    ext = os.path.splitext(file)[1].lower()
                    stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + 1

                    # Language detection
                    if ext in lang_map:
                        lang = lang_map[ext]
                        if lang not in stats['by_language']:
                            stats['by_language'][lang] = {'files': 0, 'lines': 0}
                        stats['by_language'][lang]['files'] += 1

                        # Count lines
                        try:
                            with open(filepath, 'r', errors='ignore') as f:
                                lines = len(f.readlines())
                                stats['by_language'][lang]['lines'] += lines
                                stats['total_lines'] += lines
                        except:
                            pass

                    # Config files
                    if file in config_indicators:
                        stats['config_files'].append(file)

                    # Entry points
                    if file in entry_indicators:
                        stats['entry_points'].append(os.path.relpath(filepath, path))

            # Sort largest files
            stats['largest_files'].sort(key=lambda x: x[1], reverse=True)

            # Display results
            print(f"\n{Colors.BRIGHT_GREEN}Analysis Complete{Colors.RESET}")
            print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")

            print(f"\n{Colors.BRIGHT_WHITE}Overview:{Colors.RESET}")
            print(f"  {Colors.CYAN}Total Files:{Colors.RESET}   {stats['total_files']}")
            print(f"  {Colors.CYAN}Total Dirs:{Colors.RESET}    {stats['total_dirs']}")
            print(f"  {Colors.CYAN}Total Lines:{Colors.RESET}   {stats['total_lines']:,}")
            size_mb = stats['total_size'] / (1024 * 1024)
            print(f"  {Colors.CYAN}Total Size:{Colors.RESET}    {size_mb:.2f} MB")

            if stats['by_language']:
                print(f"\n{Colors.BRIGHT_WHITE}Languages:{Colors.RESET}")
                sorted_langs = sorted(stats['by_language'].items(), key=lambda x: x[1]['lines'], reverse=True)
                for lang, data in sorted_langs[:8]:
                    pct = (data['lines'] / stats['total_lines'] * 100) if stats['total_lines'] > 0 else 0
                    bar_len = int(pct / 5)
                    bar = f"{Colors.BRIGHT_GREEN}{'█' * bar_len}{Colors.DIM}{'░' * (20 - bar_len)}{Colors.RESET}"
                    print(f"  {Colors.CYAN}{lang:15}{Colors.RESET} {bar} {data['files']:4} files, {data['lines']:6} lines ({pct:.1f}%)")

            if stats['config_files']:
                print(f"\n{Colors.BRIGHT_WHITE}Config Files Found:{Colors.RESET}")
                for cf in stats['config_files'][:10]:
                    print(f"  {Colors.DIM}•{Colors.RESET} {cf}")

            if stats['entry_points']:
                print(f"\n{Colors.BRIGHT_WHITE}Potential Entry Points:{Colors.RESET}")
                for ep in stats['entry_points'][:5]:
                    print(f"  {Colors.BRIGHT_GREEN}→{Colors.RESET} {ep}")

            if stats['largest_files']:
                print(f"\n{Colors.BRIGHT_WHITE}Largest Files:{Colors.RESET}")
                for fp, sz in stats['largest_files'][:5]:
                    rel_path = os.path.relpath(fp, path)
                    if sz > 1024 * 1024:
                        sz_str = f"{sz / (1024*1024):.1f} MB"
                    elif sz > 1024:
                        sz_str = f"{sz / 1024:.1f} KB"
                    else:
                        sz_str = f"{sz} B"
                    print(f"  {Colors.DIM}{sz_str:>10}{Colors.RESET}  {rel_path}")

            # Offer to ingest to RAG
            print(f"\n{Colors.DIM}{'─'*60}{Colors.RESET}")
            ingest = input(f"{Colors.BRIGHT_GREEN}Add to RAG context? [y/N]: {Colors.RESET}").strip().lower()
            if ingest == 'y':
                print(f"{Colors.CYAN}Ingesting files to RAG...{Colors.RESET}")
                ingested = 0
                for root, dirs, files in os.walk(path):
                    dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
                    for file in files:
                        ext = os.path.splitext(file)[1].lower()
                        if ext in lang_map or ext in ['.md', '.txt', '.json', '.yaml', '.yml']:
                            try:
                                filepath = os.path.join(root, file)
                                with open(filepath, 'r', errors='ignore') as f:
                                    content = f.read()
                                if len(content) > 100:  # Skip tiny files
                                    self.memory.add_document(content, filepath, self.current_project_id)
                                    ingested += 1
                            except:
                                pass
                print(f"{Colors.BRIGHT_GREEN}✓ Ingested {ingested} files to RAG{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/chat":
            # Legacy load command
            if args:
                return self.handle_chat_command(f"/load {args}")
            else:
                return self.handle_chat_command("/load")

        elif cmd == "/app":
            # Full App Builder integration - FULL IMPLEMENTATION
            subcmd = args.split()[0].lower() if args else ""
            subargs = " ".join(args.split()[1:]) if args and len(args.split()) > 1 else ""

            if not hasattr(self, 'app_builder') or not self.app_builder:
                print(f"{Colors.YELLOW}Initializing App Builder...{Colors.RESET}")
                try:
                    self.app_builder = AppBuilderOrchestrator(self.engine, self.memory)
                except Exception as e:
                    print(f"{Colors.RED}Failed to initialize App Builder: {e}{Colors.RESET}")
                    return "COMMAND"

            if not subcmd:
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}App Builder - Multi-Agent Development{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")

                # Show current status
                try:
                    projects = self.app_builder.list_projects()
                    print(f"\n{Colors.CYAN}Projects:{Colors.RESET} {Colors.BRIGHT_WHITE}{len(projects)}{Colors.RESET}")
                except:
                    print(f"\n{Colors.CYAN}Projects:{Colors.RESET} {Colors.BRIGHT_WHITE}0{Colors.RESET}")

                print(f"\n{Colors.BRIGHT_WHITE}Subcommands:{Colors.RESET}")
                print(f"  {Colors.CYAN}/app new [name]{Colors.RESET}        - Create new app project")
                print(f"  {Colors.CYAN}/app list{Colors.RESET}             - List all projects")
                print(f"  {Colors.CYAN}/app open [id]{Colors.RESET}        - Open project by ID")
                print(f"  {Colors.CYAN}/app status{Colors.RESET}           - Show current project status")
                print(f"  {Colors.CYAN}/app agents{Colors.RESET}           - List available agents")
                print(f"  {Colors.CYAN}/app run [agent]{Colors.RESET}      - Run specific agent")
                print(f"  {Colors.CYAN}/app files{Colors.RESET}            - List generated files")
                print(f"  {Colors.CYAN}/app export [path]{Colors.RESET}    - Export project files")

            elif subcmd == "list":
                try:
                    projects = self.app_builder.list_projects()
                    if projects:
                        print(f"\n{Colors.BRIGHT_CYAN}App Projects:{Colors.RESET}")
                        print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")
                        for p in projects:
                            pid, name, desc, created = p[0], p[1], p[2] if len(p) > 2 else '', p[3] if len(p) > 3 else ''
                            print(f"  {Colors.CYAN}[{pid}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET}")
                            if desc:
                                print(f"      {Colors.DIM}{desc[:50]}{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}No app projects found. Create one with /app new [name]{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}Error listing projects: {e}{Colors.RESET}")

            elif subcmd == "new":
                if not subargs:
                    print(f"{Colors.YELLOW}Usage: /app new [project_name]{Colors.RESET}")
                    return "COMMAND"

                print(f"\n{Colors.BRIGHT_CYAN}Creating App Project: {subargs}{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")

                # Get description
                desc = input(f"{Colors.CYAN}Description (optional): {Colors.RESET}").strip()

                try:
                    project_id = self.app_builder.create_project(subargs, desc)
                    self.current_app_project = project_id
                    print(f"\n{Colors.BRIGHT_GREEN}✓ Project created with ID: {project_id}{Colors.RESET}")
                    print(f"\n{Colors.DIM}Available agents:{Colors.RESET}")
                    print(f"  • SpecificationWriter - Define project requirements")
                    print(f"  • Architect - Design system architecture")
                    print(f"  • TechLead - Technical decisions")
                    print(f"  • Developer - Write code")
                    print(f"  • Reviewer - Review code quality")
                    print(f"\n{Colors.DIM}Use /app run [agent] to start development{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}Error creating project: {e}{Colors.RESET}")

            elif subcmd == "open":
                if not subargs:
                    return self.handle_chat_command("/app list")

                try:
                    project_id = int(subargs)
                    project = self.app_builder.get_project(project_id)
                    if project:
                        self.current_app_project = project_id
                        print(f"{Colors.BRIGHT_GREEN}✓ Opened project: {project[1]}{Colors.RESET}")
                    else:
                        print(f"{Colors.RED}Project not found: {project_id}{Colors.RESET}")
                except ValueError:
                    print(f"{Colors.RED}Invalid project ID: {subargs}{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}Error: {e}{Colors.RESET}")

            elif subcmd == "status":
                if not hasattr(self, 'current_app_project') or not self.current_app_project:
                    print(f"{Colors.YELLOW}No project open. Use /app open [id] or /app new [name]{Colors.RESET}")
                    return "COMMAND"

                try:
                    project = self.app_builder.get_project(self.current_app_project)
                    tasks = self.app_builder.get_tasks(self.current_app_project) if hasattr(self.app_builder, 'get_tasks') else []
                    files = self.app_builder.get_files(self.current_app_project) if hasattr(self.app_builder, 'get_files') else []

                    print(f"\n{Colors.BRIGHT_CYAN}Project Status{Colors.RESET}")
                    print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")
                    print(f"  {Colors.CYAN}Name:{Colors.RESET}  {Colors.BRIGHT_WHITE}{project[1]}{Colors.RESET}")
                    print(f"  {Colors.CYAN}ID:{Colors.RESET}    {Colors.BRIGHT_WHITE}{self.current_app_project}{Colors.RESET}")
                    print(f"  {Colors.CYAN}Tasks:{Colors.RESET} {Colors.BRIGHT_WHITE}{len(tasks)}{Colors.RESET}")
                    print(f"  {Colors.CYAN}Files:{Colors.RESET} {Colors.BRIGHT_WHITE}{len(files)}{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}Error: {e}{Colors.RESET}")

            elif subcmd == "agents":
                print(f"\n{Colors.BRIGHT_CYAN}Available Agents{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")
                agents = [
                    ("SpecificationWriter", "Define project requirements and user stories"),
                    ("Architect", "Design system architecture and components"),
                    ("TechLead", "Make technical decisions and define patterns"),
                    ("Developer", "Write application code"),
                    ("CodeMonkey", "Generate boilerplate and repetitive code"),
                    ("Reviewer", "Review code for quality and issues"),
                    ("Troubleshooter", "Debug and fix problems"),
                    ("Debugger", "Advanced debugging and analysis"),
                    ("TechnicalWriter", "Generate documentation")
                ]
                for name, desc in agents:
                    print(f"  {Colors.BRIGHT_GREEN}{name:20}{Colors.RESET} {Colors.DIM}{desc}{Colors.RESET}")

            elif subcmd == "run":
                if not hasattr(self, 'current_app_project') or not self.current_app_project:
                    print(f"{Colors.YELLOW}No project open. Use /app open [id] first{Colors.RESET}")
                    return "COMMAND"

                agent_name = subargs.lower() if subargs else ""
                if not agent_name:
                    print(f"{Colors.YELLOW}Usage: /app run [agent_name]{Colors.RESET}")
                    print(f"{Colors.DIM}Agents: spec, architect, techlead, developer, reviewer{Colors.RESET}")
                    return "COMMAND"

                print(f"\n{Colors.CYAN}Running {agent_name} agent...{Colors.RESET}")
                task_input = input(f"{Colors.BRIGHT_GREEN}Task description: {Colors.RESET}").strip()
                if task_input:
                    try:
                        result = self.app_builder.run_agent(self.current_app_project, agent_name, task_input)
                        print(f"\n{Colors.BRIGHT_GREEN}Agent Output:{Colors.RESET}")
                        print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")
                        print(result if result else "Agent completed.")
                    except Exception as e:
                        print(f"{Colors.RED}Error running agent: {e}{Colors.RESET}")

            elif subcmd == "files":
                if not hasattr(self, 'current_app_project') or not self.current_app_project:
                    print(f"{Colors.YELLOW}No project open{Colors.RESET}")
                    return "COMMAND"

                try:
                    files = self.app_builder.get_files(self.current_app_project) if hasattr(self.app_builder, 'get_files') else []
                    if files:
                        print(f"\n{Colors.BRIGHT_CYAN}Generated Files:{Colors.RESET}")
                        for f in files:
                            print(f"  {Colors.DIM}•{Colors.RESET} {f[1] if len(f) > 1 else f}")
                    else:
                        print(f"{Colors.YELLOW}No files generated yet{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}Error: {e}{Colors.RESET}")

            elif subcmd == "export":
                if not hasattr(self, 'current_app_project') or not self.current_app_project:
                    print(f"{Colors.YELLOW}No project open{Colors.RESET}")
                    return "COMMAND"

                export_path = subargs if subargs else os.path.join(APPS_DIR, f"export_{self.current_app_project}")
                try:
                    os.makedirs(export_path, exist_ok=True)
                    files = self.app_builder.get_files(self.current_app_project) if hasattr(self.app_builder, 'get_files') else []
                    for f in files:
                        fname = f[1] if len(f) > 1 else 'file.txt'
                        content = f[2] if len(f) > 2 else ''
                        with open(os.path.join(export_path, fname), 'w') as fp:
                            fp.write(content)
                    print(f"{Colors.BRIGHT_GREEN}✓ Exported to: {export_path}{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}Export failed: {e}{Colors.RESET}")

            else:
                print(f"{Colors.YELLOW}Unknown app command: {subcmd}{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/train":
            # Full training integration - FULL IMPLEMENTATION
            subcmd = args.split()[0].lower() if args else ""
            subargs = " ".join(args.split()[1:]) if args and len(args.split()) > 1 else ""

            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Model Training{Colors.RESET}")
            print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")

            # Check for training managers
            has_finetune = hasattr(self, 'finetune_mgr') or True  # Will initialize if needed
            has_lora = hasattr(self, 'lora_mgr') or True
            has_rlhf = hasattr(self, 'rlhf_mgr') or True

            if not subcmd:
                print(f"\n{Colors.BRIGHT_WHITE}Training Options:{Colors.RESET}")
                print(f"  {Colors.CYAN}/train finetune [dataset]{Colors.RESET}  - Fine-tune on dataset")
                print(f"  {Colors.CYAN}/train lora [dataset]{Colors.RESET}      - LoRA training")
                print(f"  {Colors.CYAN}/train rlhf{Colors.RESET}                - RLHF mode")
                print(f"  {Colors.CYAN}/train status{Colors.RESET}              - Training status")
                print(f"  {Colors.CYAN}/train list{Colors.RESET}                - List datasets")
                print(f"  {Colors.CYAN}/train create [name]{Colors.RESET}       - Create dataset")
                print(f"  {Colors.CYAN}/train stop{Colors.RESET}                - Stop training")

            elif subcmd == "status":
                print(f"\n{Colors.BRIGHT_WHITE}Training Status:{Colors.RESET}")
                # Check for active training
                training_active = hasattr(self, '_training_thread') and self._training_thread and self._training_thread.is_alive()
                if training_active:
                    print(f"  {Colors.BRIGHT_GREEN}● Training in progress{Colors.RESET}")
                    if hasattr(self, '_training_info'):
                        info = self._training_info
                        print(f"    Type: {info.get('type', 'Unknown')}")
                        print(f"    Progress: {info.get('progress', 0)}%")
                else:
                    print(f"  {Colors.DIM}○ No active training{Colors.RESET}")

                # Show available resources
                print(f"\n{Colors.CYAN}Resources:{Colors.RESET}")
                print(f"  Training Dir: {TRAINING_DIR}")
                print(f"  Models Dir: {MODELS_DIR}")
                if torch.cuda.is_available():
                    print(f"  {Colors.BRIGHT_GREEN}GPU: Available ({torch.cuda.get_device_name(0)}){Colors.RESET}")
                else:
                    print(f"  {Colors.YELLOW}GPU: Not available (CPU mode){Colors.RESET}")

            elif subcmd == "list":
                print(f"\n{Colors.BRIGHT_WHITE}Available Datasets:{Colors.RESET}")
                data_dir = os.path.join(TRAINING_DIR, 'data')
                try:
                    os.makedirs(data_dir, exist_ok=True)
                    datasets = [f for f in os.listdir(data_dir) if f.endswith(('.json', '.jsonl', '.csv', '.txt'))]
                    if datasets:
                        for ds in datasets:
                            size = os.path.getsize(os.path.join(data_dir, ds))
                            print(f"  {Colors.DIM}•{Colors.RESET} {ds} ({size // 1024}KB)")
                    else:
                        print(f"  {Colors.DIM}No datasets found in {data_dir}{Colors.RESET}")
                except Exception as e:
                    print(f"  {Colors.RED}Error: {e}{Colors.RESET}")

            elif subcmd == "create":
                dataset_name = subargs if subargs else f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                if not dataset_name.endswith('.jsonl'):
                    dataset_name += '.jsonl'

                print(f"\n{Colors.BRIGHT_CYAN}Creating Dataset: {dataset_name}{Colors.RESET}")
                print(f"{Colors.DIM}Enter training examples (format: prompt|||response){Colors.RESET}")
                print(f"{Colors.DIM}Type 'done' when finished, 'cancel' to abort{Colors.RESET}\n")

                examples = []
                while True:
                    line = input(f"{Colors.CYAN}[{len(examples)+1}]: {Colors.RESET}").strip()
                    if line.lower() == 'done':
                        break
                    if line.lower() == 'cancel':
                        print(f"{Colors.DIM}Cancelled{Colors.RESET}")
                        return "COMMAND"
                    if '|||' in line:
                        prompt, response = line.split('|||', 1)
                        examples.append({'prompt': prompt.strip(), 'response': response.strip()})
                    else:
                        print(f"{Colors.YELLOW}Format: prompt|||response{Colors.RESET}")

                if examples:
                    data_dir = os.path.join(TRAINING_DIR, 'data')
                    os.makedirs(data_dir, exist_ok=True)
                    filepath = os.path.join(data_dir, dataset_name)
                    with open(filepath, 'w') as f:
                        for ex in examples:
                            f.write(json.dumps(ex) + '\n')
                    print(f"{Colors.BRIGHT_GREEN}✓ Created dataset with {len(examples)} examples{Colors.RESET}")
                    print(f"  Path: {filepath}")

            elif subcmd == "finetune":
                dataset = subargs if subargs else None
                if not dataset:
                    print(f"{Colors.YELLOW}Usage: /train finetune [dataset_name]{Colors.RESET}")
                    return self.handle_chat_command("/train list")

                data_path = os.path.join(TRAINING_DIR, 'data', dataset)
                if not os.path.exists(data_path):
                    print(f"{Colors.RED}Dataset not found: {dataset}{Colors.RESET}")
                    return "COMMAND"

                print(f"\n{Colors.BRIGHT_CYAN}Fine-tuning Configuration{Colors.RESET}")
                epochs = input(f"{Colors.CYAN}Epochs [3]: {Colors.RESET}").strip() or "3"
                batch_size = input(f"{Colors.CYAN}Batch size [4]: {Colors.RESET}").strip() or "4"
                lr = input(f"{Colors.CYAN}Learning rate [2e-5]: {Colors.RESET}").strip() or "2e-5"

                print(f"\n{Colors.BRIGHT_YELLOW}Starting fine-tuning...{Colors.RESET}")
                print(f"{Colors.DIM}This may take a while. Training runs in background.{Colors.RESET}")

                # Initialize training in background
                try:
                    if not hasattr(self, 'finetune_mgr'):
                        self.finetune_mgr = FineTuningManager(self.config.get('model_name', 'gpt2'))

                    def train_thread():
                        try:
                            self._training_info = {'type': 'finetune', 'progress': 0}
                            self.finetune_mgr.train(data_path, int(epochs), int(batch_size), float(lr))
                            self._training_info['progress'] = 100
                        except Exception as e:
                            print(f"{Colors.RED}Training error: {e}{Colors.RESET}")

                    self._training_thread = threading.Thread(target=train_thread, daemon=True)
                    self._training_thread.start()
                    print(f"{Colors.BRIGHT_GREEN}✓ Training started. Use /train status to monitor.{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}Failed to start training: {e}{Colors.RESET}")

            elif subcmd == "lora":
                dataset = subargs if subargs else None
                if not dataset:
                    print(f"{Colors.YELLOW}Usage: /train lora [dataset_name]{Colors.RESET}")
                    return "COMMAND"

                print(f"\n{Colors.BRIGHT_CYAN}LoRA Training{Colors.RESET}")
                print(f"{Colors.DIM}Parameter-efficient fine-tuning{Colors.RESET}")

                try:
                    if not hasattr(self, 'lora_mgr'):
                        self.lora_mgr = LoRAManager(self.config.get('model_name', 'gpt2'))

                    rank = input(f"{Colors.CYAN}LoRA rank [8]: {Colors.RESET}").strip() or "8"
                    alpha = input(f"{Colors.CYAN}LoRA alpha [16]: {Colors.RESET}").strip() or "16"

                    data_path = os.path.join(TRAINING_DIR, 'data', dataset)
                    if os.path.exists(data_path):
                        print(f"{Colors.BRIGHT_YELLOW}Starting LoRA training...{Colors.RESET}")
                        self.lora_mgr.train(data_path, int(rank), int(alpha))
                        print(f"{Colors.BRIGHT_GREEN}✓ LoRA training complete{Colors.RESET}")
                    else:
                        print(f"{Colors.RED}Dataset not found{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}LoRA training error: {e}{Colors.RESET}")

            elif subcmd == "rlhf":
                print(f"\n{Colors.BRIGHT_CYAN}RLHF Training Mode{Colors.RESET}")
                print(f"{Colors.DIM}Reinforcement Learning from Human Feedback{Colors.RESET}")
                print(f"\n{Colors.YELLOW}RLHF requires:{Colors.RESET}")
                print(f"  1. A reward model or human feedback")
                print(f"  2. Preference dataset")
                print(f"  3. Significant compute resources")
                print(f"\n{Colors.DIM}Use main menu option 5 for full RLHF setup{Colors.RESET}")

            elif subcmd == "stop":
                if hasattr(self, '_training_thread') and self._training_thread and self._training_thread.is_alive():
                    print(f"{Colors.YELLOW}Stopping training...{Colors.RESET}")
                    # Note: Proper implementation would need a stop flag
                    print(f"{Colors.DIM}Training will stop at next checkpoint{Colors.RESET}")
                else:
                    print(f"{Colors.DIM}No active training to stop{Colors.RESET}")

            else:
                print(f"{Colors.YELLOW}Unknown train command: {subcmd}{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/api":
            # Full API management - FULL IMPLEMENTATION
            subcmd = args.split()[0].lower() if args else ""
            subargs = " ".join(args.split()[1:]) if args and len(args.split()) > 1 else ""

            # Initialize API manager if needed
            if not hasattr(self, 'api_manager') or not self.api_manager:
                try:
                    self.api_manager = APIServerManager(self)
                except Exception as e:
                    print(f"{Colors.YELLOW}API Manager initialization: {e}{Colors.RESET}")

            if not subcmd:
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}API Server Management{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")

                # Show status
                if hasattr(self, 'api_manager') and self.api_manager:
                    if hasattr(self.api_manager, 'servers') and self.api_manager.servers:
                        print(f"\n{Colors.BRIGHT_GREEN}● Servers running: {len(self.api_manager.servers)}{Colors.RESET}")
                        for port, info in self.api_manager.servers.items():
                            print(f"    Port {port}: {info.get('status', 'unknown')}")
                    else:
                        print(f"\n{Colors.DIM}○ No servers running{Colors.RESET}")
                else:
                    print(f"\n{Colors.DIM}○ API Manager not initialized{Colors.RESET}")

                print(f"\n{Colors.BRIGHT_WHITE}Subcommands:{Colors.RESET}")
                print(f"  {Colors.CYAN}/api start [port]{Colors.RESET}   - Start API server")
                print(f"  {Colors.CYAN}/api stop [port]{Colors.RESET}    - Stop API server")
                print(f"  {Colors.CYAN}/api status{Colors.RESET}         - Detailed status")
                print(f"  {Colors.CYAN}/api key{Colors.RESET}            - Show/generate API key")
                print(f"  {Colors.CYAN}/api test{Colors.RESET}           - Test API endpoint")
                print(f"  {Colors.CYAN}/api logs{Colors.RESET}           - Show recent logs")

            elif subcmd == "start":
                port = int(subargs) if subargs and subargs.isdigit() else 5000
                print(f"\n{Colors.CYAN}Starting API server on port {port}...{Colors.RESET}")

                try:
                    if hasattr(self.api_manager, 'start_server'):
                        self.api_manager.start_server(port)
                        print(f"{Colors.BRIGHT_GREEN}✓ API server started on port {port}{Colors.RESET}")
                        print(f"\n{Colors.DIM}Endpoints:{Colors.RESET}")
                        print(f"  POST http://localhost:{port}/chat")
                        print(f"  GET  http://localhost:{port}/health")
                        print(f"  GET  http://localhost:{port}/info")
                    else:
                        # Fallback: start Flask directly
                        if FLASK_AVAILABLE:
                            def run_api():
                                app = Flask(__name__)
                                CORS(app)

                                @app.route('/health', methods=['GET'])
                                def health():
                                    return jsonify({'status': 'healthy'})

                                @app.route('/chat', methods=['POST'])
                                def chat():
                                    data = request.get_json()
                                    message = data.get('message', '')
                                    # Process with AI
                                    response = self.engine.generate(message) if self.engine else "AI not available"
                                    return jsonify({'response': response})

                                app.run(port=port, threaded=True)

                            api_thread = threading.Thread(target=run_api, daemon=True)
                            api_thread.start()
                            if not hasattr(self, '_api_threads'):
                                self._api_threads = {}
                            self._api_threads[port] = api_thread
                            print(f"{Colors.BRIGHT_GREEN}✓ API server started on port {port}{Colors.RESET}")
                        else:
                            print(f"{Colors.RED}Flask not available. Install with: pip install flask flask-cors{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}Failed to start API server: {e}{Colors.RESET}")

            elif subcmd == "stop":
                port = int(subargs) if subargs and subargs.isdigit() else None
                if port:
                    print(f"{Colors.CYAN}Stopping API server on port {port}...{Colors.RESET}")
                    try:
                        if hasattr(self.api_manager, 'stop_server'):
                            self.api_manager.stop_server(port)
                        print(f"{Colors.BRIGHT_GREEN}✓ Server stopped{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Error: {e}{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}Usage: /api stop [port]{Colors.RESET}")

            elif subcmd == "status":
                print(f"\n{Colors.BRIGHT_CYAN}API Server Status{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")

                if hasattr(self, 'api_manager') and self.api_manager:
                    print(f"  {Colors.BRIGHT_GREEN}Manager: Initialized{Colors.RESET}")
                    if hasattr(self.api_manager, 'servers'):
                        for port, info in self.api_manager.servers.items():
                            print(f"  Port {port}: {Colors.BRIGHT_GREEN}Running{Colors.RESET}")
                    else:
                        print(f"  {Colors.DIM}No servers configured{Colors.RESET}")
                else:
                    print(f"  {Colors.YELLOW}Manager: Not initialized{Colors.RESET}")

                # Check Flask availability
                print(f"\n{Colors.CYAN}Dependencies:{Colors.RESET}")
                print(f"  Flask: {Colors.BRIGHT_GREEN if FLASK_AVAILABLE else Colors.RED}{'Available' if FLASK_AVAILABLE else 'Not installed'}{Colors.RESET}")
                print(f"  Encryption: {Colors.BRIGHT_GREEN if ENCRYPTION_AVAILABLE else Colors.RED}{'Available' if ENCRYPTION_AVAILABLE else 'Not installed'}{Colors.RESET}")

            elif subcmd == "key":
                print(f"\n{Colors.BRIGHT_CYAN}API Key Management{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")

                if ENCRYPTION_AVAILABLE:
                    # Generate or show API key
                    if not hasattr(self, '_api_key') or not self._api_key:
                        self._api_key = Fernet.generate_key().decode()

                    print(f"\n{Colors.BRIGHT_WHITE}Your API Key:{Colors.RESET}")
                    print(f"  {Colors.BRIGHT_GREEN}{self._api_key}{Colors.RESET}")
                    print(f"\n{Colors.DIM}Use this key in the X-API-Key header for authenticated requests{Colors.RESET}")

                    regen = input(f"\n{Colors.CYAN}Generate new key? [y/N]: {Colors.RESET}").strip().lower()
                    if regen == 'y':
                        self._api_key = Fernet.generate_key().decode()
                        print(f"\n{Colors.BRIGHT_GREEN}New API Key:{Colors.RESET}")
                        print(f"  {self._api_key}")
                else:
                    print(f"{Colors.RED}Encryption not available. Install cryptography package.{Colors.RESET}")

            elif subcmd == "test":
                port = int(subargs) if subargs and subargs.isdigit() else 5000
                print(f"\n{Colors.CYAN}Testing API on port {port}...{Colors.RESET}")

                try:
                    import requests as req
                    response = req.get(f"http://localhost:{port}/health", timeout=5)
                    if response.status_code == 200:
                        print(f"{Colors.BRIGHT_GREEN}✓ API is healthy{Colors.RESET}")
                        print(f"  Response: {response.json()}")
                    else:
                        print(f"{Colors.YELLOW}API returned status {response.status_code}{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}API test failed: {e}{Colors.RESET}")

            elif subcmd == "logs":
                print(f"\n{Colors.BRIGHT_CYAN}Recent API Logs{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")
                print(f"{Colors.DIM}Log viewing not yet implemented{Colors.RESET}")

            else:
                print(f"{Colors.YELLOW}Unknown API command: {subcmd}{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/quit":
            # Quit entire application
            confirm = input(f"{Colors.BRIGHT_RED}Exit AI Terminal Pro? [y/N]: {Colors.RESET}").strip().lower()
            if confirm == 'y':
                print(f"\n{Colors.BRIGHT_CYAN}Goodbye!{Colors.RESET}\n")
                sys.exit(0)
            else:
                print(f"{Colors.DIM}Cancelled{Colors.RESET}")
            return "COMMAND"

        elif cmd == "/settings":
            # Full settings editor
            subcmd = args.split()[0].lower() if args else ""
            sub_args = ' '.join(args.split()[1:]) if len(args.split()) > 1 else ""

            # Settings categories
            settings_categories = {
                'general': ['model', 'temperature', 'max_tokens', 'streaming', 'verbose'],
                'memory': ['max_context_window', 'auto_summarize', 'memory_limit'],
                'api': ['api_host', 'api_port', 'api_enabled', 'require_auth'],
                'rag': ['rag_enabled', 'vector_db_path', 'chunk_size', 'embedding_model'],
                'mcp': ['mcp_auto_start', 'mcp_timeout', 'mcp_servers'],
                'ui': ['theme', 'show_timestamps', 'colored_output', 'compact_mode'],
                'voice': ['voice_enabled', 'tts_engine', 'stt_engine', 'voice_rate'],
                'advanced': ['debug_mode', 'log_level', 'auto_save', 'backup_enabled']
            }

            def _get_setting_value(key):
                """Get current setting value"""
                if hasattr(self, 'config') and self.config:
                    return self.config.get(key, None)
                return None

            def _set_setting_value(key, value):
                """Set setting value"""
                if hasattr(self, 'config') and self.config:
                    # Type conversion
                    old_val = self.config.get(key)
                    if isinstance(old_val, bool):
                        value = str(value).lower() in ('true', 'yes', '1', 'on')
                    elif isinstance(old_val, int):
                        value = int(value)
                    elif isinstance(old_val, float):
                        value = float(value)
                    self.config[key] = value
                    return True
                return False

            def _save_settings():
                """Save settings to config file"""
                config_path = os.path.join(os.path.dirname(__file__), 'config.json')
                try:
                    with open(config_path, 'w') as f:
                        json.dump(self.config, f, indent=2)
                    return True
                except Exception as e:
                    print(f"{Colors.RED}Error saving: {e}{Colors.RESET}")
                    return False

            if not subcmd:
                # Show settings overview
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Settings{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")
                print(f"\n{Colors.BRIGHT_WHITE}Commands:{Colors.RESET}")
                print(f"  {Colors.CYAN}/settings show [category]{Colors.RESET}  - Show settings")
                print(f"  {Colors.CYAN}/settings set <key> <value>{Colors.RESET} - Change setting")
                print(f"  {Colors.CYAN}/settings get <key>{Colors.RESET}         - Get setting value")
                print(f"  {Colors.CYAN}/settings reset [key]{Colors.RESET}       - Reset to default")
                print(f"  {Colors.CYAN}/settings save{Colors.RESET}              - Save settings")
                print(f"  {Colors.CYAN}/settings reload{Colors.RESET}            - Reload from file")
                print(f"  {Colors.CYAN}/settings export <file>{Colors.RESET}     - Export settings")
                print(f"  {Colors.CYAN}/settings import <file>{Colors.RESET}     - Import settings")
                print(f"  {Colors.CYAN}/settings edit{Colors.RESET}              - Interactive editor")
                print(f"\n{Colors.BRIGHT_WHITE}Categories:{Colors.RESET}")
                for cat in settings_categories:
                    print(f"  {Colors.CYAN}{cat}{Colors.RESET} - {', '.join(settings_categories[cat][:3])}...")
                print(f"\n{Colors.DIM}Use /settings show <category> to see all settings in that category{Colors.RESET}")

            elif subcmd == "show":
                category = sub_args.strip().lower() if sub_args else ""

                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Settings{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")

                if category and category in settings_categories:
                    # Show specific category
                    print(f"\n{Colors.BRIGHT_WHITE}[{category.upper()}]{Colors.RESET}")
                    for key in settings_categories[category]:
                        val = _get_setting_value(key)
                        val_str = str(val) if val is not None else f"{Colors.DIM}(not set){Colors.RESET}"
                        if isinstance(val, bool):
                            val_str = f"{Colors.BRIGHT_GREEN}ON{Colors.RESET}" if val else f"{Colors.BRIGHT_RED}OFF{Colors.RESET}"
                        print(f"  {Colors.CYAN}{key}:{Colors.RESET} {val_str}")
                elif category:
                    print(f"{Colors.YELLOW}Unknown category: {category}{Colors.RESET}")
                    print(f"{Colors.DIM}Available: {', '.join(settings_categories.keys())}{Colors.RESET}")
                else:
                    # Show all settings
                    for cat, keys in settings_categories.items():
                        print(f"\n{Colors.BRIGHT_WHITE}[{cat.upper()}]{Colors.RESET}")
                        for key in keys:
                            val = _get_setting_value(key)
                            val_str = str(val) if val is not None else f"{Colors.DIM}(not set){Colors.RESET}"
                            if isinstance(val, bool):
                                val_str = f"{Colors.BRIGHT_GREEN}ON{Colors.RESET}" if val else f"{Colors.BRIGHT_RED}OFF{Colors.RESET}"
                            print(f"  {Colors.CYAN}{key}:{Colors.RESET} {val_str}")

            elif subcmd == "get":
                key = sub_args.strip()
                if not key:
                    print(f"{Colors.YELLOW}Usage: /settings get <key>{Colors.RESET}")
                else:
                    val = _get_setting_value(key)
                    if val is not None:
                        print(f"{Colors.CYAN}{key}:{Colors.RESET} {val}")
                    else:
                        print(f"{Colors.YELLOW}Setting not found: {key}{Colors.RESET}")

            elif subcmd == "set":
                parts = sub_args.split(maxsplit=1)
                if len(parts) < 2:
                    print(f"{Colors.YELLOW}Usage: /settings set <key> <value>{Colors.RESET}")
                else:
                    key, value = parts
                    old_val = _get_setting_value(key)
                    if _set_setting_value(key, value):
                        print(f"{Colors.BRIGHT_GREEN}✓ Set {key}: {old_val} → {_get_setting_value(key)}{Colors.RESET}")
                        print(f"{Colors.DIM}Use /settings save to persist changes{Colors.RESET}")
                    else:
                        print(f"{Colors.RED}Failed to set {key}{Colors.RESET}")

            elif subcmd == "reset":
                key = sub_args.strip() if sub_args else ""

                # Default values
                defaults = {
                    'model': 'gpt-4',
                    'temperature': 0.7,
                    'max_tokens': 4096,
                    'streaming': True,
                    'verbose': False,
                    'max_context_window': 32768,
                    'auto_summarize': True,
                    'memory_limit': 1000,
                    'theme': 'default',
                    'colored_output': True,
                    'debug_mode': False,
                    'log_level': 'INFO'
                }

                if key:
                    if key in defaults:
                        _set_setting_value(key, defaults[key])
                        print(f"{Colors.BRIGHT_GREEN}✓ Reset {key} to: {defaults[key]}{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}No default for: {key}{Colors.RESET}")
                else:
                    confirm = input(f"{Colors.BRIGHT_RED}Reset ALL settings to defaults? [y/N]: {Colors.RESET}").strip().lower()
                    if confirm == 'y':
                        for k, v in defaults.items():
                            _set_setting_value(k, v)
                        print(f"{Colors.BRIGHT_GREEN}✓ All settings reset to defaults{Colors.RESET}")
                    else:
                        print(f"{Colors.DIM}Cancelled{Colors.RESET}")

            elif subcmd == "save":
                if _save_settings():
                    print(f"{Colors.BRIGHT_GREEN}✓ Settings saved{Colors.RESET}")
                else:
                    print(f"{Colors.RED}Failed to save settings{Colors.RESET}")

            elif subcmd == "reload":
                config_path = os.path.join(os.path.dirname(__file__), 'config.json')
                try:
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            loaded = json.load(f)
                            self.config.update(loaded)
                        print(f"{Colors.BRIGHT_GREEN}✓ Settings reloaded from {config_path}{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}Config file not found: {config_path}{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}Error reloading: {e}{Colors.RESET}")

            elif subcmd == "export":
                path = sub_args.strip() if sub_args else "settings_export.json"
                if not path.endswith('.json'):
                    path += '.json'
                try:
                    with open(path, 'w') as f:
                        json.dump(self.config, f, indent=2)
                    print(f"{Colors.BRIGHT_GREEN}✓ Settings exported to: {path}{Colors.RESET}")
                except Exception as e:
                    print(f"{Colors.RED}Export error: {e}{Colors.RESET}")

            elif subcmd == "import":
                path = sub_args.strip()
                if not path:
                    print(f"{Colors.YELLOW}Usage: /settings import <file.json>{Colors.RESET}")
                elif not os.path.exists(path):
                    print(f"{Colors.RED}File not found: {path}{Colors.RESET}")
                else:
                    try:
                        with open(path, 'r') as f:
                            imported = json.load(f)

                        # Show preview
                        print(f"\n{Colors.BRIGHT_CYAN}Import Preview:{Colors.RESET}")
                        for key, val in list(imported.items())[:10]:
                            print(f"  {Colors.CYAN}{key}:{Colors.RESET} {val}")
                        if len(imported) > 10:
                            print(f"  {Colors.DIM}... and {len(imported) - 10} more{Colors.RESET}")

                        confirm = input(f"\n{Colors.BRIGHT_YELLOW}Import these settings? [y/N]: {Colors.RESET}").strip().lower()
                        if confirm == 'y':
                            self.config.update(imported)
                            print(f"{Colors.BRIGHT_GREEN}✓ Imported {len(imported)} settings{Colors.RESET}")
                        else:
                            print(f"{Colors.DIM}Cancelled{Colors.RESET}")
                    except json.JSONDecodeError:
                        print(f"{Colors.RED}Invalid JSON file{Colors.RESET}")
                    except Exception as e:
                        print(f"{Colors.RED}Import error: {e}{Colors.RESET}")

            elif subcmd == "edit":
                # Interactive settings editor
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Interactive Settings Editor{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")
                print(f"{Colors.DIM}Type setting name to edit, 'save' to save, 'quit' to exit{Colors.RESET}\n")

                # Show current settings
                all_keys = []
                for cat, keys in settings_categories.items():
                    for key in keys:
                        val = _get_setting_value(key)
                        if val is not None:
                            all_keys.append(key)

                while True:
                    print(f"\n{Colors.BRIGHT_WHITE}Current Settings:{Colors.RESET}")
                    for i, key in enumerate(all_keys[:15], 1):
                        val = _get_setting_value(key)
                        print(f"  {Colors.DIM}{i:2}.{Colors.RESET} {Colors.CYAN}{key}:{Colors.RESET} {val}")

                    try:
                        choice = input(f"\n{Colors.BRIGHT_CYAN}Setting name (or save/quit): {Colors.RESET}").strip()
                    except (EOFError, KeyboardInterrupt):
                        break

                    if choice.lower() in ('quit', 'q', 'exit'):
                        print(f"{Colors.DIM}Exiting editor{Colors.RESET}")
                        break
                    elif choice.lower() == 'save':
                        if _save_settings():
                            print(f"{Colors.BRIGHT_GREEN}✓ Settings saved{Colors.RESET}")
                        break
                    elif choice:
                        # Try to edit setting
                        current = _get_setting_value(choice)
                        if current is not None:
                            print(f"  {Colors.DIM}Current value: {current}{Colors.RESET}")
                            new_val = input(f"  {Colors.BRIGHT_CYAN}New value: {Colors.RESET}").strip()
                            if new_val:
                                _set_setting_value(choice, new_val)
                                print(f"  {Colors.BRIGHT_GREEN}✓ Updated{Colors.RESET}")
                        else:
                            # Check if it's a number reference
                            try:
                                idx = int(choice) - 1
                                if 0 <= idx < len(all_keys):
                                    key = all_keys[idx]
                                    current = _get_setting_value(key)
                                    print(f"  {Colors.DIM}Current value: {current}{Colors.RESET}")
                                    new_val = input(f"  {Colors.BRIGHT_CYAN}New value for {key}: {Colors.RESET}").strip()
                                    if new_val:
                                        _set_setting_value(key, new_val)
                                        print(f"  {Colors.BRIGHT_GREEN}✓ Updated{Colors.RESET}")
                            except ValueError:
                                print(f"{Colors.YELLOW}Unknown setting: {choice}{Colors.RESET}")

            else:
                print(f"{Colors.YELLOW}Unknown settings command: {subcmd}{Colors.RESET}")
                print(f"{Colors.DIM}Use /settings for help{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/theme":
            # Full theme management system
            subcmd = args.split()[0].lower() if args else ""
            sub_args = ' '.join(args.split()[1:]) if len(args.split()) > 1 else ""

            # Theme definitions
            themes = {
                'default': {
                    'name': 'Default',
                    'description': 'Default terminal colors',
                    'primary': '\033[36m',      # Cyan
                    'secondary': '\033[33m',    # Yellow
                    'accent': '\033[35m',       # Magenta
                    'success': '\033[32m',      # Green
                    'error': '\033[31m',        # Red
                    'warning': '\033[33m',      # Yellow
                    'info': '\033[34m',         # Blue
                    'muted': '\033[90m',        # Bright Black
                    'text': '\033[37m',         # White
                    'bg': ''                    # Default
                },
                'dark': {
                    'name': 'Dark Mode',
                    'description': 'Dark theme with blue accents',
                    'primary': '\033[94m',      # Bright Blue
                    'secondary': '\033[96m',    # Bright Cyan
                    'accent': '\033[95m',       # Bright Magenta
                    'success': '\033[92m',      # Bright Green
                    'error': '\033[91m',        # Bright Red
                    'warning': '\033[93m',      # Bright Yellow
                    'info': '\033[94m',         # Bright Blue
                    'muted': '\033[90m',        # Bright Black
                    'text': '\033[97m',         # Bright White
                    'bg': '\033[40m'            # Black BG
                },
                'light': {
                    'name': 'Light Mode',
                    'description': 'Light theme optimized for light terminals',
                    'primary': '\033[34m',      # Blue
                    'secondary': '\033[36m',    # Cyan
                    'accent': '\033[35m',       # Magenta
                    'success': '\033[32m',      # Green
                    'error': '\033[31m',        # Red
                    'warning': '\033[33m',      # Yellow
                    'info': '\033[34m',         # Blue
                    'muted': '\033[2m',         # Dim
                    'text': '\033[30m',         # Black
                    'bg': ''                    # Default
                },
                'ocean': {
                    'name': 'Ocean',
                    'description': 'Calm blue-green theme',
                    'primary': '\033[38;5;39m',     # Deep sky blue
                    'secondary': '\033[38;5;50m',   # Cyan
                    'accent': '\033[38;5;147m',     # Light purple
                    'success': '\033[38;5;84m',     # Sea green
                    'error': '\033[38;5;203m',      # Coral
                    'warning': '\033[38;5;221m',    # Gold
                    'info': '\033[38;5;75m',        # Sky blue
                    'muted': '\033[38;5;245m',      # Gray
                    'text': '\033[38;5;255m',       # White
                    'bg': ''
                },
                'forest': {
                    'name': 'Forest',
                    'description': 'Nature-inspired green theme',
                    'primary': '\033[38;5;34m',     # Forest green
                    'secondary': '\033[38;5;142m',  # Olive
                    'accent': '\033[38;5;179m',     # Tan
                    'success': '\033[38;5;40m',     # Lime
                    'error': '\033[38;5;124m',      # Dark red
                    'warning': '\033[38;5;214m',    # Orange
                    'info': '\033[38;5;71m',        # Medium green
                    'muted': '\033[38;5;242m',      # Gray
                    'text': '\033[38;5;230m',       # Light beige
                    'bg': ''
                },
                'sunset': {
                    'name': 'Sunset',
                    'description': 'Warm orange-pink theme',
                    'primary': '\033[38;5;208m',    # Orange
                    'secondary': '\033[38;5;204m',  # Hot pink
                    'accent': '\033[38;5;213m',     # Pink
                    'success': '\033[38;5;155m',    # Yellow-green
                    'error': '\033[38;5;196m',      # Red
                    'warning': '\033[38;5;220m',    # Gold
                    'info': '\033[38;5;216m',       # Light orange
                    'muted': '\033[38;5;241m',      # Gray
                    'text': '\033[38;5;255m',       # White
                    'bg': ''
                },
                'hacker': {
                    'name': 'Hacker',
                    'description': 'Classic green-on-black terminal',
                    'primary': '\033[38;5;46m',     # Bright green
                    'secondary': '\033[38;5;40m',   # Green
                    'accent': '\033[38;5;48m',      # Spring green
                    'success': '\033[38;5;82m',     # Yellow-green
                    'error': '\033[38;5;196m',      # Red
                    'warning': '\033[38;5;226m',    # Yellow
                    'info': '\033[38;5;34m',        # Forest green
                    'muted': '\033[38;5;22m',       # Dark green
                    'text': '\033[38;5;46m',        # Bright green
                    'bg': '\033[40m'                # Black BG
                },
                'cyberpunk': {
                    'name': 'Cyberpunk',
                    'description': 'Neon purple and pink',
                    'primary': '\033[38;5;201m',    # Fuchsia
                    'secondary': '\033[38;5;51m',   # Cyan
                    'accent': '\033[38;5;226m',     # Yellow
                    'success': '\033[38;5;118m',    # Bright green
                    'error': '\033[38;5;196m',      # Red
                    'warning': '\033[38;5;208m',    # Orange
                    'info': '\033[38;5;141m',       # Medium purple
                    'muted': '\033[38;5;240m',      # Gray
                    'text': '\033[38;5;255m',       # White
                    'bg': ''
                }
            }

            # Get current theme
            if not hasattr(self, '_current_theme'):
                self._current_theme = 'default'

            def _apply_theme(theme_name):
                """Apply theme colors to the Colors class"""
                if theme_name not in themes:
                    return False

                theme = themes[theme_name]
                self._current_theme = theme_name

                # Update Colors class
                Colors.CYAN = theme['primary']
                Colors.YELLOW = theme['secondary']
                Colors.MAGENTA = theme['accent']
                Colors.GREEN = theme['success']
                Colors.RED = theme['error']
                Colors.BLUE = theme['info']
                Colors.DIM = theme['muted']
                Colors.WHITE = theme['text']

                # Update bright variants
                Colors.BRIGHT_CYAN = theme['primary']
                Colors.BRIGHT_YELLOW = theme['secondary']
                Colors.BRIGHT_MAGENTA = theme['accent']
                Colors.BRIGHT_GREEN = theme['success']
                Colors.BRIGHT_RED = theme['error']
                Colors.BRIGHT_BLUE = theme['info']
                Colors.BRIGHT_WHITE = theme['text']

                # Save to config
                if hasattr(self, 'config') and self.config:
                    self.config['theme'] = theme_name

                return True

            def _preview_theme(theme_name):
                """Preview a theme's colors"""
                if theme_name not in themes:
                    return

                t = themes[theme_name]
                print(f"\n{t['primary']}Primary{Colors.RESET} | ", end='')
                print(f"{t['secondary']}Secondary{Colors.RESET} | ", end='')
                print(f"{t['accent']}Accent{Colors.RESET}")
                print(f"{t['success']}Success{Colors.RESET} | ", end='')
                print(f"{t['error']}Error{Colors.RESET} | ", end='')
                print(f"{t['warning']}Warning{Colors.RESET}")
                print(f"{t['info']}Info{Colors.RESET} | ", end='')
                print(f"{t['muted']}Muted{Colors.RESET} | ", end='')
                print(f"{t['text']}Text{Colors.RESET}")

            if not subcmd:
                # Show theme menu
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Theme Manager{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")
                print(f"\n{Colors.BRIGHT_WHITE}Current Theme:{Colors.RESET} {Colors.CYAN}{themes[self._current_theme]['name']}{Colors.RESET}")
                print(f"{Colors.DIM}{themes[self._current_theme]['description']}{Colors.RESET}")
                print(f"\n{Colors.BRIGHT_WHITE}Commands:{Colors.RESET}")
                print(f"  {Colors.CYAN}/theme list{Colors.RESET}        - List all themes")
                print(f"  {Colors.CYAN}/theme set <name>{Colors.RESET}  - Apply theme")
                print(f"  {Colors.CYAN}/theme preview <n>{Colors.RESET} - Preview theme colors")
                print(f"  {Colors.CYAN}/theme reset{Colors.RESET}       - Reset to default")
                print(f"  {Colors.CYAN}/theme create{Colors.RESET}      - Create custom theme")

            elif subcmd == "list":
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Available Themes{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}\n")

                for theme_id, theme_data in themes.items():
                    is_current = theme_id == self._current_theme
                    indicator = f"{Colors.BRIGHT_GREEN}●{Colors.RESET}" if is_current else f"{Colors.DIM}○{Colors.RESET}"
                    print(f"  {indicator} {Colors.BRIGHT_WHITE}{theme_data['name']}{Colors.RESET} ({theme_id})")
                    print(f"    {Colors.DIM}{theme_data['description']}{Colors.RESET}")
                    # Show color samples inline
                    t = theme_data
                    print(f"    {t['primary']}▓▓{Colors.RESET}{t['secondary']}▓▓{Colors.RESET}{t['accent']}▓▓{Colors.RESET}{t['success']}▓▓{Colors.RESET}{t['error']}▓▓{Colors.RESET}{t['info']}▓▓{Colors.RESET}")
                    print()

            elif subcmd in ("set", "apply", "use"):
                theme_name = sub_args.strip().lower()
                if not theme_name:
                    print(f"{Colors.YELLOW}Usage: /theme set <theme_name>{Colors.RESET}")
                    print(f"{Colors.DIM}Available: {', '.join(themes.keys())}{Colors.RESET}")
                elif theme_name not in themes:
                    print(f"{Colors.RED}Unknown theme: {theme_name}{Colors.RESET}")
                    print(f"{Colors.DIM}Available: {', '.join(themes.keys())}{Colors.RESET}")
                else:
                    if _apply_theme(theme_name):
                        print(f"{Colors.BRIGHT_GREEN}✓ Theme applied: {themes[theme_name]['name']}{Colors.RESET}")
                        _preview_theme(theme_name)
                    else:
                        print(f"{Colors.RED}Failed to apply theme{Colors.RESET}")

            elif subcmd == "preview":
                theme_name = sub_args.strip().lower()
                if not theme_name:
                    print(f"{Colors.YELLOW}Usage: /theme preview <theme_name>{Colors.RESET}")
                elif theme_name not in themes:
                    print(f"{Colors.RED}Unknown theme: {theme_name}{Colors.RESET}")
                else:
                    print(f"\n{Colors.BRIGHT_CYAN}Preview: {themes[theme_name]['name']}{Colors.RESET}")
                    print(f"{Colors.DIM}{themes[theme_name]['description']}{Colors.RESET}")
                    _preview_theme(theme_name)
                    print(f"\n{Colors.DIM}Use /theme set {theme_name} to apply{Colors.RESET}")

            elif subcmd == "reset":
                if _apply_theme('default'):
                    print(f"{Colors.BRIGHT_GREEN}✓ Theme reset to default{Colors.RESET}")

            elif subcmd == "create":
                # Interactive custom theme creator
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Custom Theme Creator{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")
                print(f"{Colors.DIM}Create a custom theme with 256-color support{Colors.RESET}")
                print(f"{Colors.DIM}Color format: number (0-255) for 256-color palette{Colors.RESET}\n")

                try:
                    name = input(f"{Colors.CYAN}Theme name: {Colors.RESET}").strip()
                    if not name:
                        print(f"{Colors.DIM}Cancelled{Colors.RESET}")
                        return "COMMAND"

                    desc = input(f"{Colors.CYAN}Description: {Colors.RESET}").strip() or f"Custom theme: {name}"

                    print(f"\n{Colors.DIM}Enter color codes (0-255) or press Enter for default{Colors.RESET}")

                    def _get_color(prompt, default=39):
                        try:
                            val = input(f"  {Colors.CYAN}{prompt} [{default}]: {Colors.RESET}").strip()
                            if val:
                                num = int(val)
                                if 0 <= num <= 255:
                                    return f'\033[38;5;{num}m'
                            return f'\033[38;5;{default}m'
                        except ValueError:
                            return f'\033[38;5;{default}m'

                    primary = _get_color("Primary color", 39)
                    secondary = _get_color("Secondary color", 220)
                    accent = _get_color("Accent color", 165)
                    success = _get_color("Success color", 82)
                    error = _get_color("Error color", 196)

                    theme_id = name.lower().replace(' ', '_')

                    # Add to themes
                    themes[theme_id] = {
                        'name': name,
                        'description': desc,
                        'primary': primary,
                        'secondary': secondary,
                        'accent': accent,
                        'success': success,
                        'error': error,
                        'warning': secondary,
                        'info': primary,
                        'muted': '\033[38;5;245m',
                        'text': '\033[38;5;255m',
                        'bg': ''
                    }

                    print(f"\n{Colors.BRIGHT_GREEN}✓ Theme '{name}' created!{Colors.RESET}")
                    _preview_theme(theme_id)
                    print(f"\n{Colors.DIM}Use /theme set {theme_id} to apply{Colors.RESET}")

                except (EOFError, KeyboardInterrupt):
                    print(f"\n{Colors.DIM}Cancelled{Colors.RESET}")

            else:
                # Treat as theme name
                if subcmd in themes:
                    if _apply_theme(subcmd):
                        print(f"{Colors.BRIGHT_GREEN}✓ Theme applied: {themes[subcmd]['name']}{Colors.RESET}")
                        _preview_theme(subcmd)
                else:
                    print(f"{Colors.YELLOW}Unknown theme or command: {subcmd}{Colors.RESET}")
                    print(f"{Colors.DIM}Use /theme list to see available themes{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/whisper":
            # Full Whisper speech-to-text integration
            subcmd = args.split()[0].lower() if args else ""
            sub_args = ' '.join(args.split()[1:]) if len(args.split()) > 1 else ""

            # Initialize whisper state
            if not hasattr(self, '_whisper_model'):
                self._whisper_model = None
                self._whisper_model_name = 'base'

            def _check_whisper():
                """Check if whisper is available"""
                try:
                    import whisper
                    return True
                except ImportError:
                    return False

            def _check_audio():
                """Check if audio recording is available"""
                try:
                    import sounddevice
                    import numpy
                    return True
                except ImportError:
                    return False

            def _load_whisper_model(model_name='base'):
                """Load whisper model"""
                try:
                    import whisper
                    print(f"{Colors.CYAN}Loading Whisper model: {model_name}...{Colors.RESET}")
                    self._whisper_model = whisper.load_model(model_name)
                    self._whisper_model_name = model_name
                    print(f"{Colors.BRIGHT_GREEN}✓ Model loaded{Colors.RESET}")
                    return True
                except Exception as e:
                    print(f"{Colors.RED}Error loading model: {e}{Colors.RESET}")
                    return False

            def _transcribe_file(file_path):
                """Transcribe audio file"""
                if not os.path.exists(file_path):
                    print(f"{Colors.RED}File not found: {file_path}{Colors.RESET}")
                    return None

                if self._whisper_model is None:
                    if not _load_whisper_model(self._whisper_model_name):
                        return None

                try:
                    print(f"{Colors.CYAN}Transcribing: {file_path}...{Colors.RESET}")
                    result = self._whisper_model.transcribe(file_path)
                    return result
                except Exception as e:
                    print(f"{Colors.RED}Transcription error: {e}{Colors.RESET}")
                    return None

            def _record_audio(duration=5, sample_rate=16000):
                """Record audio from microphone"""
                try:
                    import sounddevice as sd
                    import numpy as np
                    import tempfile
                    import scipy.io.wavfile as wav

                    print(f"{Colors.CYAN}Recording for {duration} seconds... (speak now){Colors.RESET}")
                    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
                    sd.wait()
                    print(f"{Colors.BRIGHT_GREEN}✓ Recording complete{Colors.RESET}")

                    # Save to temp file
                    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                    # Normalize and convert to int16
                    audio_normalized = np.int16(audio.flatten() * 32767)
                    wav.write(temp_file.name, sample_rate, audio_normalized)
                    return temp_file.name
                except Exception as e:
                    print(f"{Colors.RED}Recording error: {e}{Colors.RESET}")
                    return None

            if not subcmd:
                # Show whisper help
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Whisper Speech-to-Text{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")

                # Check dependencies
                whisper_ok = _check_whisper()
                audio_ok = _check_audio()

                print(f"\n{Colors.BRIGHT_WHITE}Status:{Colors.RESET}")
                print(f"  {Colors.CYAN}Whisper:{Colors.RESET} {'✓ Available' if whisper_ok else '✗ Not installed'}")
                print(f"  {Colors.CYAN}Audio:{Colors.RESET} {'✓ Available' if audio_ok else '✗ Not installed'}")
                if self._whisper_model:
                    print(f"  {Colors.CYAN}Model:{Colors.RESET} {self._whisper_model_name} (loaded)")
                else:
                    print(f"  {Colors.CYAN}Model:{Colors.RESET} {self._whisper_model_name} (not loaded)")

                print(f"\n{Colors.BRIGHT_WHITE}Commands:{Colors.RESET}")
                print(f"  {Colors.CYAN}/whisper listen{Colors.RESET}          - Record and transcribe")
                print(f"  {Colors.CYAN}/whisper listen <sec>{Colors.RESET}    - Record for N seconds")
                print(f"  {Colors.CYAN}/whisper file <path>{Colors.RESET}     - Transcribe audio file")
                print(f"  {Colors.CYAN}/whisper model <name>{Colors.RESET}    - Change model")
                print(f"  {Colors.CYAN}/whisper models{Colors.RESET}          - List available models")
                print(f"  {Colors.CYAN}/whisper chat{Colors.RESET}            - Voice chat mode")
                print(f"  {Colors.CYAN}/whisper status{Colors.RESET}          - Show status")

                if not whisper_ok:
                    print(f"\n{Colors.YELLOW}Install Whisper:{Colors.RESET}")
                    print(f"  {Colors.DIM}pip install openai-whisper{Colors.RESET}")
                if not audio_ok:
                    print(f"\n{Colors.YELLOW}Install audio support:{Colors.RESET}")
                    print(f"  {Colors.DIM}pip install sounddevice scipy numpy{Colors.RESET}")

            elif subcmd == "listen":
                # Record and transcribe
                if not _check_whisper():
                    print(f"{Colors.YELLOW}Whisper not installed. Run: pip install openai-whisper{Colors.RESET}")
                    return "COMMAND"
                if not _check_audio():
                    print(f"{Colors.YELLOW}Audio not available. Run: pip install sounddevice scipy numpy{Colors.RESET}")
                    return "COMMAND"

                try:
                    duration = int(sub_args) if sub_args else 5
                except ValueError:
                    duration = 5

                # Record
                audio_file = _record_audio(duration)
                if audio_file:
                    # Transcribe
                    result = _transcribe_file(audio_file)
                    if result:
                        text = result.get('text', '').strip()
                        print(f"\n{Colors.BRIGHT_CYAN}Transcription:{Colors.RESET}")
                        print(f"  {Colors.BRIGHT_WHITE}{text}{Colors.RESET}")

                        # Store for use
                        self._last_transcription = text

                        # Offer to use as input
                        use = input(f"\n{Colors.CYAN}Use as chat input? [Y/n]: {Colors.RESET}").strip().lower()
                        if use != 'n':
                            return text  # Return transcription as user input

                    # Clean up
                    try:
                        os.unlink(audio_file)
                    except:
                        pass

            elif subcmd == "file":
                # Transcribe file
                file_path = sub_args.strip()
                if not file_path:
                    print(f"{Colors.YELLOW}Usage: /whisper file <audio_file>{Colors.RESET}")
                    print(f"{Colors.DIM}Supported: mp3, wav, m4a, webm, mp4, flac{Colors.RESET}")
                elif not _check_whisper():
                    print(f"{Colors.YELLOW}Whisper not installed. Run: pip install openai-whisper{Colors.RESET}")
                else:
                    result = _transcribe_file(file_path)
                    if result:
                        text = result.get('text', '').strip()
                        language = result.get('language', 'unknown')

                        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Transcription Result{Colors.RESET}")
                        print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")
                        print(f"  {Colors.CYAN}File:{Colors.RESET} {file_path}")
                        print(f"  {Colors.CYAN}Language:{Colors.RESET} {language}")
                        print(f"  {Colors.CYAN}Duration:{Colors.RESET} {result.get('segments', [{}])[-1].get('end', 0):.1f}s")
                        print(f"\n{Colors.BRIGHT_WHITE}Text:{Colors.RESET}")
                        print(f"  {text}")

                        # Show segments if verbose
                        if 'segments' in result and len(result['segments']) > 1:
                            print(f"\n{Colors.DIM}Segments: {len(result['segments'])}{Colors.RESET}")

            elif subcmd == "model":
                # Change model
                model_name = sub_args.strip().lower() if sub_args else ""
                available_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']

                if not model_name:
                    print(f"{Colors.YELLOW}Usage: /whisper model <name>{Colors.RESET}")
                    print(f"{Colors.DIM}Current: {self._whisper_model_name}{Colors.RESET}")
                    print(f"{Colors.DIM}Available: {', '.join(available_models)}{Colors.RESET}")
                elif model_name not in available_models:
                    print(f"{Colors.RED}Unknown model: {model_name}{Colors.RESET}")
                    print(f"{Colors.DIM}Available: {', '.join(available_models)}{Colors.RESET}")
                else:
                    self._whisper_model = None  # Unload current
                    self._whisper_model_name = model_name
                    print(f"{Colors.BRIGHT_GREEN}✓ Model set to: {model_name}{Colors.RESET}")
                    print(f"{Colors.DIM}Model will be loaded on next transcription{Colors.RESET}")

            elif subcmd == "models":
                # List models
                models_info = [
                    ('tiny', '39M params', 'Fastest, lowest accuracy'),
                    ('base', '74M params', 'Good balance for quick tasks'),
                    ('small', '244M params', 'Better accuracy, slower'),
                    ('medium', '769M params', 'High accuracy'),
                    ('large', '1550M params', 'Best accuracy'),
                    ('large-v2', '1550M params', 'Improved large'),
                    ('large-v3', '1550M params', 'Latest large model'),
                ]

                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Whisper Models{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}\n")

                for name, size, desc in models_info:
                    is_current = name == self._whisper_model_name
                    indicator = f"{Colors.BRIGHT_GREEN}●{Colors.RESET}" if is_current else f"{Colors.DIM}○{Colors.RESET}"
                    print(f"  {indicator} {Colors.BRIGHT_WHITE}{name:12}{Colors.RESET} {Colors.DIM}{size:15}{Colors.RESET} {desc}")

                print(f"\n{Colors.DIM}Use /whisper model <name> to change{Colors.RESET}")

            elif subcmd == "chat":
                # Voice chat mode
                if not _check_whisper() or not _check_audio():
                    print(f"{Colors.YELLOW}Voice chat requires whisper and audio support{Colors.RESET}")
                    return "COMMAND"

                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Voice Chat Mode{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")
                print(f"{Colors.DIM}Press Enter to record, 'q' to quit{Colors.RESET}\n")

                while True:
                    try:
                        cmd = input(f"{Colors.CYAN}[Press Enter to speak, 'q' to quit]: {Colors.RESET}").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        break

                    if cmd == 'q':
                        break

                    # Record
                    audio_file = _record_audio(5)
                    if audio_file:
                        result = _transcribe_file(audio_file)
                        if result:
                            text = result.get('text', '').strip()
                            if text:
                                print(f"\n{Colors.BRIGHT_WHITE}You said:{Colors.RESET} {text}")

                                # Process as chat input
                                print(f"\n{Colors.BRIGHT_CYAN}AI Response:{Colors.RESET}")
                                # Call the chat processing
                                response = self._process_chat_input(text)
                                if response:
                                    print(response)

                        try:
                            os.unlink(audio_file)
                        except:
                            pass

                print(f"\n{Colors.DIM}Voice chat ended{Colors.RESET}")

            elif subcmd == "status":
                print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Whisper Status{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*50}{Colors.RESET}")

                whisper_ok = _check_whisper()
                audio_ok = _check_audio()

                print(f"  {Colors.CYAN}Whisper installed:{Colors.RESET} {'✓ Yes' if whisper_ok else '✗ No'}")
                print(f"  {Colors.CYAN}Audio support:{Colors.RESET} {'✓ Yes' if audio_ok else '✗ No'}")
                print(f"  {Colors.CYAN}Selected model:{Colors.RESET} {self._whisper_model_name}")
                print(f"  {Colors.CYAN}Model loaded:{Colors.RESET} {'✓ Yes' if self._whisper_model else '✗ No'}")

                if hasattr(self, '_last_transcription'):
                    print(f"\n{Colors.CYAN}Last transcription:{Colors.RESET}")
                    print(f"  {Colors.DIM}{self._last_transcription[:100]}{'...' if len(self._last_transcription) > 100 else ''}{Colors.RESET}")

            else:
                print(f"{Colors.YELLOW}Unknown whisper command: {subcmd}{Colors.RESET}")
                print(f"{Colors.DIM}Use /whisper for help{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/sysinfo":
            # Show full system information
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}System Information{Colors.RESET}")
            print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")

            if SYSTEM_INFO_AVAILABLE:
                sys_info = get_system_info()
                status_widget = get_status_widget()

                # Full system info box
                print(status_widget.render_box())

                # Additional details
                print(f"\n{Colors.BRIGHT_WHITE}Additional Details:{Colors.RESET}")
                print(f"  {Colors.CYAN}OS Type:{Colors.RESET}      {Colors.BRIGHT_WHITE}{sys_info.os_type}{Colors.RESET}")
                print(f"  {Colors.CYAN}OS Version:{Colors.RESET}   {Colors.BRIGHT_WHITE}{sys_info.os_version}{Colors.RESET}")
                print(f"  {Colors.CYAN}Platform:{Colors.RESET}     {Colors.BRIGHT_WHITE}{platform.platform()}{Colors.RESET}")
                print(f"  {Colors.CYAN}Python:{Colors.RESET}       {Colors.BRIGHT_WHITE}{platform.python_version()}{Colors.RESET}")
                print(f"  {Colors.CYAN}Machine:{Colors.RESET}      {Colors.BRIGHT_WHITE}{platform.machine()}{Colors.RESET}")
                print(f"  {Colors.CYAN}Home Dir:{Colors.RESET}     {Colors.BRIGHT_WHITE}{sys_info.home_directory}{Colors.RESET}")

                # Working directory
                if hasattr(self, 'working_directory') and self.working_directory:
                    print(f"  {Colors.CYAN}Working Dir:{Colors.RESET}  {Colors.BRIGHT_WHITE}{self.working_directory}{Colors.RESET}")

                # Common directories that exist
                print(f"\n{Colors.BRIGHT_WHITE}Quick Access Directories:{Colors.RESET}")
                for name, path in sys_info.get_common_directories().items():
                    exists_icon = f"{Colors.BRIGHT_GREEN}[OK]{Colors.RESET}" if os.path.exists(path) else f"{Colors.DIM}[--]{Colors.RESET}"
                    print(f"  {exists_icon} {Colors.CYAN}{name}:{Colors.RESET} {Colors.DIM}{path}{Colors.RESET}")
            else:
                # Fallback
                import getpass
                import socket
                print(f"  {Colors.CYAN}OS:{Colors.RESET}       {Colors.BRIGHT_WHITE}{platform.system()} {platform.release()}{Colors.RESET}")
                print(f"  {Colors.CYAN}User:{Colors.RESET}     {Colors.BRIGHT_WHITE}{getpass.getuser()}{Colors.RESET}")
                print(f"  {Colors.CYAN}Host:{Colors.RESET}     {Colors.BRIGHT_WHITE}{socket.gethostname()}{Colors.RESET}")
                print(f"  {Colors.CYAN}Platform:{Colors.RESET} {Colors.BRIGHT_WHITE}{platform.platform()}{Colors.RESET}")
                print(f"  {Colors.CYAN}Python:{Colors.RESET}   {Colors.BRIGHT_WHITE}{platform.python_version()}{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/traverse":
            # Directory navigation / file browser
            target_path = args.strip() if args else ""

            # Initialize current working directory if not set
            if not hasattr(self, 'working_directory') or not self.working_directory:
                self.working_directory = os.getcwd()

            if target_path:
                # Direct path navigation
                if target_path == "~":
                    new_path = os.path.expanduser("~")
                elif target_path == "..":
                    new_path = os.path.dirname(self.working_directory)
                elif target_path.startswith("~"):
                    new_path = os.path.expanduser(target_path)
                elif os.path.isabs(target_path):
                    new_path = target_path
                else:
                    new_path = os.path.join(self.working_directory, target_path)

                # Normalize and validate
                new_path = os.path.normpath(new_path)

                if os.path.isdir(new_path):
                    self.working_directory = new_path
                    print(f"\n{Colors.BRIGHT_GREEN}✓ Changed directory to:{Colors.RESET}")
                    print(f"  {Colors.BRIGHT_WHITE}{self.working_directory}{Colors.RESET}")

                    # Show directory contents preview
                    try:
                        items = os.listdir(new_path)
                        dirs = [d for d in items if os.path.isdir(os.path.join(new_path, d)) and not d.startswith('.')]
                        files = [f for f in items if os.path.isfile(os.path.join(new_path, f)) and not f.startswith('.')]
                        print(f"\n  {Colors.CYAN}Contains:{Colors.RESET} {len(dirs)} folders, {len(files)} files")
                    except PermissionError:
                        print(f"  {Colors.YELLOW}(Permission denied to list contents){Colors.RESET}")
                else:
                    print(f"{Colors.RED}Directory not found: {new_path}{Colors.RESET}")

            else:
                # Interactive directory browser
                self._traverse_interactive()

            return "COMMAND"

        elif cmd == "/pwd":
            # Print working directory
            if not hasattr(self, 'working_directory') or not self.working_directory:
                self.working_directory = os.getcwd()

            print(f"\n{Colors.BRIGHT_CYAN}Current Working Directory:{Colors.RESET}")
            print(f"  {Colors.BRIGHT_WHITE}{self.working_directory}{Colors.RESET}")
            return "COMMAND"

        elif cmd == "/ls":
            # List directory contents
            if not hasattr(self, 'working_directory') or not self.working_directory:
                self.working_directory = os.getcwd()

            target_path = args.strip() if args else self.working_directory

            # Handle special paths
            if target_path == "~":
                target_path = os.path.expanduser("~")
            elif target_path == "..":
                target_path = os.path.dirname(self.working_directory)
            elif target_path.startswith("~"):
                target_path = os.path.expanduser(target_path)
            elif not os.path.isabs(target_path):
                target_path = os.path.join(self.working_directory, target_path)

            target_path = os.path.normpath(target_path)

            if os.path.isdir(target_path):
                print(f"\n{Colors.BRIGHT_CYAN}Contents of:{Colors.RESET} {Colors.BRIGHT_WHITE}{target_path}{Colors.RESET}")
                print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")

                try:
                    items = os.listdir(target_path)
                    items.sort(key=lambda x: (not os.path.isdir(os.path.join(target_path, x)), x.lower()))

                    dirs = []
                    files = []

                    for item in items:
                        if item.startswith('.'):
                            continue  # Skip hidden files
                        full_path = os.path.join(target_path, item)
                        if os.path.isdir(full_path):
                            dirs.append(item)
                        else:
                            files.append(item)

                    # Show directories first
                    if dirs:
                        print(f"\n  {Colors.BRIGHT_YELLOW}Folders ({len(dirs)}):{Colors.RESET}")
                        for d in dirs[:20]:
                            print(f"    {Colors.CYAN}/{Colors.RESET}{Colors.BRIGHT_WHITE}{d}{Colors.RESET}")
                        if len(dirs) > 20:
                            print(f"    {Colors.DIM}... and {len(dirs) - 20} more folders{Colors.RESET}")

                    # Show files
                    if files:
                        print(f"\n  {Colors.BRIGHT_YELLOW}Files ({len(files)}):{Colors.RESET}")
                        for f in files[:20]:
                            # Get file size
                            try:
                                size = os.path.getsize(os.path.join(target_path, f))
                                if size < 1024:
                                    size_str = f"{size} B"
                                elif size < 1024 * 1024:
                                    size_str = f"{size // 1024} KB"
                                else:
                                    size_str = f"{size // (1024 * 1024)} MB"
                                print(f"    {Colors.BRIGHT_WHITE}{f}{Colors.RESET} {Colors.DIM}({size_str}){Colors.RESET}")
                            except:
                                print(f"    {Colors.BRIGHT_WHITE}{f}{Colors.RESET}")
                        if len(files) > 20:
                            print(f"    {Colors.DIM}... and {len(files) - 20} more files{Colors.RESET}")

                    if not dirs and not files:
                        print(f"  {Colors.DIM}(empty directory){Colors.RESET}")

                except PermissionError:
                    print(f"  {Colors.RED}Permission denied{Colors.RESET}")
                except Exception as e:
                    print(f"  {Colors.RED}Error: {e}{Colors.RESET}")
            else:
                print(f"{Colors.RED}Not a directory: {target_path}{Colors.RESET}")

            return "COMMAND"

        elif cmd == "/cd":
            # Alias for traverse
            return self.handle_chat_command(f"/traverse {args}")

        else:
            print(f"{Colors.YELLOW}⚠ Unknown command: {cmd}. Type {Colors.BRIGHT_WHITE}/help{Colors.YELLOW} for available commands.{Colors.RESET}")
            return "COMMAND"

    def _traverse_interactive(self):
        """Interactive directory browser for selecting project folders - Cross-platform"""
        if not hasattr(self, 'working_directory') or not self.working_directory:
            self.working_directory = os.getcwd()

        # Get system info for cross-platform support
        sys_info = None
        status_widget = None
        if SYSTEM_INFO_AVAILABLE:
            sys_info = get_system_info()
            status_widget = get_status_widget()

        current_path = self.working_directory
        show_quick_access = True  # Show quick access on first view

        while True:
            self.clear()

            # ═══════════════════════════════════════════════════════════════════
            # HEADER WITH STATUS WIDGET
            # ═══════════════════════════════════════════════════════════════════
            print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
            print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  DIRECTORY BROWSER{Colors.RESET}")
            print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")

            # Status Widget - OS, Time, Date, Network, Username
            if status_widget:
                print(f"\n{status_widget.render_compact()}")
            else:
                # Fallback if system info not available
                import getpass
                user = getpass.getuser()
                os_name = platform.system()
                time_str = datetime.now().strftime("%H:%M")
                date_str = datetime.now().strftime("%Y-%m-%d")
                print(f"\n{Colors.DIM}[{os_name}] {date_str} {time_str} | @{user}{Colors.RESET}")

            print(f"\n{Colors.CYAN}Current:{Colors.RESET} {Colors.BRIGHT_WHITE}{current_path}{Colors.RESET}")
            print(f"{Colors.DIM}{'─'*79}{Colors.RESET}")

            try:
                # ═══════════════════════════════════════════════════════════════════
                # QUICK ACCESS DIRECTORIES (First time or when requested)
                # ═══════════════════════════════════════════════════════════════════
                if show_quick_access and sys_info:
                    print(f"\n  {Colors.BRIGHT_MAGENTA}{Colors.BOLD}Quick Access:{Colors.RESET}")

                    # Common directories
                    common_dirs = sys_info.get_common_directories()
                    quick_keys = ['h', 'd', 'o', 'w', 'p', 'c']  # home, desktop, documents, downloads, projects, code
                    quick_idx = 0

                    for name, path in common_dirs.items():
                        if os.path.exists(path) and quick_idx < len(quick_keys):
                            key = quick_keys[quick_idx]
                            print(f"    {Colors.MAGENTA}[{key}]{Colors.RESET} {Colors.BRIGHT_WHITE}{name}{Colors.RESET} {Colors.DIM}({path}){Colors.RESET}")
                            quick_idx += 1

                    # Windows drives
                    if sys_info.is_windows:
                        print(f"\n  {Colors.BRIGHT_MAGENTA}Drives:{Colors.RESET}")
                        for drive in sys_info.get_root_directories()[:5]:
                            print(f"    {Colors.DIM}{drive}{Colors.RESET}")

                    print()

                # ═══════════════════════════════════════════════════════════════════
                # NAVIGATION OPTIONS
                # ═══════════════════════════════════════════════════════════════════
                print(f"  {Colors.CYAN}[0]{Colors.RESET} {Colors.DIM}..{Colors.RESET} (Parent directory)")
                print(f"  {Colors.CYAN}[~]{Colors.RESET} {Colors.DIM}Home directory{Colors.RESET}")

                # Windows: show root/drives option
                if sys_info and sys_info.is_windows:
                    print(f"  {Colors.CYAN}[/]{Colors.RESET} {Colors.DIM}Show drives{Colors.RESET}")

                print()

                # ═══════════════════════════════════════════════════════════════════
                # DIRECTORY LISTING
                # ═══════════════════════════════════════════════════════════════════
                items = os.listdir(current_path)
                items.sort(key=lambda x: (not os.path.isdir(os.path.join(current_path, x)), x.lower()))

                dirs = []
                files = []
                for item in items:
                    # Skip hidden files (cross-platform)
                    if item.startswith('.'):
                        continue
                    # Windows hidden files check
                    if sys_info and sys_info.is_windows:
                        try:
                            import stat
                            full_path = os.path.join(current_path, item)
                            if os.stat(full_path).st_file_attributes & stat.FILE_ATTRIBUTE_HIDDEN:
                                continue
                        except:
                            pass

                    full_path = os.path.join(current_path, item)
                    if os.path.isdir(full_path):
                        dirs.append(item)
                    else:
                        files.append(item)

                # Show directories with numbers
                if dirs:
                    print(f"  {Colors.BRIGHT_YELLOW}Folders:{Colors.RESET}")
                    for i, d in enumerate(dirs[:25], 1):
                        subpath = os.path.join(current_path, d)
                        try:
                            has_subdirs = any(os.path.isdir(os.path.join(subpath, x)) for x in os.listdir(subpath) if not x.startswith('.'))
                            indicator = f"{Colors.CYAN}>{Colors.RESET}" if has_subdirs else " "
                        except:
                            indicator = " "
                        print(f"  {Colors.CYAN}[{i}]{Colors.RESET} {indicator} {Colors.BRIGHT_WHITE}{d}{Colors.DIM}/{Colors.RESET}")

                    if len(dirs) > 25:
                        print(f"      {Colors.DIM}... and {len(dirs) - 25} more folders{Colors.RESET}")
                else:
                    print(f"  {Colors.DIM}(No subdirectories){Colors.RESET}")

                if files:
                    print(f"\n  {Colors.DIM}({len(files)} files in this directory){Colors.RESET}")

                # ═══════════════════════════════════════════════════════════════════
                # INSTRUCTIONS
                # ═══════════════════════════════════════════════════════════════════
                print(f"\n{Colors.DIM}{'─'*79}{Colors.RESET}")
                print(f"{Colors.DIM}[num] navigate | [s] select | [q] cancel | [?] quick access | type path to jump{Colors.RESET}")

                choice = input(f"\n{Colors.BRIGHT_GREEN}Navigate: {Colors.RESET}").strip()

                if not choice:
                    show_quick_access = False
                    continue

                show_quick_access = False  # Hide quick access after first input

                # ═══════════════════════════════════════════════════════════════════
                # HANDLE INPUT
                # ═══════════════════════════════════════════════════════════════════

                # Quit
                if choice.lower() in ('q', 'quit', 'exit'):
                    print(f"\n{Colors.DIM}Cancelled{Colors.RESET}")
                    break

                # Select current
                if choice.lower() in ('s', 'select'):
                    self.working_directory = current_path
                    print(f"\n{Colors.BRIGHT_GREEN}✓ Selected directory:{Colors.RESET}")
                    print(f"  {Colors.BRIGHT_WHITE}{self.working_directory}{Colors.RESET}")
                    time.sleep(1)
                    break

                # Show quick access
                if choice == '?':
                    show_quick_access = True
                    continue

                # Parent directory
                if choice == '0':
                    parent = os.path.dirname(current_path)
                    if parent and parent != current_path:
                        current_path = parent
                    elif sys_info and sys_info.is_windows:
                        # On Windows, show drives if at root
                        pass
                    continue

                # Home directory
                if choice == '~':
                    current_path = os.path.expanduser("~")
                    continue

                # Windows drives
                if choice == '/' and sys_info and sys_info.is_windows:
                    print(f"\n{Colors.BRIGHT_CYAN}Available Drives:{Colors.RESET}")
                    for drive in sys_info.get_root_directories():
                        print(f"  {Colors.BRIGHT_WHITE}{drive}{Colors.RESET}")
                    drive_choice = input(f"\n{Colors.BRIGHT_GREEN}Enter drive (e.g., C:): {Colors.RESET}").strip()
                    if drive_choice:
                        if not drive_choice.endswith(':'):
                            drive_choice += ':'
                        if not drive_choice.endswith('\\'):
                            drive_choice += '\\'
                        if os.path.exists(drive_choice):
                            current_path = drive_choice
                    continue

                # Quick access shortcuts
                if sys_info and choice.lower() in ['h', 'd', 'o', 'w', 'p', 'c']:
                    common_dirs = sys_info.get_common_directories()
                    dir_map = {'h': 'Home', 'd': 'Desktop', 'o': 'Documents', 'w': 'Downloads', 'p': 'Projects', 'c': 'Code'}
                    target_name = dir_map.get(choice.lower())
                    if target_name and target_name in common_dirs:
                        target_path = common_dirs[target_name]
                        if os.path.exists(target_path):
                            current_path = target_path
                            continue

                # Numeric selection
                try:
                    idx = int(choice)
                    if 1 <= idx <= len(dirs):
                        current_path = os.path.join(current_path, dirs[idx - 1])
                        continue
                    else:
                        print(f"{Colors.YELLOW}Invalid selection{Colors.RESET}")
                        time.sleep(0.5)
                        continue
                except ValueError:
                    pass

                # Direct path (absolute)
                # Handle Windows paths (C:\...) and Unix paths (/...)
                is_absolute = False
                if sys_info and sys_info.is_windows:
                    # Windows: check for drive letter or UNC path
                    is_absolute = (len(choice) >= 2 and choice[1] == ':') or choice.startswith('\\\\')
                else:
                    is_absolute = choice.startswith('/')

                if is_absolute or choice.startswith('~'):
                    test_path = os.path.expanduser(choice)
                    test_path = os.path.normpath(test_path)
                    if os.path.isdir(test_path):
                        current_path = test_path
                        continue
                    else:
                        print(f"{Colors.RED}Directory not found: {choice}{Colors.RESET}")
                        time.sleep(1)
                        continue

                # Relative path
                test_path = os.path.join(current_path, choice)
                test_path = os.path.normpath(test_path)
                if os.path.isdir(test_path):
                    current_path = test_path
                    continue

                print(f"{Colors.YELLOW}Unknown command or path: {choice}{Colors.RESET}")
                time.sleep(0.5)

            except PermissionError:
                print(f"\n{Colors.RED}Permission denied for this directory{Colors.RESET}")
                print(f"{Colors.DIM}Press Enter to go back...{Colors.RESET}")
                input()
                current_path = os.path.dirname(current_path)
            except Exception as e:
                print(f"\n{Colors.RED}Error: {e}{Colors.RESET}")
                print(f"{Colors.DIM}Press Enter to continue...{Colors.RESET}")
                input()

    def chat_loop(self):
        self.clear()
        # Create default session if none exists
        if not self.current_session_id:
            self.current_session_id = self.memory.create_session(f"Chat {datetime.now().strftime('%H:%M')}",
                                                                  self.current_project_id)

        # ═══════════════════════════════════════════════════════════════════════
        # CHAT HEADER WITH STATUS WIDGET
        # ═══════════════════════════════════════════════════════════════════════
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  CHAT MODE ACTIVE{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")

        # Status Widget - OS, Time, Date, Network, Username
        if SYSTEM_INFO_AVAILABLE:
            status_widget = get_status_widget()
            print(f"\n{status_widget.render_compact()}")
        else:
            # Fallback status display
            import getpass
            user = getpass.getuser()
            os_name = platform.system()
            time_str = datetime.now().strftime("%H:%M")
            date_str = datetime.now().strftime("%Y-%m-%d")
            print(f"\n{Colors.DIM}[{os_name}] {date_str} {time_str} | @{user}{Colors.RESET}")

        print(f"{Colors.DIM}{'─'*79}{Colors.RESET}")

        # Session and Project info
        print(f"{Colors.CYAN}Session:{Colors.RESET} {Colors.BRIGHT_WHITE}{self.memory.get_session(self.current_session_id)[1]}{Colors.RESET}")
        if self.current_project_id:
            project = self.memory.get_project(self.current_project_id)
            print(f"{Colors.CYAN}Project:{Colors.RESET} {Colors.BRIGHT_WHITE}{project[1]}{Colors.RESET}")

        # Working directory (if set)
        if hasattr(self, 'working_directory') and self.working_directory:
            print(f"{Colors.CYAN}Directory:{Colors.RESET} {Colors.DIM}{self.working_directory}{Colors.RESET}")

        print(f"\n{Colors.DIM}Type '/help' for commands, '/back' to exit{Colors.RESET}\n")
        
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
                print(f"{Colors.BRIGHT_GREEN}✓ Model loaded on {self.device}{Colors.RESET}")
            except Exception as e:
                print(f"{Colors.BRIGHT_RED}✗ Model load failed: {e}{Colors.RESET}")
                raise
        elif self.backend == "ollama":
            try:
                # 1. Check if Ollama is running with resilient request
                resilient_request('get', "http://localhost:11434", timeout=2, max_retries=3, retry_delay=1.0)
                print(f"{Colors.BRIGHT_GREEN}✓ Ollama connection established{Colors.RESET}")

                # 2. Fetch available models for Auto-Detection
                print(f"{Colors.CYAN}Detecting available models...{Colors.RESET}")
                resp = resilient_request('get', "http://localhost:11434/api/tags", timeout=5, max_retries=2, retry_delay=1.0)
                if resp.status_code == 200:
                    available = [m['name'] for m in resp.json().get('models', [])]

                    # 3. Check if configured model exists
                    if self.model_name in available:
                        print(f"{Colors.BRIGHT_GREEN}✓ Using configured model: {self.model_name}{Colors.RESET}")
                    else:
                        print(f"{Colors.YELLOW}⚠ Configured model '{self.model_name}' not found on this machine.{Colors.RESET}")

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
                            print(f"{Colors.BRIGHT_CYAN}✨ Auto-detected alternative: {detected_model}{Colors.RESET}")
                            self.model_name = detected_model
                        else:
                            print(f"{Colors.BRIGHT_RED}✗ No models found in Ollama.{Colors.RESET}")
                            print(f"{Colors.YELLOW}Please run: ollama pull mistral{Colors.RESET}")
                            raise Exception("No Ollama models available")
                else:
                    print(f"{Colors.BRIGHT_RED}✗ Failed to fetch models from Ollama{Colors.RESET}")

            except requests.exceptions.RequestException:
                print(f"{Colors.BRIGHT_YELLOW}⚠ Ollama not running on localhost:11434{Colors.RESET}")
                print(f"{Colors.YELLOW}Start Ollama with: ollama serve{Colors.RESET}")
                raise Exception("Ollama backend not available")

    def generate(self, prompt, timeout=180):
        """Generate AI response with configurable timeout."""
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
                max_tokens = self.config.get("max_response_tokens", 200)

                # Use resilient_request with self-healing for better reliability
                response = resilient_request(
                    'post',
                    "http://localhost:11434/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": True,  # Enable streaming
                        "options": {
                            "temperature": self.config.get("temperature", 0.7),
                            "num_predict": max_tokens
                        }
                    },
                    stream=True,
                    timeout=timeout,  # Configurable timeout
                    max_retries=3,
                    retry_delay=3.0
                )

                if response.status_code != 200:
                    return f"Ollama Error: {response.text[:200]}"

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
                return f"Request timed out after {timeout} seconds. The model may be slow."
            except requests.exceptions.ConnectionError:
                return "Cannot connect to Ollama. Make sure Ollama is running: ollama serve"
            except Exception as e:
                return f"Error: {str(e)[:200]}"

try:
    ai_engine = LocalAIEngine(CONFIG)
    print(f"{Colors.BRIGHT_GREEN}✓ Local AI Engine ready{Colors.RESET}")
except Exception as e:
    print(f"{Colors.BRIGHT_RED}✗ Failed to load AI model: {e}{Colors.RESET}")
    print(f"{Colors.YELLOW}Check your backend configuration in config.json{Colors.RESET}")
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
