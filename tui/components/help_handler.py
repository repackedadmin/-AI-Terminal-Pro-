"""
Enhanced Help Handler for AI Terminal Pro
Provides comprehensive, formatted help documentation for all chat commands
"""

import os
import sys


class Colors:
    """ANSI color codes for terminal output"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"


class EnhancedHelpHandler:
    """
    Enhanced help system providing comprehensive command documentation
    with formatted output and interactive navigation
    """

    def __init__(self, app=None):
        """
        Initialize the help handler

        Args:
            app: Reference to the main App instance (optional, for context-aware help)
        """
        self.app = app

    def _show_help(self, section=None):
        """
        Display help information

        Args:
            section: Optional specific section to display (None for all)
        """
        self._clear_screen()

        # Header
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}")
        print(f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}  HELP - AI TERMINAL PRO{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'='*79}{Colors.RESET}\n")

        if section:
            self._display_section(section)
        else:
            self._display_all_sections()

        # Footer
        print(f"\n{Colors.DIM}{'─'*79}{Colors.RESET}")
        print(f"{Colors.DIM}Press Enter to continue...{Colors.RESET}")

        try:
            input()
        except (EOFError, KeyboardInterrupt):
            pass

    def _display_all_sections(self):
        """Display all help sections with full detail"""

        # ═══════════════════════════════════════════════════════════════════
        # BASICS SECTION
        # ═══════════════════════════════════════════════════════════════════
        print(f"  {Colors.BRIGHT_GREEN}{Colors.BOLD}i Basics{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        print(f"    {Colors.BRIGHT_WHITE}Syntax:{Colors.RESET} {Colors.DIM}Use & to specify /Slice for command{Colors.RESET}")
        print(f"    {Colors.BRIGHT_WHITE}Context:{Colors.RESET} {Colors.DIM}Execute shell commands via | at default/ language{Colors.RESET}")
        print(f"    {Colors.BRIGHT_WHITE}ACTION:{Colors.RESET} {Colors.DIM}Use ACTION: TOOL_NAME args to trigger tool execution{Colors.RESET}")

        # ═══════════════════════════════════════════════════════════════════
        # COMMANDS SECTION
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_GREEN}{Colors.BOLD}Commands{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/new [name]", "Create a new chat session")
        self._print_command("/help or /?", "Show this help message")
        self._print_command("/back", "Return to main menu")

        # ═══════════════════════════════════════════════════════════════════
        # SESSION COMMANDS (Sub-commands)
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_CYAN}{Colors.BOLD}Session Commands{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/save [filename]", "Save current chat to file")
        self._print_subcommand("", "Exports chat history to JSON format")
        self._print_subcommand("", "Default: chat_YYYYMMDD_HHMMSS.json")

        self._print_command("/load [id|filename]", "Load chat session")
        self._print_subcommand("/load", "List all available sessions")
        self._print_subcommand("/load [id]", "Load session by ID number")
        self._print_subcommand("/load [file.json]", "Load session from JSON file")

        # ═══════════════════════════════════════════════════════════════════
        # PROJECT COMMANDS (Sub-commands)
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_CYAN}{Colors.BOLD}Project Commands{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/project [name|description]", "Create/switch project")
        self._print_subcommand("/project", "List all available projects")
        self._print_subcommand("/project [name]", "Create or switch to project")
        self._print_subcommand("/project [name|desc]", "Create with description (use | separator)")

        self._print_command("/project_save [filename]", "Export project to file")
        self._print_subcommand("", "Saves project metadata and sessions to JSON")

        self._print_command("/project_load [id|filename]", "Load project")
        self._print_subcommand("/project_load", "List projects available to load")
        self._print_subcommand("/project_load [id]", "Load project by ID")
        self._print_subcommand("/project_load [file]", "Load project from JSON file")

        # ═══════════════════════════════════════════════════════════════════
        # RAG COMMANDS (Retrieval-Augmented Generation)
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_CYAN}{Colors.BOLD}RAG Commands (Context Memory){Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/rag", "Manage extensions, filesystem, multi-extension commands")
        self._print_subcommand("/rag status", "Show RAG status and document count")
        self._print_subcommand("/rag list", "List all ingested documents")

        self._print_command("/rag search [query]", "Search vector database")
        self._print_subcommand("", "Performs semantic search through stored documents")

        self._print_command("/rag add [path]", "Add documents to context")
        self._print_subcommand("/rag add [file]", "Add single file (txt, pdf, md, py, etc.)")
        self._print_subcommand("/rag add [folder]", "Add all supported files from folder")
        self._print_subcommand("", "Supports: image, document, code files")

        self._print_command("/rag clear", "Clear all documents from RAG")
        self._print_subcommand("", "Requires confirmation, cannot be undone")

        self._print_command("/rag enable [extension]", "Enable an extension")
        self._print_subcommand("", "Usage: /rag enable <extpythonpath/extmodel>")

        self._print_command("/rag disable [extension]", "Disable an extension")

        # ═══════════════════════════════════════════════════════════════════
        # MCP COMMANDS (Model Context Protocol)
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_CYAN}{Colors.BOLD}MCP Server Management{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/mcp", "Manage configured Model Context Protocol servers")
        self._print_subcommand("/mcp status", "Show MCP server status overview")

        self._print_command("/mcp list", "List configured MCP servers and tools")
        self._print_subcommand("", "Shows all servers with their available tools")

        self._print_command("/mcp start [server]", "Start an MCP server")
        self._print_subcommand("", "Launch configured server by name")

        self._print_command("/mcp stop [server]", "Stop a running MCP server")
        self._print_subcommand("", "Terminate server process")

        self._print_command("/mcp restart [server]", "Restart an MCP server")

        self._print_command("/mcp tools [server]", "List tools from specific server")
        self._print_subcommand("", "Usage: /mcp tools <server-name>")

        self._print_command("/mcp info [server]", "Show detailed server information")
        self._print_subcommand("", "Configuration, status, and capabilities")

        # ═══════════════════════════════════════════════════════════════════
        # TOOL COMMANDS
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_CYAN}{Colors.BOLD}Tool Management{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/tools", "List all available tools")
        self._print_subcommand("", "Shows native, custom, and MCP tools")

        self._print_command("/tool [name]", "Show tool details and usage")
        self._print_subcommand("", "Display description, parameters, examples")

        self._print_command("/tool run [name] [args]", "Execute a tool directly")
        self._print_subcommand("", "Run tool with specified arguments")

        self._print_command("/tool create [type]", "Create a new custom tool")
        self._print_subcommand("/tool create python", "Create Python tool script")
        self._print_subcommand("/tool create json", "Create JSON tool definition")
        self._print_subcommand("/tool create yaml", "Create YAML tool definition")

        self._print_command("/tool edit [name]", "Edit an existing tool")
        self._print_command("/tool delete [name]", "Delete a custom tool")

        self._print_command("/browser [url]", "Open URL in Playwright browser")
        self._print_subcommand("", "Web automation and scraping capabilities")

        # ═══════════════════════════════════════════════════════════════════
        # TUI CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_CYAN}{Colors.BOLD}TUI Configuration{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/config", "Show current configuration")
        self._print_subcommand("/config [key]", "Show specific config value")
        self._print_subcommand("/config [key] [value]", "Update configuration setting")

        self._print_command("/config list", "List all configuration options")

        self._print_command("/theme [name]", "Change UI theme (if available)")
        self._print_subcommand("/theme dark", "Dark theme")
        self._print_subcommand("/theme light", "Light theme")

        self._print_command("/settings", "Open settings menu")

        # ═══════════════════════════════════════════════════════════════════
        # SYSTEM COMMANDS
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_CYAN}{Colors.BOLD}System Commands{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/clear", "Clear the chat display")
        self._print_subcommand("", "Clears screen and redraws header")

        self._print_command("/status", "Show system status information")
        self._print_subcommand("", "Model, memory, backend, session info")

        self._print_command("/model", "Show current model information")
        self._print_subcommand("/model [name]", "Switch to different model")
        self._print_subcommand("/model list", "List available models")

        self._print_command("/memory", "Show memory/context usage")
        self._print_subcommand("/memory clear", "Clear conversation memory")

        self._print_command("/history", "Show conversation history")
        self._print_subcommand("/history [n]", "Show last n messages")
        self._print_subcommand("/history clear", "Clear history (keeps RAG)")

        self._print_command("/export [format]", "Export conversation")
        self._print_subcommand("/export json", "Export as JSON")
        self._print_subcommand("/export md", "Export as Markdown")
        self._print_subcommand("/export txt", "Export as plain text")

        # ═══════════════════════════════════════════════════════════════════
        # NAVIGATION / TRAVERSE
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_CYAN}{Colors.BOLD}Directory Navigation{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/traverse", "Browse and select project directory")
        self._print_subcommand("", "Interactive file browser to navigate folders")
        self._print_subcommand("/traverse [path]", "Jump directly to specified path")
        self._print_subcommand("/traverse ..", "Go up one directory level")
        self._print_subcommand("/traverse ~", "Go to home directory")

        self._print_command("/pwd", "Show current working directory")
        self._print_command("/ls", "List files in current directory")
        self._print_subcommand("/ls [path]", "List files in specified path")
        self._print_command("/cd [path]", "Alias for /traverse")

        self._print_command("/sysinfo", "Show full system information")
        self._print_subcommand("", "OS, user, network, time, directories")

        # ═══════════════════════════════════════════════════════════════════
        # PERSONA & ANALYSIS
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_CYAN}{Colors.BOLD}Persona & Analysis{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/persona", "Analyze project and create tailored persona")
        self._print_subcommand("/persona create", "Generate persona from project files")
        self._print_subcommand("/persona load [file]", "Load persona from file")
        self._print_subcommand("/persona save [file]", "Save current persona")
        self._print_subcommand("/persona clear", "Clear active persona")

        self._print_command("/analyze [path]", "Analyze codebase or documents")
        self._print_subcommand("", "Creates context-aware understanding")

        # ═══════════════════════════════════════════════════════════════════
        # LEGACY/CHAT COMMANDS
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_CYAN}{Colors.BOLD}Legacy Commands{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/chat [session]", "Load test session file (if exists)")
        self._print_subcommand("", "Also accessible as /load command")

        # ═══════════════════════════════════════════════════════════════════
        # ADVANCED FEATURES
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_YELLOW}{Colors.BOLD}Advanced Features{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/camera", "Launch Camera Assistant")
        self._print_subcommand("", "Vision only, optimized for low-end PCs")
        self._print_subcommand("", "OpenCV-based computer vision")

        self._print_command("/voice or /tts", "Launch Voice Assistant")
        self._print_subcommand("", "Text-to-Speech only, no camera")
        self._print_subcommand("", "Supports pyttsx3, kokoro engines")

        self._print_command("/vision", "Launch Vision + Voice Assistant")
        self._print_subcommand("", "Combined camera + voice capabilities")
        self._print_subcommand("", "Full multimodal interaction")

        self._print_command("/whisper", "Speech-to-text input")
        self._print_subcommand("", "Whisper-based voice recognition")

        # ═══════════════════════════════════════════════════════════════════
        # APP BUILDER
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_YELLOW}{Colors.BOLD}App Builder (Multi-Agent){Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/app", "Access App Builder from chat")
        self._print_subcommand("/app new [name]", "Create new app project")
        self._print_subcommand("/app list", "List app projects")
        self._print_subcommand("/app open [id]", "Open existing project")
        self._print_subcommand("/app status", "Show current app status")

        # ═══════════════════════════════════════════════════════════════════
        # TRAINING COMMANDS
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_YELLOW}{Colors.BOLD}Model Training{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/train", "Access training features")
        self._print_subcommand("/train finetune", "Fine-tune model on dataset")
        self._print_subcommand("/train lora", "LoRA parameter-efficient training")
        self._print_subcommand("/train rlhf", "RLHF training mode")
        self._print_subcommand("/train status", "Show training status")

        # ═══════════════════════════════════════════════════════════════════
        # API COMMANDS
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_YELLOW}{Colors.BOLD}API Management{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/api", "API server management")
        self._print_subcommand("/api start [port]", "Start API server")
        self._print_subcommand("/api stop", "Stop API server")
        self._print_subcommand("/api status", "Show API server status")
        self._print_subcommand("/api key", "Show/generate API key")

        # ═══════════════════════════════════════════════════════════════════
        # EXIT COMMANDS
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_RED}{Colors.BOLD}Exit{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        self._print_command("/back", "Return to main menu")
        self._print_command("/exit", "Exit chat (same as /back)")
        self._print_command("/quit", "Exit entire application")

        # ═══════════════════════════════════════════════════════════════════
        # TIPS SECTION
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n  {Colors.BRIGHT_MAGENTA}{Colors.BOLD}Tips{Colors.RESET}")
        print(f"  {Colors.DIM}{'─'*40}{Colors.RESET}")
        print(f"    {Colors.DIM}• Use ACTION: TOOL_NAME args to execute tools in chat{Colors.RESET}")
        print(f"    {Colors.DIM}• Documents loaded via RAG enhance AI responses{Colors.RESET}")
        print(f"    {Colors.DIM}• Projects help organize related conversations{Colors.RESET}")
        print(f"    {Colors.DIM}• Configure MCP servers for external tool integrations{Colors.RESET}")
        print(f"    {Colors.DIM}• Use /help [command] for detailed command help{Colors.RESET}")

    def _display_section(self, section_key):
        """Display help for a specific section or command"""
        section_key = section_key.lower().strip().lstrip('/')

        # Check if it's a command-specific help request
        command_help = self._get_command_help(section_key)
        if command_help:
            self._show_command_help(section_key, command_help)
            return

        print(f"{Colors.YELLOW}No specific help found for: {section_key}{Colors.RESET}")
        print(f"{Colors.DIM}Use /help to see all commands{Colors.RESET}")

    def _get_command_help(self, cmd):
        """Get detailed help for a specific command"""
        help_data = {
            "new": {
                "usage": "/new [session_name]",
                "description": "Create a new chat session",
                "details": [
                    "Creates a fresh chat session for conversation",
                    "Session is automatically saved to database",
                    "Can be associated with a project"
                ],
                "examples": [
                    "/new                    Create with auto-generated name",
                    "/new MyChat             Create session named 'MyChat'",
                    "/new 'Debug Session'    Create with spaces in name"
                ],
                "related": ["/save", "/load", "/project"]
            },
            "save": {
                "usage": "/save [filename]",
                "description": "Save the current chat session to a JSON file",
                "details": [
                    "Exports full chat history including metadata",
                    "Files saved to ai_sandbox directory by default",
                    "Can be loaded later with /load command"
                ],
                "examples": [
                    "/save                   Auto-generate filename",
                    "/save mychat.json       Save to specific file",
                    "/save debug_session     .json added automatically"
                ],
                "related": ["/load", "/export"]
            },
            "load": {
                "usage": "/load [id|filename]",
                "description": "Load a previously saved chat session",
                "details": [
                    "Can load by session ID from database",
                    "Can load from exported JSON file",
                    "Running without args lists available sessions"
                ],
                "examples": [
                    "/load                   List all sessions",
                    "/load 5                 Load session ID 5",
                    "/load mychat.json       Load from file"
                ],
                "related": ["/save", "/new"]
            },
            "project": {
                "usage": "/project [name|description]",
                "description": "Create, switch, or list projects",
                "details": [
                    "Projects group related sessions together",
                    "Project context is included in AI prompts",
                    "Use | to separate name and description"
                ],
                "examples": [
                    "/project                        List all projects",
                    "/project WebApp                 Create/switch to project",
                    "/project WebApp|E-commerce site Create with description"
                ],
                "related": ["/project_save", "/project_load"]
            },
            "rag": {
                "usage": "/rag [subcommand] [args]",
                "description": "Manage RAG (Retrieval-Augmented Generation) context memory",
                "details": [
                    "RAG enables semantic search through your documents",
                    "Documents are chunked and embedded for retrieval",
                    "Retrieved context enhances AI responses"
                ],
                "subcommands": [
                    "/rag                    Show RAG status",
                    "/rag list               List ingested documents",
                    "/rag search [query]     Search documents",
                    "/rag add [path]         Add file or folder",
                    "/rag clear              Clear all documents",
                    "/rag enable [ext]       Enable extension",
                    "/rag disable [ext]      Disable extension"
                ],
                "examples": [
                    "/rag add ./docs/        Add all docs from folder",
                    "/rag search 'API auth'  Search for API auth info",
                    "/rag list               Show all documents"
                ],
                "related": ["/memory", "/analyze"]
            },
            "mcp": {
                "usage": "/mcp [subcommand] [server_name]",
                "description": "Manage Model Context Protocol (MCP) servers",
                "details": [
                    "MCP enables external tool integrations",
                    "Servers communicate via JSON-RPC 2.0",
                    "Configure servers in mcp_servers.json"
                ],
                "subcommands": [
                    "/mcp                    Show MCP status",
                    "/mcp list               List servers and tools",
                    "/mcp start [name]       Start a server",
                    "/mcp stop [name]        Stop a server",
                    "/mcp restart [name]     Restart a server",
                    "/mcp tools [name]       List server's tools",
                    "/mcp info [name]        Server details"
                ],
                "examples": [
                    "/mcp list               Show all servers",
                    "/mcp start filesystem   Start filesystem server",
                    "/mcp tools github       List GitHub tools"
                ],
                "related": ["/tools", "/config"]
            },
            "tools": {
                "usage": "/tools",
                "description": "List all available tools",
                "details": [
                    "Shows native, custom, and MCP tools",
                    "Tools can be executed with ACTION: syntax",
                    "Custom tools can be created in Python/JSON/YAML"
                ],
                "examples": [
                    "/tools                  List all tools",
                    "/tool read_file         Show tool details",
                    "/tool run search .      Execute tool"
                ],
                "related": ["/tool", "/mcp"]
            },
            "tool": {
                "usage": "/tool [name] or /tool [subcommand] [args]",
                "description": "Manage and execute tools",
                "subcommands": [
                    "/tool [name]            Show tool details",
                    "/tool run [name] [args] Execute tool directly",
                    "/tool create [type]     Create new tool",
                    "/tool edit [name]       Edit existing tool",
                    "/tool delete [name]     Delete tool"
                ],
                "examples": [
                    "/tool read_file         Show read_file details",
                    "/tool create python     Create Python tool",
                    "/tool run search query  Run search tool"
                ],
                "related": ["/tools", "/mcp"]
            },
            "clear": {
                "usage": "/clear",
                "description": "Clear the chat display",
                "details": [
                    "Clears the terminal screen",
                    "Redraws the chat header",
                    "Does not clear conversation history"
                ],
                "examples": ["/clear"],
                "related": ["/history clear", "/memory clear"]
            },
            "status": {
                "usage": "/status",
                "description": "Show system status information",
                "details": [
                    "Displays current model and backend",
                    "Shows active session and project",
                    "Reports memory and configuration"
                ],
                "examples": ["/status"],
                "related": ["/config", "/model"]
            },
            "model": {
                "usage": "/model [name]",
                "description": "View or change the current AI model",
                "details": [
                    "Without args, shows current model info",
                    "With name, switches to specified model",
                    "Model change may require reload"
                ],
                "examples": [
                    "/model                  Show current model",
                    "/model llama3           Switch to llama3",
                    "/model list             List available models"
                ],
                "related": ["/config", "/status"]
            },
            "config": {
                "usage": "/config [key] [value]",
                "description": "View or modify configuration settings",
                "details": [
                    "Without args, shows all settings",
                    "With key only, shows that setting",
                    "With key and value, updates setting"
                ],
                "examples": [
                    "/config                     Show all config",
                    "/config temperature         Show temperature",
                    "/config temperature 0.8     Set temperature"
                ],
                "related": ["/settings", "/model"]
            },
            "camera": {
                "usage": "/camera",
                "description": "Launch the Camera Assistant",
                "details": [
                    "Vision-only mode (no voice)",
                    "Optimized for low-end PCs",
                    "Uses OpenCV for capture",
                    "AI analyzes camera frames"
                ],
                "examples": ["/camera"],
                "related": ["/voice", "/vision"]
            },
            "voice": {
                "usage": "/voice or /tts",
                "description": "Launch the Voice Assistant",
                "details": [
                    "Text-to-Speech mode (no camera)",
                    "Supports multiple TTS engines",
                    "pyttsx3 and kokoro support"
                ],
                "examples": ["/voice", "/tts"],
                "related": ["/camera", "/vision", "/whisper"]
            },
            "vision": {
                "usage": "/vision",
                "description": "Launch the Vision + Voice Assistant",
                "details": [
                    "Combined camera and voice mode",
                    "Full multimodal interaction",
                    "Requires more system resources"
                ],
                "examples": ["/vision"],
                "related": ["/camera", "/voice"]
            },
            "browser": {
                "usage": "/browser [url]",
                "description": "Open URL in Playwright browser",
                "details": [
                    "Automated browser for web interaction",
                    "Supports scraping and automation",
                    "Playwright-powered headless browser"
                ],
                "examples": [
                    "/browser https://example.com",
                    "/browser google.com"
                ],
                "related": ["/tools"]
            },
            "api": {
                "usage": "/api [subcommand]",
                "description": "Manage the API server",
                "subcommands": [
                    "/api                    Show API status",
                    "/api start [port]       Start server on port",
                    "/api stop               Stop server",
                    "/api status             Detailed status",
                    "/api key                Show/generate API key"
                ],
                "examples": [
                    "/api start 5000         Start on port 5000",
                    "/api key                Show API key"
                ],
                "related": ["/config"]
            },
            "app": {
                "usage": "/app [subcommand]",
                "description": "Access App Builder multi-agent system",
                "subcommands": [
                    "/app                    App Builder status",
                    "/app new [name]         Create new project",
                    "/app list               List projects",
                    "/app open [id]          Open project",
                    "/app status             Current project status"
                ],
                "examples": [
                    "/app new MyWebApp       Create new app",
                    "/app list               Show all apps"
                ],
                "related": ["/project"]
            },
            "train": {
                "usage": "/train [subcommand]",
                "description": "Access model training features",
                "subcommands": [
                    "/train                  Training menu",
                    "/train finetune         Fine-tuning mode",
                    "/train lora             LoRA training",
                    "/train rlhf             RLHF training",
                    "/train status           Training status"
                ],
                "examples": [
                    "/train finetune         Start fine-tuning",
                    "/train status           Check progress"
                ],
                "related": ["/model"]
            },
            "traverse": {
                "usage": "/traverse [path]",
                "description": "Navigate and select project directories",
                "details": [
                    "Opens interactive file browser when no path given",
                    "Navigate using number keys to select directories",
                    "Updates working directory for the chat session",
                    "Supports direct path, .., and ~ shortcuts"
                ],
                "subcommands": [
                    "/traverse               Interactive directory browser",
                    "/traverse [path]        Jump to specified directory",
                    "/traverse ..            Go up one directory level",
                    "/traverse ~             Go to home directory"
                ],
                "examples": [
                    "/traverse                       Open browser at current dir",
                    "/traverse /home/user/projects   Jump to projects folder",
                    "/traverse ~/Desktop             Go to Desktop",
                    "/traverse ..                    Go to parent directory"
                ],
                "related": ["/pwd", "/ls", "/project"]
            },
            "pwd": {
                "usage": "/pwd",
                "description": "Print current working directory",
                "details": [
                    "Shows the current directory for the chat session",
                    "This is where relative paths are resolved from"
                ],
                "examples": ["/pwd"],
                "related": ["/traverse", "/ls"]
            },
            "ls": {
                "usage": "/ls [path]",
                "description": "List files and directories",
                "details": [
                    "Lists contents of current or specified directory",
                    "Shows directories with / suffix",
                    "Shows file sizes and counts"
                ],
                "examples": [
                    "/ls                     List current directory",
                    "/ls /home/user          List specific path",
                    "/ls ..                  List parent directory"
                ],
                "related": ["/traverse", "/pwd"]
            },
            "cd": {
                "usage": "/cd [path]",
                "description": "Change directory (alias for /traverse)",
                "details": [
                    "Shortcut alias for /traverse command",
                    "Works exactly the same as /traverse"
                ],
                "examples": [
                    "/cd /home/user/projects",
                    "/cd ..",
                    "/cd ~"
                ],
                "related": ["/traverse", "/pwd", "/ls"]
            },
            "sysinfo": {
                "usage": "/sysinfo",
                "description": "Show full system information",
                "details": [
                    "Displays comprehensive system status",
                    "Shows OS type and version (Windows 10/11, macOS, Linux)",
                    "Shows current user and hostname",
                    "Shows network connectivity status",
                    "Shows date and time",
                    "Lists quick access directories"
                ],
                "examples": ["/sysinfo"],
                "related": ["/status", "/traverse"]
            }
        }
        return help_data.get(cmd)

    def _show_command_help(self, cmd, help_data):
        """Display detailed help for a specific command"""
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}Command: /{cmd}{Colors.RESET}")
        print(f"{Colors.DIM}{'─'*60}{Colors.RESET}")

        print(f"\n{Colors.BRIGHT_WHITE}Usage:{Colors.RESET} {help_data['usage']}")
        print(f"\n{Colors.BRIGHT_WHITE}Description:{Colors.RESET}")
        print(f"  {help_data['description']}")

        if 'details' in help_data:
            print(f"\n{Colors.BRIGHT_WHITE}Details:{Colors.RESET}")
            for detail in help_data['details']:
                print(f"  {Colors.DIM}• {detail}{Colors.RESET}")

        if 'subcommands' in help_data:
            print(f"\n{Colors.BRIGHT_WHITE}Subcommands:{Colors.RESET}")
            for subcmd in help_data['subcommands']:
                print(f"  {Colors.DIM}{subcmd}{Colors.RESET}")

        if 'examples' in help_data:
            print(f"\n{Colors.BRIGHT_WHITE}Examples:{Colors.RESET}")
            for example in help_data['examples']:
                print(f"  {Colors.GREEN}{example}{Colors.RESET}")

        if 'related' in help_data:
            print(f"\n{Colors.BRIGHT_WHITE}Related Commands:{Colors.RESET}")
            print(f"  {Colors.CYAN}{', '.join(help_data['related'])}{Colors.RESET}")

    def _print_command(self, command, description):
        """Print a formatted command entry"""
        visible_len = len(command)
        padding = " " * max(1, 30 - visible_len)
        print(f"    {Colors.BRIGHT_GREEN}{command}{Colors.RESET}{padding}{Colors.DIM}{description}{Colors.RESET}")

    def _print_subcommand(self, subcommand, description):
        """Print a formatted subcommand entry"""
        if subcommand:
            visible_len = len(subcommand)
            padding = " " * max(1, 30 - visible_len)
            print(f"      {Colors.CYAN}{subcommand}{Colors.RESET}{padding}{Colors.DIM}{description}{Colors.RESET}")
        else:
            print(f"      {Colors.DIM}  └─ {description}{Colors.RESET}")

    def _clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')


def show_quick_help():
    """Display quick help without an App instance"""
    handler = EnhancedHelpHandler()
    handler._show_help()


if __name__ == "__main__":
    show_quick_help()
