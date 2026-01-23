"""
Self-Healing Module for AI Terminal Pro
Provides auto-recovery, error handling, and system resilience features
"""

import os
import sys
import time
import sqlite3
import socket
import threading
import subprocess
import shutil
import json
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable, Any, Optional, Dict, List


class Colors:
    """ANSI color codes"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


class SelfHealingConfig:
    """Configuration for self-healing behavior"""
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0  # seconds
    RETRY_BACKOFF = 2.0  # exponential backoff multiplier
    DB_RECONNECT_ATTEMPTS = 5
    NETWORK_TIMEOUT = 10
    MCP_RESTART_DELAY = 2.0
    HEALTH_CHECK_INTERVAL = 60  # seconds
    AUTO_BACKUP_INTERVAL = 300  # 5 minutes
    MAX_MEMORY_MB = 500  # Memory threshold for cleanup
    LOG_FILE = "self_healing.log"


class SelfHealingLogger:
    """Logger for self-healing events"""

    def __init__(self, log_file: str = None):
        self.log_file = log_file or SelfHealingConfig.LOG_FILE
        self.log_dir = os.path.dirname(os.path.abspath(__file__))
        self.log_path = os.path.join(self.log_dir, '..', '..', self.log_file)

    def log(self, level: str, message: str, component: str = "general"):
        """Log a self-healing event"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level.upper()}] [{component}] {message}"

        try:
            with open(self.log_path, 'a') as f:
                f.write(log_entry + "\n")
        except:
            pass  # Fail silently if can't write log

        # Print to console based on level
        if level == "error":
            print(f"{Colors.BRIGHT_RED}[HEAL] {message}{Colors.RESET}")
        elif level == "warning":
            print(f"{Colors.BRIGHT_YELLOW}[HEAL] {message}{Colors.RESET}")
        elif level == "success":
            print(f"{Colors.BRIGHT_GREEN}[HEAL] {message}{Colors.RESET}")
        elif level == "info":
            print(f"{Colors.DIM}[HEAL] {message}{Colors.RESET}")


# Global logger instance
_logger = SelfHealingLogger()


def retry_on_failure(max_retries: int = None, delay: float = None,
                     exceptions: tuple = (Exception,),
                     on_retry: Callable = None,
                     on_failure: Callable = None):
    """
    Decorator that retries a function on failure with exponential backoff

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries
        exceptions: Tuple of exceptions to catch and retry
        on_retry: Callback function called on each retry (receives attempt number, exception)
        on_failure: Callback function called on final failure (receives exception)
    """
    max_retries = max_retries or SelfHealingConfig.MAX_RETRIES
    delay = delay or SelfHealingConfig.RETRY_DELAY

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e

                    if attempt < max_retries:
                        _logger.log("warning",
                                   f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}",
                                   "retry")

                        if on_retry:
                            try:
                                on_retry(attempt + 1, e)
                            except:
                                pass

                        time.sleep(current_delay)
                        current_delay *= SelfHealingConfig.RETRY_BACKOFF
                    else:
                        _logger.log("error",
                                   f"{func.__name__} failed after {max_retries + 1} attempts: {e}",
                                   "retry")

                        if on_failure:
                            try:
                                return on_failure(e)
                            except:
                                pass
                        raise

            return None
        return wrapper
    return decorator


class DatabaseHealer:
    """Self-healing for SQLite database connections"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.backup_dir = os.path.join(os.path.dirname(db_path), 'backups')
        self.conn = None
        self._ensure_backup_dir()

    def _ensure_backup_dir(self):
        """Ensure backup directory exists"""
        try:
            os.makedirs(self.backup_dir, exist_ok=True)
        except:
            pass

    def connect(self) -> sqlite3.Connection:
        """Get a database connection with auto-recovery"""
        if self.conn:
            try:
                # Test if connection is still valid
                self.conn.execute("SELECT 1")
                return self.conn
            except:
                self.conn = None

        # Try to connect with recovery
        for attempt in range(SelfHealingConfig.DB_RECONNECT_ATTEMPTS):
            try:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
                self.conn.execute("PRAGMA journal_mode=WAL")
                self.conn.execute("PRAGMA synchronous=NORMAL")
                _logger.log("success", f"Database connected: {self.db_path}", "database")
                return self.conn
            except sqlite3.DatabaseError as e:
                _logger.log("warning", f"Database connection failed (attempt {attempt + 1}): {e}", "database")

                if "database is locked" in str(e).lower():
                    time.sleep(1)
                    continue
                elif "database disk image is malformed" in str(e).lower():
                    self._attempt_repair()
                else:
                    time.sleep(0.5)

        # Last resort: create new database
        _logger.log("error", "All recovery attempts failed, creating fresh database", "database")
        return self._create_fresh_database()

    def _attempt_repair(self):
        """Attempt to repair a corrupted database"""
        _logger.log("info", "Attempting database repair...", "database")

        try:
            # Backup corrupted file
            backup_path = os.path.join(self.backup_dir,
                                       f"corrupted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
            shutil.copy2(self.db_path, backup_path)

            # Try to recover using sqlite3 dump
            temp_db = self.db_path + ".recovery"

            # Export what we can
            try:
                old_conn = sqlite3.connect(self.db_path)
                with open(temp_db + ".sql", 'w') as f:
                    for line in old_conn.iterdump():
                        f.write(line + '\n')
                old_conn.close()

                # Import to new database
                new_conn = sqlite3.connect(temp_db)
                with open(temp_db + ".sql", 'r') as f:
                    new_conn.executescript(f.read())
                new_conn.close()

                # Replace corrupted with recovered
                os.remove(self.db_path)
                os.rename(temp_db, self.db_path)
                os.remove(temp_db + ".sql")

                _logger.log("success", "Database repair successful", "database")
            except Exception as e:
                _logger.log("error", f"Database repair failed: {e}", "database")
        except Exception as e:
            _logger.log("error", f"Backup failed during repair: {e}", "database")

    def _create_fresh_database(self) -> sqlite3.Connection:
        """Create a fresh database with proper schema"""
        try:
            # Backup old file if exists
            if os.path.exists(self.db_path):
                backup_path = os.path.join(self.backup_dir,
                                          f"old_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
                try:
                    shutil.move(self.db_path, backup_path)
                except:
                    os.remove(self.db_path)

            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            _logger.log("success", "Fresh database created", "database")
            return self.conn
        except Exception as e:
            _logger.log("error", f"Failed to create fresh database: {e}", "database")
            raise

    def create_backup(self) -> Optional[str]:
        """Create a backup of the current database"""
        try:
            if not os.path.exists(self.db_path):
                return None

            backup_path = os.path.join(self.backup_dir,
                                      f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
            shutil.copy2(self.db_path, backup_path)

            # Clean old backups (keep last 10)
            self._cleanup_old_backups()

            _logger.log("info", f"Database backup created: {backup_path}", "database")
            return backup_path
        except Exception as e:
            _logger.log("error", f"Backup failed: {e}", "database")
            return None

    def _cleanup_old_backups(self, keep: int = 10):
        """Remove old backups, keeping only the most recent ones"""
        try:
            backups = sorted([
                os.path.join(self.backup_dir, f)
                for f in os.listdir(self.backup_dir)
                if f.startswith('backup_') and f.endswith('.db')
            ], reverse=True)

            for old_backup in backups[keep:]:
                try:
                    os.remove(old_backup)
                except:
                    pass
        except:
            pass

    def execute_safe(self, query: str, params: tuple = (),
                     default: Any = None) -> Any:
        """Execute a query with auto-recovery on failure"""
        for attempt in range(SelfHealingConfig.DB_RECONNECT_ATTEMPTS):
            try:
                conn = self.connect()
                cursor = conn.cursor()
                cursor.execute(query, params)

                if query.strip().upper().startswith("SELECT"):
                    return cursor.fetchall()
                else:
                    conn.commit()
                    return cursor.lastrowid
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    time.sleep(0.5 * (attempt + 1))
                    continue
                else:
                    self.conn = None  # Force reconnection
                    if attempt == SelfHealingConfig.DB_RECONNECT_ATTEMPTS - 1:
                        _logger.log("error", f"Query failed: {e}", "database")
                        return default
            except Exception as e:
                _logger.log("error", f"Query error: {e}", "database")
                return default

        return default


class NetworkHealer:
    """Self-healing for network operations"""

    _connection_cache = {}
    _last_check = None
    _is_online = None

    @classmethod
    def check_connectivity(cls, force: bool = False) -> bool:
        """Check internet connectivity with caching"""
        now = datetime.now()

        # Use cached result if recent
        if not force and cls._last_check and cls._is_online is not None:
            if (now - cls._last_check).seconds < 30:
                return cls._is_online

        # Test connectivity
        test_hosts = [
            ("8.8.8.8", 53),      # Google DNS
            ("1.1.1.1", 53),      # Cloudflare DNS
            ("208.67.222.222", 53) # OpenDNS
        ]

        for host, port in test_hosts:
            try:
                socket.setdefaulttimeout(SelfHealingConfig.NETWORK_TIMEOUT)
                socket.create_connection((host, port), timeout=3)
                cls._is_online = True
                cls._last_check = now
                return True
            except:
                continue

        cls._is_online = False
        cls._last_check = now
        return False

    @classmethod
    def wait_for_connection(cls, timeout: int = 60,
                           check_interval: int = 5) -> bool:
        """Wait for network connection to be restored"""
        _logger.log("info", "Waiting for network connection...", "network")

        start_time = time.time()
        while time.time() - start_time < timeout:
            if cls.check_connectivity(force=True):
                _logger.log("success", "Network connection restored", "network")
                return True
            time.sleep(check_interval)

        _logger.log("error", f"Network connection not restored within {timeout}s", "network")
        return False

    @classmethod
    @retry_on_failure(max_retries=3, exceptions=(socket.error, ConnectionError, TimeoutError))
    def safe_request(cls, func: Callable, *args, **kwargs) -> Any:
        """Execute a network request with auto-retry"""
        if not cls.check_connectivity():
            if not cls.wait_for_connection(timeout=30):
                raise ConnectionError("No network connectivity")
        return func(*args, **kwargs)


class MCPHealer:
    """Self-healing for MCP server connections"""

    def __init__(self, registry=None):
        self.registry = registry
        self._restart_attempts = {}
        self._max_restarts = 3
        self._restart_window = 300  # 5 minutes

    def _can_restart(self, server_name: str) -> bool:
        """Check if server can be restarted (rate limiting)"""
        now = datetime.now()

        if server_name not in self._restart_attempts:
            self._restart_attempts[server_name] = []

        # Clean old attempts
        self._restart_attempts[server_name] = [
            t for t in self._restart_attempts[server_name]
            if (now - t).seconds < self._restart_window
        ]

        return len(self._restart_attempts[server_name]) < self._max_restarts

    def health_check(self, server_name: str) -> bool:
        """Check if an MCP server is healthy"""
        if not self.registry or not hasattr(self.registry, 'mcp_clients'):
            return False

        if server_name not in self.registry.mcp_clients:
            return False

        client = self.registry.mcp_clients[server_name]

        try:
            # Check if process is running
            if hasattr(client, 'process') and client.process:
                return client.process.poll() is None
            return False
        except:
            return False

    def restart_server(self, server_name: str) -> bool:
        """Attempt to restart an MCP server"""
        if not self._can_restart(server_name):
            _logger.log("warning",
                       f"MCP server '{server_name}' restart rate limit reached",
                       "mcp")
            return False

        _logger.log("info", f"Attempting to restart MCP server: {server_name}", "mcp")

        try:
            # Stop existing server
            if server_name in self.registry.mcp_clients:
                try:
                    self.registry.mcp_clients[server_name].stop()
                except:
                    pass

            time.sleep(SelfHealingConfig.MCP_RESTART_DELAY)

            # Restart server
            if hasattr(self.registry, 'start_mcp_server'):
                self.registry.start_mcp_server(server_name)
            elif hasattr(self.registry, '_start_mcp_client'):
                self.registry._start_mcp_client(server_name)

            # Record restart attempt
            self._restart_attempts[server_name].append(datetime.now())

            # Verify restart
            time.sleep(1)
            if self.health_check(server_name):
                _logger.log("success", f"MCP server '{server_name}' restarted successfully", "mcp")
                return True
            else:
                _logger.log("error", f"MCP server '{server_name}' failed to restart", "mcp")
                return False
        except Exception as e:
            _logger.log("error", f"Error restarting MCP server '{server_name}': {e}", "mcp")
            return False

    def auto_heal_all(self) -> Dict[str, bool]:
        """Check and heal all MCP servers"""
        results = {}

        if not self.registry or not hasattr(self.registry, 'mcp_clients'):
            return results

        for server_name in list(self.registry.mcp_clients.keys()):
            if not self.health_check(server_name):
                results[server_name] = self.restart_server(server_name)
            else:
                results[server_name] = True

        return results


class ConfigHealer:
    """Self-healing for configuration files"""

    DEFAULT_CONFIG = {
        "first_run": False,
        "backend": "ollama",
        "model_name": "llama3",
        "system_prompt": "You are a helpful AI assistant.",
        "enable_dangerous_commands": False,
        "max_context_window": 32768,
        "max_response_tokens": 2000,
        "temperature": 0.7,
        "cpu_threads": 4,
        "default_editor_command": "",
        "theme": "default"
    }

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.backup_path = config_path + ".backup"

    def load_and_heal(self) -> dict:
        """Load config with automatic healing of missing/invalid values"""
        config = {}

        # Try to load existing config
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
            except json.JSONDecodeError as e:
                _logger.log("warning", f"Config file corrupted: {e}", "config")
                config = self._attempt_recovery()
            except Exception as e:
                _logger.log("error", f"Failed to load config: {e}", "config")

        # Heal missing keys
        healed = False
        for key, default_value in self.DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = default_value
                healed = True
                _logger.log("info", f"Healed missing config key: {key}", "config")

        # Validate and fix invalid values
        config, type_healed = self._validate_types(config)
        healed = healed or type_healed

        # Save healed config
        if healed:
            self.save(config)

        return config

    def _attempt_recovery(self) -> dict:
        """Attempt to recover config from backup or create new"""
        # Try backup
        if os.path.exists(self.backup_path):
            try:
                with open(self.backup_path, 'r') as f:
                    config = json.load(f)
                _logger.log("success", "Config recovered from backup", "config")
                return config
            except:
                pass

        # Create new config
        _logger.log("info", "Creating new config from defaults", "config")
        return dict(self.DEFAULT_CONFIG)

    def _validate_types(self, config: dict) -> tuple:
        """Validate config value types and fix if needed"""
        healed = False
        type_map = {
            "first_run": bool,
            "backend": str,
            "model_name": str,
            "system_prompt": str,
            "enable_dangerous_commands": bool,
            "max_context_window": int,
            "max_response_tokens": int,
            "temperature": float,
            "cpu_threads": int,
            "theme": str
        }

        for key, expected_type in type_map.items():
            if key in config and not isinstance(config[key], expected_type):
                try:
                    config[key] = expected_type(config[key])
                    healed = True
                except:
                    config[key] = self.DEFAULT_CONFIG.get(key)
                    healed = True

        # Range validation
        if config.get("temperature", 0) < 0 or config.get("temperature", 0) > 2:
            config["temperature"] = 0.7
            healed = True

        if config.get("max_response_tokens", 0) < 100:
            config["max_response_tokens"] = 2000
            healed = True

        return config, healed

    def save(self, config: dict):
        """Save config with backup"""
        try:
            # Backup current config
            if os.path.exists(self.config_path):
                shutil.copy2(self.config_path, self.backup_path)

            # Save new config
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            _logger.log("error", f"Failed to save config: {e}", "config")


class MemoryHealer:
    """Self-healing for memory management"""

    @staticmethod
    def get_memory_usage_mb() -> float:
        """Get current process memory usage in MB"""
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            return usage.ru_maxrss / 1024  # Convert to MB
        except:
            try:
                import psutil
                process = psutil.Process(os.getpid())
                return process.memory_info().rss / (1024 * 1024)
            except:
                return 0

    @staticmethod
    def cleanup_if_needed(threshold_mb: float = None) -> bool:
        """Run garbage collection if memory usage is high"""
        import gc

        threshold = threshold_mb or SelfHealingConfig.MAX_MEMORY_MB
        current_usage = MemoryHealer.get_memory_usage_mb()

        if current_usage > threshold:
            _logger.log("info", f"Memory cleanup triggered ({current_usage:.1f}MB)", "memory")
            gc.collect()
            new_usage = MemoryHealer.get_memory_usage_mb()
            freed = current_usage - new_usage
            _logger.log("success", f"Freed {freed:.1f}MB of memory", "memory")
            return True

        return False


class HealthMonitor:
    """Background health monitoring with auto-healing"""

    def __init__(self, app=None):
        self.app = app
        self.running = False
        self._thread = None
        self.db_healer = None
        self.mcp_healer = None
        self.config_healer = None
        self._last_backup = None

    def start(self, db_path: str = None, config_path: str = None, registry=None):
        """Start the health monitor"""
        if db_path:
            self.db_healer = DatabaseHealer(db_path)
        if config_path:
            self.config_healer = ConfigHealer(config_path)
        if registry:
            self.mcp_healer = MCPHealer(registry)

        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        _logger.log("info", "Health monitor started", "monitor")

    def stop(self):
        """Stop the health monitor"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
        _logger.log("info", "Health monitor stopped", "monitor")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._run_health_checks()
                time.sleep(SelfHealingConfig.HEALTH_CHECK_INTERVAL)
            except Exception as e:
                _logger.log("error", f"Health check error: {e}", "monitor")
                time.sleep(10)

    def _run_health_checks(self):
        """Run all health checks"""
        # Memory cleanup
        MemoryHealer.cleanup_if_needed()

        # Database backup
        if self.db_healer:
            now = datetime.now()
            if (not self._last_backup or
                (now - self._last_backup).seconds > SelfHealingConfig.AUTO_BACKUP_INTERVAL):
                self.db_healer.create_backup()
                self._last_backup = now

        # MCP server health
        if self.mcp_healer:
            self.mcp_healer.auto_heal_all()

        # Network connectivity
        NetworkHealer.check_connectivity()

    def get_status(self) -> dict:
        """Get current health status"""
        return {
            "running": self.running,
            "network": NetworkHealer.check_connectivity(),
            "memory_mb": MemoryHealer.get_memory_usage_mb(),
            "last_backup": self._last_backup.isoformat() if self._last_backup else None
        }


# Global health monitor instance
_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get or create the global health monitor"""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor


def start_health_monitor(db_path: str = None, config_path: str = None, registry=None):
    """Start the global health monitor"""
    monitor = get_health_monitor()
    monitor.start(db_path, config_path, registry)
    return monitor


# Convenience decorators for self-healing
def with_db_recovery(db_healer: DatabaseHealer):
    """Decorator to add database recovery to functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except sqlite3.Error as e:
                _logger.log("warning", f"DB error in {func.__name__}: {e}, attempting recovery", "database")
                db_healer.conn = None  # Force reconnection
                return func(*args, **kwargs)
        return wrapper
    return decorator


def with_network_recovery(func: Callable) -> Callable:
    """Decorator to add network recovery to functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, socket.error, TimeoutError) as e:
            _logger.log("warning", f"Network error: {e}, waiting for connection", "network")
            if NetworkHealer.wait_for_connection(timeout=30):
                return func(*args, **kwargs)
            raise
    return wrapper


if __name__ == "__main__":
    # Test self-healing features
    print("Testing Self-Healing Module...")

    print("\n1. Network Check:")
    print(f"   Online: {NetworkHealer.check_connectivity()}")

    print("\n2. Memory Check:")
    print(f"   Usage: {MemoryHealer.get_memory_usage_mb():.1f}MB")

    print("\n3. Config Healer Test:")
    healer = ConfigHealer("test_config.json")
    config = healer.load_and_heal()
    print(f"   Loaded config with {len(config)} keys")

    # Cleanup test file
    try:
        os.remove("test_config.json")
        os.remove("test_config.json.backup")
    except:
        pass

    print("\nSelf-healing module test complete!")
