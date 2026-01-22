"""
System Information Module for AI Terminal Pro
Cross-platform OS detection, system info, and status display
"""

import os
import sys
import platform
import socket
import getpass
from datetime import datetime


class Colors:
    """ANSI color codes for terminal output"""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    BG_BLUE = "\033[44m"
    BG_CYAN = "\033[46m"
    BG_BLACK = "\033[40m"


class OSType:
    """Operating system type constants"""
    WINDOWS = "windows"
    MACOS = "macos"
    LINUX = "linux"
    UNKNOWN = "unknown"


class SystemInfo:
    """
    Cross-platform system information detection and utilities
    Supports Windows 10/11, macOS, and Linux
    """

    def __init__(self):
        self._os_type = None
        self._os_name = None
        self._os_version = None
        self._hostname = None
        self._username = None
        self._home_dir = None
        self._detect_system()

    def _detect_system(self):
        """Detect the operating system and gather system information"""
        system = platform.system().lower()

        if system == "windows":
            self._os_type = OSType.WINDOWS
            self._detect_windows()
        elif system == "darwin":
            self._os_type = OSType.MACOS
            self._detect_macos()
        elif system == "linux":
            self._os_type = OSType.LINUX
            self._detect_linux()
        else:
            self._os_type = OSType.UNKNOWN
            self._os_name = platform.system()
            self._os_version = platform.release()

        # Common detection
        self._hostname = socket.gethostname()
        self._username = getpass.getuser()
        self._home_dir = os.path.expanduser("~")

    def _detect_windows(self):
        """Detect Windows version (10/11)"""
        self._os_name = "Windows"
        release = platform.release()
        version = platform.version()

        # Windows 11 detection (build 22000+)
        try:
            build = int(version.split('.')[-1]) if version else 0
            if build >= 22000:
                self._os_name = "Windows 11"
            elif release == "10":
                self._os_name = "Windows 10"
            else:
                self._os_name = f"Windows {release}"
        except:
            self._os_name = f"Windows {release}"

        self._os_version = version

    def _detect_macos(self):
        """Detect macOS version"""
        self._os_name = "macOS"
        mac_ver = platform.mac_ver()[0]

        # Map version to name
        version_names = {
            "14": "Sonoma",
            "13": "Ventura",
            "12": "Monterey",
            "11": "Big Sur",
            "10.15": "Catalina",
            "10.14": "Mojave",
        }

        major_ver = mac_ver.split('.')[0] if mac_ver else ""
        minor_ver = f"{major_ver}.{mac_ver.split('.')[1]}" if mac_ver and len(mac_ver.split('.')) > 1 else major_ver

        if major_ver in version_names:
            self._os_name = f"macOS {version_names[major_ver]}"
        elif minor_ver in version_names:
            self._os_name = f"macOS {version_names[minor_ver]}"

        self._os_version = mac_ver

    def _detect_linux(self):
        """Detect Linux distribution"""
        self._os_name = "Linux"
        self._os_version = platform.release()

        # Try to get distro info
        try:
            # Try /etc/os-release first (most modern distros)
            if os.path.exists('/etc/os-release'):
                with open('/etc/os-release', 'r') as f:
                    lines = f.readlines()
                    info = {}
                    for line in lines:
                        if '=' in line:
                            key, value = line.strip().split('=', 1)
                            info[key] = value.strip('"')

                    if 'PRETTY_NAME' in info:
                        self._os_name = info['PRETTY_NAME']
                    elif 'NAME' in info:
                        self._os_name = info['NAME']
                        if 'VERSION' in info:
                            self._os_name += f" {info['VERSION']}"

            # Fallback to lsb_release
            elif os.path.exists('/etc/lsb-release'):
                with open('/etc/lsb-release', 'r') as f:
                    for line in f:
                        if line.startswith('DISTRIB_DESCRIPTION='):
                            self._os_name = line.split('=')[1].strip().strip('"')
                            break
        except:
            pass

    @property
    def os_type(self) -> str:
        """Get OS type constant (windows/macos/linux/unknown)"""
        return self._os_type

    @property
    def os_name(self) -> str:
        """Get friendly OS name"""
        return self._os_name

    @property
    def os_version(self) -> str:
        """Get OS version string"""
        return self._os_version

    @property
    def hostname(self) -> str:
        """Get system hostname"""
        return self._hostname

    @property
    def username(self) -> str:
        """Get current username"""
        return self._username

    @property
    def home_directory(self) -> str:
        """Get user's home directory"""
        return self._home_dir

    @property
    def is_windows(self) -> bool:
        return self._os_type == OSType.WINDOWS

    @property
    def is_macos(self) -> bool:
        return self._os_type == OSType.MACOS

    @property
    def is_linux(self) -> bool:
        return self._os_type == OSType.LINUX

    def get_path_separator(self) -> str:
        """Get the appropriate path separator for the OS"""
        return '\\' if self.is_windows else '/'

    def get_root_directories(self) -> list:
        """Get root/drive directories based on OS"""
        if self.is_windows:
            # Get available drives on Windows
            drives = []
            try:
                import string
                for letter in string.ascii_uppercase:
                    drive = f"{letter}:\\"
                    if os.path.exists(drive):
                        drives.append(drive)
            except:
                drives = ["C:\\"]
            return drives
        else:
            # Unix-like systems
            return ["/"]

    def get_common_directories(self) -> dict:
        """Get common user directories based on OS"""
        home = self._home_dir

        if self.is_windows:
            return {
                "Home": home,
                "Desktop": os.path.join(home, "Desktop"),
                "Documents": os.path.join(home, "Documents"),
                "Downloads": os.path.join(home, "Downloads"),
                "Projects": os.path.join(home, "Projects"),  # Common dev folder
                "Code": os.path.join(home, "Code"),  # VS Code default
            }
        elif self.is_macos:
            return {
                "Home": home,
                "Desktop": os.path.join(home, "Desktop"),
                "Documents": os.path.join(home, "Documents"),
                "Downloads": os.path.join(home, "Downloads"),
                "Developer": os.path.join(home, "Developer"),  # Xcode default
                "Projects": os.path.join(home, "Projects"),
            }
        else:  # Linux
            return {
                "Home": home,
                "Desktop": os.path.join(home, "Desktop"),
                "Documents": os.path.join(home, "Documents"),
                "Downloads": os.path.join(home, "Downloads"),
                "Projects": os.path.join(home, "Projects"),
                "Code": os.path.join(home, "code"),
            }

    def normalize_path(self, path: str) -> str:
        """Normalize a path for the current OS"""
        # Expand user home
        path = os.path.expanduser(path)
        # Expand environment variables
        path = os.path.expandvars(path)
        # Normalize separators and resolve
        path = os.path.normpath(path)
        return path

    def check_network_connectivity(self) -> tuple:
        """
        Check network connectivity
        Returns: (is_connected: bool, status: str)
        """
        try:
            # Try to connect to a reliable host
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            return True, "Connected"
        except OSError:
            pass

        try:
            # Fallback: try to resolve a hostname
            socket.gethostbyname("google.com")
            return True, "Connected"
        except socket.gaierror:
            pass

        return False, "Offline"

    def get_current_time(self) -> str:
        """Get current time formatted"""
        return datetime.now().strftime("%H:%M:%S")

    def get_current_date(self) -> str:
        """Get current date formatted"""
        return datetime.now().strftime("%Y-%m-%d")

    def get_datetime_formatted(self) -> str:
        """Get formatted date and time"""
        return datetime.now().strftime("%a %b %d, %Y  %H:%M")


class StatusWidget:
    """
    TUI Status Widget for displaying system information
    Shows: OS, Time, Date, Network, Username
    """

    def __init__(self, system_info: SystemInfo = None):
        self.sys_info = system_info or SystemInfo()

    def get_os_icon(self) -> str:
        """Get OS-specific icon/emoji"""
        if self.sys_info.is_windows:
            return "[W]"  # Windows
        elif self.sys_info.is_macos:
            return "[M]"  # Mac
        elif self.sys_info.is_linux:
            return "[L]"  # Linux
        return "[?]"

    def get_network_icon(self, connected: bool) -> str:
        """Get network status icon"""
        return "[+]" if connected else "[x]"

    def render_compact(self) -> str:
        """Render a compact single-line status bar"""
        net_connected, net_status = self.sys_info.check_network_connectivity()
        net_icon = self.get_network_icon(net_connected)
        net_color = Colors.BRIGHT_GREEN if net_connected else Colors.BRIGHT_RED

        os_icon = self.get_os_icon()
        time_str = self.sys_info.get_current_time()
        date_str = self.sys_info.get_current_date()
        user = self.sys_info.username
        os_name = self.sys_info.os_name

        # Truncate OS name if too long
        if len(os_name) > 20:
            os_name = os_name[:17] + "..."

        status_line = (
            f"{Colors.BG_BLACK}{Colors.BRIGHT_CYAN} {os_icon} {os_name} {Colors.RESET}"
            f"{Colors.BG_BLACK}{Colors.BRIGHT_WHITE} | {Colors.RESET}"
            f"{Colors.BG_BLACK}{Colors.BRIGHT_YELLOW} {date_str} {time_str} {Colors.RESET}"
            f"{Colors.BG_BLACK}{Colors.BRIGHT_WHITE} | {Colors.RESET}"
            f"{Colors.BG_BLACK}{net_color} {net_icon} {net_status} {Colors.RESET}"
            f"{Colors.BG_BLACK}{Colors.BRIGHT_WHITE} | {Colors.RESET}"
            f"{Colors.BG_BLACK}{Colors.BRIGHT_MAGENTA} @{user} {Colors.RESET}"
        )

        return status_line

    def render_box(self) -> str:
        """Render a boxed status widget"""
        net_connected, net_status = self.sys_info.check_network_connectivity()
        net_color = Colors.BRIGHT_GREEN if net_connected else Colors.BRIGHT_RED

        os_name = self.sys_info.os_name
        time_str = self.sys_info.get_current_time()
        date_str = self.sys_info.get_current_date()
        user = self.sys_info.username
        hostname = self.sys_info.hostname

        # Build the box
        width = 50
        border_color = Colors.BRIGHT_CYAN

        lines = [
            f"{border_color}{'─' * width}{Colors.RESET}",
            f"{Colors.BRIGHT_YELLOW}  System Status{Colors.RESET}",
            f"{border_color}{'─' * width}{Colors.RESET}",
            f"  {Colors.CYAN}OS:{Colors.RESET}       {Colors.BRIGHT_WHITE}{os_name}{Colors.RESET}",
            f"  {Colors.CYAN}User:{Colors.RESET}     {Colors.BRIGHT_MAGENTA}{user}{Colors.RESET}@{Colors.DIM}{hostname}{Colors.RESET}",
            f"  {Colors.CYAN}Date:{Colors.RESET}     {Colors.BRIGHT_WHITE}{date_str}{Colors.RESET}",
            f"  {Colors.CYAN}Time:{Colors.RESET}     {Colors.BRIGHT_WHITE}{time_str}{Colors.RESET}",
            f"  {Colors.CYAN}Network:{Colors.RESET}  {net_color}{net_status}{Colors.RESET}",
            f"{border_color}{'─' * width}{Colors.RESET}",
        ]

        return '\n'.join(lines)

    def render_minimal(self) -> str:
        """Render minimal status info"""
        net_connected, _ = self.sys_info.check_network_connectivity()
        net_icon = self.get_network_icon(net_connected)
        net_color = Colors.BRIGHT_GREEN if net_connected else Colors.BRIGHT_RED

        time_str = self.sys_info.get_current_time()
        user = self.sys_info.username

        return (
            f"{Colors.DIM}[{Colors.RESET}"
            f"{Colors.BRIGHT_CYAN}{self.get_os_icon()}{Colors.RESET}"
            f"{Colors.DIM}]{Colors.RESET} "
            f"{Colors.BRIGHT_WHITE}{time_str}{Colors.RESET} "
            f"{net_color}{net_icon}{Colors.RESET} "
            f"{Colors.BRIGHT_MAGENTA}@{user}{Colors.RESET}"
        )


# Global instance for easy access
_system_info = None

def get_system_info() -> SystemInfo:
    """Get the global SystemInfo instance"""
    global _system_info
    if _system_info is None:
        _system_info = SystemInfo()
    return _system_info


def get_status_widget() -> StatusWidget:
    """Get a StatusWidget instance"""
    return StatusWidget(get_system_info())


# Quick access functions
def is_windows() -> bool:
    return get_system_info().is_windows

def is_macos() -> bool:
    return get_system_info().is_macos

def is_linux() -> bool:
    return get_system_info().is_linux

def get_os_type() -> str:
    return get_system_info().os_type

def get_home_dir() -> str:
    return get_system_info().home_directory


if __name__ == "__main__":
    # Test the module
    info = SystemInfo()
    widget = StatusWidget(info)

    print("\n=== System Information ===")
    print(f"OS Type: {info.os_type}")
    print(f"OS Name: {info.os_name}")
    print(f"OS Version: {info.os_version}")
    print(f"Username: {info.username}")
    print(f"Hostname: {info.hostname}")
    print(f"Home: {info.home_directory}")

    print("\n=== Status Widget (Box) ===")
    print(widget.render_box())

    print("\n=== Status Widget (Compact) ===")
    print(widget.render_compact())

    print("\n=== Status Widget (Minimal) ===")
    print(widget.render_minimal())

    print("\n=== Common Directories ===")
    for name, path in info.get_common_directories().items():
        exists = "[OK]" if os.path.exists(path) else "[--]"
        print(f"  {exists} {name}: {path}")
