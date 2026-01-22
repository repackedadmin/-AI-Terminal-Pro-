"""
App Integration Module for AI Terminal Pro
Provides integration between the main App and TUI components
"""

from .components.help_handler import EnhancedHelpHandler


def setup_tui_features(app, use_textual=False):
    """
    Setup TUI features for the main application

    Args:
        app: The main App instance
        use_textual: Whether to use full Textual UI (False for enhanced help only)

    Returns:
        The app instance with TUI features attached
    """
    # Initialize enhanced help handler
    try:
        app._help_handler = EnhancedHelpHandler(app)
    except Exception as e:
        app._help_handler = None
        print(f"Warning: Enhanced help handler failed to initialize: {e}")

    # Set flags for TUI availability
    app.use_textual = use_textual
    app.textual_available = False  # Textual menu disabled, using text-based

    return app


def get_help_handler(app=None):
    """
    Get or create an EnhancedHelpHandler instance

    Args:
        app: Optional App instance for context

    Returns:
        EnhancedHelpHandler instance
    """
    return EnhancedHelpHandler(app)


__all__ = ['setup_tui_features', 'get_help_handler']
