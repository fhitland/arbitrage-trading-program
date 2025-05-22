"""
Simple logging configuration for the real-time arbitrage system.
"""

import logging
import sys

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    # Configure root logger
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.DEBUG)
        
        # Configure logging format
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.DEBUG)
        root.addHandler(console_handler)
    
    # Get logger for this module
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    return logger