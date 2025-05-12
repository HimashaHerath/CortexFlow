#!/usr/bin/env python3
"""
Setup script for the CortexFlow Evaluation Framework.

This script ensures all necessary directories and dependencies are in place
for running the evaluation framework.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Add parent directory to sys.path to import cortexflow modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("setup_evaluation")

def setup_directories(
    results_dir: str = "evaluation_results",
    logs_dir: str = None,
    create_subdirs: bool = True
):
    """
    Set up directories for the evaluation framework.
    
    Args:
        results_dir: Directory for evaluation results
        logs_dir: Directory for logs (default: results_dir/logs)
        create_subdirs: Whether to create benchmark-specific subdirectories
    """
    # Create main results directory
    os.makedirs(results_dir, exist_ok=True)
    logger.info(f"Created main results directory: {results_dir}")
    
    # Create logs directory
    if logs_dir is None:
        logs_dir = os.path.join(results_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger.info(f"Created logs directory: {logs_dir}")
    
    # Create benchmark-specific subdirectories if requested
    if create_subdirs:
        subdirs = [
            "multi_hop_benchmarks",
            "consistency_evaluations",
            "reasoning_analysis",
            "test_suites",
            "visualizations"
        ]
        
        for subdir in subdirs:
            subdir_path = os.path.join(results_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            logger.info(f"Created subdirectory: {subdir_path}")

def create_default_config(
    config_path: str = "evaluation_config.json",
    db_path: str = None,
    overwrite: bool = False
):
    """
    Create a default configuration file for the evaluation framework.
    
    Args:
        config_path: Path to the configuration file
        db_path: Path to the knowledge database
        overwrite: Whether to overwrite existing config
    
    Returns:
        True if config was created, False otherwise
    """
    # Check if config already exists
    if os.path.exists(config_path) and not overwrite:
        logger.info(f"Configuration file already exists at {config_path}")
        return False
    
    # Create default configuration
    config = {
        "results_dir": "evaluation_results",
        "db_path": db_path,
        "reasoning_log_db": "evaluation_results/reasoning_logs.db",
        "benchmarks": {
            "multi_hop": {
                "num_tests": 30,
                "synthetic_ratio": 0.5,
                "max_hops": 3
            },
            "consistency": {
                "time_window_days": 30,
                "take_snapshot": True
            },
            "reasoning": {
                "n_sessions": 10
            }
        },
        "visualization": {
            "enabled": True,
            "formats": ["png", "html"],
            "interactive": True
        },
        "test_generation": {
            "num_multi_hop": 20,
            "num_counterfactual": 10,
            "custom_domains": []
        },
        "logging": {
            "level": "INFO",
            "file_logging": True,
            "console_logging": True
        },
        "created_at": datetime.now().isoformat()
    }
    
    # Write configuration to file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created default configuration at {config_path}")
    return True

def check_dependencies():
    """
    Check if all required dependencies are installed.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    required_packages = [
        "numpy",
        "matplotlib",
        "networkx",
        "sqlite3"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
        logger.warning("Install missing packages with: pip install " + " ".join(missing_packages))
        return False
    
    logger.info("All required dependencies are installed")
    return True

def initialize_database(db_path: str):
    """
    Initialize the evaluation database if it doesn't exist.
    
    Args:
        db_path: Path to the database
    
    Returns:
        True if initialization was successful, False otherwise
    """
    if not db_path:
        logger.warning("No database path provided")
        return False
    
    try:
        import sqlite3
        
        # Check if database already exists
        if os.path.exists(db_path):
            logger.info(f"Database already exists at {db_path}")
            return True
        
        # Create database directory if needed
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        # Create database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create basic tables - actual schema will be created by the framework
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            value TEXT,
            timestamp REAL
        )
        ''')
        
        # Add initialization timestamp
        cursor.execute(
            "INSERT INTO evaluation_metadata (key, value, timestamp) VALUES (?, ?, ?)",
            ("initialized", "true", datetime.now().timestamp())
        )
        
        conn.commit()
        conn.close()
        
        logger.info(f"Initialized database at {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False

def main():
    """Main function to set up the evaluation framework."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Setup CortexFlow Evaluation Framework')
    parser.add_argument('--results_dir', type=str, default='evaluation_results', help='Directory for evaluation results')
    parser.add_argument('--logs_dir', type=str, help='Directory for logs')
    parser.add_argument('--db_path', type=str, help='Path to knowledge database')
    parser.add_argument('--config', type=str, default='evaluation_config.json', help='Path to configuration file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing configuration')
    parser.add_argument('--no_subdirs', action='store_true', help='Do not create benchmark-specific subdirectories')
    
    args = parser.parse_args()
    
    # Check dependencies
    dependencies_ok = check_dependencies()
    if not dependencies_ok:
        logger.warning("Some dependencies are missing. Continuing setup but framework may not work.")
    
    # Setup directories
    setup_directories(args.results_dir, args.logs_dir, not args.no_subdirs)
    
    # Create default configuration
    create_default_config(args.config, args.db_path, args.overwrite)
    
    # Initialize database if path provided
    if args.db_path:
        initialize_database(args.db_path)
    
    logger.info("Setup completed successfully")
    logger.info(f"To run the evaluation framework, use: python benchmark/evaluation_framework.py --config {args.config}")

if __name__ == "__main__":
    main() 