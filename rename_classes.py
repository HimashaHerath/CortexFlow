#!/usr/bin/env python3
import os
import re

def rename_classes_in_file(file_path):
    """Replace class names from 'AdaptiveContext' to 'CortexFlow' patterns."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Define patterns to replace
        patterns = [
            (r'class CortexFlowConfig', 'class CortexFlowConfig'),
            (r'class CortexFlowManager', 'class CortexFlowManager'),
            # For typehints and instantiations
            (r"CortexFlowConfig", "CortexFlowConfig"),
            (r"CortexFlowManager", "CortexFlowManager"),
            # For imports
            (r"from cortexflow import CortexFlowManager, CortexFlowConfig", 
             "from cortexflow import CortexFlowManager, CortexFlowConfig"),
            (r"from cortexflow.manager import CortexFlowManager", 
             "from cortexflow.manager import CortexFlowManager"),
            (r"from cortexflow.config import CortexFlowConfig", 
             "from cortexflow.config import CortexFlowConfig"),
        ]
        
        updated_content = content
        for old_pattern, new_pattern in patterns:
            updated_content = re.sub(old_pattern, new_pattern, updated_content)
        
        if content != updated_content:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(updated_content)
            print(f"Updated classes in: {file_path}")
        
        return True
    except Exception as e:
        print(f"Error updating classes in {file_path}: {e}")
        return False

def process_directory(directory):
    """Process all Python files in a directory recursively."""
    success_count = 0
    failure_count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py') or file.endswith('.md'):
                file_path = os.path.join(root, file)
                if rename_classes_in_file(file_path):
                    success_count += 1
                else:
                    failure_count += 1
    
    return success_count, failure_count

if __name__ == "__main__":
    # Directories to process
    directories = ["cortexflow", "tests", "benchmark", "docs"]
    py_files = [f for f in os.listdir(".") if f.endswith(".py") or f.endswith(".md")]
    
    total_success = 0
    total_failure = 0
    
    # Process Python files in root directory
    for py_file in py_files:
        if rename_classes_in_file(py_file):
            total_success += 1
        else:
            total_failure += 1
    
    # Process directories
    for directory in directories:
        if os.path.exists(directory):
            success, failure = process_directory(directory)
            total_success += success
            total_failure += failure
        else:
            print(f"Directory not found: {directory}")
    
    print(f"\nSummary:")
    print(f"  Successfully updated class names in: {total_success} files")
    print(f"  Failed to update class names in: {total_failure} files") 