#!/usr/bin/env python3
import os
import re

def update_imports_in_file(file_path):
    """Replace 'adaptive_context' with 'cortexflow' in import statements."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Replace import statements
        updated_content = re.sub(
            r'from adaptive_context', 
            'from cortexflow', 
            content
        )
        updated_content = re.sub(
            r'import cortexflow', 
            'import cortexflow', 
            content
        )
        
        # Replace logging statements
        updated_content = re.sub(
            r"logging.getLogger\('adaptive_context'\)", 
            "logging.getLogger('cortexflow')", 
            updated_content
        )
        
        if content != updated_content:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(updated_content)
            print(f"Updated: {file_path}")
        
        return True
    except Exception as e:
        print(f"Error updating {file_path}: {e}")
        return False

def process_directory(directory):
    """Process all Python files in a directory recursively."""
    success_count = 0
    failure_count = 0
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if update_imports_in_file(file_path):
                    success_count += 1
                else:
                    failure_count += 1
    
    return success_count, failure_count

if __name__ == "__main__":
    # Directories to process
    directories = ["cortexflow", "tests", "benchmark", "docs"]
    py_files = [f for f in os.listdir(".") if f.endswith(".py")]
    
    total_success = 0
    total_failure = 0
    
    # Process Python files in root directory
    for py_file in py_files:
        if update_imports_in_file(py_file):
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
    print(f"  Successfully updated: {total_success} files")
    print(f"  Failed to update: {total_failure} files") 