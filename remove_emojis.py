"""
Script to remove all emojis from markdown and python files
"""

import os
import re

def remove_emojis_from_file(file_path):
    """Remove emojis from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Common emoji patterns in our files
        emoji_patterns = [
            r'🎯', r'✅', r'⭐', r'📊', r'🚀', r'❌', r'⚠️', r'💡', r'📋', 
            r'🎉', r'🏆', r'📈', r'🔍', r'📁', r'🧪', r'🤖', r'📺', r'🎬', 
            r'🔧', r'🎓', r'💯', r'🔥', r'📄', r'🌟', r'✨', r'🛠️', r'🎨',
            r'📦', r'🗂️', r'🔗', r'🎮', r'🎯', r'🧠', r'💻', r'🔬', r'📖',
            r'🚗', r'🏃', r'👥', r'🔒', r'🛡️', r'⚡', r'🎪', r'🎭', r'🎸',
            r'🎲', r'🎯', r'🏁', r'🎊', r'🎈', r'🌈', r'⭐', r'🌟', r'💫'
        ]
        
        # Remove each emoji pattern
        for emoji in emoji_patterns:
            content = re.sub(emoji + r'\s*', '', content)
        
        # Clean up any double spaces or formatting issues
        content = re.sub(r'\n\n\n+', '\n\n', content)  # Remove triple+ newlines
        content = re.sub(r'  +', ' ', content)  # Remove multiple spaces
        
        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"Cleaned: {file_path}")
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    """Remove emojis from all relevant files"""
    
    # Files to clean
    files_to_clean = [
        'COMPLETION_SUMMARY.md',
        'OPTIMIZATION_REPORT.md', 
        'OPTIMIZATION_ANALYSIS.md',
        'PROJECT_REPORT.md',
        'QUICK_REFERENCE.md',
        'TEST_RESULTS.md',
        'ARCHITECTURE.md',
        'FIXES_SUMMARY.md',
        'test_script.py',
        'enhanced_test.py',
        'optimized_test.py',
        'final_optimized_test.py'
    ]
    
    print("Removing emojis from project files...")
    
    for filename in files_to_clean:
        if os.path.exists(filename):
            remove_emojis_from_file(filename)
        else:
            print(f"File not found: {filename}")
    
    print("\nEmoji removal complete!")

if __name__ == '__main__':
    main()