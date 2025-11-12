#!/usr/bin/env python3
"""
Add CHI_REPO_PATH environment variable to all ctest definitions.
"""

import re
import sys

def add_chi_repo_path_to_tests(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()

    # Pattern to match add_test blocks
    # Match: add_test(\n    NAME test_name\n    ...\n  )
    pattern = r'(add_test\(\s*\n\s*NAME\s+(\w+)\s*\n(?:.*?\n)*?\s*\))'

    def replace_test(match):
        full_match = match.group(1)
        test_name = match.group(2)

        # Check if this test already has set_tests_properties for CHI_REPO_PATH
        # by looking ahead in the content
        idx = match.end()
        next_lines = content[idx:idx+200]

        if f'set_tests_properties({test_name}' in next_lines and 'CHI_REPO_PATH' in next_lines:
            # Already has the property
            return full_match

        # Add set_tests_properties after the add_test block
        properties_block = f'''  set_tests_properties({test_name} PROPERTIES
    ENVIRONMENT "CHI_REPO_PATH=${{CMAKE_BINARY_DIR}}/bin"
  )
'''
        return full_match + '\n' + properties_block

    # Replace all matches
    updated_content = re.sub(pattern, replace_test, content, flags=re.MULTILINE)

    with open(output_file, 'w') as f:
        f.write(updated_content)

    print(f"Updated {output_file}")

if __name__ == '__main__':
    input_path = '/home/llogan/Documents/Projects/iowarp/iowarp-runtime/test/unit/CMakeLists.txt'
    output_path = input_path  # Overwrite the same file

    add_chi_repo_path_to_tests(input_path, output_path)
