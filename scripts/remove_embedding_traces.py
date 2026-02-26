#!/usr/bin/env python3
"""Script to remove tracing from embedding methods."""

import re

file_path = "ingestion/embedding_service.py"

with open(file_path, 'r') as f:
    content = f.read()

# Pattern to match trace_operation blocks
# This will match: with trace_operation(...) as trace: ... try: ... except: ...
pattern = r'(\s+)with trace_operation\([^)]+(?:\n[^)]+)*\) as trace:\n(\1    try:(?:.*?\n)*?\1        return .*?\n\n\1    except Exception as e:(?:.*?\n)*?\1        return self\._get_mock_embedding\([^)]*\))'

def fix_indentation(match):
    indent = match.group(1)
    body = match.group(2)
    # Remove one level of indentation from the body
    fixed = body.replace(f'\n{indent}    ', f'\n{indent}')
    # Remove trace.update calls
    fixed = re.sub(r'\n.*?if trace:.*?\n.*?trace\.update\([^)]+\)', '', fixed, flags=re.DOTALL)
    return fixed

content = re.sub(pattern, fix_indentation, content, flags=re.DOTALL)

with open(file_path, 'w') as f:
    f.write(content)

print(f"✓ Removed tracing from {file_path}")
