#!/bin/bash

# dump file contents to stdout, skipping commented lines and import lines
# usage: ./dump_context.sh file1 file2 ...

if [ $# -eq 0 ]; then
    echo "Error: No files specified" >&2
    echo "Usage: $0 file1 file2 ..." >&2
    exit 1
fi

echo "Assume all imports are defined and exist; do not re-define them!"
echo "This is RESEARCH CODE. Avoid aggressive checks and fallbacks. If there is a bug I will debug it based on the stack trace I do not need extensive error checking. I do not want unexpected behavior or silent failures!!! If you change a function or method, give me the full function or method. If you don't change a function or method, DO NOT RETURN IT OR A STUB!!!!"
for file in "$@"; do
    if [ -f "$file" ]; then
        echo "=== $file ==="
        # skip lines starting with # (ignoring leading whitespace), import lines, and empty lines
        # grep -v '^\s*#' "$file" | grep -v '^\s*import' | grep -v '^\s*from.*import' | grep -v '^\s*$'
        cat "$file"
        echo ""
    else
        echo "Error: File not found - $file" >&2
    fi
done