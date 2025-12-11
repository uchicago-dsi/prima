#!/bin/bash
# Remove old gallery to force regeneration
rm -f debug_viz/gallery.html

# Regenerate gallery with a small sample
python visualize_gallery.py --max-exams 2

echo "---"
echo "Checking generated JavaScript for syntax errors..."
if [ -f debug_viz/gallery.html ]; then
    # Check for the specific line that was causing issues
    echo "Sample of generated exam data:"
    grep -A 5 "const allExams = \[" debug_viz/gallery.html | head -20
fi
