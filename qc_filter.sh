#!/bin/bash
# qc_filter.sh - Quick launcher for QC'ing specific filter categories

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

FILTER_DIR="data/filter_tests"

# Check if filter tests exist
if [ ! -d "$FILTER_DIR" ]; then
    echo -e "${YELLOW}No filter tests found. Running test_positioning_filters.py...${NC}"
    eval "$(micromamba shell hook -s bash)"
    micromamba activate prima
    python test_positioning_filters.py
fi

# List available filters
echo -e "${BLUE}Available filters:${NC}"
echo ""

filters=()
i=1
for f in "$FILTER_DIR"/*.txt; do
    if [ -f "$f" ]; then
        basename=$(basename "$f" .txt)
        count=$(grep -v "^#" "$f" | wc -l)
        echo "  $i) $(echo $basename | sed 's/_/ /g') ($count exams)"
        filters+=("$f")
        ((i++))
    fi
done

echo ""
echo -e "${GREEN}Select filter number (or 'q' to quit):${NC} "
read choice

if [ "$choice" = "q" ]; then
    echo "Cancelled."
    exit 0
fi

# Validate choice
if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#filters[@]}" ]; then
    echo "Invalid choice"
    exit 1
fi

# Get selected filter file
selected_filter="${filters[$((choice-1))]}"

echo ""
echo -e "${GREEN}Starting QC server for: $(basename "$selected_filter")${NC}"
echo ""

# Activate environment and start server
eval "$(micromamba shell hook -s bash)"
micromamba activate prima

python qc/qc_gallery.py --serve --exam-list "$selected_filter"
