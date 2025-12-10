#!/bin/bash
# Commit final 200-aggregation test results when test completes

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "Committing Final 200-Aggregation Results"
echo "=========================================="
echo ""

# Find the latest versioned directory
LATEST_DIR=$(find future_works/logs -type d -name "run_200agg_*" -maxdepth 1 | sort | tail -1)

if [ -z "$LATEST_DIR" ]; then
    echo "❌ No versioned test directory found!"
    exit 1
fi

echo "📁 Latest test directory: $LATEST_DIR"
echo ""

# Check if test is complete (has SUMMARY.txt)
if [ -f "$LATEST_DIR/SUMMARY.txt" ]; then
    echo "✅ Test appears to be COMPLETE"
    echo ""
    echo "Summary:"
    cat "$LATEST_DIR/SUMMARY.txt"
    echo ""
else
    echo "⚠️  Test may still be running (no SUMMARY.txt found)"
    echo "   Proceed anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Aborted."
        exit 1
    fi
fi

# Stage the results
echo "📦 Staging results..."
git add future_works/logs/run_200agg_*
git add future_works/logs/run_200agg_output.log 2>/dev/null

# Show what will be committed
echo ""
echo "Files to be committed:"
git status --short future_works/logs/ | grep "run_200agg"

echo ""
echo "Commit message will be:"
echo "  'Add 200-aggregation test results for improved TrustWeight'"
echo ""
echo "Proceed with commit? (y/n)"
read -r response

if [ "$response" = "y" ]; then
    TIMESTAMP=$(basename "$LATEST_DIR" | sed 's/run_200agg_//')
    git commit -m "Add 200-aggregation test results for improved TrustWeight

- Complete test run with auto-tuning and compression
- Versioned results in: $(basename "$LATEST_DIR")
- Test timestamp: $TIMESTAMP
- All evaluation and participation logs included"
    
    echo ""
    echo "✅ Final results committed successfully!"
    echo ""
    echo "To push to remote:"
    echo "  git push origin <branch-name>"
else
    echo "Aborted."
    exit 1
fi

