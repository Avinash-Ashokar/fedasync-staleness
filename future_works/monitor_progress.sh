#!/bin/bash
# Monitor progress of 200-aggregation test

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT_LOG="$LOG_DIR/run_200agg_output.log"
VERSIONED_DIRS=$(find "$LOG_DIR" -type d -name "run_200agg_*" 2>/dev/null | sort)

echo "=========================================="
echo "200-Aggregation Test Progress Monitor"
echo "=========================================="
echo ""

# Check if test is running
if pgrep -f "run_200_aggregations.py" > /dev/null; then
    echo "✅ Test is RUNNING"
    echo ""
    
    # Show latest output
    if [ -f "$OUTPUT_LOG" ]; then
        echo "Latest output (last 20 lines):"
        echo "----------------------------------------"
        tail -20 "$OUTPUT_LOG" 2>/dev/null || echo "No output yet..."
        echo "----------------------------------------"
    fi
else
    echo "❌ Test is NOT running"
    echo ""
fi

# Show versioned directories
if [ -n "$VERSIONED_DIRS" ]; then
    echo "Versioned result directories:"
    for dir in $VERSIONED_DIRS; do
        echo "  📁 $dir"
        if [ -f "$dir/SUMMARY.txt" ]; then
            echo "     ✅ Complete"
            echo "     $(head -5 "$dir/SUMMARY.txt" | tail -1)"
        else
            echo "     ⏳ In progress..."
        fi
    done
else
    echo "No versioned directories found yet"
fi

echo ""
echo "=========================================="

