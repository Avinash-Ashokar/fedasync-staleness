#!/bin/bash
# Commit all test results and documentation

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "Committing Future Works Test Results"
echo "=========================================="
echo ""

# Check if test is still running
if pgrep -f "run_200_aggregations.py" > /dev/null; then
    echo "⚠️  WARNING: Test is still running!"
    echo "   Do you want to commit current results anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Aborted."
        exit 1
    fi
fi

# Stage all future_works files
echo "📦 Staging files..."
git add future_works/

# Check what will be committed
echo ""
echo "Files to be committed:"
git status --short future_works/ | head -20

if [ $(git status --short future_works/ | wc -l) -gt 20 ]; then
    echo "... and more"
fi

echo ""
echo "Commit message will be:"
echo "  'Add future works: improved TrustWeight with auto-tuning and compression'"
echo "  '  - Extended test results (200 aggregations)'"
echo "  '  - Versioned test logs'"
echo "  - Updated documentation'"
echo ""
echo "Proceed with commit? (y/n)"
read -r response

if [ "$response" = "y" ]; then
    git commit -m "Add future works: improved TrustWeight with auto-tuning and compression

- Extended test results (200 aggregations)
- Versioned test logs in future_works/logs/
- Auto-tuning parameter evolution analysis
- Compression effectiveness validation
- Updated documentation and analysis files"
    
    echo ""
    echo "✅ Committed successfully!"
    echo ""
    echo "To push to remote:"
    echo "  git push origin <branch-name>"
else
    echo "Aborted."
    exit 1
fi

