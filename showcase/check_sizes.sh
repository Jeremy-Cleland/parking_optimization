#!/bin/bash
# Check file sizes in showcase directory for upload planning

echo "📁 Showcase Directory File Sizes"
echo "=================================="

if [ -d "showcase/latest_run" ]; then
    echo ""
    echo "📊 Selected Visualizations:"
    du -h showcase/latest_run/selected_visuals/* 2>/dev/null | sort -hr

    echo ""
    echo "📋 Data Files:"
    du -h showcase/latest_run/*.json 2>/dev/null

    echo ""
    echo "📈 Total Size:"
    du -sh showcase/latest_run

    echo ""
    echo "🎯 Recommended for Upload:"
    echo "  • Interactive Map: interactive_parking_map.html"
    echo "  • Dashboard: summary_dashboard.png"
    echo "  • Network Analysis: network_analysis_map.png"
    echo "  • Performance: performance_metrics.png"
    echo "  • Metadata: metadata.json"

    echo ""
    echo "💡 For presentations, focus on:"
    echo "  • summary_dashboard.png (executive overview)"
    echo "  • geographic_dashboard.png (multi-panel analysis)"
    echo "  • algorithm_complexity.png (technical depth)"

else
    echo "❌ No showcase data found."
    echo "Run 'make showcase' first to prepare files."
fi

echo ""
echo "🚀 Upload Command Examples:"
echo "  Git: git add showcase/ && git commit -m 'Add showcase results'"
echo "  Share: zip -r parking_showcase.zip showcase/"
