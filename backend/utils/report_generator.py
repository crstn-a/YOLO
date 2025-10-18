"""
Enhanced PDF report generator for YOLOv11 Fingerspelling Benchmark
"""
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
import datetime
import json
import os
from typing import Dict, Any

def generate_pdf_report(results: Dict[str, Any]) -> str:
    """Generate comprehensive PDF benchmark report"""
    
    # Create report filename
    timestamp = int(datetime.datetime.now().timestamp())
    report_name = f"results/reports/benchmark_report_{timestamp}.pdf"
    os.makedirs("results/reports", exist_ok=True)
    
    # Create document
    doc = SimpleDocTemplate(report_name, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("YOLOv11 Fingerspelling Benchmark Report", styles["Title"]))
    story.append(Spacer(1, 20))
    
    # Report metadata
    if "benchmark_info" in results:
        info = results["benchmark_info"]
        story.append(Paragraph(f"<b>Report Generated:</b> {datetime.datetime.fromtimestamp(info.get('timestamp', timestamp)).strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
        story.append(Paragraph(f"<b>Benchmark Duration:</b> {info.get('duration', 0):.2f} seconds", styles["Normal"]))
        story.append(Spacer(1, 20))
    
    # Research Grade Assessment
    if "research_grade_assessment" in results:
        assessment = results["research_grade_assessment"]
        story.append(Paragraph("🏆 Research Grade Assessment", styles["Heading1"]))
        
        # Create assessment table
        assessment_data = [
            ["Criteria", "Required", "Achieved", "Status"],
            ["Static Accuracy", "≥ 90%", f"{assessment.get('criteria_details', {}).get('static_accuracy', {}).get('achieved', 0):.1%}", 
             "✅ PASS" if assessment.get('static_accuracy_passes', False) else "❌ FAIL"],
            ["Dynamic Accuracy", "≥ 85%", f"{assessment.get('criteria_details', {}).get('dynamic_accuracy', {}).get('achieved', 0):.1%}", 
             "✅ PASS" if assessment.get('dynamic_accuracy_passes', False) else "❌ FAIL"],
            ["FPS Performance", "≥ 25", f"{assessment.get('criteria_details', {}).get('fps', {}).get('achieved', 0):.1f}", 
             "✅ PASS" if assessment.get('fps_passes', False) else "❌ FAIL"],
            ["DTW Similarity", "≥ 70%", f"{assessment.get('criteria_details', {}).get('dtw_similarity', {}).get('achieved', 0):.1%}", 
             "✅ PASS" if assessment.get('dtw_similarity_passes', False) else "❌ FAIL"]
        ]
        
        assessment_table = Table(assessment_data)
        assessment_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(assessment_table)
        story.append(Spacer(1, 20))
        
        # Overall assessment
        overall_status = "✅ RESEARCH GRADE READY" if assessment.get('research_grade_ready', False) else "❌ NOT READY FOR DEPLOYMENT"
        story.append(Paragraph(f"<b>Overall Assessment:</b> {overall_status}", styles["Heading2"]))
        story.append(Spacer(1, 20))
    
    # Static Evaluation Results
    if "static_evaluation" in results and "error" not in results["static_evaluation"]:
        static_results = results["static_evaluation"]
        story.append(Paragraph("📊 Static Fingerspelling Evaluation (A-I, K-Y)", styles["Heading1"]))
        
        if "metrics" in static_results:
            metrics = static_results["metrics"]
            story.append(Paragraph(f"<b>Overall Accuracy:</b> {metrics.get('overall_accuracy', 0):.3f}", styles["Normal"]))
            story.append(Paragraph(f"<b>Precision:</b> {metrics.get('overall_precision', 0):.3f}", styles["Normal"]))
            story.append(Paragraph(f"<b>Recall:</b> {metrics.get('overall_recall', 0):.3f}", styles["Normal"]))
            story.append(Paragraph(f"<b>F1-Score:</b> {metrics.get('overall_f1_score', 0):.3f}", styles["Normal"]))
            story.append(Paragraph(f"<b>Samples Evaluated:</b> {static_results.get('samples_evaluated', 0)}", styles["Normal"]))
            story.append(Paragraph(f"<b>Meets Threshold (90%):</b> {'✅ YES' if metrics.get('meets_threshold', False) else '❌ NO'}", styles["Normal"]))
        
        story.append(Spacer(1, 20))
    
    # Dynamic Evaluation Results
    if "dynamic_evaluation" in results and "error" not in results["dynamic_evaluation"]:
        dynamic_results = results["dynamic_evaluation"]
        story.append(Paragraph("🔄 Dynamic Gesture Evaluation (J, Z)", styles["Heading1"]))
        
        if "metrics" in dynamic_results:
            metrics = dynamic_results["metrics"]
            story.append(Paragraph(f"<b>Sequence Accuracy:</b> {metrics.get('sequence_accuracy', 0):.3f}", styles["Normal"]))
            story.append(Paragraph(f"<b>DTW Similarity:</b> {metrics.get('dtw_similarity', 0):.3f}", styles["Normal"]))
            story.append(Paragraph(f"<b>Completion Rate:</b> {metrics.get('completion_rate', 0):.3f}", styles["Normal"]))
            story.append(Paragraph(f"<b>Sequences Evaluated:</b> {metrics.get('sequences_evaluated', 0)}", styles["Normal"]))
            story.append(Paragraph(f"<b>Meets Threshold (85%):</b> {'✅ YES' if metrics.get('meets_threshold', False) else '❌ NO'}", styles["Normal"]))
        
        story.append(Spacer(1, 20))
    
    # Efficiency Evaluation Results
    if "efficiency_evaluation" in results and "error" not in results["efficiency_evaluation"]:
        efficiency_results = results["efficiency_evaluation"]
        story.append(Paragraph("⚡ Real-time Performance Evaluation", styles["Heading1"]))
        
        if "metrics" in efficiency_results:
            metrics = efficiency_results["metrics"]
            story.append(Paragraph(f"<b>Average FPS:</b> {metrics.get('average_fps', 0):.2f}", styles["Normal"]))
            story.append(Paragraph(f"<b>Min FPS:</b> {metrics.get('min_fps', 0):.2f}", styles["Normal"]))
            story.append(Paragraph(f"<b>Max FPS:</b> {metrics.get('max_fps', 0):.2f}", styles["Normal"]))
            story.append(Paragraph(f"<b>Average Latency:</b> {metrics.get('average_latency', 0)*1000:.2f} ms", styles["Normal"]))
            story.append(Paragraph(f"<b>Real-time Ready:</b> {'✅ YES' if metrics.get('real_time_ready', False) else '❌ NO'}", styles["Normal"]))
            story.append(Paragraph(f"<b>Meets Target FPS (25):</b> {'✅ YES' if metrics.get('meets_target_fps', False) else '❌ NO'}", styles["Normal"]))
        
        story.append(Spacer(1, 20))
    
    # Configuration Summary
    story.append(Paragraph("⚙️ Benchmark Configuration", styles["Heading1"]))
    story.append(Paragraph(f"<b>Static Threshold:</b> 90%", styles["Normal"]))
    story.append(Paragraph(f"<b>Dynamic Threshold:</b> 85%", styles["Normal"]))
    story.append(Paragraph(f"<b>Target FPS:</b> 25", styles["Normal"]))
    story.append(Paragraph(f"<b>DTW Threshold:</b> 70%", styles["Normal"]))
    story.append(Spacer(1, 20))
    
    # Limitations and Scope
    story.append(Paragraph("📋 Benchmark Scope & Limitations", styles["Heading1"]))
    story.append(Paragraph("• <b>Scope:</b> ASL Fingerspelling A-Z only", styles["Normal"]))
    story.append(Paragraph("• <b>Static Letters:</b> A-I, K-Y (24 letters)", styles["Normal"]))
    story.append(Paragraph("• <b>Dynamic Letters:</b> J, Z (2 letters)", styles["Normal"]))
    story.append(Paragraph("• <b>Limitations:</b> Does not cover full ASL vocabulary", styles["Normal"]))
    story.append(Paragraph("• <b>Hardware Dependency:</b> Performance varies by CPU/GPU", styles["Normal"]))
    story.append(Spacer(1, 20))
    
    # Footer
    story.append(Paragraph("Generated by YOLOv11 Fingerspelling Benchmark Framework", styles["Normal"]))
    story.append(Paragraph(f"Report ID: {timestamp}", styles["Normal"]))
    
    # Build PDF
    doc.build(story)
    return report_name

def generate_summary_report(results: Dict[str, Any]) -> str:
    """Generate a summary report for quick overview"""
    
    timestamp = int(datetime.datetime.now().timestamp())
    report_name = f"results/reports/summary_{timestamp}.pdf"
    os.makedirs("results/reports", exist_ok=True)
    
    doc = SimpleDocTemplate(report_name, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("YOLOv11 Fingerspelling - Quick Summary", styles["Title"]))
    story.append(Spacer(1, 20))
    
    # Key metrics table
    summary_data = [["Metric", "Value", "Status"]]
    
    # Add static results
    if "static_evaluation" in results and "error" not in results["static_evaluation"]:
        static_acc = results["static_evaluation"]["metrics"].get("overall_accuracy", 0)
        summary_data.append(["Static Accuracy", f"{static_acc:.1%}", "✅" if static_acc >= 0.9 else "❌"])
    
    # Add dynamic results
    if "dynamic_evaluation" in results and "error" not in results["dynamic_evaluation"]:
        dynamic_acc = results["dynamic_evaluation"]["metrics"].get("sequence_accuracy", 0)
        summary_data.append(["Dynamic Accuracy", f"{dynamic_acc:.1%}", "✅" if dynamic_acc >= 0.85 else "❌"])
    
    # Add efficiency results
    if "efficiency_evaluation" in results and "error" not in results["efficiency_evaluation"]:
        fps = results["efficiency_evaluation"]["metrics"].get("average_fps", 0)
        summary_data.append(["Average FPS", f"{fps:.1f}", "✅" if fps >= 25 else "❌"])
    
    # Overall assessment
    if "research_grade_assessment" in results:
        ready = results["research_grade_assessment"].get("research_grade_ready", False)
        summary_data.append(["Research Grade Ready", "YES" if ready else "NO", "✅" if ready else "❌"])
    
    summary_table = Table(summary_data)
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    doc.build(story)
    return report_name
