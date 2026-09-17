"""
PDF Report Generator for RoadVision AI
Uses ReportLab to generate professional inspection reports.
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    Image as RLImage, PageBreak, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from datetime import datetime
from pathlib import Path
import io

# ============================================================================
# COLOR PALETTE (matches app theme)
# ============================================================================
WARM_BG = colors.HexColor("#f6efe8")
WARM_CARD = colors.HexColor("#faf5ef")
WARM_BORDER = colors.HexColor("#e8ddd0")
TEXT_PRIMARY = colors.HexColor("#3d2c1e")
TEXT_SECONDARY = colors.HexColor("#9a8776")
ACCENT = colors.HexColor("#b8956e")
HIGH_RED = colors.HexColor("#a65a4a")
MEDIUM_AMBER = colors.HexColor("#9a7a4a")
GREEN_OK = colors.HexColor("#7a9a7a")


# ============================================================================
# STYLES
# ============================================================================
def get_styles():
    styles = getSampleStyleSheet()
    
    styles.add(ParagraphStyle(
        name='ReportTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=TEXT_PRIMARY,
        fontName='Helvetica-Bold',
        spaceAfter=4,
    ))
    
    styles.add(ParagraphStyle(
        name='ReportSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=TEXT_SECONDARY,
        fontName='Helvetica',
        spaceAfter=12,
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=TEXT_PRIMARY,
        fontName='Helvetica-Bold',
        spaceBefore=12,
        spaceAfter=6,
    ))
    
    styles.add(ParagraphStyle(
        name='CellText',
        parent=styles['Normal'],
        fontSize=8,
        textColor=TEXT_PRIMARY,
        fontName='Helvetica',
    ))
    
    styles.add(ParagraphStyle(
        name='FooterText',
        parent=styles['Normal'],
        fontSize=8,
        textColor=TEXT_SECONDARY,
        alignment=TA_CENTER,
    ))
    
    return styles


# ============================================================================
# HEADER & FOOTER
# ============================================================================
def add_header_footer(canvas, doc):
    """Draw header and footer on every page."""
    canvas.saveState()
    width, height = A4
    
    # Header line
    canvas.setStrokeColor(WARM_BORDER)
    canvas.setLineWidth(0.5)
    canvas.line(15*mm, height - 12*mm, width - 15*mm, height - 12*mm)
    
    # Header text
    canvas.setFont('Helvetica-Bold', 8)
    canvas.setFillColor(TEXT_SECONDARY)
    canvas.drawString(15*mm, height - 10*mm, "RoadVision AI")
    canvas.drawRightString(width - 15*mm, height - 10*mm, 
                            f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}")
    
    # Footer line
    canvas.line(15*mm, 12*mm, width - 15*mm, 12*mm)
    
    # Footer text
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(TEXT_SECONDARY)
    canvas.drawString(15*mm, 8*mm, "RoadVision AI · YOLOv8 · PostgreSQL")
    canvas.drawCentredString(width / 2, 8*mm, f"Page {doc.page}")
    canvas.drawRightString(width - 15*mm, 8*mm, "Confidential")
    
    canvas.restoreState()


# ============================================================================
# SUMMARY CARDS
# ============================================================================
def build_summary_table(total, high_count, medium_count, unique_classes):
    """Build summary stat cards as a table."""
    data = [
        ["Total Detections", "High Priority", "Medium Priority", "Damage Types"],
        [str(total), str(high_count), str(medium_count), str(unique_classes)],
    ]
    
    table = Table(data, colWidths=[45*mm, 45*mm, 45*mm, 45*mm])
    table.setStyle(TableStyle([
        # Header row (labels)
        ('BACKGROUND', (0, 0), (-1, 0), WARM_CARD),
        ('TEXTCOLOR', (0, 0), (-1, 0), TEXT_SECONDARY),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
        
        # Value row
        ('BACKGROUND', (0, 1), (-1, 1), colors.white),
        ('TEXTCOLOR', (0, 1), (-1, 1), TEXT_PRIMARY),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, 1), 20),
        ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
        ('TOPPADDING', (0, 1), (-1, 1), 4),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 10),
        
        # Borders
        ('BOX', (0, 0), (-1, -1), 0.5, WARM_BORDER),
        ('LINEAFTER', (0, 0), (-2, -1), 0.5, WARM_BORDER),
    ]))
    
    return table


# ============================================================================
# DETECTION TABLE
# ============================================================================
def build_detection_table(rows):
    """Build the main detection table."""
    header = ["ID", "Damage Class", "Confidence", "Severity", "Location", "Timestamp", "Device"]
    
    data = [header]
    for r in rows:
        # r = (id, class_name, confidence, severity, lat, lon, timestamp, device_id)
        id_, cls, conf, sev, lat, lon, ts, dev = r
        
        location = f"{lat:.4f}, {lon:.4f}" if lat and lat != 0 else "N/A"
        timestamp = ts.strftime("%d %b %Y, %H:%M") if ts else "N/A"
        
        data.append([
            str(id_),
            cls,
            f"{conf:.1%}",
            sev,
            location,
            timestamp,
            dev or "N/A",
        ])
    
    table = Table(
        data,
        colWidths=[12*mm, 42*mm, 22*mm, 20*mm, 32*mm, 32*mm, 20*mm],
        repeatRows=1,
    )
    
    style = [
        # Header
        ('BACKGROUND', (0, 0), (-1, 0), TEXT_PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 8.5),
        ('ALIGN', (0, 0), (-1, 0), 'LEFT'),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        
        # Body
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('TEXTCOLOR', (0, 1), (-1, -1), TEXT_PRIMARY),
        ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
        
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.4, WARM_BORDER),
        ('LINEBELOW', (0, 0), (-1, 0), 0.8, TEXT_PRIMARY),
    ]
    
    # Alternate row shading
    for i in range(1, len(data)):
        if i % 2 == 0:
            style.append(('BACKGROUND', (0, i), (-1, i), WARM_CARD))
    
    # Highlight High severity rows
    for i, r in enumerate(rows, start=1):
        if r[3] == "High":
            style.append(('TEXTCOLOR', (3, i), (3, i), HIGH_RED))
            style.append(('FONTNAME', (3, i), (3, i), 'Helvetica-Bold'))
        elif r[3] == "Medium":
            style.append(('TEXTCOLOR', (3, i), (3, i), MEDIUM_AMBER))
            style.append(('FONTNAME', (3, i), (3, i), 'Helvetica-Bold'))
    
    table.setStyle(TableStyle(style))
    return table


# ============================================================================
# CHARTS
# ============================================================================
def build_class_chart(rows, width=250, height=160):
    """Build bar chart for class distribution."""
    counts = {}
    for r in rows:
        counts[r[1]] = counts.get(r[1], 0) + 1
    
    if not counts:
        return None
    
    drawing = Drawing(width, height)
    
    chart = VerticalBarChart()
    chart.x = 30
    chart.y = 25
    chart.height = height - 45
    chart.width = width - 45
    chart.data = [list(counts.values())]
    chart.categoryAxis.categoryNames = list(counts.keys())
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.angle = 15
    chart.categoryAxis.labels.dy = -8
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = ACCENT
    chart.barWidth = 12
    
    drawing.add(chart)
    return drawing


def build_severity_chart(rows, width=250, height=160):
    """Build pie chart for severity distribution."""
    counts = {}
    for r in rows:
        counts[r[3]] = counts.get(r[3], 0) + 1
    
    if not counts:
        return None
    
    drawing = Drawing(width, height)
    
    pie = Pie()
    pie.x = width / 2 - 20
    pie.y = 15
    pie.width = 90
    pie.height = 90
    pie.data = list(counts.values())
    pie.labels = [f"{k} ({v})" for k, v in counts.items()]
    pie.slices.fontSize = 8
    pie.slices.fontName = 'Helvetica-Bold'
    pie.slices.labelRadius = 1.15
    pie.slices.strokeColor = colors.white
    pie.slices.strokeWidth = 1.5
    
    # Color mapping
    color_map = {"High": HIGH_RED, "Medium": MEDIUM_AMBER}
    for i, label in enumerate(counts.keys()):
        pie.slices[i].fillColor = color_map.get(label, ACCENT)
    
    drawing.add(pie)
    return drawing


# ============================================================================
# MAIN PDF GENERATOR
# ============================================================================
def generate_report(rows, output_path=None):
    """
    Generate PDF report from selected rows.
    
    Args:
        rows: List of tuples (id, class_name, confidence, severity, lat, lon, timestamp, device_id)
        output_path: Path to save PDF. If None, returns BytesIO buffer.
    
    Returns:
        Path to PDF file or BytesIO buffer.
    """
    styles = get_styles()
    
    # Output target
    if output_path is None:
        buffer = io.BytesIO()
        doc_target = buffer
    else:
        doc_target = str(output_path)
    
    doc = SimpleDocTemplate(
        doc_target,
        pagesize=A4,
        leftMargin=15*mm,
        rightMargin=15*mm,
        topMargin=20*mm,
        bottomMargin=18*mm,
        title="RoadVision AI - Detection Report",
        author="RoadVision AI",
    )
    
    story = []
    
    # ============================================================
    # TITLE
    # ============================================================
    story.append(Paragraph("RoadVision AI", styles['ReportTitle']))
    story.append(Paragraph(
        f"Road Damage Inspection Report · {len(rows)} Detection(s) · "
        f"Generated {datetime.now().strftime('%d %B %Y at %H:%M')}",
        styles['ReportSubtitle']
    ))
    story.append(Spacer(1, 6*mm))
    
    # ============================================================
    # SUMMARY
    # ============================================================
    story.append(Paragraph("Summary", styles['SectionHeading']))
    
    total = len(rows)
    high_count = sum(1 for r in rows if r[3] == "High")
    medium_count = sum(1 for r in rows if r[3] == "Medium")
    unique_classes = len(set(r[1] for r in rows))
    
    story.append(build_summary_table(total, high_count, medium_count, unique_classes))
    story.append(Spacer(1, 8*mm))
    
    # ============================================================
    # DETECTION TABLE
    # ============================================================
    story.append(Paragraph("Detected Damages", styles['SectionHeading']))
    story.append(build_detection_table(rows))
    story.append(Spacer(1, 8*mm))
    
    # ============================================================
    # CHARTS
    # ============================================================
    story.append(Paragraph("Analytics", styles['SectionHeading']))
    
    class_chart = build_class_chart(rows, width=250, height=160)
    severity_chart = build_severity_chart(rows, width=250, height=160)
    
    if class_chart or severity_chart:
        chart_data = [[
            class_chart if class_chart else "",
            severity_chart if severity_chart else "",
        ]]
        chart_table = Table(chart_data, colWidths=[85*mm, 85*mm])
        chart_table.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(chart_table)
    
    story.append(Spacer(1, 8*mm))
    
    # ============================================================
    # NOTES
    # ============================================================
    story.append(Paragraph("Recommendations", styles['SectionHeading']))
    story.append(Paragraph(
        f"Based on the analysis, {high_count} location(s) require immediate attention "
        f"(High priority) and {medium_count} location(s) should be scheduled for "
        f"repair soon (Medium priority). High priority damages include potholes and "
        f"alligator cracks which pose significant safety risks to vehicles and pedestrians.",
        styles['Normal']
    ))
    
    # ============================================================
    # BUILD PDF
    # ============================================================
    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)
    
    if output_path is None:
        buffer.seek(0)
        return buffer
    
    return output_path


# ============================================================================
# TEST FUNCTION
# ============================================================================
if __name__ == "__main__":
    # Test with sample data
    from datetime import datetime
    
    sample_rows = [
        (1, "Pothole", 0.85, "High", 30.5594, 76.7206, datetime.now(), "phone"),
        (2, "Alligator Crack", 0.72, "High", 30.5612, 76.7234, datetime.now(), "cctv"),
        (3, "Longitudinal Crack", 0.61, "Medium", 30.5578, 76.7189, datetime.now(), "phone"),
        (4, "Transverse Crack", 0.55, "Medium", 30.5601, 76.7212, datetime.now(), "webcam"),
    ]
    
    output = generate_report(sample_rows, "test_report.pdf")
    print(f"Report generated: {output}")