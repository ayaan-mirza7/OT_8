"""
PDF Generator for Technical Report.
Generates an academic, publication-quality PDF report from TECHNICAL_REPORT.md.
Student: Ayaan Mirza (Roll No: 241070043)
"""

import os
import re
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Adds professional running header and footer with page numbers (Page X of Y)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setFillColor(colors.HexColor("#64748b"))

        # Running Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "Optimization Techniques | Lab Assignment 8: The 14-Day Vibe-Coding Challenge")
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "Ayaan Mirza (241070043)")
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 40, 8.5 * inch - 54, 11 * inch - 40)

        # Running Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 34, page_text)
        self.drawString(54, 34, "Confidential & Academic Submission — M.Tech Optimization Techniques")
        self.setLineWidth(0.5)
        self.line(54, 46, 8.5 * inch - 54, 46)

        self.restoreState()

def clean_markdown_text(text: str) -> str:
    """Escapes XML and cleans markdown formatting for ReportLab Paragraphs."""
    t = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    # Bold **text**
    t = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', t)
    # Inline code `text`
    t = re.sub(r'`(.*?)`', r'<font name="Courier" color="#1e293b">\1</font>', t)
    # Math symbols cleanup for clean readable typography
    t = t.replace('$\\min', 'min').replace('$\\sum', 'sum').replace('$', '')
    return t

def generate_pdf():
    pdf_filename = "Technical_Report_Ayaan_Mirza_241070043.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#334155"),
        spaceAfter=14
    )
    meta_box_style = ParagraphStyle(
        'MetaBoxText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b")
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#0369a1"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )
    q_title_style = ParagraphStyle(
        'QuestionTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#b91c1c"),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        'BulletText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        leftIndent=12,
        spaceAfter=3
    )

    story = []

    # Title Banner
    story.append(Paragraph("ASSIGNMENT 8: OPTIMIZE ANYTHING", title_style))
    story.append(Paragraph("The 14-Day Vibe-Coding Challenge — Comprehensive Technical Report", subtitle_style))

    # Student Metadata Table Box
    meta_data = [
        [
            Paragraph("<b>Student Name:</b> Ayaan Mirza", meta_box_style),
            Paragraph("<b>Roll No:</b> 241070043", meta_box_style)
        ],
        [
            Paragraph("<b>Course:</b> Optimization Techniques (Lab 8)", meta_box_style),
            Paragraph("<b>Submission:</b> Public GitHub Repository & Technical Report", meta_box_style)
        ],
        [
            Paragraph("<b>Methodology:</b> Mandatory Vibe Coding / AI-Assisted Development", meta_box_style),
            Paragraph("<b>Verification:</b> Pytest Automated Suite (12/12 Passing)", meta_box_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[250, 250])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Read markdown content
    with open("TECHNICAL_REPORT.md", "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse and build flowables
    in_exec_summary = False
    in_section1 = False
    in_section2 = False

    for line in lines:
        raw = line.strip()
        if not raw:
            continue

        # Skip main doc title lines already formatted in banner
        if raw.startswith("# Comprehensive Technical Report") or raw.startswith("**Course:**") or raw.startswith("**Student Name:**") or raw.startswith("**Roll No:**") or raw.startswith("**Repository:**") or raw.startswith("**Development Methodology:**"):
            continue

        if raw.startswith("## Executive Summary"):
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceBefore=8, spaceAfter=8))
            story.append(Paragraph("Executive Summary", h1_style))
            in_exec_summary = True
            continue

        if raw.startswith("## 1. Track-by-Track"):
            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceBefore=10, spaceAfter=8))
            story.append(Paragraph("1. Track-by-Track Technical Breakdown & Experimental Outcomes", h1_style))
            continue

        if raw.startswith("## 2. Answers to the 10 Compulsory Final Questions"):
            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceBefore=10, spaceAfter=8))
            story.append(Paragraph("2. Compulsory Final Questions (Section 11 Answers)", h1_style))
            continue

        if raw.startswith("## 3. Conclusion"):
            story.append(Spacer(1, 10))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0284c7"), spaceBefore=10, spaceAfter=8))
            story.append(Paragraph("3. Conclusion & Vibe Coding Reflections", h1_style))
            continue

        if raw.startswith("### Track"):
            story.append(Paragraph(clean_markdown_text(raw.replace("### ", "")), h2_style))
            continue

        if raw.startswith("### Q"):
            story.append(Paragraph(clean_markdown_text(raw.replace("### ", "")), q_title_style))
            continue

        # Bullets
        if raw.startswith("- ") or raw.startswith("* "):
            bullet_body = clean_markdown_text(raw[2:])
            story.append(Paragraph(f"• &nbsp; {bullet_body}", bullet_style))
            continue

        # Numbered list
        if re.match(r'^\d+\.\s', raw):
            num_body = clean_markdown_text(re.sub(r'^\d+\.\s', '', raw))
            story.append(Paragraph(num_body, bullet_style))
            continue

        # Normal text paragraph
        clean_p = clean_markdown_text(raw)
        story.append(Paragraph(clean_p, body_style))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF Generator] Successfully generated: {pdf_filename}")

if __name__ == '__main__':
    generate_pdf()
