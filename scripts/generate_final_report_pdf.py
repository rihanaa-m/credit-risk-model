"""Convert final_report.md to a polished Medium-style PDF."""

from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "reports" / "final_report.md"
OUTPUT = ROOT / "reports" / "final_report.pdf"
PLOTS = ROOT / "analysis_outputs" / "task2"
IMG_WIDTH = 15.5 * cm


def parse_markdown(filepath: Path) -> list:
    """Parse markdown into blocks (headings, paragraphs, tables, code blocks)."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    blocks = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Headings
        if line.startswith("# "):
            blocks.append(("h1", line[2:]))
            i += 1
        elif line.startswith("## "):
            blocks.append(("h2", line[3:]))
            i += 1
        elif line.startswith("### "):
            blocks.append(("h3", line[4:]))
            i += 1
        
        # Code blocks
        elif line.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                code_lines.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(code_lines)))
            i += 1
        
        # Tables
        elif line.startswith("|"):
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].startswith("|"):
                table_lines.append(lines[i])
                i += 1
            blocks.append(("table", table_lines))
        
        # Images
        elif line.startswith("!["):
            blocks.append(("image", line))
            i += 1
        
        # Horizontal rule
        elif line.strip().startswith("---"):
            blocks.append(("spacer", "lg"))
            i += 1
        
        # Blank lines
        elif not line.strip():
            i += 1
        
        # Paragraphs
        else:
            para_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "```", "!")):
                para_lines.append(lines[i])
                i += 1
            if para_lines:
                blocks.append(("paragraph", "\n".join(para_lines)))
    
    return blocks


def parse_table(table_lines: list) -> Table:
    """Parse markdown table into reportlab Table."""
    rows = []
    for line in table_lines:
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        rows.append(cells)
    
    if len(rows) > 1:
        # Remove separator row (second row is usually ---)
        if all(c.startswith("-") for c in rows[1]):
            rows = [rows[0]] + rows[2:]
    
    t = Table(rows, colWidths=[3 * cm] * len(rows[0]) if rows else [])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a365d")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf2f7")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def main() -> None:
    blocks = parse_markdown(MD_PATH)
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontSize=20,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#1a365d"),
        spaceAfter=6,
        spaceBefore=12,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "H1",
        parent=styles["Heading1"],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6,
        textColor=colors.HexColor("#1a365d"),
    )
    h2_style = ParagraphStyle(
        "H2",
        parent=styles["Heading2"],
        fontSize=11,
        spaceBefore=8,
        spaceAfter=4,
        textColor=colors.HexColor("#2b6cb0"),
    )
    h3_style = ParagraphStyle(
        "H3",
        parent=styles["Heading3"],
        fontSize=10,
        spaceBefore=6,
        spaceAfter=2,
        textColor=colors.HexColor("#2d3748"),
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        alignment=TA_JUSTIFY,
    )
    code_style = ParagraphStyle(
        "Code",
        parent=styles["Normal"],
        fontSize=7.5,
        fontName="Courier",
        leftIndent=0.5 * cm,
        rightIndent=0.5 * cm,
        backColor=colors.HexColor("#f7fafc"),
        spaceAfter=6,
    )
    
    story = []
    
    for block_type, content in blocks:
        if block_type == "h1":
            if "final report" in content.lower() or "building" in content.lower():
                story.append(Paragraph(content, title_style))
            else:
                story.append(Paragraph(content, h1_style))
        
        elif block_type == "h2":
            story.append(Paragraph(content, h2_style))
        
        elif block_type == "h3":
            story.append(Paragraph(content, h3_style))
        
        elif block_type == "paragraph":
            # Replace markdown bold/italic more carefully
            # Replace **text** with <b>text</b>
            content = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', content)
            # Escape remaining special chars
            content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            # Restore bold tags
            content = content.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
            story.append(Paragraph(content, body_style))
            story.append(Spacer(1, 0.1 * cm))
        
        elif block_type == "code":
            # Escape special characters
            content = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            code_para = Paragraph(f"<pre>{content}</pre>", code_style)
            story.append(code_para)
            story.append(Spacer(1, 0.15 * cm))
        
        elif block_type == "table":
            try:
                table = parse_table(content)
                story.append(table)
                story.append(Spacer(1, 0.2 * cm))
            except Exception as e:
                print(f"Warning: could not parse table: {e}")
        
        elif block_type == "image":
            # Extract image path from markdown ![alt](path)
            match = re.search(r"!\[.*?\]\((.*?)\)", content)
            if match:
                img_path = match.group(1)
                # Try to find the image file
                abs_path = PLOTS / img_path.split("/")[-1]
                if abs_path.exists():
                    try:
                        img = Image(str(abs_path), width=IMG_WIDTH, height=IMG_WIDTH * 0.48)
                        img.hAlign = "CENTER"
                        story.append(Spacer(1, 0.1 * cm))
                        story.append(img)
                        story.append(Spacer(1, 0.15 * cm))
                    except Exception as e:
                        print(f"Warning: could not add image {abs_path}: {e}")
        
        elif block_type == "spacer":
            if content == "lg":
                story.append(Spacer(1, 0.3 * cm))
            else:
                story.append(Spacer(1, 0.15 * cm))
    
    # Build PDF
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
    )
    doc.build(story)
    print(f"✅ Wrote {OUTPUT}")
    print(f"   Size: {OUTPUT.stat().st_size / (1024*1024):.1f} MB")


if __name__ == "__main__":
    main()
