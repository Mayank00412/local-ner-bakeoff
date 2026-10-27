"""Typeset the existing consolidated HTML as a bookmarked A4 study PDF."""
import html
import json
import re
import textwrap
from html.parser import HTMLParser
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, Preformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output/pdf/NER_Bakeoff_Interview_Study_Guide.pdf'
NAVY = colors.HexColor('#183153')
GRAY = colors.HexColor('#536170')


def normalize(text):
    return re.sub('[\u2010\u2011\u2012\u2013\u2014\u2212]', '-', text).replace('\u00a0', ' ')


class Blocks(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.blocks, self.mode, self.parts, self.anchor = [], None, [], None
        self.skip = None
        self.rows, self.cells, self.cell = None, None, None

    def handle_starttag(self, tag, attrs):
        if self.skip:
            return
        if tag in ('head', 'nav'):
            self.skip = tag
            return
        if tag == 'table':
            self.mode, self.rows = 'table', []
        elif self.mode == 'table':
            if tag == 'tr':
                self.cells = []
            elif tag in ('th', 'td'):
                self.cell = []
        elif tag in ('h1', 'h2', 'h3', 'p', 'pre', 'li'):
            self.mode, self.parts = tag, []
            self.anchor = dict(attrs).get('id')

    def handle_data(self, data):
        if self.skip:
            return
        if self.mode == 'table' and self.cell is not None:
            self.cell.append(data)
        elif self.mode:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if self.skip:
            if tag == self.skip:
                self.skip = None
            return
        if self.mode == 'table':
            if tag in ('th', 'td') and self.cell is not None:
                self.cells.append(normalize(''.join(self.cell).strip()))
                self.cell = None
            elif tag == 'tr' and self.cells is not None:
                self.rows.append(self.cells)
            elif tag == 'table':
                self.blocks.append(('table', self.rows, None))
                self.mode = None
        elif tag == self.mode:
            value = ''.join(self.parts)
            self.blocks.append((tag, normalize(value.strip()), self.anchor))
            self.mode = None


class GuideDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=A4, leftMargin=45, rightMargin=45,
                         topMargin=48, bottomMargin=46,
                         title='Local NER Bake-off - Consolidated Report and Interview Study Guide',
                         author='Project report prepared with Codex')
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='Guide', frames=frame, onPage=self.decorate))

    def decorate(self, canvas, doc):
        canvas.saveState()
        width, height = A4
        if doc.page > 1:
            canvas.setFont('Arial', 8)
            canvas.setFillColor(GRAY)
            canvas.drawString(45, height-29, 'LOCAL NER BAKE-OFF  |  PROJECT REPORT & INTERVIEW GUIDE')
            canvas.setStrokeColor(colors.HexColor('#D8DEE6'))
            canvas.line(45, height-35, width-45, height-35)
        canvas.setFont('Arial', 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(45, 26, 'Five local model pipelines - recorded results, methods, and study material')
        canvas.drawRightString(width-45, 26, str(doc.page))
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if hasattr(flowable, 'section_anchor'):
            title, anchor = flowable.section_title, flowable.section_anchor
            self.canv.bookmarkPage(anchor)
            self.canv.addOutlineEntry(title, anchor, level=0, closed=False)
            self.notify('TOCEntry', (0, title, self.page, anchor))


def main():
    fonts = Path('C:/Windows/Fonts')
    for name, file in [('Arial','arial.ttf'), ('Arial-Bold','arialbd.ttf'),
                       ('Arial-Italic','ariali.ttf'), ('Consolas','consola.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fonts/file)))
    pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic')
    styles = getSampleStyleSheet()
    body = ParagraphStyle('GuideBody', fontName='Arial', fontSize=9.6, leading=14,
                          textColor=colors.HexColor('#202A35'), spaceAfter=7)
    h2 = ParagraphStyle('GuideH2', parent=body, fontName='Arial-Bold', fontSize=13.3,
                        leading=17, spaceBefore=15, spaceAfter=8, keepWithNext=True, textColor=NAVY)
    h3 = ParagraphStyle('GuideH3', parent=body, fontName='Arial-Bold', fontSize=10.1,
                        leading=14, spaceBefore=8, spaceAfter=4, keepWithNext=True, textColor=NAVY)
    bullet = ParagraphStyle('GuideBullet', parent=body, leftIndent=11, firstLineIndent=-8,
                           spaceAfter=5)
    mono = ParagraphStyle('GuideCode', fontName='Consolas', fontSize=7.2, leading=10,
                         leftIndent=7, rightIndent=7, spaceBefore=4, spaceAfter=9,
                         backColor=colors.HexColor('#F2F4F7'), borderPadding=6)
    cover_title = ParagraphStyle('Cover', parent=body, fontName='Arial-Bold', fontSize=27,
                                leading=34, textColor=NAVY, spaceAfter=17)
    cover_sub = ParagraphStyle('CoverSub', parent=body, fontSize=15, leading=22, spaceAfter=18)
    small = ParagraphStyle('Small', parent=body, fontSize=8.3, leading=11.5)
    width = A4[0]-90
    story = [Spacer(1, 52), Paragraph('LOCAL NER BAKE-OFF', cover_title),
        Paragraph('Consolidated project report<br/>and interview study guide', cover_sub),
        Paragraph('28 sections | 33 practice questions | Five locally executed model pipelines', body),
        Paragraph('Prepared 28 October 2026', small), Spacer(1, 24)]
    results = json.loads((ROOT/'results/results.json').read_text())
    data = [['Model', 'Micro F1', 'Precision', 'Recall']]
    for r in results:
        label = 'GLiNER' if r['model']=='gliner' else r['model'].replace('qwen', 'Qwen ').replace('05b','0.5B').replace('15b','1.5B').replace('3b','3B').replace('7b','7B')
        data.append([label, *[f'{r[k]:.4f}' for k in ('ALL_f1','ALL_precision','ALL_recall')]])
    summary = Table(data, colWidths=[width*.4, width*.2, width*.2, width*.2])
    summary.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,-1),'Arial'), ('FONTSIZE',(0,0),(-1,-1),10),
        ('FONTNAME',(0,0),(-1,0),'Arial-Bold'), ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EEF5')),
        ('TOPPADDING',(0,0),(-1,-1),9), ('BOTTOMPADDING',(0,0),(-1,-1),9),
        ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#CBD5E1')),
        ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#E0E5EB'))]))
    story += [summary, Spacer(1, 23),
        Paragraph('Read the complete project evidence first. For interview preparation, start with sections 18-19, work through sections 21-23, and rehearse the questions and live demonstration in sections 24-25.', body),
        Paragraph('The 3B/7B runs are a disclosed post-review extension using Q4_K_M and llama.cpp. Original runs use float32 PyTorch. A partial 7B attempt was excluded and restarted with the same settings. All final results and limitations are preserved in this guide.', small),
        PageBreak(), Paragraph('Contents', h2)]
    toc = TableOfContents()
    toc.levelStyles = [ParagraphStyle('TOC', fontName='Arial', fontSize=9,
                                     leading=13, leftIndent=0, firstLineIndent=0, spaceBefore=0)]
    story += [toc, PageBreak()]
    parser = Blocks()
    parser.feed((ROOT/'FULL_REPORT.html').read_text(encoding='utf-8'))
    section_count, question_count, table_count = 0, 0, 0
    for kind, value, anchor in parser.blocks:
        if kind == 'h1':
            continue
        if kind == 'h2':
            if value.startswith('18.'):
                story.append(PageBreak())
            p = Paragraph(html.escape(value), h2)
            p.section_title, p.section_anchor = value, anchor
            story.append(p)
            section_count += 1
        elif kind == 'h3':
            story.append(Paragraph(html.escape(value), h3))
            question_count += bool(re.match(r'Q\d+\.', value))
        elif kind == 'p':
            story.append(Paragraph(html.escape(value), body))
        elif kind == 'li':
            story.append(Paragraph('&#8226; ' + html.escape(value), bullet))
        elif kind == 'pre':
            lines = []
            for line in value.expandtabs(4).splitlines():
                indent = min(len(line)-len(line.lstrip())+2, 24)
                lines.extend(textwrap.wrap(line, width=106, subsequent_indent=' '*indent,
                    replace_whitespace=False, drop_whitespace=False) or [''])
            story.append(Preformatted('\n'.join(lines), mono))
        elif kind == 'table':
            count = len(value[0])
            first_weight = 2.5 if max(len(row[0]) for row in value)>35 else 1.5
            weights = [first_weight]+[1]*(count-1)
            col_widths = [width*w/sum(weights) for w in weights]
            ts = ParagraphStyle('Cell', parent=body, fontSize=7.6 if count>=7 else 8.6,
                                leading=10.3 if count>=7 else 12, spaceAfter=0,
                                splitLongWords=True)
            th = ParagraphStyle('HeaderCell', parent=ts, fontName='Arial-Bold', textColor=NAVY)
            cells = [[Paragraph(html.escape(cell), th if i==0 else ts) for cell in row]
                     for i, row in enumerate(value)]
            t = Table(cells, colWidths=col_widths, repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([
                ('VALIGN',(0,0),(-1,-1),'TOP'), ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EEF5')),
                ('GRID',(0,0),(-1,-1),.35,colors.HexColor('#D1D9E2')),
                ('LEFTPADDING',(0,0),(-1,-1),6), ('RIGHTPADDING',(0,0),(-1,-1),6),
                ('TOPPADDING',(0,0),(-1,-1),7), ('BOTTOMPADDING',(0,0),(-1,-1),7)]))
            story += [t, Spacer(1,10)]
            table_count += 1
    assert section_count == 28 and question_count == 33
    OUT.parent.mkdir(parents=True, exist_ok=True)
    GuideDoc(OUT).multiBuild(story)
    print(f'Created {OUT}')
    print(f'Converted {section_count} sections, {question_count} interview questions, {table_count} report tables.')


if __name__ == '__main__':
    main()
