import argparse
import markdown
import re
import os
import tempfile
import sys
import shutil
from pathlib import Path
import urllib.parse
import urllib.request
import urllib.error

# Try to import required libraries, install if not available
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image as PILImage
    from bs4 import BeautifulSoup
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab", "pillow", "beautifulsoup4"])
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.units import inch, cm
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from PIL import Image as PILImage
    from bs4 import BeautifulSoup

def register_multilingual_fonts():
    """Register fonts that support Vietnamese, Korean, and other Unicode characters."""
    import platform
    
    try:
        # Check if we're on Windows
        if platform.system() == 'Windows':
            # Windows font paths
            windows_font_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
            
            # Priority 1: Arial Unicode MS (best support for Vietnamese and Korean on Windows)
            arial_unicode = os.path.join(windows_font_dir, 'ARIALUNI.TTF')
            if os.path.exists(arial_unicode):
                pdfmetrics.registerFont(TTFont('ArialUnicode', arial_unicode))
                print("Using Arial Unicode MS font (full Vietnamese/Korean support)")
                return 'ArialUnicode', 'ArialUnicode'
            
            # Priority 2: Segoe UI (good Vietnamese support, included in Windows)
            segoe_regular = os.path.join(windows_font_dir, 'segoeui.ttf')
            segoe_bold = os.path.join(windows_font_dir, 'segoeuib.ttf')
            if os.path.exists(segoe_regular) and os.path.exists(segoe_bold):
                pdfmetrics.registerFont(TTFont('SegoeUI', segoe_regular))
                pdfmetrics.registerFont(TTFont('SegoeUI-Bold', segoe_bold))
                print("Using Segoe UI fonts (good Vietnamese support)")
                return 'SegoeUI-Bold', 'SegoeUI'
            
            # Priority 3: Times New Roman (basic Vietnamese support)
            times_regular = os.path.join(windows_font_dir, 'times.ttf')
            times_bold = os.path.join(windows_font_dir, 'timesbd.ttf')
            if os.path.exists(times_regular) and os.path.exists(times_bold):
                pdfmetrics.registerFont(TTFont('TimesNewRoman', times_regular))
                pdfmetrics.registerFont(TTFont('TimesNewRoman-Bold', times_bold))
                print("Using Times New Roman fonts (basic Vietnamese support)")
                return 'TimesNewRoman-Bold', 'TimesNewRoman'
            
            # Priority 4: Arial (limited Vietnamese support)
            arial_regular = os.path.join(windows_font_dir, 'arial.ttf')
            arial_bold = os.path.join(windows_font_dir, 'arialbd.ttf')
            if os.path.exists(arial_regular) and os.path.exists(arial_bold):
                pdfmetrics.registerFont(TTFont('Arial', arial_regular))
                pdfmetrics.registerFont(TTFont('Arial-Bold', arial_bold))
                print("Using Arial fonts (limited Vietnamese support)")
                return 'Arial-Bold', 'Arial'
        
        else:
            # Linux/Unix font paths
            # Priority 1: Try regular Noto Sans (limited Korean support but better than DejaVu)
            noto_regular = '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf'
            noto_bold = '/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf'

            if os.path.exists(noto_regular) and os.path.exists(noto_bold):
                pdfmetrics.registerFont(TTFont('NotoSans', noto_regular))
                pdfmetrics.registerFont(TTFont('NotoSans-Bold', noto_bold))
                print("Using regular Noto Sans fonts (limited Korean support)")
                return 'NotoSans-Bold', 'NotoSans'

            # Priority 2: DejaVu Sans (fallback)
            dejavu_regular = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
            dejavu_bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

            if os.path.exists(dejavu_regular) and os.path.exists(dejavu_bold):
                pdfmetrics.registerFont(TTFont('DejaVuSans', dejavu_regular))
                pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', dejavu_bold))
                print("Using DejaVu Sans fonts (minimal Korean support)")
                return 'DejaVuSans-Bold', 'DejaVuSans'

            # Priority 3: Liberation fonts
            liberation_regular = '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'
            liberation_bold = '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'

            if os.path.exists(liberation_regular) and os.path.exists(liberation_bold):
                pdfmetrics.registerFont(TTFont('LiberationSans', liberation_regular))
                pdfmetrics.registerFont(TTFont('LiberationSans-Bold', liberation_bold))
                print("Using Liberation Sans fonts (no Korean support)")
                return 'LiberationSans-Bold', 'LiberationSans'

    except Exception as e:
        print(f"Warning: Could not register multilingual fonts: {e}")

    # Final fallback to standard fonts (may not display Korean/Vietnamese correctly)
    print("Warning: Using standard fonts - Korean and Vietnamese characters may not display correctly")
    return 'Helvetica-Bold', 'Helvetica'

# Register fonts and set font constants
FONT_BOLD, FONT_NORMAL = register_multilingual_fonts()

def fix_markdown_image_paths(md_content, base_dir):
    """Fix image paths in markdown content before conversion to HTML."""
    # Pattern to find image references in markdown
    pattern = r'!\[(.*?)\]\(([^)]+)\)'

    def replace_path(match):
        alt_text = match.group(1)
        rel_path = match.group(2)

        # Handle both relative paths starting with ../ and other paths
        if rel_path.startswith('../'):
            # For paths like ../public/images/file.png
            abs_path = os.path.normpath(os.path.join(base_dir, rel_path))
        else:
            # For other paths, assume they're relative to the base_dir
            abs_path = os.path.normpath(os.path.join(base_dir, rel_path))

        # Make sure the path exists
        if os.path.exists(abs_path):
            # Use absolute path with file:// protocol
            return f'![{alt_text}](file://{abs_path})'
        else:
            print(f"Warning: Image file not found: {abs_path}")
            return match.group(0)  # Return original if file not found

    # Replace relative paths with absolute paths
    fixed_md = re.sub(pattern, replace_path, md_content)
    return fixed_md

def copy_images_to_temp_dir(md_content, base_dir):
    """Copy images referenced in markdown to a temporary directory."""
    # Create a temporary directory for images
    temp_dir = tempfile.mkdtemp()

    # Pattern to find image references in markdown
    pattern = r'!\[(.*?)\]\(([^)]+)\)'
    image_paths = re.findall(pattern, md_content)

    # Dictionary to map original paths to new paths
    path_mapping = {}

    for alt_text, rel_path in image_paths:
        # URL decode the path first
        decoded_path = urllib.parse.unquote(rel_path)

        # Handle both relative paths starting with ../ and other paths
        if decoded_path.startswith('../'):
            # For paths like ../public/images/file.png
            abs_path = os.path.normpath(os.path.join(base_dir, decoded_path))
        else:
            # For other paths, assume they're relative to the base_dir
            abs_path = os.path.normpath(os.path.join(base_dir, decoded_path))

        # Make sure the file exists (try the path as-is first, since filenames may be URL-encoded)
        if os.path.exists(abs_path):
            # Get the filename
            filename = os.path.basename(abs_path)
            # Create a new path in the temporary directory
            new_path = os.path.join(temp_dir, filename)

            # Copy the file
            import shutil
            shutil.copy2(abs_path, new_path)

            # Add to path mapping
            path_mapping[rel_path] = filename
        else:
            # Try with the original encoded path as well
            if rel_path.startswith('../'):
                abs_path_encoded = os.path.normpath(os.path.join(base_dir, rel_path))
            else:
                abs_path_encoded = os.path.normpath(os.path.join(base_dir, rel_path))

            if os.path.exists(abs_path_encoded):
                # Get the filename
                filename = os.path.basename(abs_path_encoded)
                # Create a new path in the temporary directory
                new_path = os.path.join(temp_dir, filename)

                # Copy the file
                import shutil
                shutil.copy2(abs_path_encoded, new_path)

                # Add to path mapping
                path_mapping[rel_path] = filename

    return temp_dir, path_mapping

def replace_emojis_with_text(text):
    """Replace common emojis with text equivalents for PDF compatibility."""
    emoji_replacements = {
        '🌾': '[농촌]',
        '🏥': '[병원]',
        '⭐': '[★]',
        '❌': '[X]',
        '✅': '[✓]',
        '🔋': '[배터리]',
        '🔌': '[전원]',
        '🏆': '[우승]',
        '💡': '[아이디어]',
        '📱': '[모바일]',
        '💻': '[컴퓨터]',
        '🚀': '[로켓]',
        '⚡': '[번개]',
        '🎯': '[타겟]',
        '📊': '[차트]',
        '📈': '[상승]',
        '📉': '[하락]',
        '🔍': '[검색]',
        '⚙️': '[설정]',
        '🛠️': '[도구]',
        '📋': '[클립보드]',
        '📝': '[메모]',
        '💾': '[저장]',
        '🌐': '[웹]',
        '📡': '[신호]',
        '🔒': '[잠금]',
        '🔓': '[해제]',
        '⚠️': '[경고]',
        '🚨': '[알람]',
        '✨': '[반짝]',
        '🎉': '[축하]',
        '👍': '[좋음]',
        '👎': '[나쁨]',
        '💪': '[힘]',
        '🧠': '[뇌]',
        '❤️': '[하트]',
        '💚': '[녹색하트]',
        '💙': '[파란하트]',
        '💛': '[노란하트]',
        '🔥': '[불]',
        '💧': '[물방울]',
        '☀️': '[태양]',
        '🌙': '[달]',
        '⭐⭐⭐⭐⭐': '[★★★★★]',
        '⭐⭐⭐⭐': '[★★★★]',
        '⭐⭐⭐': '[★★★]',
        '⭐⭐': '[★★]',
    }

    for emoji, replacement in emoji_replacements.items():
        text = text.replace(emoji, replacement)

    return text

def process_element_with_links(element, style):
    """Process an HTML element and preserve links as clickable in PDF."""
    # Check if element contains links
    links = element.find_all('a')
    if not links:
        # No links, return simple paragraph
        return Paragraph(element.text, style)

    # Build paragraph with links
    from reportlab.lib.colors import blue

    # Create a copy of the style for links
    link_style = ParagraphStyle(
        name=style.name + '_Link',
        parent=style,
        textColor=blue,
        underline=True
    )

    # Process the element content with links
    content_parts = []
    current_pos = 0
    element_text = str(element)

    for link in links:
        # Find the position of this link in the element
        link_text = link.text
        link_url = link.get('href', '')

        # Add text before the link
        before_link = element.text[:element.text.find(link_text, current_pos)]
        if before_link:
            content_parts.append(before_link)

        # Add the clickable link
        if link_url:
            content_parts.append(f'<a href="{link_url}" color="blue">{link_text}</a>')
        else:
            content_parts.append(link_text)

        current_pos = element.text.find(link_text, current_pos) + len(link_text)

    # Add remaining text after the last link
    remaining_text = element.text[current_pos:]
    if remaining_text:
        content_parts.append(remaining_text)

    # Join all parts and create paragraph
    full_content = ''.join(content_parts)
    return Paragraph(full_content, style)

def is_numbered_section_header(element):
    """Check if a paragraph element contains a numbered section header like '1. Clinical Findings:' or if it's a list item with bold header"""
    if element.name == 'p':
        # Check if the paragraph contains a strong element with numbered section pattern
        strong_elem = element.find('strong')
        if strong_elem:
            # Check if the strong text matches the pattern "number. text" (colon is optional)
            strong_text = strong_elem.get_text().strip()
            match = re.match(r'^(\d+)\.\s+(.+?)(:?)$', strong_text)
            if match:
                number, title, colon = match.groups()
                return True, (int(number), title)

    elif element.name == 'li':
        # Check if this is a list item that contains a bold section header
        strong_elem = element.find('strong')
        if strong_elem:
            # Get the full text of the list item
            full_text = element.get_text().strip()
            # Check if it starts with a number and contains bold text
            match = re.match(r'^(\d+)\.\s+(.+?)(:?)$', full_text)
            if match:
                number, title_part, colon = match.groups()
                # Extract just the bold part as the title
                bold_text = strong_elem.get_text().strip()
                # Remove colon if present
                if bold_text.endswith(':'):
                    bold_text = bold_text[:-1]
                return True, (int(number), bold_text)

    return False, None

def preprocess_lists(md_content):
    """Preprocess markdown content to ensure lists are properly formatted."""
    # Split content into lines
    lines = md_content.split('\n')
    processed_lines = []

    # Process each line
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this line starts a list item
        list_match = re.match(r'^(\s*)([\*\-\+]|\d+\.)\s+(.*)', line)

        if list_match:
            # This is a list item
            indent, marker, content = list_match.groups()

            # Ensure there's a blank line before the list if needed
            if i > 0 and processed_lines and not processed_lines[-1].strip() == '':
                processed_lines.append('')

            # Add the list item
            processed_lines.append(line)

            # Check if this is the last line or if the next line is not a list item
            if i == len(lines) - 1 or not re.match(r'^(\s*)([\*\-\+]|\d+\.)\s+(.*)', lines[i+1]):
                # Add a blank line after the list
                processed_lines.append('')
        else:
            # Not a list item, add as is
            processed_lines.append(line)

        i += 1

    # Join lines back into a string
    return '\n'.join(processed_lines)

def convert_md_to_pdf(input_file, output_file=None, debug=False):
    """Convert a Markdown file to PDF."""
    print(f"Starting PDF conversion: {input_file} -> {output_file}")
    
    # Set default output filename if not provided
    if output_file is None:
        output_file = Path(input_file).with_suffix('.pdf')

    # Convert to absolute paths
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)

    # Get the directory of the input file for resolving relative paths
    input_dir = os.path.dirname(input_file)
    print(f"Input directory: {input_dir}")

    # Read markdown content
    print("Reading markdown content...")
    with open(input_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    print(f"Markdown content length: {len(md_content)} characters")

    # Replace emojis with text equivalents for PDF compatibility
    print("Replacing emojis...")
    md_content = replace_emojis_with_text(md_content)

    # Preprocess lists to ensure proper formatting
    print("Preprocessing lists...")
    md_content = preprocess_lists(md_content)

    # Copy images to a temporary directory
    print("Copying images to temporary directory...")
    temp_img_dir, path_mapping = copy_images_to_temp_dir(md_content, input_dir)
    print(f"Images copied to: {temp_img_dir}")

    # Replace image paths in markdown with paths to the temporary directory
    print("Replacing image paths...")
    modified_md_content = md_content
    for orig_path, new_filename in path_mapping.items():
        # Replace the original path with the new path
        modified_md_content = modified_md_content.replace(
            f']({orig_path})',
            f']({new_filename})'
        )

    # Convert modified markdown to HTML with all necessary extensions
    print("Converting markdown to HTML...")
    html_content = markdown.markdown(
        modified_md_content,
        extensions=[
            'tables',
            'fenced_code',
            'markdown.extensions.nl2br',  # Convert newlines to <br>
            'markdown.extensions.sane_lists',  # Better list handling
            'markdown.extensions.smarty',  # Smart quotes, dashes, etc.
        ]
    )
    print(f"HTML content length: {len(html_content)} characters")

    # Save intermediate HTML for debugging if requested
    if debug:
        html_debug_file = os.path.splitext(output_file)[0] + '.html'
        os.makedirs(os.path.dirname(html_debug_file), exist_ok=True)
        with open(html_debug_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Saved debug HTML to {html_debug_file}")

    # Create a temporary HTML file with proper styling
    print("Creating styled HTML file...")
    with tempfile.NamedTemporaryFile(suffix='.html', mode='w+', encoding='utf-8', delete=False) as temp_html:
        temp_html_path = temp_html.name
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ 
                    font-family: 'Times New Roman', serif; 
                    margin: 40px; 
                    line-height: 1.6;
                    color: #333;
                    background-color: #ffffff;
                }}
                h1 {{ 
                    color: #2c3e50; 
                    border-bottom: 3px solid #3498db; 
                    padding-bottom: 10px;
                    font-size: 24px;
                    font-weight: bold;
                }}
                h2 {{ 
                    color: #34495e; 
                    border-left: 4px solid #3498db; 
                    padding-left: 15px;
                    font-size: 18px;
                    font-weight: bold;
                    margin-top: 25px;
                }}
                h3 {{ 
                    color: #2c3e50; 
                    font-size: 14px;
                    font-weight: bold;
                    margin-top: 20px;
                }}
                code {{ 
                    background-color: #ecf0f1; 
                    padding: 2px 6px; 
                    border-radius: 4px; 
                    font-family: 'Courier New', monospace;
                }}
                pre {{ 
                    background-color: #f8f9fa; 
                    padding: 15px; 
                    border-radius: 6px; 
                    border-left: 4px solid #3498db;
                }}
                img {{ 
                    max-width: 100%; 
                    display: block; 
                    margin: 20px auto; 
                    border: 2px solid #bdc3c7;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                }}
                table {{ 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0; 
                    border: 2px solid #3498db;
                    border-radius: 6px;
                    overflow: hidden;
                }}
                th, td {{ 
                    border: 1px solid #bdc3c7; 
                    padding: 12px; 
                    text-align: left; 
                }}
                th {{ 
                    background-color: #3498db; 
                    color: white; 
                    font-weight: bold;
                    text-transform: uppercase;
                    font-size: 12px;
                }}
                td {{
                    background-color: #f8f9fa;
                }}
                ul, ol {{
                    margin: 15px 0;
                    padding-left: 30px;
                }}
                li {{
                    margin: 8px 0;
                }}
                blockquote {{
                    border-left: 4px solid #3498db;
                    margin: 20px 0;
                    padding: 10px 20px;
                    background-color: #ecf0f1;
                    font-style: italic;
                }}
                hr {{
                    border: none;
                    border-top: 2px solid #3498db;
                    margin: 30px 0;
                }}
                .disclaimer {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 6px;
                    padding: 15px;
                    margin: 20px 0;
                }}
                .metadata {{
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 6px;
                    padding: 15px;
                    margin: 20px 0;
                    font-size: 12px;
                    color: #6c757d;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        temp_html.write(styled_html)
    print(f"Styled HTML file created: {temp_html_path}")

    # Change working directory to the temporary image directory
    original_cwd = os.getcwd()
    os.chdir(temp_img_dir)
    print(f"Changed working directory to: {temp_img_dir}")

    # Create a PDF document
    try:
        print("Setting up PDF document...")
        # Set up the document
        doc = SimpleDocTemplate(
            output_file,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        # Get styles
        print("Setting up styles...")
        styles = getSampleStyleSheet()

        # Create custom styles with standard fonts
        title_style = ParagraphStyle(
            name='Heading1',
            fontName=FONT_BOLD,  # Use bold for headings
            fontSize=14,
            leading=18,
            alignment=1,  # Center alignment
        )

        heading2_style = ParagraphStyle(
            name='Heading2',
            fontName=FONT_BOLD,  # Use bold for headings
            fontSize=12,
            leading=16,
        )

        heading3_style = ParagraphStyle(
            name='Heading3',
            fontName=FONT_BOLD,  # Use bold for headings
            fontSize=11,
            leading=14,
        )

        # Style for numbered section headers (like "1. Clinical Findings:")
        numbered_section_style = ParagraphStyle(
            name='NumberedSection',
            fontName=FONT_BOLD,
            fontSize=11,
            leading=15,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.Color(0.1, 0.1, 0.1),  # Darker color for section headers
        )

        normal_style = ParagraphStyle(
            name='Normal',
            fontName=FONT_NORMAL,  # Use normal font for body text
            fontSize=10,
            leading=13,
            textColor=colors.Color(0.2, 0.2, 0.2),  # Slightly lighter color
        )

        # Create a list to hold the PDF elements
        elements = []

        # Parse the HTML content to extract text and images
        print("Parsing HTML content...")
        soup = BeautifulSoup(html_content, 'html.parser')

        # Process the HTML elements
        print("Processing HTML elements...")
        element_count = 0
        for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'img', 'table', 'ul', 'ol', 'pre', 'code', 'li']):
            element_count += 1
            if element_count % 10 == 0:
                print(f"Processed {element_count} elements...")
                
            if element.name == 'h1':
                elements.append(Paragraph(element.text, title_style))
                elements.append(Spacer(1, 12))
            elif element.name == 'h2':
                elements.append(Paragraph(element.text, heading2_style))
                elements.append(Spacer(1, 10))
            elif element.name == 'h3':
                elements.append(Paragraph(element.text, heading3_style))
                elements.append(Spacer(1, 8))
            elif element.name == 'p':
                # Skip paragraphs that are inside list items to avoid duplication
                if element.parent and element.parent.name == 'li':
                    continue

                # Skip paragraphs that contain code blocks to avoid duplication
                if element.find('code') or element.find('pre'):
                    continue

                # Check if this is a numbered section header (like "1. Clinical Findings")
                is_numbered, section_info = is_numbered_section_header(element)
                if is_numbered:
                    number, title = section_info
                    # Format as "1. Clinical Findings" preserving original format
                    section_text = f"{number}. {title}"
                    elements.append(Paragraph(section_text, numbered_section_style))
                    continue

                # Process paragraph with HTML tags (like <strong>)
                text = element.get_text()
                
                # Handle <strong> tags by converting them to bold text
                strong_tags = element.find_all('strong')
                if strong_tags:
                    # Replace <strong> tags with bold formatting
                    for strong in strong_tags:
                        strong_text = strong.get_text()
                        # Use ReportLab's bold formatting
                        text = text.replace(strong_text, f"<b>{strong_text}</b>")
                
                if text.strip():
                    elements.append(Paragraph(text, normal_style))
                    elements.append(Spacer(1, 6))
            elif element.name == 'li':
                # Check if this list item is a numbered section header
                is_numbered, section_info = is_numbered_section_header(element)
                if is_numbered:
                    number, title = section_info
                    # Format as numbered section header
                    section_text = f"{number}. {title}"
                    elements.append(Paragraph(section_text, numbered_section_style))
                    continue
                # If not a section header, it will be processed by the parent ul/ol

            elif element.name in ['ul', 'ol']:
                # Process lists (if any are properly formatted in HTML)
                list_items = element.find_all('li')
                if list_items:
                    for i, li in enumerate(list_items):
                        # Skip if this list item is a numbered section header (already processed above)
                        is_numbered, _ = is_numbered_section_header(li)
                        if is_numbered:
                            continue

                        # Get the text content, ignoring any nested paragraph tags
                        if li.find('p'):
                            # If there are nested paragraphs, join their text
                            item_text = ' '.join([p.text.strip() for p in li.find_all('p')])
                        else:
                            # Otherwise use the direct text
                            item_text = li.text.strip()

                        bullet = '•' if element.name == 'ul' else f"{i+1}."
                        list_text = f"{bullet} {item_text}"
                        elements.append(Paragraph(list_text, normal_style))
                        elements.append(Spacer(1, 3))
                    elements.append(Spacer(1, 6))
            elif element.name == 'img':
                img_src = element.get('src')
                if img_src and os.path.exists(img_src):
                    print(f"Processing image: {img_src}")
                    # Get the image dimensions
                    img = PILImage.open(img_src)
                    width, height = img.size

                    # Scale the image to fit the page width
                    max_width = 450  # Maximum width in points
                    if width > max_width:
                        ratio = max_width / width
                        width = max_width
                        height = height * ratio

                    # Add the image to the PDF
                    elements.append(Image(img_src, width=width, height=height))
                    elements.append(Spacer(1, 12))
            elif element.name == 'table':
                # Process table with link support
                rows = []
                table_rows = element.find_all('tr')
                if table_rows:
                    for tr in table_rows:
                        row = []
                        cells = tr.find_all(['td', 'th'])
                        if cells:
                            for td in cells:
                                # Check if cell contains links
                                links = td.find_all('a')
                                if links:
                                    # Process cell with links
                                    cell_content = ""
                                    for link in links:
                                        link_text = link.text
                                        link_url = link.get('href', '')
                                        if link_url:
                                            # Create clickable link in table cell
                                            cell_content += f'<a href="{link_url}" color="blue">{link_text}</a>'
                                        else:
                                            cell_content += link_text
                                    # Add any remaining text
                                    remaining_text = td.text
                                    for link in links:
                                        remaining_text = remaining_text.replace(link.text, '')
                                    cell_content = remaining_text + cell_content
                                    row.append(Paragraph(cell_content, normal_style))
                                else:
                                    # Regular cell without links
                                    row.append(td.text.strip())
                        if row:  # Only add row if it has content
                            rows.append(row)

                if rows:
                    # Create a table
                    table = Table(rows)

                    # Add style to the table
                    table_style = TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),  # Bold headers stay black
                        ('TEXTCOLOR', (0, 1), (-1, -1), colors.Color(0.2, 0.2, 0.2)),  # Lighter color for content
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),  # Use bold for table headers
                        ('FONTNAME', (0, 1), (-1, -1), FONT_NORMAL),  # Use normal font for table content
                        ('FONTSIZE', (0, 0), (-1, -1), 9),  # Smaller font size for tables
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # Vertical alignment for cells with links
                    ])
                    table.setStyle(table_style)

                    elements.append(table)
                    elements.append(Spacer(1, 12))
            elif element.name == 'pre':
                # Process code blocks and preserve formatting
                code_style = ParagraphStyle(
                    name='CodeBlock',
                    fontName='Courier',  # Use monospace font for code
                    fontSize=9,
                    leading=12,
                    leftIndent=20,
                    rightIndent=20,
                    spaceBefore=6,
                    spaceAfter=6,
                    backColor=colors.lightgrey,
                    borderColor=colors.grey,
                    borderWidth=1,
                    borderPadding=10,
                    textColor=colors.Color(0.2, 0.2, 0.2),  # Lighter color for code text
                )

                # Get the text content and preserve line breaks
                code_text = element.text
                if code_text.strip():
                    # Replace line breaks with <br/> tags for proper formatting
                    formatted_code = code_text.replace('\n', '<br/>')
                    elements.append(Paragraph(formatted_code, code_style))
                    elements.append(Spacer(1, 12))

        print(f"Total elements processed: {element_count}")
        print(f"Total PDF elements created: {len(elements)}")

        # Build the PDF
        print("Building PDF document...")
        doc.build(elements)
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error converting to PDF: {e}")
        # Try saving the HTML file as a fallback
        html_fallback_file = os.path.splitext(output_file)[0] + '.fallback.html'
        os.makedirs(os.path.dirname(html_fallback_file), exist_ok=True)
        with open(html_fallback_file, 'w', encoding='utf-8') as f:
            f.write(styled_html)
        print(f"Saved fallback HTML to {html_fallback_file}")
    finally:
        # Change back to the original working directory
        os.chdir(original_cwd)

        # Clean up the temporary files
        if os.path.exists(temp_html_path):
            os.unlink(temp_html_path)

        # Clean up the temporary image directory (but preserve font directory)
        if os.path.exists(temp_img_dir):
            try:
                shutil.rmtree(temp_img_dir)
            except PermissionError:
                # Windows-specific issue: files might still be in use
                # Try to remove individual files first
                try:
                    import time
                    time.sleep(0.1)  # Brief pause
                    for root, dirs, files in os.walk(temp_img_dir, topdown=False):
                        for file in files:
                            try:
                                os.unlink(os.path.join(root, file))
                            except PermissionError:
                                pass  # Skip files that are still in use
                        for dir in dirs:
                            try:
                                os.rmdir(os.path.join(root, dir))
                            except OSError:
                                pass  # Skip directories that aren't empty
                    try:
                        os.rmdir(temp_img_dir)
                    except OSError:
                        pass  # Directory might not be empty, that's ok
                except Exception:
                    pass  # Cleanup failed, but PDF was created successfully
        
        # Note: We intentionally do NOT clean up the persistent font directory
        # (.fonts/korean) as it should persist across runs for better performance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Markdown to PDF")
    parser.add_argument("input_file", help="Path to the Markdown file")
    parser.add_argument("-o", "--output", help="Path to the output PDF file")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    convert_md_to_pdf(args.input_file, args.output, args.debug)