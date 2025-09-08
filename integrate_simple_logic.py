#!/usr/bin/env python3
"""
Integration: Replace the complex margin detection in pdfval.py with the simplified approach
"""
import sys
sys.path.append('.')

from simple_layout import extract_body_text_spans, detect_layout_type, create_expected_boundaries, check_boundary_violations

def create_simple_validation_function():
    """
    Create a replacement for the complex margin intrusion detection
    """
    function_code = '''
def simple_boundary_validation(page):
    """
    Simplified boundary validation: 
    1. Detect text layout (single/multi-column)  
    2. Create expected boundaries (green boxes)
    3. Flag only narrow text that extends outside boundaries
    """
    # Step 1: Extract body text spans
    raw = page.get_text("dict")
    blocks = raw.get("blocks", [])
    
    page_height = page.rect.height
    page_width = page.rect.width
    header_zone = page_height * 0.1
    footer_zone = page_height * 0.9
    
    body_spans = []
    for block in blocks:
        if block["type"] == 0:  # text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    bbox = span.get("bbox", [])
                    text = span.get("text", "")
                    
                    if (bbox and text.strip() and 
                        len(bbox) == 4 and
                        bbox[2] - bbox[0] > 2 and  # reasonable width
                        bbox[3] - bbox[1] > 2):    # reasonable height
                        
                        span_center_y = (bbox[1] + bbox[3]) / 2
                        
                        # Only include body text (not headers/footers)
                        if header_zone < span_center_y < footer_zone:
                            body_spans.append((fitz.Rect(bbox), text))
    
    # Step 2: Detect layout type
    page_center = page_width / 2
    clear_left_spans = []
    clear_right_spans = []
    
    for bbox, text in body_spans:
        span_width = bbox.x1 - bbox.x0
        span_center = (bbox.x0 + bbox.x1) / 2
        
        # Skip very wide spans (full-width paragraphs)
        if span_width > page_width * 0.6:
            continue
            
        # Only consider spans clearly positioned left or right
        margin = 50
        
        if span_center < page_center - margin:
            clear_left_spans.append((bbox, text))
        elif span_center > page_center + margin:
            clear_right_spans.append((bbox, text))
    
    layout_type = 'single'
    divider = None
    
    if len(clear_left_spans) >= 5 and len(clear_right_spans) >= 5:
        left_edges = [bbox.x1 for bbox, text in clear_left_spans]
        right_edges = [bbox.x0 for bbox, text in clear_right_spans]
        
        left_column_end = max(left_edges)
        right_column_start = min(right_edges)
        
        if right_column_start > left_column_end + 5:
            divider = (left_column_end + right_column_start) / 2
            layout_type = 'multi'
    
    # Step 3: Create expected boundaries
    if not body_spans:
        return []
    
    x_coords = [bbox.x0 for bbox, text in body_spans] + [bbox.x1 for bbox, text in body_spans]
    y_coords = [bbox.y0 for bbox, text in body_spans] + [bbox.y1 for bbox, text in body_spans]
    
    text_left = min(x_coords)
    text_right = max(x_coords)
    text_top = min(y_coords)
    text_bottom = max(y_coords)
    
    margin = 12.0  # Expanded margin to reduce boundary violations
    expected_boundaries = []
    
    if layout_type == 'single':
        box = fitz.Rect(
            max(36.0, text_left - margin),
            text_top - margin,
            min(page_width - 36.0, text_right + margin),
            text_bottom + margin
        )
        expected_boundaries = [box]
    elif layout_type == 'multi' and divider:
        left_box = fitz.Rect(
            max(36.0, text_left - margin),
            text_top - margin,
            divider,
            text_bottom + margin
        )
        right_box = fitz.Rect(
            divider,
            text_top - margin,
            min(page_width - 36.0, text_right + margin),
            text_bottom + margin
        )
        expected_boundaries = [left_box, right_box]
    
    # Step 4: Check violations (only narrow spans)
    violations = []
    
    for bbox, text in body_spans:
        span_width = bbox.x1 - bbox.x0
        
        # Skip wide spans (intentional full-width content)
        if span_width > page_width * 0.6:
            continue
        
        # Check if contained within expected boundaries
        contained = False
        for boundary in expected_boundaries:
            if boundary.contains(bbox):
                contained = True
                break
        
        if not contained:
            violations.append({
                "type": "simple_boundary_violation",
                "detail": f"Text extends outside column boundaries: '{text[:40]}'",
                "bbox": list(bbox)
            })
    
    return violations
'''
    return function_code

if __name__ == "__main__":
    # Test the simple validation on TT4
    import fitz
    from simple_layout import analyze_page_layout
    
    doc = fitz.open('/home/lee2mr/Github/PDFValidator/testScripture/TT4_MAT-images2_GAL_ptxp.pdf')
    
    total_violations = 0
    
    # Test on a few pages
    for page_idx in [4, 5, 6]:  # Pages 5, 6, 7
        page = doc[page_idx]
        result = analyze_page_layout(page)
        
        print(f"Page {page_idx + 1}: {result['layout_type']}, {len(result['violations'])} violations")
        total_violations += len(result['violations'])
    
    print(f"\nTotal violations across 3 pages: {total_violations}")
    print("Compare this to the 1113 margin intrusions from the complex system!")
    
    doc.close()
    
    # Output the integration code
    print("\n" + "="*60)
    print("INTEGRATION CODE (to replace complex margin detection):")
    print("="*60)
    print(create_simple_validation_function())