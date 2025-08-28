#!/usr/bin/env python3
"""
Visual test: Create green boxes and see how they should look for TT4 page 6
"""
import fitz
from simple_layout import extract_body_text_spans

def visualize_page_layout():
    """Create a visual test showing text spans and proposed boundaries"""
    doc = fitz.open('/home/lee2mr/Github/PDFValidator/testScripture/TT4_MAT-images2_GAL_ptxp.pdf')
    page = doc[5]  # Page 6
    
    # Create output document
    out_doc = fitz.open()
    out_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
    out_page.show_pdf_page(page.rect, doc, 5)
    
    # Get body text spans
    body_spans = extract_body_text_spans(page)
    
    print(f"=== Page 6 Text Span Analysis ===")
    print(f"Total body spans: {len(body_spans)}")
    
    # Analyze by position
    page_width = page.rect.width
    page_center = page_width / 2
    
    print(f"Page width: {page_width:.1f}, center: {page_center:.1f}")
    
    # Show all spans with positions
    left_spans = []
    right_spans = []
    center_spans = []
    
    for i, (bbox, text) in enumerate(body_spans[:20]):  # First 20 for analysis
        span_width = bbox.x1 - bbox.x0
        span_center = (bbox.x0 + bbox.x1) / 2
        
        # Classify spans
        if span_width > page_width * 0.6:
            category = "WIDE"
            center_spans.append((bbox, text))
        elif span_center < page_center - 50:  # Clearly left
            category = "LEFT"
            left_spans.append((bbox, text))
        elif span_center > page_center + 50:  # Clearly right
            category = "RIGHT"
            right_spans.append((bbox, text))
        else:
            category = "CENTER"
            center_spans.append((bbox, text))
        
        print(f"{i+1:2d}. {category:6s} x={bbox.x0:5.1f}-{bbox.x1:5.1f} (w={span_width:5.1f}) \"{text[:30]}\"")
    
    print(f"\nSummary:")
    print(f"Left spans: {len(left_spans)}")
    print(f"Right spans: {len(right_spans)}")  
    print(f"Center/wide spans: {len(center_spans)}")
    
    # Now try to draw reasonable boundaries
    if len(left_spans) > 0 and len(right_spans) > 0:
        # Multi-column layout
        left_x_coords = [bbox.x0 for bbox, text in left_spans] + [bbox.x1 for bbox, text in left_spans]
        right_x_coords = [bbox.x0 for bbox, text in right_spans] + [bbox.x1 for bbox, text in right_spans]
        
        left_min = min(left_x_coords)
        left_max = max(left_x_coords)
        right_min = min(right_x_coords)
        right_max = max(right_x_coords)
        
        print(f"\nColumn analysis:")
        print(f"Left column: {left_min:.1f} to {left_max:.1f}")
        print(f"Right column: {right_min:.1f} to {right_max:.1f}")
        
        # Find gap
        gap_start = left_max
        gap_end = right_min
        divider = (gap_start + gap_end) / 2
        
        print(f"Gap: {gap_start:.1f} to {gap_end:.1f}, divider: {divider:.1f}")
        
        # Draw column boundaries
        all_y = [bbox.y0 for bbox, text in body_spans] + [bbox.y1 for bbox, text in body_spans]
        text_top = min(all_y) - 10
        text_bottom = max(all_y) + 10
        
        # Left column box
        left_box = fitz.Rect(left_min - 5, text_top, divider, text_bottom)
        out_page.draw_rect(left_box, color=(0, 0.8, 0), width=3)
        
        # Right column box  
        right_box = fitz.Rect(divider, text_top, right_max + 5, text_bottom)
        out_page.draw_rect(right_box, color=(0, 0.8, 0), width=3)
        
        # Draw divider line
        out_page.draw_line(fitz.Point(divider, text_top), fitz.Point(divider, text_bottom), color=(1, 0, 0), width=2)
        
        print(f"Drew multi-column layout with divider at {divider:.1f}")
    
    out_doc.save("output/visualize_layout_test.pdf")
    out_doc.close()
    doc.close()
    
    print("Saved visualization: output/visualize_layout_test.pdf")

if __name__ == "__main__":
    visualize_page_layout()