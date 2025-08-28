#!/usr/bin/env python3
"""
Streamlined Column Detection with Predictable Facing Page Layout

This module identifies text layout patterns using document-wide analysis:
- Analyzes sample pages to detect facing page layout and gutters
- Uses predictable left/right alternating pattern for facing pages  
- Creates visual text boundary detection
- Generates green boundary annotations

Usage:
    python columnDetection_new.py input.pdf [output.pdf]
"""

import fitz  # PyMuPDF
import sys
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class TextRegion:
    """Represents a text region with its type and boundaries"""
    bbox: fitz.Rect
    text_type: str  # 'body_single', 'body_left', 'body_right', 'header', 'footer', 'title'
    confidence: float = 1.0


class DocumentLayout:
    """Analyzes document-wide layout patterns"""
    
    def __init__(self, doc: fitz.Document):
        self.doc = doc
        self.has_facing_pages = False
        self.has_gutters = False
        self.left_page_left_margin = 36.0
        self.left_page_right_margin = 36.0
        self.right_page_left_margin = 36.0
        self.right_page_right_margin = 36.0
        self._analyze_document_layout()
    
    def _analyze_document_layout(self):
        """Analyze first 10 pages to determine document layout pattern"""
        sample_pages = min(10, len(self.doc))
        if sample_pages < 4:
            return
            
        page_margins = []
        
        for page_num in range(sample_pages):
            page = self.doc[page_num]
            spans = self._get_body_spans(page)
            if spans:
                left_edges = [s['bbox'].x0 for s in spans]
                right_edges = [s['bbox'].x1 for s in spans]
                left_margin = min(left_edges)
                right_margin = page.rect.width - max(right_edges)
                page_margins.append((left_margin, right_margin))
        
        if len(page_margins) >= 4:
            # Separate odd and even pages (1-indexed)
            odd_margins = [page_margins[i] for i in range(0, len(page_margins), 2)]  # Pages 1, 3, 5...
            even_margins = [page_margins[i] for i in range(1, len(page_margins), 2)]  # Pages 2, 4, 6...
            
            if len(odd_margins) >= 2 and len(even_margins) >= 2:
                odd_left_avg = sum(m[0] for m in odd_margins) / len(odd_margins)
                odd_right_avg = sum(m[1] for m in odd_margins) / len(odd_margins)
                even_left_avg = sum(m[0] for m in even_margins) / len(even_margins)
                even_right_avg = sum(m[1] for m in even_margins) / len(even_margins)
                
                # Check if there's a significant alternating pattern
                left_diff = abs(odd_left_avg - even_left_avg)
                right_diff = abs(odd_right_avg - even_right_avg)
                
                if left_diff > 10 or right_diff > 10:  # Significant margin difference
                    self.has_facing_pages = True
                    self.has_gutters = True
                    
                    # Assign margins: odd pages = left-facing, even pages = right-facing
                    self.left_page_left_margin = odd_left_avg
                    self.left_page_right_margin = odd_right_avg
                    self.right_page_left_margin = even_left_avg
                    self.right_page_right_margin = even_right_avg
                    
                    print(f"Detected facing pages with gutters:")
                    print(f"  Left pages (odd): L={odd_left_avg:.1f}, R={odd_right_avg:.1f}")
                    print(f"  Right pages (even): L={even_left_avg:.1f}, R={even_right_avg:.1f}")
    
    def _get_body_spans(self, page: fitz.Page) -> List[Dict]:
        """Get body text spans (excluding headers/footers)"""
        raw = page.get_text("dict")
        spans = []
        
        # Define body region as middle 70% of page height
        page_height = page.rect.height
        body_top = page_height * 0.15
        body_bottom = page_height * 0.85
        
        for block in raw.get("blocks", []):
            if block["type"] == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        bbox = span.get("bbox", [])
                        text = span.get("text", "").strip()
                        if bbox and text:
                            center_y = (bbox[1] + bbox[3]) / 2
                            if body_top <= center_y <= body_bottom:
                                spans.append({
                                    'bbox': fitz.Rect(bbox),
                                    'text': text,
                                    'center_y': center_y
                                })
        return spans


class ColumnDetector:
    """Detects column layout and creates text region boundaries"""
    
    def __init__(self, page: fitz.Page, page_number: int = 1, document_layout: DocumentLayout = None):
        self.page = page
        self.page_number = page_number  # 1-indexed
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        self.document_layout = document_layout
        
        # Zone definitions
        self.header_zone_end = self.page_height * 0.15
        self.footer_zone_start = self.page_height * 0.85
        
        # Determine page position based on document layout
        if document_layout and document_layout.has_facing_pages:
            self.is_facing_page_layout = True
            self.is_left_page = (page_number % 2 == 1)  # Odd pages = left-facing
        else:
            self.is_facing_page_layout = False
            self.is_left_page = False
    
    def detect_text_regions(self) -> List[TextRegion]:
        """Main method to detect and return text regions"""
        spans_by_zone = self._classify_spans_by_zone()
        layout = self._detect_column_layout(spans_by_zone['body'])
        divider = self._find_column_divider(spans_by_zone['body']) if layout == 'double' else None
        
        return self._create_boundaries(layout, divider, spans_by_zone)
    
    def _classify_spans_by_zone(self) -> Dict[str, List[Dict]]:
        """Classify spans into header, body, and footer zones"""
        raw = self.page.get_text("dict")
        zones = {'header': [], 'body': [], 'footer': []}
        
        for block in raw.get("blocks", []):
            if block["type"] == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        bbox = span.get("bbox", [])
                        text = span.get("text", "").strip()
                        if bbox and text:
                            center_y = (bbox[1] + bbox[3]) / 2
                            span_data = {
                                'bbox': fitz.Rect(bbox),
                                'text': text,
                                'center_y': center_y
                            }
                            
                            if center_y < self.header_zone_end:
                                zones['header'].append(span_data)
                            elif center_y > self.footer_zone_start:
                                zones['footer'].append(span_data)
                            else:
                                zones['body'].append(span_data)
        
        return zones
    
    def _detect_column_layout(self, body_spans: List[Dict]) -> str:
        """Detect if body text is single or double column"""
        if not body_spans or len(body_spans) < 10:
            return 'single'
        
        # Check for visual separator first - strong indicator of double column
        visual_divider = self._detect_visual_separator()
        if visual_divider:
            # If there's a clear visual divider, check if text is balanced around it
            left_spans = sum(1 for span in body_spans 
                           if (span['bbox'].x0 + span['bbox'].x1) / 2 < visual_divider)
            right_spans = sum(1 for span in body_spans 
                            if (span['bbox'].x0 + span['bbox'].x1) / 2 > visual_divider)
            
            if left_spans >= 8 and right_spans >= 8:
                return 'double'
        
        # Fallback to text distribution analysis
        x_positions = []
        for span in body_spans:
            center_x = (span['bbox'].x0 + span['bbox'].x1) / 2
            x_positions.append(center_x)
        
        x_positions.sort()
        page_center = self.page_width / 2
        
        # Count spans in left and right halves
        left_count = sum(1 for x in x_positions if x < page_center)
        right_count = sum(1 for x in x_positions if x >= page_center)
        
        # Look for gap around center
        center_tolerance = self.page_width * 0.08  # 8% of page width
        center_spans = sum(1 for x in x_positions if abs(x - page_center) < center_tolerance)
        
        # Double column criteria:
        # 1. Both sides have reasonable content
        # 2. Few spans cross the center area (indicating a gap)
        # 3. Content is distributed across page width
        total_spans = len(body_spans)
        center_ratio = center_spans / total_spans
        balance_ratio = min(left_count, right_count) / max(left_count, right_count) if max(left_count, right_count) > 0 else 0
        
        if (left_count >= 8 and right_count >= 8 and 
            center_ratio < 0.2 and  # Less than 20% of spans in center (relaxed for visual dividers)
            balance_ratio > 0.3):    # Reasonable balance between columns
            return 'double'
        else:
            return 'single'
    
    def _find_column_divider(self, body_spans: List[Dict]) -> Optional[float]:
        """Find the divider between columns using visual elements and gap detection"""
        if not body_spans:
            return None
        
        # Visual separator (black vertical line) is the absolute source of truth
        visual_divider = self._detect_visual_separator()
        if visual_divider:
            return visual_divider
        
        # Only use text gap analysis when no visual divider exists
        return self._detect_text_gap_divider(body_spans)
    
    def _detect_visual_separator(self) -> Optional[float]:
        """Detect visual separator lines like vertical rules"""
        paths = self.page.get_drawings()
        
        # Look for vertical lines that could be column separators
        # Don't restrict to page center - could be anywhere in middle 60% of page
        left_boundary = self.page_width * 0.2   # 20% from left
        right_boundary = self.page_width * 0.8  # 20% from right
        
        candidates = []
        
        for path in paths:
            # Look for vertical stroke paths
            if path['type'] == 's':  # stroke path
                items = path.get('items', [])
                for item_type, *points in items:
                    if item_type == 'l' and len(points) >= 2:  # line
                        p1, p2 = points[:2]
                        # Check if it's a substantial vertical line in the middle area
                        if (abs(p1.x - p2.x) < 3 and  # Nearly vertical (allow slight angle)
                            abs(p1.y - p2.y) > self.page_height * 0.3 and  # At least 30% of page height
                            left_boundary <= p1.x <= right_boundary):  # In middle area
                            candidates.append(p1.x)
        
        # If we found vertical lines, return the one closest to page center
        if candidates:
            page_center = self.page_width / 2
            return min(candidates, key=lambda x: abs(x - page_center))
        
        return None
    
    def _detect_text_gap_divider(self, body_spans: List[Dict]) -> Optional[float]:
        """Detect column divider based on largest horizontal text gap"""
        # Collect all horizontal text positions
        all_x_positions = []
        for span in body_spans:
            all_x_positions.extend([span['bbox'].x0, span['bbox'].x1])
        
        all_x_positions.sort()
        
        # Find the largest gap in the middle portion of the page
        page_center = self.page_width / 2
        center_zone_start = page_center - (self.page_width * 0.25)  # 25% left of center
        center_zone_end = page_center + (self.page_width * 0.25)    # 25% right of center
        
        largest_gap = 0
        best_divider = None
        
        for i in range(1, len(all_x_positions)):
            gap_start = all_x_positions[i-1]
            gap_end = all_x_positions[i]
            gap_size = gap_end - gap_start
            gap_center = (gap_start + gap_end) / 2
            
            # Only consider gaps in the center zone and larger than current best
            if (gap_size > largest_gap and 
                gap_size > 10 and  # Minimum meaningful gap
                center_zone_start <= gap_center <= center_zone_end):
                largest_gap = gap_size
                best_divider = gap_center
        
        # Verify this creates a reasonable column split
        if best_divider:
            left_spans = sum(1 for span in body_spans 
                           if (span['bbox'].x0 + span['bbox'].x1) / 2 < best_divider)
            right_spans = sum(1 for span in body_spans 
                            if (span['bbox'].x0 + span['bbox'].x1) / 2 > best_divider)
            
            # Need reasonable balance between columns
            if left_spans >= 5 and right_spans >= 5:
                return best_divider
        
        return None
    
    def _create_boundaries(self, layout: str, divider: Optional[float], 
                          spans_by_zone: Dict[str, List[Dict]]) -> List[TextRegion]:
        """Create text region boundaries with visual margins"""
        regions = []
        body_spans = spans_by_zone['body']
        
        if not body_spans:
            return regions
        
        # Get text bounds
        all_body_x = [coord for span in body_spans for coord in [span['bbox'].x0, span['bbox'].x1]]
        all_body_y = [coord for span in body_spans for coord in [span['bbox'].y0, span['bbox'].y1]]
        
        # Calculate boundaries with visual clearance
        boundary_margin = 12.0  # Visual clearance around text
        min_y = min(all_body_y) - boundary_margin
        max_y = max(all_body_y) + boundary_margin
        
        # Determine horizontal boundaries based on document layout
        if self.document_layout and self.document_layout.has_facing_pages:
            if self.is_left_page:
                # Left-facing page: use document's left page margins
                text_left = min(all_body_x)
                text_right = max(all_body_x)
                min_x = max(0, text_left - boundary_margin)
                max_x = min(self.page_width, text_right + boundary_margin)
            else:
                # Right-facing page: use document's right page margins  
                text_left = min(all_body_x)
                text_right = max(all_body_x)
                min_x = max(0, text_left - boundary_margin)
                max_x = min(self.page_width, text_right + boundary_margin)
        else:
            # Standard single-page layout
            min_x = max(36.0, min(all_body_x) - boundary_margin)
            max_x = min(self.page_width - 36.0, max(all_body_x) + boundary_margin)
        
        # Create body regions
        if layout == 'double' and divider:
            # Check if this divider came from a visual separator (definitive center)
            visual_divider = self._detect_visual_separator()
            
            if visual_divider and abs(visual_divider - divider) < 1:  # Same divider
                # Visual separator: create symmetric columns around center
                left_boundary_distance = divider - min_x
                symmetric_right_boundary = divider + left_boundary_distance
                
                # Ensure we don't exceed page boundaries
                final_right_boundary = min(symmetric_right_boundary, self.page_width)
                
                left_region = TextRegion(
                    bbox=fitz.Rect(min_x, min_y, divider, max_y),
                    text_type='body_left'
                )
                right_region = TextRegion(
                    bbox=fitz.Rect(divider, min_y, final_right_boundary, max_y),
                    text_type='body_right'
                )
                regions.extend([left_region, right_region])
            else:
                # Text-based divider: use actual text boundaries
                left_region = TextRegion(
                    bbox=fitz.Rect(min_x, min_y, divider, max_y),
                    text_type='body_left'
                )
                right_region = TextRegion(
                    bbox=fitz.Rect(divider, min_y, max_x, max_y),
                    text_type='body_right'
                )
                regions.extend([left_region, right_region])
        else:
            single_region = TextRegion(
                bbox=fitz.Rect(min_x, min_y, max_x, max_y),
                text_type='body_single'
            )
            regions.append(single_region)
        
        # Add header and footer regions if they exist
        for zone_name in ['header', 'footer']:
            zone_spans = spans_by_zone[zone_name]
            if zone_spans:
                zone_x = [coord for span in zone_spans for coord in [span['bbox'].x0, span['bbox'].x1]]
                zone_y = [coord for span in zone_spans for coord in [span['bbox'].y0, span['bbox'].y1]]
                
                zone_min_x = max(0, min(zone_x) - boundary_margin)
                zone_max_x = min(self.page_width, max(zone_x) + boundary_margin)
                zone_min_y = min(zone_y) - boundary_margin
                zone_max_y = max(zone_y) + boundary_margin
                
                zone_region = TextRegion(
                    bbox=fitz.Rect(zone_min_x, zone_min_y, zone_max_x, zone_max_y),
                    text_type=zone_name
                )
                regions.append(zone_region)
        
        return regions


def annotate_page_boundaries(page: fitz.Page, regions: List[TextRegion]) -> None:
    """Add green boundary annotations to a page"""
    colors = {
        'body_single': [0, 0.8, 0],      # Green
        'body_left': [0, 0.8, 0],        # Green
        'body_right': [0, 0.8, 0],       # Green
        'header': [0, 0.6, 0.8],         # Teal
        'footer': [0, 0.6, 0.8],         # Teal
        'title': [0.8, 0.4, 0]           # Orange
    }
    
    for region in regions:
        color = colors.get(region.text_type, [0, 0.5, 0])
        annot = page.add_rect_annot(region.bbox)
        annot.set_colors(stroke=color)
        annot.set_border(width=1.5)
        annot.update()


def process_pdf(input_path: str, output_path: str) -> None:
    """Process a PDF file and add column detection annotations"""
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' not found.")
        return
    
    try:
        doc = fitz.open(input_path)
        print(f"Processing {input_path} ({len(doc)} pages)...")
        
        # Analyze document layout first
        document_layout = DocumentLayout(doc)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            detector = ColumnDetector(page, page_num + 1, document_layout)
            regions = detector.detect_text_regions()
            
            facing_info = ""
            if detector.is_facing_page_layout:
                facing_info = f" ({'left' if detector.is_left_page else 'right'} facing page)"
            
            print(f"Page {page_num + 1}: {len(regions)} text regions detected{facing_info}")
            
            # Show detected regions
            for region in regions:
                print(f"  {region.text_type}: {region.bbox}")
            
            # Add annotations
            annotate_page_boundaries(page, regions)
        
        # Save annotated PDF
        doc.save(output_path)
        doc.close()
        print(f"Annotated PDF saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing PDF: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python columnDetection_new.py input.pdf [output.pdf]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else f"annotated_{os.path.basename(input_file)}"
    
    process_pdf(input_file, output_file)