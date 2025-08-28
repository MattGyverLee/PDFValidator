#!/usr/bin/env python3
"""
Streamlined Column Detection and Boundary Visualization

This module identifies text layout patterns and creates green boundary annotations:
- Single column body text
- Double column body text (with or without center divider)
- Full-width titles and headers
- Headers and footers

Usage:
    python columnDetection.py input.pdf [output.pdf]
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


class ColumnDetector:
    """Detects column layout and text boundaries in PDF pages"""
    
    def __init__(self, page: fitz.Page, page_number: int = 0):
        self.page = page
        self.page_number = page_number
        self.page_width = page.rect.width
        self.page_height = page.rect.height
        self.page_center = self.page_width / 2
        
        # Zone definitions
        self.header_zone = self.page_height * 0.12  # Top 12%
        self.footer_zone = self.page_height * 0.88  # Bottom 12%
        self.body_zone = (self.header_zone, self.footer_zone)
        
        # Extract all text spans with positions
        self.all_spans = self._extract_text_spans()
        
        # Detect if this is a facing-page layout
        self.is_facing_page_layout = self._detect_facing_page_layout()
        self.is_left_page = self._is_left_facing_page() if self.is_facing_page_layout else False
        
    def _extract_text_spans(self) -> List[Dict]:
        """Extract all text spans with their bounding boxes and text"""
        spans = []
        raw = self.page.get_text("dict")
        
        for block in raw.get("blocks", []):
            if block["type"] == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        bbox = span.get("bbox", [])
                        text = span.get("text", "").strip()
                        
                        if bbox and text and len(bbox) == 4:
                            spans.append({
                                'bbox': fitz.Rect(bbox),
                                'text': text,
                                'center_x': (bbox[0] + bbox[2]) / 2,
                                'center_y': (bbox[1] + bbox[3]) / 2,
                                'width': bbox[2] - bbox[0],
                                'height': bbox[3] - bbox[1]
                            })
        return spans
    
    def _detect_facing_page_layout(self) -> bool:
        """Detect if this PDF uses facing page layout by analyzing margin asymmetry"""
        if len(self.all_spans) < 10:
            return False
            
        # Get body text spans
        body_spans = [s for s in self.all_spans 
                     if self.header_zone < s['center_y'] < self.footer_zone]
        
        if len(body_spans) < 5:
            return False
        
        # Check if text is significantly offset from center
        left_distances = [s['bbox'].x0 for s in body_spans]
        right_distances = [self.page_width - s['bbox'].x1 for s in body_spans]
        
        avg_left_margin = sum(left_distances) / len(left_distances)
        avg_right_margin = sum(right_distances) / len(right_distances)
        
        # If margins differ significantly, this suggests facing page layout
        margin_ratio = max(avg_left_margin, avg_right_margin) / max(min(avg_left_margin, avg_right_margin), 1)
        
        return margin_ratio > 1.02  # Even 2% difference suggests facing pages
    
    def _is_left_facing_page(self) -> bool:
        """Determine if this is a left-facing page (larger right margin for binding)"""
        if not self.is_facing_page_layout:
            return False
            
        body_spans = [s for s in self.all_spans 
                     if self.header_zone < s['center_y'] < self.footer_zone]
        
        if not body_spans:
            return False
        
        left_distances = [s['bbox'].x0 for s in body_spans]
        right_distances = [self.page_width - s['bbox'].x1 for s in body_spans]
        
        avg_left_margin = sum(left_distances) / len(left_distances)
        avg_right_margin = sum(right_distances) / len(right_distances)
        
        # Left page: larger right margin (for binding)
        return avg_right_margin > avg_left_margin
    
    def _classify_spans_by_zone(self) -> Dict[str, List[Dict]]:
        """Classify spans into header, body, and footer zones"""
        zones = {'header': [], 'body': [], 'footer': []}
        
        for span in self.all_spans:
            if span['center_y'] < self.header_zone:
                zones['header'].append(span)
            elif span['center_y'] > self.footer_zone:
                zones['footer'].append(span)
            else:
                zones['body'].append(span)
                
        return zones
    
    def _detect_column_layout(self, body_spans: List[Dict]) -> str:
        """Determine if page has single or double column layout"""
        if len(body_spans) < 10:
            return 'single'
        
        # Classify spans by width - wide spans suggest titles/headers
        narrow_spans = [s for s in body_spans if s['width'] <= self.page_width * 0.6]
        
        if len(narrow_spans) < 5:
            return 'single'
        
        # Check distribution across page width
        left_spans = [s for s in narrow_spans if s['center_x'] < self.page_center * 0.8]
        right_spans = [s for s in narrow_spans if s['center_x'] > self.page_center * 1.2]
        
        # Need significant content in both halves for double column
        if len(left_spans) >= 5 and len(right_spans) >= 5:
            return 'double'
        
        return 'single'
    
    def _find_column_divider(self, body_spans: List[Dict]) -> Optional[float]:
        """Find the divider position between columns by detecting the actual gap"""
        # Get spans that are likely column content (not full-width)
        narrow_spans = [s for s in body_spans if s['width'] <= self.page_width * 0.6]
        
        if len(narrow_spans) < 10:
            return None
        
        # More aggressive column classification - look for clear separation
        page_center = self.page_width / 2
        
        # Find spans that are clearly on the left side
        left_column_spans = []
        for span in narrow_spans:
            # Left column: center is well left of page center AND right edge doesn't extend far right
            if (span['center_x'] < page_center - 20 and span['bbox'].x1 < page_center + 10):
                left_column_spans.append(span)
        
        # Find spans that are clearly on the right side  
        right_column_spans = []
        for span in narrow_spans:
            # Right column: center is well right of page center AND left edge starts after some point
            if (span['center_x'] > page_center + 20 and span['bbox'].x0 > page_center - 10):
                right_column_spans.append(span)
        
        if len(left_column_spans) < 3 or len(right_column_spans) < 3:
            return None
        
        # Find the actual gap between columns
        left_right_edges = [s['bbox'].x1 for s in left_column_spans]
        right_left_edges = [s['bbox'].x0 for s in right_column_spans]
        
        # Use percentile approach to avoid outliers
        left_right_edges.sort()
        right_left_edges.sort()
        
        # Use 90th percentile of left edges and 10th percentile of right edges
        left_boundary = left_right_edges[int(len(left_right_edges) * 0.9)]
        right_boundary = right_left_edges[int(len(right_left_edges) * 0.1)]
        
        gap_size = right_boundary - left_boundary
        
        if gap_size > 3:  # Must have some gap
            divider = (left_boundary + right_boundary) / 2
            return divider
        
        return None
    
    def _create_boundaries(self, layout: str, divider: Optional[float], 
                          spans_by_zone: Dict[str, List[Dict]]) -> List[TextRegion]:
        """Create text region boundaries based on detected layout"""
        regions = []
        margin = 8.0  # Boundary margin
        edge_extension = 8.5  # 3mm extension for edge characters (3mm = ~8.5pts)
        
        body_spans = spans_by_zone['body']
        if not body_spans:
            return regions
        
        # Get overall text bounds
        all_body_x = []
        all_body_y = []
        for span in body_spans:
            all_body_x.extend([span['bbox'].x0, span['bbox'].x1])
            all_body_y.extend([span['bbox'].y0, span['bbox'].y1])
        
        # Calculate margins with minimal extension - just outside text edges
        min_y = min(all_body_y) - margin
        max_y = max(all_body_y) + margin
        
        # Use large enough margin to ensure no character overlap
        consistent_margin = 8.0  # Large margin to clear all edge characters
        
        if self.is_facing_page_layout:
            # For facing pages, use consistent small margins - text shift provides the spacing
            if self.is_left_page:
                # Left page: text shifted right, so left has natural space, right is tight
                min_x = max(36.0, min(all_body_x) - consistent_margin)
                max_x = min(self.page_width, max(all_body_x) + consistent_margin)
            else:
                # Right page: text shifted left, so right has natural space, left is tight
                # Use larger left margin to ensure no overlap with leftmost characters
                left_margin = consistent_margin + 4.0  # Extra 4pts for left edge
                min_x = max(0, min(all_body_x) - left_margin)
                max_x = min(self.page_width, max(all_body_x) + consistent_margin)
        else:
            # Standard margins
            min_x = max(36.0, min(all_body_x) - consistent_margin)
            max_x = min(self.page_width - 36.0, max(all_body_x) + consistent_margin)
        
        if layout == 'double' and divider:
            # Create left and right column regions
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
            # Single column region
            single_region = TextRegion(
                bbox=fitz.Rect(min_x, min_y, max_x, max_y),
                text_type='body_single'
            )
            regions.append(single_region)
        
        # Add header and footer regions if they have content
        if spans_by_zone['header']:
            header_spans = spans_by_zone['header']
            header_x = [span['bbox'].x0 for span in header_spans] + [span['bbox'].x1 for span in header_spans]
            header_y = [span['bbox'].y0 for span in header_spans] + [span['bbox'].y1 for span in header_spans]
            
            # Apply same margin logic to headers/footers
            consistent_margin = 8.0
            if self.is_facing_page_layout:
                if self.is_left_page:
                    header_min_x = max(36.0, min(header_x) - consistent_margin)
                    header_max_x = min(self.page_width, max(header_x) + consistent_margin)
                else:
                    header_min_x = max(0, min(header_x) - consistent_margin)
                    header_max_x = min(self.page_width, max(header_x) + consistent_margin)
            else:
                header_min_x = max(36.0, min(header_x) - consistent_margin)
                header_max_x = min(self.page_width - 36.0, max(header_x) + consistent_margin)
            
            header_region = TextRegion(
                bbox=fitz.Rect(header_min_x, min(header_y) - margin,
                              header_max_x, max(header_y) + margin),
                text_type='header'
            )
            regions.append(header_region)
        
        if spans_by_zone['footer']:
            footer_spans = spans_by_zone['footer']
            footer_x = [span['bbox'].x0 for span in footer_spans] + [span['bbox'].x1 for span in footer_spans]
            footer_y = [span['bbox'].y0 for span in footer_spans] + [span['bbox'].y1 for span in footer_spans]
            
            # Apply same margin logic to footers
            consistent_margin = 8.0
            if self.is_facing_page_layout:
                if self.is_left_page:
                    footer_min_x = max(36.0, min(footer_x) - consistent_margin)
                    footer_max_x = min(self.page_width, max(footer_x) + consistent_margin)
                else:
                    footer_min_x = max(0, min(footer_x) - consistent_margin)
                    footer_max_x = min(self.page_width, max(footer_x) + consistent_margin)
            else:
                footer_min_x = max(36.0, min(footer_x) - consistent_margin)
                footer_max_x = min(self.page_width - 36.0, max(footer_x) + consistent_margin)
            
            footer_region = TextRegion(
                bbox=fitz.Rect(footer_min_x, min(footer_y) - margin,
                              footer_max_x, max(footer_y) + margin),
                text_type='footer'
            )
            regions.append(footer_region)
        
        # Identify full-width titles within body zone
        body_wide_spans = [s for s in body_spans if s['width'] > self.page_width * 0.6]
        if body_wide_spans:
            # Group consecutive wide spans into title regions
            title_groups = []
            current_group = []
            
            body_wide_spans.sort(key=lambda s: s['center_y'])
            for span in body_wide_spans:
                if (not current_group or 
                    abs(span['center_y'] - current_group[-1]['center_y']) < 30):
                    current_group.append(span)
                else:
                    if current_group:
                        title_groups.append(current_group)
                    current_group = [span]
            
            if current_group:
                title_groups.append(current_group)
            
            # Create title regions
            for i, group in enumerate(title_groups):
                title_x = [s['bbox'].x0 for s in group] + [s['bbox'].x1 for s in group]
                title_y = [s['bbox'].y0 for s in group] + [s['bbox'].y1 for s in group]
                
                # Apply same margin logic to titles
                consistent_margin = 8.0
                if self.is_facing_page_layout:
                    if self.is_left_page:
                        title_min_x = max(36.0, min(title_x) - consistent_margin)
                        title_max_x = min(self.page_width, max(title_x) + consistent_margin)
                    else:
                        title_min_x = max(0, min(title_x) - consistent_margin)
                        title_max_x = min(self.page_width, max(title_x) + consistent_margin)
                else:
                    title_min_x = max(36.0, min(title_x) - consistent_margin)
                    title_max_x = min(self.page_width - 36.0, max(title_x) + consistent_margin)
                
                title_region = TextRegion(
                    bbox=fitz.Rect(title_min_x, min(title_y) - margin,
                                  title_max_x, max(title_y) + margin),
                    text_type='title'
                )
                regions.append(title_region)
        
        return regions
    
    def detect_text_regions(self) -> List[TextRegion]:
        """Main method to detect all text regions on the page"""
        spans_by_zone = self._classify_spans_by_zone()
        layout = self._detect_column_layout(spans_by_zone['body'])
        
        divider = None
        if layout == 'double':
            divider = self._find_column_divider(spans_by_zone['body'])
            if divider is None:
                layout = 'single'  # Fallback if divider detection fails
        
        regions = self._create_boundaries(layout, divider, spans_by_zone)
        return regions


def annotate_page_boundaries(page: fitz.Page, regions: List[TextRegion]) -> None:
    """Add green boundary annotations to the page"""
    colors = {
        'body_single': (0, 0.8, 0),    # Bright green
        'body_left': (0, 0.7, 0),      # Medium green  
        'body_right': (0, 0.7, 0),     # Medium green
        'header': (0, 0.5, 0.2),       # Green-teal
        'footer': (0, 0.5, 0.2),       # Green-teal
        'title': (0.2, 0.8, 0),        # Yellow-green
    }
    
    for region in regions:
        color = colors.get(region.text_type, (0, 0.6, 0))
        
        # Add rectangle annotation
        annot = page.add_rect_annot(region.bbox)
        annot.set_colors(stroke=color)
        annot.set_border(width=2.0)
        annot.set_opacity(0.3)
        annot.update()


def process_pdf(input_path: str, output_path: str) -> None:
    """Process entire PDF and add boundary annotations"""
    doc = fitz.open(input_path)
    
    print(f"Processing {input_path} ({len(doc)} pages)...")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        detector = ColumnDetector(page, page_num + 1)  # Pass 1-based page number
        regions = detector.detect_text_regions()
        
        facing_info = ""
        if detector.is_facing_page_layout:
            facing_info = f" ({'left' if detector.is_left_page else 'right'} facing page)"
        
        print(f"Page {page_num + 1}: {len(regions)} text regions detected{facing_info}")
        for region in regions:
            print(f"  {region.text_type}: {region.bbox}")
        
        annotate_page_boundaries(page, regions)
    
    doc.save(output_path)
    doc.close()
    print(f"Annotated PDF saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python columnDetection.py input.pdf [output.pdf]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_boundaries{ext}"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        sys.exit(1)
    
    try:
        process_pdf(input_file, output_file)
    except Exception as e:
        print(f"Error processing PDF: {e}")
        sys.exit(1)