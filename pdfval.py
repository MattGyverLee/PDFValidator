#!/usr/bin/env python3
# verify_pdf.py
"""
Deterministic PDF checks using PyMuPDF (MuPDF):
- Margin intrusions (text/images too close to edges)
- Overlapping text spans / text over images
- Non-embedded fonts
- Low-DPI images
- Page box sanity (Crop < Media, etc.)

Usage:
  python verify_pdf.py input.pdf --margin-pts 36 --dpi-threshold 300 --annotate out_annotated.pdf --json out_report.json
"""

import json, math, argparse, sys, re, os, glob
import fitz  # PyMuPDF
import numpy as np
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class TextRegion:
    """Represents a text region with its type and boundaries"""
    bbox: fitz.Rect
    text_type: str  # 'body_single', 'body_left', 'body_right', 'header', 'footer', 'title', 'section_break', 'full_width'
    confidence: float = 1.0
    spans_count: int = 0  # Number of text spans in this region


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


class EnhancedColumnDetector:
    """Enhanced column detector that replaces the old detect_column_boundaries function"""
    
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
    
    def detect_column_boundaries(self) -> List[float]:
        """Main method to detect column boundaries - returns list of divider positions"""
        body_spans = self._get_body_spans()
        layout = self._detect_column_layout(body_spans)
        
        if layout == 'double':
            divider = self._find_column_divider(body_spans)
            return [divider] if divider else []
        else:
            return []  # Single column
    
    def _get_body_spans(self) -> List[Dict]:
        """Get body text spans for analysis"""
        raw = self.page.get_text("dict")
        spans = []
        
        for block in raw.get("blocks", []):
            if block["type"] == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        bbox = span.get("bbox", [])
                        text = span.get("text", "").strip()
                        if bbox and text:
                            center_y = (bbox[1] + bbox[3]) / 2
                            if self.header_zone_end <= center_y <= self.footer_zone_start:
                                spans.append({
                                    'bbox': fitz.Rect(bbox),
                                    'text': text,
                                    'center_y': center_y
                                })
        return spans
    
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
        total_spans = len(body_spans)
        center_ratio = center_spans / total_spans
        balance_ratio = min(left_count, right_count) / max(left_count, right_count) if max(left_count, right_count) > 0 else 0
        
        if (left_count >= 8 and right_count >= 8 and 
            center_ratio < 0.2 and  # Less than 20% of spans in center
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


def simple_boundary_validation(page):
    """Simplified boundary validation - replaces complex margin detection"""
    raw = page.get_text("dict")
    blocks = raw.get("blocks", [])
    
    page_height = page.rect.height
    page_width = page.rect.width
    header_zone = page_height * 0.1
    footer_zone = page_height * 0.9
    
    body_spans = []
    for block in blocks:
        if block["type"] == 0:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    bbox = span.get("bbox", [])
                    text = span.get("text", "")
                    
                    if (bbox and text.strip() and len(bbox) == 4 and
                        bbox[2] - bbox[0] > 2 and bbox[3] - bbox[1] > 2):
                        
                        span_center_y = (bbox[1] + bbox[3]) / 2
                        if header_zone < span_center_y < footer_zone:
                            body_spans.append((fitz.Rect(bbox), text))
    
    # Detect layout 
    page_center = page_width / 2
    clear_left = [bbox for bbox, text in body_spans 
                  if bbox.x1 - bbox.x0 <= page_width * 0.6 and (bbox.x0 + bbox.x1) / 2 < page_center - 50]
    clear_right = [bbox for bbox, text in body_spans 
                   if bbox.x1 - bbox.x0 <= page_width * 0.6 and (bbox.x0 + bbox.x1) / 2 > page_center + 50]
    
    divider = None
    if len(clear_left) >= 5 and len(clear_right) >= 5:
        left_end = max(bbox.x1 for bbox in clear_left)
        right_start = min(bbox.x0 for bbox in clear_right)
        if right_start > left_end + 5:
            divider = (left_end + right_start) / 2
    
    # Create boundaries
    if not body_spans:
        return []
    
    all_x = [bbox.x0 for bbox, text in body_spans] + [bbox.x1 for bbox, text in body_spans]
    all_y = [bbox.y0 for bbox, text in body_spans] + [bbox.y1 for bbox, text in body_spans]
    
    margin = 12.0  # Expanded margin to reduce boundary violations
    if divider:  # Multi-column
        boundaries = [
            fitz.Rect(max(36.0, min(all_x) - margin), min(all_y) - margin, divider, max(all_y) + margin),
            fitz.Rect(divider, min(all_y) - margin, min(page_width - 36.0, max(all_x) + margin), max(all_y) + margin)
        ]
    else:  # Single column
        boundaries = [fitz.Rect(max(36.0, min(all_x) - margin), min(all_y) - margin, 
                                min(page_width - 36.0, max(all_x) + margin), max(all_y) + margin)]
    
    # Check violations
    violations = []
    for bbox, text in body_spans:
        if bbox.x1 - bbox.x0 > page_width * 0.6:  # Skip wide spans
            continue
        
        if not any(boundary.contains(bbox) for boundary in boundaries):
            violations.append({
                "type": "boundary_violation", 
                "detail": f"Text outside boundaries: '{text[:40]}'",
                "bbox": list(bbox)
            })
    
    return violations

def rects_intersect(a, b, iou_thresh=0.05):
    # a,b are fitz.Rect; compute IoU-ish to avoid tiny touches
    inter = a & b
    if inter.is_empty: return False
    inter_area = inter.get_area()
    union_area = a.get_area() + b.get_area() - inter_area
    return (inter_area / union_area) >= iou_thresh

def detect_column_boundaries(page, page_number=1, document_layout=None):
    """
    Enhanced column detection using the improved logic from columnDetection.py
    
    Args:
        page: PyMuPDF page object
        page_number: 1-indexed page number (for facing page detection)
        document_layout: DocumentLayout object for gutter-aware detection
    
    Returns:
        List of column divider positions (empty list for single column)
    """
    detector = EnhancedColumnDetector(page, page_number, document_layout)
    return detector.detect_column_boundaries()

def text_spans_actually_overlap(rect1, text1, rect2, text2, column_boundaries=None):
    """
    Refined text overlap detection:
    - Horizontal overlap: Always problematic (especially cross-column)
    - Vertical overlap: Only when actual text areas touch (not just headroom)
    """
    # Skip empty or whitespace-only text spans (common in PDFs with encoding issues)
    text1_clean = text1.strip()
    text2_clean = text2.strip()
    if not text1_clean or not text2_clean:
        return False
    
    # First check if bounding boxes intersect at all
    inter = rect1 & rect2
    if inter.is_empty:
        return False
    
    # Skip identical text (repeated headers, page numbers)
    if text1_clean == text2_clean:
        return False
    
    # Skip if one text is completely contained in the other (OCR artifacts)
    text1_lower = text1_clean.lower()
    text2_lower = text2_clean.lower()
    if text1_lower in text2_lower or text2_lower in text1_lower:
        return False
    
    # Calculate intersection dimensions
    inter_width = inter.x1 - inter.x0
    inter_height = inter.y1 - inter.y0
    
    # HORIZONTAL OVERLAP DETECTION (always problematic - minimal text box padding)
    if inter_width > 0.1:  # Very strict - text boxes have minimal horizontal padding
        # Skip if both spans are very wide (likely full-width paragraphs, not column text)
        rect1_width = rect1.x1 - rect1.x0
        rect2_width = rect2.x1 - rect2.x0
        
        # If both spans are very wide (>60% page width), they're likely paragraph blocks
        # Horizontal overlap between paragraph blocks is usually just formatting
        page_width = 419.5  # Approximate page width - could be passed as parameter
        if rect1_width > page_width * 0.6 and rect2_width > page_width * 0.6:
            # These are likely paragraph blocks, treat as vertical overlap instead
            pass  # Fall through to vertical overlap logic
        else:
            # Narrow text blocks with horizontal overlap - this is always problematic
            center_x1 = (rect1.x0 + rect1.x1) / 2
            center_x2 = (rect2.x0 + rect2.x1) / 2
            
            # Cross-column overlap is especially serious (first column extending into second)
            if column_boundaries:
                col1 = 0
                col2 = 0
                for boundary in column_boundaries:
                    if center_x1 > boundary:
                        col1 += 1
                    if center_x2 > boundary:
                        col2 += 1
                
                # Any cross-column overlap is a major issue
                if col1 != col2:
                    return True
            
            # Even same-column horizontal overlap is problematic with minimal padding
            # Only allow very tiny overlaps (possible rounding errors)
            if inter_width > 0.5:
                return True
    
    # VERTICAL OVERLAP DETECTION - only flag when text actually touches
    if inter_height > 1.0:  # Some vertical overlap exists
        # Handle empty spans differently - they might be structural elements
        text1_empty = not text1.strip()
        text2_empty = not text2.strip()
        
        # If both are empty, only flag if boxes are substantial (not artifacts)
        if text1_empty and text2_empty:
            area1 = (rect1.x1 - rect1.x0) * (rect1.y1 - rect1.y0)
            area2 = (rect2.x1 - rect2.x0) * (rect2.y1 - rect2.y0)
            
            # Skip tiny empty boxes
            if area1 < 20 or area2 < 20:
                return False
            
            # For structural elements, be more lenient with vertical overlap
            # Only flag if there's also significant horizontal overlap
            if inter_width > 3.0:
                return True
            return False
        
        # For text content, estimate actual text positions within boxes
        box1_height = rect1.y1 - rect1.y0
        box2_height = rect2.y1 - rect2.y0
        
        # Estimate actual text area within text boxes (accounting for headroom/footspace)
        # Text boxes have significant vertical padding (headroom above, footspace below)
        # Only flag when estimated actual text areas would touch and affect legibility
        
        # Conservative estimation: text occupies middle portion of text box
        # Larger text boxes likely have more headroom
        def estimate_text_area_ratio(box_height):
            if box_height < 8:   # Very small boxes - mostly text
                return 0.8
            elif box_height < 15:  # Normal single line - some headroom
                return 0.6
            else:  # Large boxes - significant headroom/leading
                return 0.4
        
        text1_ratio = estimate_text_area_ratio(box1_height)
        text2_ratio = estimate_text_area_ratio(box2_height)
        
        # Calculate estimated text bounds (centered in box)
        text1_padding = box1_height * (1.0 - text1_ratio) / 2
        text1_top = rect1.y0 + text1_padding
        text1_bottom = rect1.y1 - text1_padding
        
        text2_padding = box2_height * (1.0 - text2_ratio) / 2
        text2_top = rect2.y0 + text2_padding
        text2_bottom = rect2.y1 - text2_padding
        
        # Check if estimated text areas actually overlap
        text_overlap_top = max(text1_top, text2_top)
        text_overlap_bottom = min(text1_bottom, text2_bottom)
        
        if text_overlap_bottom > text_overlap_top:
            # Estimated text areas do overlap vertically
            text_overlap_height = text_overlap_bottom - text_overlap_top
            
            # Only flag if overlap would significantly affect legibility
            # Be more conservative - allow overlap unless text areas really touch
            min_legible_separation = 4.0  # More generous minimum separation for readability
            if text_overlap_height > min_legible_separation:
                return True
    
    return False

def identify_problematic_line(rect1, text1, rect2, text2, column_boundaries):
    """
    Identify text boxes that cross the column boundary (gutter space).
    NO text should cross the vertical line between columns.
    Returns: 0 for rect1, 1 for rect2, None if neither crosses boundary
    """
    if not column_boundaries:
        return None
    
    main_boundary = column_boundaries[0]
    
    # Check if either text box crosses the column boundary
    # Left column text should not extend past boundary
    # Right column text should not extend before boundary
    
    crosses_boundary_1 = rect1.x0 < main_boundary < rect1.x1
    crosses_boundary_2 = rect2.x0 < main_boundary < rect2.x1
    
    # If both cross, return the one that crosses more severely
    if crosses_boundary_1 and crosses_boundary_2:
        # Calculate how much each crosses
        cross_amount_1 = min(rect1.x1 - main_boundary, main_boundary - rect1.x0)
        cross_amount_2 = min(rect2.x1 - main_boundary, main_boundary - rect2.x0)
        return 0 if cross_amount_1 > cross_amount_2 else 1
    
    # Return whichever one crosses the boundary
    if crosses_boundary_1:
        return 0
    elif crosses_boundary_2:
        return 1
    
    # If neither crosses the boundary, determine column assignment
    center_x1 = (rect1.x0 + rect1.x1) / 2
    center_x2 = (rect2.x0 + rect2.x1) / 2
    
    col1 = 0 if center_x1 < main_boundary else 1
    col2 = 0 if center_x2 < main_boundary else 1
    
    # If they're in the same column, flag the one extending further toward boundary
    if col1 == col2:
        if col1 == 0:  # Both in left column
            # Flag the one extending closer to the boundary
            if rect1.x1 > rect2.x1:
                return 0
            else:
                return 1
        else:  # Both in right column
            # Flag the one extending closer to the boundary  
            if rect1.x0 < rect2.x0:
                return 0
            else:
                return 1
    
    return None

def validate_text_justification(page, column_boundaries=None):
    """
    Validate text justification by checking if text extends into margins.
    Text should be properly justified and not extend beyond expected margins.
    """
    issues = []
    
    if not column_boundaries:
        # Detect column boundaries if not provided
        column_boundaries = detect_column_boundaries(page)
    
    if not column_boundaries:
        return issues
    
    main_boundary = column_boundaries[0]
    page_width = page.rect.width
    
    # Get all text spans first to analyze layout
    raw = page.get_text("dict")
    blocks = raw.get("blocks", [])
    
    all_spans = []
    for b in blocks:
        if b["type"] == 0:  # text block
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    bbox = fitz.Rect(span["bbox"])
                    text = span.get("text", "")
                    if bbox.width > 5:  # Only substantial spans
                        all_spans.append((bbox, text))
    
    if not all_spans:
        return issues
    
    # Check if this is actually a multi-column layout
    # Analyze column positioning
    col1_spans = []
    col2_spans = []
    
    for bbox, text in all_spans:
        center_x = (bbox.x0 + bbox.x1) / 2
        if center_x < main_boundary:
            col1_spans.append((bbox, text))
        else:
            col2_spans.append((bbox, text))
    
    # If we don't have a reasonable distribution between columns, skip gutter validation
    # This handles single-column layouts or where boundary detection failed
    total_spans = len(all_spans)
    if total_spans > 0:
        left_ratio = len(col1_spans) / total_spans
        right_ratio = len(col2_spans) / total_spans
        
        # If more than 80% of text is on one side, treat as single column
        if left_ratio > 0.8 or right_ratio > 0.8:
            return issues  # Skip gutter validation for single-column layouts
        
        # If we have very few spans total, the boundary detection may be unreliable
        if total_spans < 10:
            return issues
    
    # Check for text extending into gutter area (column boundary violations)
    gutter_tolerance = 5.0  # Allow 5pt buffer from column boundary
    
    for bbox, text in all_spans:
        if bbox.x1 - bbox.x0 > 10:  # Only check substantial text spans
            # Check if text crosses or gets too close to column boundary
            if bbox.x0 < main_boundary < bbox.x1:
                # Text definitely crosses the boundary
                issues.append({
                    "type": "text_extends_to_gutter", 
                    "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    "detail": f"Text crosses column boundary: extends from x={bbox.x0:.1f} to x={bbox.x1:.1f}, crossing boundary at {main_boundary:.1f}"
                })
            elif abs(bbox.x1 - main_boundary) < gutter_tolerance and bbox.x0 < main_boundary:
                # Left column text getting too close to boundary
                issues.append({
                    "type": "text_extends_to_gutter", 
                    "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1],
                    "detail": f"Text extends to x={bbox.x1:.1f}, too close to column boundary at {main_boundary:.1f}"
                })
            elif abs(bbox.x0 - main_boundary) < gutter_tolerance and bbox.x1 > main_boundary:
                # Right column text getting too close to boundary
                issues.append({
                    "type": "text_extends_to_gutter",
                    "bbox": [bbox.x0, bbox.y0, bbox.x1, bbox.y1], 
                    "detail": f"Text starts at x={bbox.x0:.1f}, too close to column boundary at {main_boundary:.1f}"
                })
    
    return issues

def analyze_gutter_margins(doc, min_margin_pts=36.0):
    """
    Analyze document to determine if it uses guttered margins for book binding.
    Returns margin info for proper validation of book layouts.
    """
    margin_analysis = {
        "uses_gutters": False,
        "odd_page_margins": {},  # left, right, top, bottom
        "even_page_margins": {},
        "consistent_margins": {},  # for non-guttered layouts
        "margin_issues": []
    }
    
    if len(doc) < 2:
        # Single page - assume consistent margins
        page = doc[0]
        pr = page.rect
        margin_analysis["consistent_margins"] = {
            "left": min_margin_pts,
            "right": min_margin_pts, 
            "top": min_margin_pts,
            "bottom": min_margin_pts
        }
        return margin_analysis
    
    # Sample first few pages to detect margin pattern
    sample_pages = min(6, len(doc))
    odd_left_margins = []
    odd_right_margins = []
    even_left_margins = []
    even_right_margins = []
    
    for page_idx in range(sample_pages):
        page = doc[page_idx]
        page_num = page_idx + 1  # 1-based page numbering
        
        # Get text blocks to estimate actual margins
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])
        
        if not blocks:
            continue
            
        # Find leftmost and rightmost text positions
        left_positions = []
        right_positions = []
        
        for block in blocks:
            if block["type"] == 0:  # text block
                bbox = block["bbox"]
                left_positions.append(bbox[0])
                right_positions.append(bbox[2])
        
        if not left_positions or not right_positions:
            continue
            
        # Estimate margins based on text placement
        page_width = page.rect.width
        left_margin = max(0, min(left_positions)) if left_positions else min_margin_pts
        right_margin = max(0, page_width - max(right_positions)) if right_positions else min_margin_pts
        
        if page_num % 2 == 1:  # Odd page
            odd_left_margins.append(left_margin)
            odd_right_margins.append(right_margin)
        else:  # Even page
            even_left_margins.append(left_margin)
            even_right_margins.append(right_margin)
    
    # Analyze margin patterns
    if odd_left_margins and odd_right_margins and even_left_margins and even_right_margins:
        avg_odd_left = sum(odd_left_margins) / len(odd_left_margins)
        avg_odd_right = sum(odd_right_margins) / len(odd_right_margins)
        avg_even_left = sum(even_left_margins) / len(even_left_margins)
        avg_even_right = sum(even_right_margins) / len(even_right_margins)
        
        # Check if margins are mirrored (gutter pattern)
        # Odd pages: left=gutter (larger), right=outer (smaller)
        # Even pages: left=outer (smaller), right=gutter (larger)
        gutter_threshold = 10.0  # Points difference to consider guttered
        
        odd_left_larger = avg_odd_left - avg_odd_right > gutter_threshold
        even_right_larger = avg_even_right - avg_even_left > gutter_threshold
        
        if odd_left_larger and even_right_larger:
            # Detected gutter pattern
            margin_analysis["uses_gutters"] = True
            margin_analysis["odd_page_margins"] = {
                "left": avg_odd_left,    # gutter (inner)
                "right": avg_odd_right,  # outer
                "top": min_margin_pts,   # assume standard
                "bottom": min_margin_pts
            }
            margin_analysis["even_page_margins"] = {
                "left": avg_even_left,   # outer  
                "right": avg_even_right, # gutter (inner)
                "top": min_margin_pts,
                "bottom": min_margin_pts
            }
        else:
            # No gutter pattern detected - use consistent margins
            avg_left = max(min_margin_pts * 0.5, (avg_odd_left + avg_even_left) / 2)
            avg_right = max(min_margin_pts * 0.5, (avg_odd_right + avg_even_right) / 2)
            margin_analysis["consistent_margins"] = {
                "left": avg_left,
                "right": avg_right,
                "top": min_margin_pts,
                "bottom": min_margin_pts
            }
    
    # If no margin data was collected, use reasonable defaults
    if not margin_analysis.get("uses_gutters") and not margin_analysis.get("consistent_margins"):
        margin_analysis["consistent_margins"] = {
            "left": min_margin_pts * 0.75,
            "right": min_margin_pts * 0.75,
            "top": min_margin_pts * 0.75,
            "bottom": min_margin_pts * 0.75
        }
    
    return margin_analysis

def check_guttered_margins(page, page_number, margin_analysis, min_acceptable_margin=18.0):
    """
    Check margins with awareness of gutter layouts for book binding
    """
    issues = []
    pr = page.rect
    page_num = page_number  # 1-based
    
    # Determine expected margins based on analysis
    if margin_analysis["uses_gutters"]:
        if page_num % 2 == 1:  # Odd page
            expected_margins = margin_analysis["odd_page_margins"]
        else:  # Even page  
            expected_margins = margin_analysis["even_page_margins"]
    else:
        expected_margins = margin_analysis["consistent_margins"]
    
    if not expected_margins:
        # Fallback to default margins
        expected_margins = {"left": 36.0, "right": 36.0, "top": 36.0, "bottom": 36.0}
    
    # Check if overall margins are too small (report once per page)
    min_margin = min(expected_margins.values())
    if min_margin < min_acceptable_margin:
        issues.append({
            "type": "insufficient_margins",
            "detail": f"Page {page_num}: Margins too small (minimum: {min_margin:.1f}pt, required: {min_acceptable_margin}pt)"
        })
    
    # Create margin zones using expected margins
    left_margin_zone = fitz.Rect(pr.x0, pr.y0, pr.x0 + expected_margins["left"], pr.y1)
    right_margin_zone = fitz.Rect(pr.x1 - expected_margins["right"], pr.y0, pr.x1, pr.y1)
    top_margin_zone = fitz.Rect(pr.x0, pr.y0, pr.x1, pr.y0 + expected_margins["top"])
    bottom_margin_zone = fitz.Rect(pr.x0, pr.y1 - expected_margins["bottom"], pr.x1, pr.y1)
    
    # Get text and image elements
    raw = page.get_text("dict")
    blocks = raw.get("blocks", [])
    
    text_rects = []
    image_rects = []
    
    for b in blocks:
        if b["type"] == 0:  # text block
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    bbox = fitz.Rect(span["bbox"])
                    text_rects.append((bbox, span.get("text", "")))
        elif b["type"] == 1:  # image block
            ibox = fitz.Rect(b["bbox"])
            image_rects.append(ibox)
    
    # Check for unusual intrusions (not just any intrusion)
    intrusion_threshold = 5.0  # Points - only flag significant intrusions
    
    for rect, text in text_rects:
        # Check each margin zone
        margin_intrusions = []
        
        if rect.intersects(left_margin_zone):
            intrusion_depth = max(0, rect.x1 - left_margin_zone.x1)
            if intrusion_depth > intrusion_threshold:
                margin_intrusions.append(f"left margin by {intrusion_depth:.1f}pt")
        
        if rect.intersects(right_margin_zone):
            intrusion_depth = max(0, right_margin_zone.x0 - rect.x0) 
            if intrusion_depth > intrusion_threshold:
                margin_intrusions.append(f"right margin by {intrusion_depth:.1f}pt")
                
        if rect.intersects(top_margin_zone):
            intrusion_depth = max(0, rect.y1 - top_margin_zone.y1)
            if intrusion_depth > intrusion_threshold:
                margin_intrusions.append(f"top margin by {intrusion_depth:.1f}pt")
                
        if rect.intersects(bottom_margin_zone):
            intrusion_depth = max(0, bottom_margin_zone.y0 - rect.y0)
            if intrusion_depth > intrusion_threshold:
                margin_intrusions.append(f"bottom margin by {intrusion_depth:.1f}pt")
        
        if margin_intrusions:
            issues.append({
                "type": "margin_intrusion_text",
                "detail": f"Text '{text[:50]}' intrudes into {', '.join(margin_intrusions)}",
                "bbox": list(rect)
            })
    
    # Check image intrusions  
    for rect in image_rects:
        margin_intrusions = []
        
        if rect.intersects(left_margin_zone):
            intrusion_depth = max(0, rect.x1 - left_margin_zone.x1)
            if intrusion_depth > intrusion_threshold:
                margin_intrusions.append(f"left margin by {intrusion_depth:.1f}pt")
                
        if rect.intersects(right_margin_zone):
            intrusion_depth = max(0, right_margin_zone.x0 - rect.x0)
            if intrusion_depth > intrusion_threshold:
                margin_intrusions.append(f"right margin by {intrusion_depth:.1f}pt")
                
        if rect.intersects(top_margin_zone):
            intrusion_depth = max(0, rect.y1 - top_margin_zone.y1)
            if intrusion_depth > intrusion_threshold:
                margin_intrusions.append(f"top margin by {intrusion_depth:.1f}pt")
                
        if rect.intersects(bottom_margin_zone):
            intrusion_depth = max(0, bottom_margin_zone.y0 - rect.y0)
            if intrusion_depth > intrusion_threshold:
                margin_intrusions.append(f"bottom margin by {intrusion_depth:.1f}pt")
        
        if margin_intrusions:
            issues.append({
                "type": "margin_intrusion_image", 
                "detail": f"Image intrudes into {', '.join(margin_intrusions)}",
                "bbox": list(rect)
            })
    
    return issues

def px_per_inch_from_matrix(img_matrix):
    # MuPDF stores image transform; extract scale if available
    # Fall back to None if not meaningful
    # Typical matrix: [sx, 0, 0, sy, tx, ty]; sx,sy in pixels per point; 72 pts = 1 inch
    sx = img_matrix.x
    sy = img_matrix.y
    if sx and sy:
        ppi_x = sx * 72.0
        ppi_y = sy * 72.0
        return ppi_x, ppi_y
    return None, None

def check_pdfx_compliance(doc):
    """Check PDF/X compliance indicators"""
    issues = []
    try:
        # Check metadata for PDF/X conformance
        metadata = doc.metadata
        if metadata:
            # Look for PDF/X conformance in metadata
            gts_pdfx = metadata.get('gts_pdfx', '').lower()
            if not gts_pdfx:
                issues.append({"type": "pdfx_compliance", "detail": "No PDF/X conformance declaration found"})
            elif 'pdf/x' not in gts_pdfx:
                issues.append({"type": "pdfx_compliance", "detail": f"Non-standard PDF/X declaration: {gts_pdfx}"})
        
        # Check for OutputIntent (required for PDF/X)
        try:
            # This is a heuristic - full PDF/X validation requires deeper PDF parsing
            catalog = doc.pdf_catalog()
            if catalog and 'OutputIntents' not in str(catalog):
                issues.append({"type": "pdfx_compliance", "detail": "Missing OutputIntents (required for PDF/X)"})
        except:
            issues.append({"type": "pdfx_compliance", "detail": "Could not verify OutputIntents"})
            
    except Exception as e:
        issues.append({"type": "pdfx_compliance", "detail": f"PDF/X check failed: {str(e)}"})
    
    return issues

def analyze_color_usage(page):
    """Analyze color usage patterns on page"""
    issues = []
    try:
        # Get all drawing operations
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])
        
        rgb_found = False
        cmyk_found = False
        
        # Check text color usage
        for block in blocks:
            if block["type"] == 0:  # text block
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        color = span.get("color", 0)
                        # Check if color is not black (0) or white (16777215)
                        if color != 0 and color != 16777215:
                            # Convert color to RGB components
                            r = (color >> 16) & 255
                            g = (color >> 8) & 255  
                            b = color & 255
                            if r > 0 or g > 0 or b > 0:
                                rgb_found = True
        
        # Check images for color space
        images = page.get_images()
        for img in images:
            try:
                img_doc = fitz.open("pdf", page.parent.extract_image(img[0])["image"])
                pix = img_doc[0].get_pixmap()
                if pix.colorspace and pix.colorspace.name:
                    cs_name = pix.colorspace.name.lower()
                    if 'rgb' in cs_name or 'srgb' in cs_name:
                        rgb_found = True
                    elif 'cmyk' in cs_name:
                        cmyk_found = True
                img_doc.close()
            except:
                pass
        
        if rgb_found:
            issues.append({"type": "color_mode_rgb", "detail": "RGB content found (should be K=100% for B&W sections)"})
        
        if not cmyk_found and rgb_found:
            issues.append({"type": "color_mode_compliance", "detail": "No CMYK color space detected for color content"})
            
    except Exception as e:
        issues.append({"type": "color_analysis_error", "detail": f"Color analysis failed: {str(e)}"})
    
    return issues

def validate_page_numbering(doc):
    """Validate page numbering sequences - improved to avoid false positives"""
    issues = []
    likely_page_numbers = []
    
    for pno in range(len(doc)):
        page = doc[pno]
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])
        
        page_rect = page.rect
        page_width = page_rect.width
        page_height = page_rect.height
        
        # Look for numbers in typical page number locations (headers/footers)
        header_zone = page_height * 0.1   # Top 10%
        footer_zone = page_height * 0.9   # Bottom 10% 
        side_margin = page_width * 0.1    # Side 10%
        
        candidates = []
        
        for block in blocks:
            if block["type"] == 0:  # text block
                bbox = block["bbox"]
                block_y = (bbox[1] + bbox[3]) / 2  # center Y
                block_x = (bbox[0] + bbox[2]) / 2  # center X
                
                # Check if block is in header, footer, or margins (typical page number locations)
                in_header = block_y < header_zone
                in_footer = block_y > footer_zone
                in_side_margin = block_x < side_margin or block_x > (page_width - side_margin)
                
                if in_header or in_footer or in_side_margin:
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            
                            # Look for isolated numbers (likely page numbers)
                            # More restrictive patterns to avoid citations, years, etc.
                            patterns = [
                                r'^(\d{1,3})$',              # Just a number by itself
                                r'^\-\s*(\d{1,3})\s*\-$',    # -123-
                                r'^(\d{1,3})\s*$',           # Number with minimal text
                                r'^\s*(\d{1,3})\s*\|',       # 123 | (common in headers)
                                r'\|\s*(\d{1,3})\s*$',       # | 123 (common in headers)
                            ]
                            
                            for pattern in patterns:
                                match = re.search(pattern, text)
                                if match:
                                    num = int(match.group(1))
                                    # Reasonable page number range
                                    if 1 <= num <= 9999:
                                        candidates.append((pno, num, bbox, text))
                                        break
        
        # If we found multiple candidates on same page, pick the most likely one
        if candidates:
            # Prefer numbers in footer, then header, then sides
            footer_candidates = [c for c in candidates if c[2][3] > footer_zone * page_height]
            header_candidates = [c for c in candidates if c[2][1] < header_zone * page_height]
            
            if footer_candidates:
                # Pick the one closest to center horizontally
                best = min(footer_candidates, key=lambda x: abs(x[2][0] - page_width/2))
                likely_page_numbers.append(best)
            elif header_candidates:
                best = min(header_candidates, key=lambda x: abs(x[2][0] - page_width/2))
                likely_page_numbers.append(best)
            elif candidates:
                # Fallback to any candidate
                likely_page_numbers.append(candidates[0])
    
    # Now validate sequence with much more restrictive logic
    if len(likely_page_numbers) > 1:
        # Sort by physical page order
        likely_page_numbers.sort(key=lambda x: x[0])
        
        # Check for reasonable sequence issues
        for i in range(1, len(likely_page_numbers)):
            prev_page_idx, prev_num, prev_bbox, prev_text = likely_page_numbers[i-1]
            curr_page_idx, curr_num, curr_bbox, curr_text = likely_page_numbers[i]
            
            # Only flag if numbers are going backwards or duplicate
            # Allow for reasonable gaps (like chapter breaks)
            if curr_num < prev_num or (curr_num == prev_num and curr_page_idx != prev_page_idx):
                issues.append({
                    "type": "page_sequence_error", 
                    "detail": f"Page {curr_page_idx+1}: number {curr_num} not sequential (previous: {prev_num} on page {prev_page_idx+1})"
                })
            # Also flag extremely large jumps that might indicate wrong number detection
            elif curr_num > prev_num + 50:  # Large gap might indicate false detection
                issues.append({
                    "type": "page_sequence_warning", 
                    "detail": f"Large gap in page numbering: {prev_num} to {curr_num} (pages {prev_page_idx+1} to {curr_page_idx+1})"
                })
    
    return issues

def check_baseline_alignment(page, grid_threshold=2.0):
    """Check if text aligns to baseline grid"""
    issues = []
    baselines = []
    
    try:
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])
        
        for block in blocks:
            if block["type"] == 0:  # text block
                for line in block.get("lines", []):
                    bbox = line.get("bbox")
                    if bbox:
                        baseline_y = bbox[3]  # bottom of line bbox
                        baselines.append(baseline_y)
        
        if len(baselines) > 5:  # Need sufficient samples
            baselines = sorted(baselines)
            
            # Calculate most common grid interval
            intervals = []
            for i in range(1, len(baselines)):
                diff = baselines[i] - baselines[i-1]
                if 8 < diff < 50:  # Reasonable line spacing range
                    intervals.append(diff)
            
            if intervals:
                # Find most common interval (approximate grid)
                interval_counts = Counter([round(x) for x in intervals])
                common_interval = interval_counts.most_common(1)[0][0]
                
                # Check alignment to grid
                misaligned = 0
                for baseline in baselines:
                    grid_position = baseline % common_interval
                    if min(grid_position, common_interval - grid_position) > grid_threshold:
                        misaligned += 1
                
                if misaligned > len(baselines) * 0.2:  # More than 20% misaligned
                    issues.append({
                        "type": "baseline_grid_violation",
                        "detail": f"{misaligned}/{len(baselines)} lines not aligned to {common_interval}pt grid"
                    })
                    
    except Exception as e:
        issues.append({"type": "baseline_check_error", "detail": f"Baseline check failed: {str(e)}"})
    
    return issues

def check_column_balance(page, balance_threshold=0.15):
    """Check column balance"""
    issues = []
    
    try:
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])
        
        # Group text blocks by approximate column
        page_width = page.rect.width
        columns = defaultdict(list)
        
        for block in blocks:
            if block["type"] == 0:  # text block
                bbox = block["bbox"]
                center_x = (bbox[0] + bbox[2]) / 2
                
                # Simple 2-column detection
                if center_x < page_width / 2:
                    columns["left"].append(block)
                else:
                    columns["right"].append(block)
        
        if len(columns) == 2 and all(len(col) > 2 for col in columns.values()):
            # Count lines in each column
            left_lines = sum(len(block.get("lines", [])) for block in columns["left"])
            right_lines = sum(len(block.get("lines", [])) for block in columns["right"])
            
            if left_lines > 0 and right_lines > 0:
                imbalance = abs(left_lines - right_lines) / max(left_lines, right_lines)
                if imbalance > balance_threshold:
                    issues.append({
                        "type": "column_imbalance",
                        "detail": f"Column imbalance: {left_lines} vs {right_lines} lines ({imbalance:.1%})"
                    })
                    
    except Exception as e:
        issues.append({"type": "column_balance_error", "detail": f"Column balance check failed: {str(e)}"})
    
    return issues

def check_page(page, margin_pts, dpi_threshold, page_number=None, total_pages=None):
    issues = []
    pr = page.rect  # page rectangle (CropBox)
    W, H = pr.width, pr.height

    # Page boxes sanity (Crop should be within Media if available)
    try:
        media = page.mediabox  # not always available
        crop  = page.cropbox
        if media and crop:
            if not media.contains(crop):
                issues.append({"type":"page_box_error", "detail":"CropBox not within MediaBox"})
    except Exception:
        pass

    # Margin zones
    left_m   = fitz.Rect(pr.x0, pr.y0, pr.x0 + margin_pts, pr.y1)
    right_m  = fitz.Rect(pr.x1 - margin_pts, pr.y0, pr.x1, pr.y1)
    top_m    = fitz.Rect(pr.x0, pr.y0, pr.x1, pr.y0 + margin_pts)
    bottom_m = fitz.Rect(pr.x0, pr.y1 - margin_pts, pr.x1, pr.y1)

    # Gather text blocks / spans / glyphs and images
    raw = page.get_text("dict")
    blocks = raw.get("blocks", [])

    text_rects = []     # span-level rects
    glyph_rects = []    # per-glyph rects (optional / heavier)
    image_rects = []    # image rects
    image_ppi   = []    # (rect, ppi_x, ppi_y)

    for b in blocks:
        if b["type"] == 0:
            # text block: lines -> spans -> bbox, chars
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    bbox = fitz.Rect(span["bbox"])
                    text_rects.append((bbox, span.get("text","")))
                    # glyphs may not always be present; enable if you want stricter overlap
                    for g in span.get("chars", []):
                        gb = fitz.Rect(g["bbox"])
                        glyph_rects.append((gb, g.get("c","")))
        elif b["type"] == 1:
            # image block
            ibox = fitz.Rect(b["bbox"])
            image_rects.append(ibox)
            # Estimate placed PPI from transform matrix if available (not always)
            # NOTE: For embedded image native resolution, you'd need to resolve the image XObject.
            # PyMuPDF 1.23+ has page.get_images() then doc.extract_image(xref) for width/height.
            # Here we compute placed PPI as a proxy.
            # Fallback to None if we can't estimate.
            ppi_x, ppi_y = None, None
            if "transform" in b:
                try:
                    m = fitz.Matrix(b["transform"])
                    ppi_x, ppi_y = px_per_inch_from_matrix(m)
                except Exception:
                    pass
            image_ppi.append((ibox, ppi_x, ppi_y))

    # 1) Simplified boundary validation (replaces margin intrusion detection)
    boundary_violations = simple_boundary_validation(page)
    issues.extend(boundary_violations)

    # 1b) Margin intrusions (images)
    for r in image_rects:
        if r.intersects(left_m) or r.intersects(right_m) or r.intersects(top_m) or r.intersects(bottom_m):
            issues.append({"type":"margin_intrusion_image", "bbox":list(r)})

    # 2) Text overlap detection - identify problematic lines only
    # Detect column boundaries for better overlap analysis
    column_boundaries = detect_column_boundaries(page)
    
    # Find lines that extend beyond their proper boundaries
    problematic_lines = set()  # Track which lines are the culprits
    
    # O(n^2) over spans per page; typically fine. For big pages, you can grid-index.
    for i in range(len(text_rects)):
        ri, ti = text_rects[i]
        for j in range(i+1, len(text_rects)):
            rj, tj = text_rects[j]
            # Check if these spans overlap
            if text_spans_actually_overlap(ri, ti, rj, tj, column_boundaries):
                # Determine which line is the problematic one (extends too far)
                culprit_idx = identify_problematic_line(ri, ti, rj, tj, column_boundaries)
                if culprit_idx is not None:
                    # Mark the problematic line
                    actual_idx = i if culprit_idx == 0 else j
                    problematic_lines.add(actual_idx)
    
    # Report only lines that actually cross the column boundary
    main_boundary = column_boundaries[0] if column_boundaries else None
    if main_boundary:
        for idx in problematic_lines:
            ri, ti = text_rects[idx]
            # Double-check that this line actually crosses the boundary
            if ri.x0 < main_boundary < ri.x1:
                issues.append({
                    "type": "text_crosses_column_boundary", 
                    "detail": f"Text crosses column separator at {main_boundary:.1f}pt: '{ti[:50]}'",
                    "bbox": list(ri)
                })
    else:
        # Fallback if no boundary detected
        for idx in problematic_lines:
            ri, ti = text_rects[idx]
            issues.append({
                "type": "text_crosses_column_boundary", 
                "detail": f"Text crosses column separator: '{ti[:50]}'",
                "bbox": list(ri)
            })

    # 3) Text over image overlap
    for rtxt, txt in text_rects:
        for rimg in image_rects:
            if rects_intersect(rtxt, rimg, iou_thresh=0.02):
                issues.append({"type":"overlap_text_image", "detail":txt[:60], "bboxes":[list(rtxt), list(rimg)]})

    # 4) Low-DPI images (placed ppi proxy)
    for rimg, ppx, ppy in image_ppi:
        if ppx and ppy:
            if ppx < dpi_threshold or ppy < dpi_threshold:
                issues.append({"type":"low_dpi_image", "detail":f"{int(ppx)}x{int(ppy)} ppi", "bbox":list(rimg)})
        else:
            # Could not estimate; optional warning
            issues.append({"type":"dpi_unknown_image", "bbox":list(rimg)})

    # 5) Font embedding presence (resource-level heuristic)
    # PyMuPDF: page.get_fonts() -> [(xref, name, is_embedded, ...)]
    # NOTE: Not all PDFs make this trivial; this is a useful but not perfect proxy.
    try:
        fonts = page.get_fonts(full=True)
        for f in fonts:
            # tuple format can vary by version; guard indexes
            is_embedded = False
            if len(f) >= 3:
                is_embedded = bool(f[2])
            if not is_embedded:
                issues.append({"type":"font_not_embedded", "detail":str(f)})
    except Exception:
        issues.append({"type":"font_check_error", "detail":"Failed to read font info"})

    return issues

def detect_image_screening(page):
    """Detect screened/halftoned images using frequency analysis"""
    issues = []
    
    try:
        images = page.get_images()
        for img_index, img in enumerate(images):
            try:
                # Extract image data
                img_data = page.parent.extract_image(img[0])
                img_bytes = img_data["image"]
                
                # Open as PIL/PyMuPDF image for analysis
                img_doc = fitz.open("pdf", img_bytes)
                pix = img_doc[0].get_pixmap()
                
                if pix.width > 50 and pix.height > 50:  # Skip tiny images
                    # Convert to numpy array for analysis
                    samples = np.frombuffer(pix.samples, dtype=np.uint8)
                    
                    if pix.n == 1:  # Grayscale
                        samples = samples.reshape((pix.height, pix.width))
                    elif pix.n >= 3:  # RGB/CMYK
                        samples = samples.reshape((pix.height, pix.width, pix.n))
                        # Convert to grayscale for screening analysis
                        if pix.n == 3:  # RGB
                            samples = np.dot(samples[...,:3], [0.299, 0.587, 0.114])
                        else:  # CMYK - use K channel
                            samples = samples[..., -1]
                    
                    # Analyze histogram for screening patterns
                    if samples.ndim == 2:
                        hist, bins = np.histogram(samples, bins=256, range=(0, 255))
                        
                        # Check for bimodal distribution (typical of screened images)
                        # Find peaks in histogram
                        peak_count = 0
                        for i in range(1, len(hist)-1):
                            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > len(samples.flat) * 0.01:
                                peak_count += 1
                        
                        # Check for limited tonal range (screening artifact)
                        non_zero = np.count_nonzero(hist)
                        if non_zero < 64:  # Very limited tonal range
                            issues.append({
                                "type": "screened_image_detected",
                                "detail": f"Image {img_index+1}: Limited tonal range ({non_zero} levels)",
                                "bbox": list(fitz.Rect(page.get_image_bbox(img)))
                            })
                        elif peak_count <= 2 and non_zero < 128:
                            issues.append({
                                "type": "possible_screened_image",
                                "detail": f"Image {img_index+1}: Possible screening ({peak_count} peaks, {non_zero} levels)",
                                "bbox": list(fitz.Rect(page.get_image_bbox(img)))
                            })
                
                img_doc.close()
                
            except Exception as e:
                issues.append({
                    "type": "image_screening_check_error",
                    "detail": f"Image {img_index+1} analysis failed: {str(e)}"
                })
                
    except Exception as e:
        issues.append({"type": "screening_analysis_error", "detail": f"Screening analysis failed: {str(e)}"})
    
    return issues

def compare_text_similarity(text1, text2, threshold=0.8):
    """Compare text similarity for consistency checks"""
    if not text1 or not text2:
        return 0.0
    
    # Normalize text
    text1 = re.sub(r'\s+', ' ', text1.strip().lower())
    text2 = re.sub(r'\s+', ' ', text2.strip().lower())
    
    return SequenceMatcher(None, text1, text2).ratio()

# Thumb tab validation removed - too specialized for general implementation
def check_thumb_tabs_removed():
    """REMOVED: Thumb tab validation was too specialized"""
    return []  # Return empty list for compatibility
    
    try:
        # Convert mm to points (1mm = ~2.83 points)
        min_bleed_pts = min_bleed_mm * 2.83
        min_text_margin_pts = min_text_margin_mm * 2.83
        
        page_rect = page.rect
        cropbox = page.cropbox
        
        # Look for elements extending beyond crop box (potential thumb tabs)
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])
        
        tab_elements = []
        
        # Check text blocks
        for block in blocks:
            if block["type"] == 0:  # text block
                bbox = fitz.Rect(block["bbox"])
                # Check if extends beyond crop box
                if (bbox.x0 < cropbox.x0 or bbox.x1 > cropbox.x1 or 
                    bbox.y0 < cropbox.y0 or bbox.y1 > cropbox.y1):
                    tab_elements.append(("text", bbox, block))
        
        # Check images
        for img in page.get_images():
            bbox = page.get_image_bbox(img)
            bbox_rect = fitz.Rect(bbox)
            if (bbox_rect.x0 < cropbox.x0 or bbox_rect.x1 > cropbox.x1 or 
                bbox_rect.y0 < cropbox.y0 or bbox_rect.y1 > cropbox.y1):
                tab_elements.append(("image", bbox_rect, None))
        
        # Analyze potential thumb tabs
        for elem_type, bbox, data in tab_elements:
            # Check bleed distance
            bleed_left = max(0, cropbox.x0 - bbox.x0)
            bleed_right = max(0, bbox.x1 - cropbox.x1)
            bleed_top = max(0, cropbox.y0 - bbox.y0)
            bleed_bottom = max(0, bbox.y1 - cropbox.y1)
            
            max_bleed = max(bleed_left, bleed_right, bleed_top, bleed_bottom)
            
            if max_bleed > 0:  # Element extends beyond crop
                if max_bleed < min_bleed_pts:
                    issues.append({
                        "type": "thumb_tab_insufficient_bleed",
                        "detail": f"Thumb tab bleed {max_bleed/2.83:.1f}mm < required {min_bleed_mm}mm",
                        "bbox": list(bbox)
                    })
                
                # Check text margin for text elements
                if elem_type == "text" and data:
                    # Check distance from trim edge
                    trim_distance = min_bleed_pts - max_bleed  # Approximate
                    if trim_distance < min_text_margin_pts:
                        issues.append({
                            "type": "thumb_tab_text_margin",
                            "detail": f"Thumb tab text too close to trim edge",
                            "bbox": list(bbox)
                        })
                        
    except Exception as e:
        issues.append({"type": "thumb_tab_check_error", "detail": f"Thumb tab check failed: {str(e)}"})
    
    return issues

def check_page_enhanced(page, margin_pts, dpi_threshold, page_number=None, total_pages=None, enable_advanced=True, doc=None, margin_analysis=None):
    """Enhanced page checking with all validation features"""
    # Run simplified boundary validation instead of complex margin checks
    if margin_analysis:
        margin_issues = simple_boundary_validation(page)
        # Run original checks without margin checking (pass margin_pts=0 to skip)
        other_issues = check_page_original(page, 0, dpi_threshold)
        issues = margin_issues + other_issues
    else:
        # Fallback to original checks if no margin analysis
        issues = check_page_original(page, margin_pts, dpi_threshold)
    
    # Add new advanced checks if enabled
    if enable_advanced:
        # Original advanced checks
        issues.extend(analyze_color_usage(page))
        issues.extend(check_baseline_alignment(page))
        issues.extend(check_column_balance(page))
        issues.extend(detect_image_screening(page))
        # Thumb tab validation removed - too specialized for general implementation
        
        # New advanced checks (items 1-5)
        # 1. Hyphenation at page breaks
        next_page = None
        if doc and page_number and page_number < len(doc):
            next_page = doc[page_number]  # page_number is 1-based, doc is 0-based
        issues.extend(check_hyphenation_at_breaks(page, next_page))
        
        # 3. Enhanced line spacing consistency
        issues.extend(check_line_spacing_consistency(page))
        
        # 4. Advanced image quality assessment
        issues.extend(assess_image_quality_advanced(page))
        
        # 5. Header consistency detection
        issues.extend(check_header_consistency(page, page_number, total_pages))
        
        # 6. Text justification validation
        issues.extend(validate_text_justification(page))
    
    return issues

# Rename original function for backwards compatibility  
def check_page_original(page, margin_pts, dpi_threshold):
    """Original page checking function - moved from check_page"""
    issues = []
    pr = page.rect  # page rectangle (CropBox)
    W, H = pr.width, pr.height

    # Page boxes sanity (Crop should be within Media if available)
    try:
        media = page.mediabox  # not always available
        crop  = page.cropbox
        if media and crop:
            if not media.contains(crop):
                issues.append({"type":"page_box_error", "detail":"CropBox not within MediaBox"})
    except Exception:
        pass

    # Get text spans first to calculate adaptive margins (same as green box logic)
    raw = page.get_text("dict")
    blocks = raw.get("blocks", [])
    page_height = pr.height
    header_zone = page_height * 0.1
    footer_zone = page_height * 0.9
    
    body_text_spans = []
    for b in blocks:
        if b["type"] == 0:  # text block
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    bbox = fitz.Rect(span["bbox"]) if span.get("bbox") else None
                    text = span.get("text", "")
                    if bbox and bbox.width > 2 and bbox.height > 2 and text.strip():
                        span_center_y = (bbox.y0 + bbox.y1) / 2
                        if header_zone < span_center_y < footer_zone:
                            body_text_spans.append(bbox)
    
    # Calculate adaptive margins using EXACT same logic as green boxes
    # Import detect_column_boundaries function (already defined above)
    column_boundaries = detect_column_boundaries(page)
    
    if body_text_spans:
        body_x_coords = [r.x0 for r in body_text_spans] + [r.x1 for r in body_text_spans]
        body_x_coords.sort()
        
        text_left = min(body_x_coords)
        text_right = max(body_x_coords)
        
        if column_boundaries:
            # Multi-column: use exact green box logic
            actual_divider = column_boundaries[0]
            adaptive_left_margin = max(36.0, text_left - 5)  # Small padding (same as green box)
            adaptive_right_margin = min(pr.width - 36.0, text_right + 5)  # Small padding
        else:
            # Single column: use exact green box logic  
            adaptive_left_margin = max(36.0, text_left - 8)  # 8pt padding (same as green box)
            adaptive_right_margin = min(pr.width - 36.0, text_right + 8)  # 8pt padding
    else:
        # Fallback to standard margins
        adaptive_left_margin = margin_pts
        adaptive_right_margin = pr.width - margin_pts
    
    # Create margin zones using adaptive margins (aligned with green box positioning)
    left_m   = fitz.Rect(pr.x0, pr.y0, adaptive_left_margin, pr.y1)
    right_m  = fitz.Rect(adaptive_right_margin, pr.y0, pr.x1, pr.y1)
    top_m    = fitz.Rect(pr.x0, pr.y0, pr.x1, pr.y0 + margin_pts)
    bottom_m = fitz.Rect(pr.x0, pr.y1 - margin_pts, pr.x1, pr.y1)

    # Gather text blocks / spans / glyphs and images (reusing parsed blocks)
    blocks = raw.get("blocks", [])

    text_rects = []     # span-level rects
    glyph_rects = []    # per-glyph rects (optional / heavier)
    image_rects = []    # image rects
    image_ppi   = []    # (rect, ppi_x, ppi_y)

    for b in blocks:
        if b["type"] == 0:
            # text block: lines -> spans -> bbox, chars
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    bbox = fitz.Rect(span["bbox"])
                    text_rects.append((bbox, span.get("text","")))
                    # glyphs may not always be present; enable if you want stricter overlap
                    for g in span.get("chars", []):
                        gb = fitz.Rect(g["bbox"])
                        glyph_rects.append((gb, g.get("c","")))
        elif b["type"] == 1:
            # image block
            ibox = fitz.Rect(b["bbox"])
            image_rects.append(ibox)
            # Estimate placed PPI from transform matrix if available (not always)
            # NOTE: For embedded image native resolution, you'd need to resolve the image XObject.
            # PyMuPDF 1.23+ has page.get_images() then doc.extract_image(xref) for width/height.
            # Here we compute placed PPI as a proxy.
            # Fallback to None if we can't estimate.
            ppi_x, ppi_y = None, None
            if "transform" in b:
                try:
                    m = fitz.Matrix(b["transform"])
                    ppi_x, ppi_y = px_per_inch_from_matrix(m)
                except Exception:
                    pass
            image_ppi.append((ibox, ppi_x, ppi_y))

    # 1) Simplified boundary validation (replaces margin intrusion detection)
    boundary_violations = simple_boundary_validation(page)
    issues.extend(boundary_violations)

    # 1b) Margin intrusions (images)
    for r in image_rects:
        if r.intersects(left_m) or r.intersects(right_m) or r.intersects(top_m) or r.intersects(bottom_m):
            issues.append({"type":"margin_intrusion_image", "bbox":list(r)})

    # 2) Text overlap detection - identify problematic lines only
    # Detect column boundaries for better overlap analysis
    column_boundaries = detect_column_boundaries(page)
    
    # Find lines that extend beyond their proper boundaries
    problematic_lines = set()  # Track which lines are the culprits
    
    # O(n^2) over spans per page; typically fine. For big pages, you can grid-index.
    for i in range(len(text_rects)):
        ri, ti = text_rects[i]
        for j in range(i+1, len(text_rects)):
            rj, tj = text_rects[j]
            # Check if these spans overlap
            if text_spans_actually_overlap(ri, ti, rj, tj, column_boundaries):
                # Determine which line is the problematic one (extends too far)
                culprit_idx = identify_problematic_line(ri, ti, rj, tj, column_boundaries)
                if culprit_idx is not None:
                    # Mark the problematic line
                    actual_idx = i if culprit_idx == 0 else j
                    problematic_lines.add(actual_idx)
    
    # Report only lines that actually cross the column boundary
    main_boundary = column_boundaries[0] if column_boundaries else None
    if main_boundary:
        for idx in problematic_lines:
            ri, ti = text_rects[idx]
            # Double-check that this line actually crosses the boundary
            if ri.x0 < main_boundary < ri.x1:
                issues.append({
                    "type": "text_crosses_column_boundary", 
                    "detail": f"Text crosses column separator at {main_boundary:.1f}pt: '{ti[:50]}'",
                    "bbox": list(ri)
                })
    else:
        # Fallback if no boundary detected
        for idx in problematic_lines:
            ri, ti = text_rects[idx]
            issues.append({
                "type": "text_crosses_column_boundary", 
                "detail": f"Text crosses column separator: '{ti[:50]}'",
                "bbox": list(ri)
            })

    # 3) Text over image overlap
    for rtxt, txt in text_rects:
        for rimg in image_rects:
            if rects_intersect(rtxt, rimg, iou_thresh=0.02):
                issues.append({"type":"overlap_text_image", "detail":txt[:60], "bboxes":[list(rtxt), list(rimg)]})

    # 4) Low-DPI images (placed ppi proxy)
    for rimg, ppx, ppy in image_ppi:
        if ppx and ppy:
            if ppx < dpi_threshold or ppy < dpi_threshold:
                issues.append({"type":"low_dpi_image", "detail":f"{int(ppx)}x{int(ppy)} ppi", "bbox":list(rimg)})
        else:
            # Could not estimate; optional warning
            issues.append({"type":"dpi_unknown_image", "bbox":list(rimg)})

    # 5) Font embedding presence (resource-level heuristic)
    # PyMuPDF: page.get_fonts() -> [(xref, name, is_embedded, ...)]
    # NOTE: Not all PDFs make this trivial; this is a useful but not perfect proxy.
    try:
        fonts = page.get_fonts(full=True)
        for f in fonts:
            # tuple format can vary by version; guard indexes
            is_embedded = False
            if len(f) >= 3:
                is_embedded = bool(f[2])
            if not is_embedded:
                issues.append({"type":"font_not_embedded", "detail":str(f)})
    except Exception:
        issues.append({"type":"font_check_error", "detail":"Failed to read font info"})

    return issues

# Update check_page to use enhanced version
def check_page(page, margin_pts, dpi_threshold, page_number=None, total_pages=None, enable_advanced=True, doc=None, margin_analysis=None):
    """Main page checking function with enhanced validation"""
    return check_page_enhanced(page, margin_pts, dpi_threshold, page_number, total_pages, enable_advanced, doc, margin_analysis)

def annotate_pdf(input_path, output_path, per_page_issues, document_issues=None, report_summary=None):
    """Enhanced PDF annotation with comprehensive issue highlighting and summary pages"""
    doc = fitz.open(input_path)
    
    # Enhanced color mapping for all validation types
    color_by_type = {
        # Original checks - Red family
        "margin_intrusion_text": (1,0,0),
        "margin_intrusion_image": (1,0,0),
        "insufficient_margins": (0.8,0,0),
        "overlap_text_text": (0,0,1),
        "text_crosses_column_boundary": (1,0,0.5),  # Red-purple for column boundary violations
        "text_extends_to_gutter": (0.9,0,0.4),  # Dark pink for gutter extension
        "overlap_text_image": (0.5,0,0.5),
        "low_dpi_image": (1,0.5,0),
        "dpi_unknown_image": (0.5,0.5,0.5),
        "font_not_embedded": (0,0,0),
        "page_box_error": (0,0,0),
        
        # Document structure - Purple family
        "pdfx_compliance": (0.8,0,0.8),
        "page_sequence_error": (0.5,0.5,1),
        "page_sequence_warning": (0.7,0.7,1),
        "inconsistent_page_size": (0.8,0.2,0.8),
        "inconsistent_crop_size": (0.8,0.4,0.8),
        "page_size_check_error": (0.5,0.5,0.5),
        
        # Color and print - Orange family
        "color_mode_rgb": (1,0.5,0),
        "color_mode_compliance": (1,0.7,0),
        "screened_image_detected": (1,0,0.5),
        "possible_screened_image": (1,0.5,0.5),
        "color_analysis_error": (0.5,0.5,0.5),
        "screening_analysis_error": (0.5,0.5,0.5),
        "image_screening_check_error": (0.5,0.5,0.5),
        
        # Layout and typography - Cyan/Green family
        "baseline_grid_violation": (0,0.8,0.8),
        "column_imbalance": (0.8,0.8,0),
        "line_spacing_inconsistency": (0.2,0.8,0.8),
        "line_spacing_too_tight": (1,0.8,0),
        "line_spacing_too_loose": (1,0.6,0),
        "hyphen_at_page_break": (0,0.8,0),
        "baseline_check_error": (0.5,0.5,0.5),
        "column_balance_error": (0.5,0.5,0.5),
        "line_spacing_check_error": (0.5,0.5,0.5),
        "hyphen_check_error": (0.5,0.5,0.5),
        
        # Headers and content - Blue family
        "header_too_long": (0.6,0.2,0.8),
        "minimal_header_content": (0.4,0.4,0.8),
        "missing_header": (0.8,0.4,0.4),
        "header_check_error": (0.5,0.5,0.5),
        "high_text_similarity": (0.4,0.6,1),
        "text_comparison_error": (0.5,0.5,0.5),
        
        # Image quality - Red/Yellow family
        "blurry_image": (1,0.2,0.2),
        "low_contrast_image": (0.8,0.6,0),
        "clipped_highlights": (0.9,0.9,0.2),
        "clipped_shadows": (0.2,0.2,0.9),
        "image_quality_check_error": (0.5,0.5,0.5),
        "image_quality_analysis_error": (0.5,0.5,0.5),
        
        # Print production - (removed thumb tab validation)
    }
    
    # Annotate page-level issues
    for pno, issues in per_page_issues.items():
        page = doc[pno]
        
        # Track annotations on page to avoid overlap
        page_annotations = []
        
        for iss in issues:
            col = color_by_type.get(iss["type"], (0.2,0.2,0.2))  # Default dark gray
            
            # Handle issues with specific bounding boxes
            if "bbox" in iss and iss["bbox"]:
                try:
                    r = fitz.Rect(iss["bbox"])
                    if not r.is_empty:
                        # Draw visual indicators
                        page.draw_rect(r, color=col, width=2)
                        page.draw_circle(r.tl, radius=4, color=col, width=2)
                        page.draw_circle(r.br, radius=4, color=col, width=2)
                        
                        # Add annotation
                        label = f"{iss['type']}"
                        if "detail" in iss and iss["detail"]:
                            label += f": {iss['detail'][:50]}{'...' if len(iss.get('detail', '')) > 50 else ''}"
                        
                        # Find non-overlapping position for annotation
                        annot_point = find_annotation_position(r.tl, page_annotations, page.rect)
                        page.add_text_annot(annot_point, label)
                        page_annotations.append(annot_point)
                except Exception as e:
                    print(f"Warning: Could not draw bbox for {iss['type']}: {e}")
            
            # Handle issues with multiple bounding boxes
            elif "bboxes" in iss and iss["bboxes"]:
                try:
                    for i, bb in enumerate(iss["bboxes"]):
                        r = fitz.Rect(bb)
                        if not r.is_empty:
                            page.draw_rect(r, color=col, width=2)
                            if i == 0:  # Only annotate first bbox to avoid clutter
                                label = f"{iss['type']}"
                                if "detail" in iss and iss["detail"]:
                                    label += f": {iss['detail'][:50]}{'...' if len(iss.get('detail', '')) > 50 else ''}"
                                
                                annot_point = find_annotation_position(r.tl, page_annotations, page.rect)
                                page.add_text_annot(annot_point, label)
                                page_annotations.append(annot_point)
                except Exception as e:
                    print(f"Warning: Could not draw bboxes for {iss['type']}: {e}")
            
            # Handle issues without specific location (page-level)
            else:
                # Add floating annotation for issues without specific location
                label = f"PAGE ISSUE - {iss['type']}"
                if "detail" in iss and iss["detail"]:
                    label += f": {iss['detail'][:80]}{'...' if len(iss.get('detail', '')) > 80 else ''}"
                
                # Place at top-right corner of page, stacking vertically
                page_width = page.rect.width
                annot_x = page_width - 100
                annot_y = 20 + len(page_annotations) * 15
                annot_point = fitz.Point(annot_x, annot_y)
                
                # Draw a small colored rectangle as indicator
                indicator_rect = fitz.Rect(annot_x - 10, annot_y - 5, annot_x - 5, annot_y + 5)
                page.draw_rect(indicator_rect, color=col, fill=col)
                
                page.add_text_annot(annot_point, label)
                page_annotations.append(annot_point)
    
    # Draw green column outline boxes on each page
    for pno in range(len(doc)):
        page = doc[pno]
        page_width = page.rect.width
        page_height = page.rect.height
        
        # Get all text spans, excluding headers/footers
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])
        
        # Define header/footer zones (top 10% and bottom 10% of page)
        header_zone = page_height * 0.1
        footer_zone = page_height * 0.9
        
        body_text_spans = []
        for b in blocks:
            if b["type"] == 0:  # text block
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        bbox = fitz.Rect(span["bbox"])
                        if bbox.width > 2 and bbox.height > 2:  # Skip tiny spans
                            # Skip headers and footers
                            span_center_y = (bbox.y0 + bbox.y1) / 2
                            if header_zone < span_center_y < footer_zone:
                                body_text_spans.append(bbox)
        
        if not body_text_spans:
            continue  # Skip pages with no body text
        
        # Standard margins
        left_margin = 36.0  # 0.5 inch
        right_margin = page_width - 36.0
        
        # For facing pages, use consistent column divider but adaptive margins
        # Detect the logical column divider position (should be consistent)
        column_boundaries = detect_column_boundaries(page)
        
        # Determine if this is a facing page layout and adapt accordingly
        page_number = pno + 1  # 1-based page numbering
        is_right_facing = (page_number % 2 == 1)  # Odd pages are right-facing
        
        # Analyze actual text positioning to determine gutter-aware margins
        if body_text_spans:
            # Find the leftmost and rightmost text positions (excluding outliers)
            body_x_coords = [r.x0 for r in body_text_spans] + [r.x1 for r in body_text_spans]
            body_x_coords.sort()
            
            # Use 5th and 95th percentile to exclude outliers
            left_text_edge = body_x_coords[max(0, int(len(body_x_coords) * 0.05))]
            right_text_edge = body_x_coords[min(len(body_x_coords)-1, int(len(body_x_coords) * 0.95))]
            
            # Adaptive margins based on actual text distribution for this page
            adaptive_left_margin = max(left_margin, left_text_edge - 5)  # At least 5pt padding
            adaptive_right_margin = min(right_margin, right_text_edge + 5)  # At least 5pt padding
        else:
            adaptive_left_margin = left_margin
            adaptive_right_margin = right_margin
        
        if not column_boundaries:
            # Single column - create a well-sized box with reasonable margins
            body_y_coords = [r.y0 for r in body_text_spans] + [r.y1 for r in body_text_spans]
            body_x_coords = [r.x0 for r in body_text_spans] + [r.x1 for r in body_text_spans]
            
            if body_y_coords and body_x_coords:
                # Use actual text bounds but add generous margins for single column
                text_left = min(body_x_coords)
                text_right = max(body_x_coords)
                text_top = min(body_y_coords)
                text_bottom = max(body_y_coords)
                
                # Add margins to ensure text doesn't escape (slightly more generous)
                margin_left = max(left_margin, text_left - 8)  # 8pt padding
                margin_right = min(right_margin, text_right + 8)  # 8pt padding
                
                column_box = fitz.Rect(
                    margin_left,
                    text_top - 10,
                    margin_right, 
                    text_bottom + 10
                )
                page.draw_rect(column_box, color=(0, 0.7, 0), width=2)
        else:
            # Multi-column with balanced widths - gutter may shift but columns are equal width
            actual_divider_position = column_boundaries[0]
            
            # Get actual text bounds to determine proper margins
            body_x_coords = [r.x0 for r in body_text_spans] + [r.x1 for r in body_text_spans]
            if body_x_coords:
                text_left = min(body_x_coords)
                text_right = max(body_x_coords)
                
                # Calculate balanced column widths
                total_text_width = text_right - text_left
                available_width = text_right - text_left
                
                # Each column gets equal width
                column_width = available_width / 2
                
                # Position columns symmetrically around the divider but with equal widths
                left_column_left_edge = max(left_margin, text_left - 5)  # Small padding
                left_column_right_edge = actual_divider_position
                right_column_left_edge = actual_divider_position  
                right_column_right_edge = min(right_margin, text_right + 5)  # Small padding
                
                # Ensure columns are balanced by adjusting if necessary
                left_width = left_column_right_edge - left_column_left_edge
                right_width = right_column_right_edge - right_column_left_edge
                
                # If widths are significantly different, balance them
                if abs(left_width - right_width) > 20:  # More than 20pt difference
                    target_width = (left_width + right_width) / 2
                    
                    # Adjust column edges to create equal widths
                    gap_center = actual_divider_position
                    left_column_left_edge = gap_center - target_width
                    left_column_right_edge = gap_center
                    right_column_left_edge = gap_center
                    right_column_right_edge = gap_center + target_width
                    
                    # Ensure we don't go outside reasonable page bounds
                    left_column_left_edge = max(left_margin, left_column_left_edge)
                    right_column_right_edge = min(right_margin, right_column_right_edge)
            else:
                # Fallback to adaptive margins
                left_column_left_edge = adaptive_left_margin
                left_column_right_edge = actual_divider_position
                right_column_left_edge = actual_divider_position
                right_column_right_edge = adaptive_right_margin
            
            # Get Y bounds from body text
            all_body_y = [r.y0 for r in body_text_spans] + [r.y1 for r in body_text_spans]
            
            if all_body_y:
                text_top = min(all_body_y) - 10
                text_bottom = max(all_body_y) + 10
                
                # Left column box - adaptive to this page
                if left_column_left_edge < left_column_right_edge:
                    left_column_box = fitz.Rect(
                        left_column_left_edge,
                        text_top,
                        left_column_right_edge,
                        text_bottom
                    )
                    page.draw_rect(left_column_box, color=(0, 0.7, 0), width=2)
                
                # Right column box - adaptive to this page
                if right_column_left_edge < right_column_right_edge:
                    right_column_box = fitz.Rect(
                        right_column_left_edge,
                        text_top,
                        right_column_right_edge,
                        text_bottom
                    )
                    page.draw_rect(right_column_box, color=(0, 0.7, 0), width=2)
    
    # Add summary pages for document-level issues and overall report
    if document_issues or report_summary:
        add_summary_pages(doc, document_issues or [], report_summary or {}, color_by_type)
    
    doc.save(output_path)
    doc.close()

def find_annotation_position(preferred_point, existing_annotations, page_rect, min_distance=20):
    """Find a non-overlapping position for annotation"""
    candidate = preferred_point
    
    # Check if position conflicts with existing annotations
    max_attempts = 10
    attempt = 0
    
    while attempt < max_attempts:
        conflict = False
        for existing in existing_annotations:
            if abs(candidate.x - existing.x) < min_distance and abs(candidate.y - existing.y) < min_distance:
                conflict = True
                break
        
        if not conflict:
            return candidate
        
        # Try moving the candidate position
        if attempt % 2 == 0:
            candidate = fitz.Point(candidate.x + min_distance, candidate.y)
        else:
            candidate = fitz.Point(preferred_point.x, candidate.y + min_distance)
        
        # Keep within page bounds
        if candidate.x > page_rect.width - 50:
            candidate = fitz.Point(50, candidate.y + min_distance)
        if candidate.y > page_rect.height - 50:
            candidate = fitz.Point(candidate.x, 50)
        
        attempt += 1
    
    return candidate

def add_summary_pages(doc, document_issues, report_summary, color_by_type):
    """Add summary pages with document-level issues and overall statistics"""
    
    # Page 1: Document-Level Issues
    if document_issues:
        summary_page1 = doc.new_page(width=595, height=842)  # A4 size
        
        # Title
        title_rect = fitz.Rect(50, 50, 545, 100)
        summary_page1.insert_textbox(title_rect, "PDF VALIDATION REPORT\nDocument-Level Issues", 
                                   fontsize=18, align=1)
        
        y_pos = 120
        line_height = 20
        
        # Group issues by type
        issue_groups = {}
        for issue in document_issues:
            issue_type = issue["type"]
            if issue_type not in issue_groups:
                issue_groups[issue_type] = []
            issue_groups[issue_type].append(issue)
        
        # Display grouped issues
        for issue_type, issues_list in issue_groups.items():
            if y_pos > 750:  # Start new page if needed
                summary_page1 = doc.new_page(width=595, height=842)
                y_pos = 50
            
            # Issue type header with color indicator
            color = color_by_type.get(issue_type, (0.2, 0.2, 0.2))
            
            # Color indicator rectangle
            color_rect = fitz.Rect(50, y_pos, 70, y_pos + 15)
            summary_page1.draw_rect(color_rect, color=color, fill=color)
            
            # Issue type title
            type_rect = fitz.Rect(80, y_pos, 545, y_pos + 15)
            summary_page1.insert_textbox(type_rect, f"{issue_type.replace('_', ' ').title()} ({len(issues_list)} issues)", 
                                       fontsize=12)
            y_pos += 25
            
            # List individual issues (limit to first 5 per type)
            for i, issue in enumerate(issues_list[:5]):
                if y_pos > 750:
                    summary_page1 = doc.new_page(width=595, height=842)
                    y_pos = 50
                
                detail_text = issue.get("detail", "No details available")[:100]
                if len(issue.get("detail", "")) > 100:
                    detail_text += "..."
                
                detail_rect = fitz.Rect(80, y_pos, 545, y_pos + line_height)
                summary_page1.insert_textbox(detail_rect, f"• {detail_text}", 
                                           fontsize=10)
                y_pos += line_height
            
            if len(issues_list) > 5:
                more_rect = fitz.Rect(80, y_pos, 545, y_pos + line_height)
                summary_page1.insert_textbox(more_rect, f"... and {len(issues_list) - 5} more", 
                                           fontsize=10, color=(0.5, 0.5, 0.5))
                y_pos += line_height
            
            y_pos += 10  # Extra spacing between issue types
    
    # Page 2: Summary Statistics
    if report_summary:
        summary_page2 = doc.new_page(width=595, height=842)
        
        # Title
        title_rect = fitz.Rect(50, 50, 545, 100)
        summary_page2.insert_textbox(title_rect, "PDF VALIDATION REPORT\nSummary Statistics", 
                                   fontsize=18, align=1)
        
        y_pos = 120
        
        # Overall statistics
        stats_text = f"""OVERALL RESULTS:
Total Issues Found: {report_summary.get('total_issues', 0)}
Document Issues: {report_summary.get('document_issues', 0)}
Pages with Issues: {report_summary.get('pages_with_issues', 0)}
Validation Status: {'✓ PASSED' if report_summary.get('ok', False) else '✗ ISSUES FOUND'}
"""
        
        stats_rect = fitz.Rect(50, y_pos, 545, y_pos + 100)
        summary_page2.insert_textbox(stats_rect, stats_text, fontsize=12)
        y_pos += 120
        
        # Issue breakdown
        if "issue_breakdown" in report_summary and report_summary["issue_breakdown"]:
            breakdown_title_rect = fitz.Rect(50, y_pos, 545, y_pos + 20)
            summary_page2.insert_textbox(breakdown_title_rect, "ISSUE BREAKDOWN BY TYPE:", 
                                       fontsize=14)
            y_pos += 30
            
            # Sort issues by frequency
            sorted_issues = sorted(report_summary["issue_breakdown"].items(), 
                                 key=lambda x: x[1], reverse=True)
            
            for issue_type, count in sorted_issues[:15]:  # Show top 15
                if y_pos > 750:
                    summary_page2 = doc.new_page(width=595, height=842)
                    y_pos = 50
                
                # Color indicator
                color = color_by_type.get(issue_type, (0.2, 0.2, 0.2))
                color_rect = fitz.Rect(50, y_pos, 70, y_pos + 15)
                summary_page2.draw_rect(color_rect, color=color, fill=color)
                
                # Issue count and type
                issue_rect = fitz.Rect(80, y_pos, 545, y_pos + 15)
                display_name = issue_type.replace('_', ' ').title()
                summary_page2.insert_textbox(issue_rect, f"{display_name}: {count}", 
                                           fontsize=11)
                y_pos += 20
            
            if len(sorted_issues) > 15:
                more_rect = fitz.Rect(80, y_pos, 545, y_pos + 20)
                summary_page2.insert_textbox(more_rect, f"... and {len(sorted_issues) - 15} more issue types", 
                                           fontsize=10, color=(0.5, 0.5, 0.5))

def main():
    ap = argparse.ArgumentParser(description="Enhanced PDF Validator - Comprehensive print production validation")
    ap.add_argument("pdf", help="Input PDF file or directory containing PDFs")
    ap.add_argument("--margin-pts", type=float, default=36.0, help="Margin threshold in points (72pt = 1 inch)")
    ap.add_argument("--dpi-threshold", type=int, default=300, help="Min acceptable placed PPI for images")
    ap.add_argument("--annotate", help="Write annotated PDF here (ignored in batch mode)")
    ap.add_argument("--json", help="Write JSON report here (ignored in batch mode)")
    ap.add_argument("--basic-only", action="store_true", help="Run only basic checks (disable advanced validation)")
    ap.add_argument("--check-text-similarity", action="store_true", help="Enable text consistency checks between pages")
    ap.add_argument("--batch", action="store_true", help="Batch mode: process all PDFs in specified directory")
    ap.add_argument("--output-dir", default="output", help="Output directory for batch mode (default: output)")
    ap.add_argument("--rating-summary", action="store_true", help="Generate rating summary ranking PDFs from worst to best (batch mode only)")
    args = ap.parse_args()

    # Check if batch mode or single file mode
    if args.batch or (os.path.isdir(args.pdf)):
        return batch_process_pdfs(args)
    else:
        return process_single_pdf(args)

def batch_process_pdfs(args):
    """Process all PDFs in a directory with organized output structure"""
    
    input_dir = args.pdf if os.path.isdir(args.pdf) else "."
    output_base = args.output_dir
    
    # Find all PDF files
    pdf_pattern = os.path.join(input_dir, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return 1
    
    # Create organized output structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_base, f"batch_{timestamp}")
    reports_dir = os.path.join(batch_dir, "reports")
    annotations_dir = os.path.join(batch_dir, "annotations")
    summary_dir = os.path.join(batch_dir, "summary")
    
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True) 
    os.makedirs(summary_dir, exist_ok=True)
    
    print(f"\n=== BATCH PDF VALIDATION ===")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {batch_dir}")
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Mode: {'Basic' if args.basic_only else 'Advanced'} validation")
    print("\nProcessing files...")
    
    batch_summary = {
        "timestamp": timestamp,
        "input_directory": input_dir,
        "output_directory": batch_dir,
        "total_files": len(pdf_files),
        "processed_files": 0,
        "failed_files": 0,
        "total_issues": 0,
        "files": []
    }
    
    for i, pdf_path in enumerate(pdf_files, 1):
        filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(filename)[0]
        
        print(f"\n[{i}/{len(pdf_files)}] Processing: {filename}")
        
        try:
            # Process single PDF
            single_args = type('Args', (), {
                'pdf': pdf_path,
                'margin_pts': args.margin_pts,
                'dpi_threshold': args.dpi_threshold,
                'basic_only': args.basic_only,
                'check_text_similarity': args.check_text_similarity,
                'json': os.path.join(reports_dir, f"{base_name}_report.json"),
                'annotate': os.path.join(annotations_dir, f"{base_name}_annotated.pdf")
            })()
            
            result = process_single_pdf(single_args, verbose=False)
            
            # Read the generated report for summary
            if os.path.exists(single_args.json):
                with open(single_args.json, 'r') as f:
                    report_data = json.load(f)
                    
                file_summary = {
                    "filename": filename,
                    "status": "success",
                    "total_issues": report_data.get("summary", {}).get("total_issues", 0),
                    "pages": report_data.get("pages", 0),
                    "pages_with_issues": report_data.get("summary", {}).get("pages_with_issues", 0),
                    "ok": report_data.get("summary", {}).get("ok", False)
                }
                
                batch_summary["total_issues"] += file_summary["total_issues"]
                batch_summary["processed_files"] += 1
                
                # Print brief results
                status_icon = "\u2713" if file_summary["ok"] else "\u26a0"
                print(f"  {status_icon} {file_summary['total_issues']} issues found ({file_summary['pages']} pages)")
                
            else:
                file_summary = {
                    "filename": filename,
                    "status": "error",
                    "error": "Report file not generated"
                }
                batch_summary["failed_files"] += 1
                print(f"  \u2717 Failed to generate report")
                
            batch_summary["files"].append(file_summary)
            
        except Exception as e:
            print(f"  \u2717 Error: {str(e)}")
            batch_summary["failed_files"] += 1
            batch_summary["files"].append({
                "filename": filename,
                "status": "error",
                "error": str(e)
            })
    
    # Generate batch summary report
    summary_file = os.path.join(summary_dir, "batch_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    
    # Generate batch summary text report  
    summary_text_file = os.path.join(summary_dir, "batch_summary.txt")
    generate_batch_summary_text(batch_summary, summary_text_file)
    
    # Generate rating summary if requested
    if args.rating_summary:
        rating_file = os.path.join(summary_dir, "rating_summary.txt")
        generate_rating_summary(batch_summary, rating_file)
        print(f"  Rating Summary: {summary_dir}/rating_summary.txt")
    
    # Final summary
    print(f"\n=== BATCH PROCESSING COMPLETE ===")
    print(f"Processed: {batch_summary['processed_files']}/{batch_summary['total_files']} files")
    print(f"Failed: {batch_summary['failed_files']} files")
    print(f"Total issues found: {batch_summary['total_issues']}")
    print(f"\nResults saved to: {batch_dir}")
    print(f"  Reports: {reports_dir}")
    print(f"  Annotations: {annotations_dir}") 
    print(f"  Summary: {summary_dir}")
    
    return 0 if batch_summary["failed_files"] == 0 else 1

def generate_batch_summary_text(batch_summary, output_file):
    """Generate human-readable batch summary report"""
    with open(output_file, 'w') as f:
        f.write("PDF VALIDATION BATCH REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Timestamp: {batch_summary['timestamp']}\n")
        f.write(f"Input Directory: {batch_summary['input_directory']}\n")
        f.write(f"Output Directory: {batch_summary['output_directory']}\n\n")
        
        f.write("SUMMARY:\n")
        f.write(f"  Total Files: {batch_summary['total_files']}\n")
        f.write(f"  Processed Successfully: {batch_summary['processed_files']}\n")
        f.write(f"  Failed: {batch_summary['failed_files']}\n")
        f.write(f"  Total Issues Found: {batch_summary['total_issues']}\n\n")
        
        # File-by-file results
        f.write("INDIVIDUAL RESULTS:\n")
        f.write("-" * 30 + "\n")
        
        for file_info in batch_summary['files']:
            filename = file_info['filename']
            status = file_info['status']
            
            if status == 'success':
                issues = file_info['total_issues']
                pages = file_info['pages']
                status_text = "PASSED" if file_info['ok'] else f"{issues} ISSUES"
                f.write(f"{filename:<40} [{status_text}] ({pages} pages)\n")
            else:
                error = file_info.get('error', 'Unknown error')
                f.write(f"{filename:<40} [ERROR] {error}\n")

def generate_rating_summary(batch_summary, output_file):
    """Generate rating summary ranking PDFs from worst to best"""
    with open(output_file, 'w') as f:
        f.write("PDF VALIDATION RATING SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write("PDFs ranked from WORST to BEST quality\n")
        f.write("(Based on total issues found per page)\n\n")
        
        # Filter successful files and calculate ratings
        successful_files = [f for f in batch_summary['files'] if f['status'] == 'success']
        
        if not successful_files:
            f.write("No successfully processed files to rate.\n")
            return
        
        # Calculate issue density (issues per page) for ranking
        rated_files = []
        for file_info in successful_files:
            pages = file_info.get('pages', 1)
            issues = file_info.get('total_issues', 0)
            issue_density = issues / pages if pages > 0 else issues
            
            # Determine quality grade
            if issues == 0:
                grade = "EXCELLENT"
                grade_symbol = "★★★★★"
            elif issue_density < 1:
                grade = "VERY GOOD"
                grade_symbol = "★★★★☆"
            elif issue_density < 3:
                grade = "GOOD"
                grade_symbol = "★★★☆☆"
            elif issue_density < 6:
                grade = "FAIR" 
                grade_symbol = "★★☆☆☆"
            elif issue_density < 10:
                grade = "POOR"
                grade_symbol = "★☆☆☆☆"
            else:
                grade = "CRITICAL"
                grade_symbol = "☆☆☆☆☆"
            
            rated_files.append({
                'filename': file_info['filename'],
                'total_issues': issues,
                'pages': pages,
                'issue_density': issue_density,
                'grade': grade,
                'grade_symbol': grade_symbol,
                'ok': file_info.get('ok', False)
            })
        
        # Sort by issue density (worst first)
        rated_files.sort(key=lambda x: x['issue_density'], reverse=True)
        
        # Generate ranking
        f.write(f"Total Files Rated: {len(rated_files)}\n")
        f.write(f"Rating Scale: Issues per page (0 = Perfect, >10 = Critical)\n\n")
        
        f.write("RANKING (Worst to Best):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Rank':<4} {'File':<35} {'Grade':<12} {'Issues':<8} {'Pages':<6} {'Density':<8}\n")
        f.write("-" * 80 + "\n")
        
        for rank, file_info in enumerate(rated_files, 1):
            filename = file_info['filename']
            if len(filename) > 32:
                filename = filename[:29] + "..."
                
            f.write(f"{rank:<4} {filename:<35} {file_info['grade']:<12} "
                   f"{file_info['total_issues']:<8} {file_info['pages']:<6} "
                   f"{file_info['issue_density']:<8.2f}\n")
        
        f.write("-" * 80 + "\n\n")
        
        # Summary statistics
        f.write("QUALITY DISTRIBUTION:\n")
        f.write("-" * 30 + "\n")
        
        grade_counts = {}
        for file_info in rated_files:
            grade = file_info['grade']
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        
        grade_order = ["EXCELLENT", "VERY GOOD", "GOOD", "FAIR", "POOR", "CRITICAL"]
        for grade in grade_order:
            count = grade_counts.get(grade, 0)
            if count > 0:
                percentage = (count / len(rated_files)) * 100
                f.write(f"{grade:<12}: {count:>3} files ({percentage:>5.1f}%)\n")
        
        f.write("\n")
        
        # Best and worst files
        if len(rated_files) > 0:
            best_file = rated_files[-1]  # Last in sorted list (lowest density)
            worst_file = rated_files[0]   # First in sorted list (highest density)
            
            f.write("HIGHLIGHTS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Best Quality:  {best_file['filename']}\n")
            f.write(f"               {best_file['grade']} - {best_file['total_issues']} issues "
                   f"in {best_file['pages']} pages ({best_file['issue_density']:.2f} per page)\n\n")
            f.write(f"Worst Quality: {worst_file['filename']}\n")
            f.write(f"               {worst_file['grade']} - {worst_file['total_issues']} issues "
                   f"in {worst_file['pages']} pages ({worst_file['issue_density']:.2f} per page)\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            
            critical_files = [f for f in rated_files if f['grade'] == 'CRITICAL']
            poor_files = [f for f in rated_files if f['grade'] == 'POOR']
            
            if critical_files:
                f.write(f"• {len(critical_files)} file(s) need immediate attention (CRITICAL quality)\n")
            if poor_files:
                f.write(f"• {len(poor_files)} file(s) have significant quality issues (POOR quality)\n")
            
            good_files = [f for f in rated_files if f['grade'] in ['EXCELLENT', 'VERY GOOD', 'GOOD']]
            if good_files:
                f.write(f"• {len(good_files)} file(s) meet acceptable quality standards\n")
            
            if not critical_files and not poor_files:
                f.write("• All files meet acceptable quality standards\n")

def process_single_pdf(args, verbose=True):
    """Process a single PDF file"""
    doc = fitz.open(args.pdf)
    enable_advanced = not args.basic_only
    
    # Create document layout analyzer for enhanced column detection
    document_layout = DocumentLayout(doc)
    
    # Analyze margin structure for gutter-aware validation
    margin_analysis = analyze_gutter_margins(doc, args.margin_pts)
    
    report = {
        "file": args.pdf,
        "pages": len(doc),
        "checks": {
            "margin_pts": args.margin_pts,
            "dpi_threshold": args.dpi_threshold,
            "advanced_checks": enable_advanced,
            "text_similarity_checks": args.check_text_similarity,
        },
        "issues": {
            # page index (0-based) -> list of issues
        },
        "document_issues": [],
        "summary": {}
    }

    # Document-level checks
    if enable_advanced:
        # PDF/X compliance check
        doc_issues = check_pdfx_compliance(doc)
        report["document_issues"].extend(doc_issues)
        
        # Page numbering validation
        numbering_issues = validate_page_numbering(doc)
        report["document_issues"].extend(numbering_issues)
        
        # Page size validation (new item 2)
        page_size_issues = validate_page_sizes(doc)
        report["document_issues"].extend(page_size_issues)
        
        # Text similarity checks if enabled
        if args.check_text_similarity and len(doc) >= 2:
            # Compare first two pages as an example
            page0_text = doc[0].get_text().strip()
            page1_text = doc[1].get_text().strip()
            if page0_text and page1_text:
                similarity = compare_text_similarity(page0_text, page1_text)
                if similarity > 0.9:  # Very similar - might indicate copy/paste errors
                    report["document_issues"].append({
                        "type": "high_text_similarity",
                        "detail": f"Pages 1-2 are {similarity:.1%} similar - check for duplicated content"
                    })

    total_issues = len(report["document_issues"])
    
    # Page-level checks
    for pno in range(len(doc)):
        page = doc[pno]
        issues = check_page_enhanced(page, args.margin_pts, args.dpi_threshold, pno+1, len(doc), enable_advanced, doc, margin_analysis, document_layout)
        if issues:
            report["issues"][str(pno)] = issues
            total_issues += len(issues)

    report["summary"]["total_issues"] = total_issues
    report["summary"]["document_issues"] = len(report["document_issues"])
    report["summary"]["pages_with_issues"] = len(report["issues"])
    report["summary"]["ok"] = (total_issues == 0)
    
    # Issue breakdown by type
    issue_types = defaultdict(int)
    for page_issues in report["issues"].values():
        for issue in page_issues:
            issue_types[issue["type"]] += 1
    for doc_issue in report["document_issues"]:
        issue_types[doc_issue["type"]] += 1
    
    report["summary"]["issue_breakdown"] = dict(issue_types)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if args.annotate and (report["issues"] or report["document_issues"]):
        annotate_pdf(args.pdf, args.annotate, 
                    {int(k): v for k,v in report["issues"].items()}, 
                    report["document_issues"], 
                    report["summary"])

    # Enhanced console summary (only if verbose)
    if verbose:
        print(f"\n=== PDF Validation Report for {args.pdf} ===")
        print(f"Pages: {len(doc)}")
        print(f"Advanced checks: {'enabled' if enable_advanced else 'disabled'}")
        print(f"Total issues found: {total_issues}")
        
        if report["document_issues"]:
            print(f"\nDocument-level issues: {len(report['document_issues'])}")
            for issue in report["document_issues"][:5]:  # Show first 5
                print(f"  - {issue['type']}: {issue['detail']}")
            if len(report["document_issues"]) > 5:
                print(f"  ... and {len(report['document_issues']) - 5} more")
        
        if report["issues"]:
            print(f"\nPage-level issues: {sum(len(issues) for issues in report['issues'].values())}")
            print(f"Pages affected: {len(report['issues'])}")
            
            # Show issue type breakdown
            if issue_types:
                print("\nIssue breakdown:")
                for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {issue_type}: {count}")
        
        if total_issues == 0:
            print("\n✓ No issues found - PDF appears to meet validation criteria")
        else:
            print(f"\n⚠ {total_issues} issues found - see detailed report for specifics")
    
    if verbose:
        print(json.dumps(report["summary"], indent=2))
    
    doc.close()
    return 0 if total_issues == 0 else 1

def check_hyphenation_at_breaks(page, next_page=None):
    """Check for hyphenated words at end of page breaks"""
    issues = []
    
    try:
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])
        
        # Find text blocks and get last lines
        last_lines = []
        page_height = page.rect.height
        
        for block in blocks:
            if block["type"] == 0:  # text block
                lines = block.get("lines", [])
                if lines:
                    # Get lines in bottom 20% of page
                    bottom_threshold = page_height * 0.8
                    for line in lines:
                        line_y = line.get("bbox", [0,0,0,0])[3]  # bottom Y coordinate
                        if line_y > bottom_threshold:
                            last_lines.append(line)
        
        # Check for hyphens at end of lines
        for line in last_lines:
            spans = line.get("spans", [])
            if spans:
                last_span = spans[-1]
                text = last_span.get("text", "").strip()
                
                # Check if line ends with hyphen
                if text.endswith("-"):
                    # Additional check: make sure it's not just a dash
                    if len(text) > 1 and text[-2].isalnum():
                        # Check if next page starts with continuation
                        continuation_found = False
                        if next_page:
                            next_text = next_page.get_text(textpage_flags=0)[:200]  # First 200 chars
                            first_word = next_text.split()[0] if next_text.split() else ""
                            if first_word and first_word[0].islower():
                                continuation_found = True
                        
                        issues.append({
                            "type": "hyphen_at_page_break",
                            "detail": f"Hyphenated word at page end: '{text}' (continuation: {continuation_found})",
                            "bbox": list(last_span["bbox"])
                        })
                        
    except Exception as e:
        issues.append({"type": "hyphen_check_error", "detail": f"Hyphenation check failed: {str(e)}"})
    
    return issues

def validate_page_sizes(doc):
    """Validate consistent page sizes across document"""
    issues = []
    
    try:
        if len(doc) < 2:
            return issues
            
        # Get dimensions from first page as reference
        first_page = doc[0]
        ref_mediabox = first_page.mediabox
        ref_cropbox = first_page.cropbox
        
        ref_media_size = (ref_mediabox.width, ref_mediabox.height)
        ref_crop_size = (ref_cropbox.width, ref_cropbox.height)
        
        tolerance = 2.0  # 2 points tolerance
        
        for pno in range(1, len(doc)):
            page = doc[pno]
            
            # Check MediaBox consistency
            media = page.mediabox
            media_size = (media.width, media.height)
            
            if (abs(media_size[0] - ref_media_size[0]) > tolerance or 
                abs(media_size[1] - ref_media_size[1]) > tolerance):
                issues.append({
                    "type": "inconsistent_page_size",
                    "detail": f"Page {pno+1} MediaBox {media_size[0]:.1f}x{media_size[1]:.1f} differs from reference {ref_media_size[0]:.1f}x{ref_media_size[1]:.1f}",
                    "page": pno+1
                })
            
            # Check CropBox consistency
            crop = page.cropbox
            crop_size = (crop.width, crop.height)
            
            if (abs(crop_size[0] - ref_crop_size[0]) > tolerance or 
                abs(crop_size[1] - ref_crop_size[1]) > tolerance):
                issues.append({
                    "type": "inconsistent_crop_size", 
                    "detail": f"Page {pno+1} CropBox {crop_size[0]:.1f}x{crop_size[1]:.1f} differs from reference {ref_crop_size[0]:.1f}x{ref_crop_size[1]:.1f}",
                    "page": pno+1
                })
                
    except Exception as e:
        issues.append({"type": "page_size_check_error", "detail": f"Page size validation failed: {str(e)}"})
    
    return issues

def check_line_spacing_consistency(page, tolerance_pts=2.0):
    """Enhanced line spacing consistency checking"""
    issues = []
    
    try:
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])
        
        all_spacings = []
        block_spacings = []
        
        for block in blocks:
            if block["type"] == 0:  # text block
                lines = block.get("lines", [])
                if len(lines) > 1:
                    # Calculate spacing within this block
                    spacings = []
                    for i in range(1, len(lines)):
                        prev_line = lines[i-1]
                        curr_line = lines[i]
                        
                        prev_bottom = prev_line.get("bbox", [0,0,0,0])[3]
                        curr_top = curr_line.get("bbox", [0,0,0,0])[1]
                        
                        spacing = curr_top - prev_bottom
                        if 0 < spacing < 50:  # Reasonable range
                            spacings.append(spacing)
                            all_spacings.append(spacing)
                    
                    if spacings:
                        block_spacings.append({
                            "spacings": spacings,
                            "mean": np.mean(spacings),
                            "std": np.std(spacings),
                            "bbox": block["bbox"]
                        })
        
        if len(all_spacings) > 3:
            overall_mean = np.mean(all_spacings)
            overall_std = np.std(all_spacings)
            
            # Check for significant variations
            if overall_std > tolerance_pts:
                inconsistent_blocks = []
                for block_info in block_spacings:
                    if abs(block_info["mean"] - overall_mean) > tolerance_pts * 1.5:
                        inconsistent_blocks.append(block_info)
                
                if inconsistent_blocks:
                    issues.append({
                        "type": "line_spacing_inconsistency",
                        "detail": f"Inconsistent line spacing: overall {overall_mean:.1f}±{overall_std:.1f}pts, {len(inconsistent_blocks)} blocks vary significantly",
                        "bbox": inconsistent_blocks[0]["bbox"] if inconsistent_blocks else None
                    })
            
            # Check for extremely tight or loose spacing
            min_spacing = min(all_spacings)
            max_spacing = max(all_spacings)
            
            if min_spacing < 2.0:
                issues.append({
                    "type": "line_spacing_too_tight",
                    "detail": f"Very tight line spacing detected: {min_spacing:.1f}pts"
                })
            
            if max_spacing > 30.0:
                issues.append({
                    "type": "line_spacing_too_loose", 
                    "detail": f"Very loose line spacing detected: {max_spacing:.1f}pts"
                })
                
    except Exception as e:
        issues.append({"type": "line_spacing_check_error", "detail": f"Line spacing check failed: {str(e)}"})
    
    return issues

def assess_image_quality_advanced(page):
    """Advanced image quality assessment beyond DPI"""
    issues = []
    
    try:
        images = page.get_images()
        
        for img_index, img in enumerate(images):
            try:
                # Extract image data
                img_data = page.parent.extract_image(img[0])
                img_bytes = img_data["image"]
                
                # Open image for analysis
                img_doc = fitz.open("pdf", img_bytes)
                pix = img_doc[0].get_pixmap()
                
                if pix.width > 20 and pix.height > 20:  # Skip tiny images
                    samples = np.frombuffer(pix.samples, dtype=np.uint8)
                    
                    if pix.n == 1:  # Grayscale
                        samples = samples.reshape((pix.height, pix.width))
                    elif pix.n >= 3:  # RGB/CMYK
                        samples = samples.reshape((pix.height, pix.width, pix.n))
                        # Convert to grayscale for analysis
                        if pix.n == 3:  # RGB
                            samples = np.dot(samples[...,:3], [0.299, 0.587, 0.114])
                        else:  # CMYK - use composite
                            samples = 255 - samples[..., -1]  # Invert K channel
                    
                    if samples.ndim == 2 and samples.size > 400:  # Sufficient data
                        # 1. Blur detection using Laplacian variance
                        from scipy import ndimage
                        laplacian_var = ndimage.laplace(samples.astype(float)).var()
                        
                        if laplacian_var < 100:  # Low variance indicates blur
                            issues.append({
                                "type": "blurry_image",
                                "detail": f"Image {img_index+1}: Low sharpness (variance: {laplacian_var:.1f})",
                                "bbox": list(fitz.Rect(page.get_image_bbox(img)))
                            })
                        
                        # 2. Contrast analysis
                        contrast = samples.std()
                        if contrast < 15:  # Low contrast
                            issues.append({
                                "type": "low_contrast_image",
                                "detail": f"Image {img_index+1}: Low contrast (std: {contrast:.1f})",
                                "bbox": list(fitz.Rect(page.get_image_bbox(img)))
                            })
                        
                        # 3. Histogram analysis for quality issues
                        hist, bins = np.histogram(samples, bins=256, range=(0, 255))
                        
                        # Check for clipped highlights/shadows
                        total_pixels = samples.size
                        highlight_clip = hist[-5:].sum() / total_pixels  # Last 5 bins
                        shadow_clip = hist[:5].sum() / total_pixels     # First 5 bins
                        
                        if highlight_clip > 0.05:  # More than 5% clipped whites
                            issues.append({
                                "type": "clipped_highlights",
                                "detail": f"Image {img_index+1}: {highlight_clip:.1%} clipped highlights",
                                "bbox": list(fitz.Rect(page.get_image_bbox(img)))
                            })
                        
                        if shadow_clip > 0.05:  # More than 5% clipped blacks
                            issues.append({
                                "type": "clipped_shadows",
                                "detail": f"Image {img_index+1}: {shadow_clip:.1%} clipped shadows", 
                                "bbox": list(fitz.Rect(page.get_image_bbox(img)))
                            })
                
                img_doc.close()
                
            except Exception as e:
                issues.append({
                    "type": "image_quality_check_error",
                    "detail": f"Image {img_index+1} quality analysis failed: {str(e)}"
                })
                
    except Exception as e:
        issues.append({"type": "image_quality_analysis_error", "detail": f"Image quality analysis failed: {str(e)}"})
    
    return issues

def check_header_consistency(page, page_number=None, total_pages=None):
    """Detect and validate header consistency"""
    issues = []
    
    try:
        raw = page.get_text("dict")
        blocks = raw.get("blocks", [])
        
        page_height = page.rect.height
        header_threshold = page_height * 0.15  # Top 15% of page
        
        # Find potential header text
        header_elements = []
        
        for block in blocks:
            if block["type"] == 0:  # text block
                bbox = block["bbox"]
                if bbox[1] < header_threshold:  # Top of page
                    # Look for header-like patterns
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                # Check if it looks like a header
                                font_size = span.get("size", 0)
                                
                                # Headers are typically positioned left, center, or right
                                x_pos = span["bbox"][0]
                                page_width = page.rect.width
                                
                                position = "left"
                                if x_pos > page_width * 0.4 and x_pos < page_width * 0.6:
                                    position = "center"
                                elif x_pos > page_width * 0.7:
                                    position = "right"
                                
                                header_elements.append({
                                    "text": text,
                                    "font_size": font_size,
                                    "position": position,
                                    "bbox": span["bbox"],
                                    "page": page_number
                                })
        
        # Store header info for cross-page consistency checking
        # (This would be enhanced to compare across multiple pages)
        if header_elements:
            # Check for reasonable header content
            for header in header_elements:
                text = header["text"].lower()
                
                # Flag potential issues
                if len(text) > 100:  # Headers shouldn't be too long
                    issues.append({
                        "type": "header_too_long",
                        "detail": f"Header text exceeds 100 characters: '{text[:50]}...'",
                        "bbox": header["bbox"]
                    })
                
        # Check for missing headers on pages that should have them
        if page_number and page_number > 1 and not header_elements:
            issues.append({
                "type": "missing_header",
                "detail": f"No header detected on page {page_number}"
            })
            
    except Exception as e:
        issues.append({"type": "header_check_error", "detail": f"Header consistency check failed: {str(e)}"})
    
    return issues

def check_text_comparison(doc, page1_idx=0, page2_idx=1):
    """Compare text between two pages for consistency"""
    issues = []
    
    if len(doc) <= max(page1_idx, page2_idx):
        return issues
    
    try:
        page1_text = doc[page1_idx].get_text().strip()
        page2_text = doc[page2_idx].get_text().strip()
        
        if page1_text and page2_text:
            similarity = compare_text_similarity(page1_text, page2_text)
            
            # Look for specific patterns that might indicate cover/title page comparison
            # This is a basic heuristic - could be enhanced with more sophisticated matching
            if similarity < 0.3:  # Very different - this is expected
                pass
            elif similarity > 0.8:  # Very similar - might be an issue
                issues.append({
                    "type": "high_text_similarity",
                    "detail": f"Pages {page1_idx+1} and {page2_idx+1} are {similarity:.1%} similar"
                })
                
    except Exception as e:
        issues.append({"type": "text_comparison_error", "detail": f"Text comparison failed: {str(e)}"})
    
    return issues

if __name__ == "__main__":
    sys.exit(main())
