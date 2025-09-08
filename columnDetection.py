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
        
        # Enhanced vertical analysis
        precise_zones = self._analyze_vertical_zones(spans_by_zone)
        full_width_elements = self._detect_full_width_elements(precise_zones)
        intervening_elements = self._detect_intervening_elements(precise_zones)
        
        layout = self._detect_column_layout(precise_zones['body'])
        divider = self._find_column_divider(precise_zones['body']) if layout == 'double' else None
        
        return self._create_enhanced_boundaries(layout, divider, precise_zones, full_width_elements, intervening_elements)
    
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
    
    def _analyze_vertical_zones(self, zones: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """Analyze vertical zones to find precise boundaries and detect intervening elements"""
        enhanced_zones = {'header': [], 'body': [], 'footer': [], 'intervening': []}
        
        # Get all spans sorted by vertical position
        all_spans = []
        for zone_name, spans in zones.items():
            for span in spans:
                span_copy = span.copy()
                span_copy['original_zone'] = zone_name
                all_spans.append(span_copy)
        
        all_spans.sort(key=lambda s: s['center_y'])
        
        if not all_spans:
            return enhanced_zones
        
        # Find actual header/footer boundaries based on text distribution
        page_height = self.page_height
        header_threshold = page_height * 0.25  # Look in top 25%
        footer_threshold = page_height * 0.75  # Look in bottom 25%
        
        # Identify gaps in text flow that indicate zone boundaries
        y_positions = [span['center_y'] for span in all_spans]
        gaps = self._find_significant_vertical_gaps(y_positions)
        
        # Determine precise zone boundaries
        header_bottom = self._find_header_boundary(all_spans, header_threshold, gaps)
        footer_top = self._find_footer_boundary(all_spans, footer_threshold, gaps)
        
        # Classify spans into precise zones
        for span in all_spans:
            y = span['center_y']
            
            if y <= header_bottom:
                enhanced_zones['header'].append(span)
            elif y >= footer_top:
                enhanced_zones['footer'].append(span)
            else:
                # Check if this span breaks column flow (intervening element)
                if self._is_intervening_element(span, all_spans):
                    enhanced_zones['intervening'].append(span)
                else:
                    enhanced_zones['body'].append(span)
        
        return enhanced_zones
    
    def _find_significant_vertical_gaps(self, y_positions: List[float]) -> List[Tuple[float, float]]:
        """Find significant vertical gaps in text flow"""
        if len(y_positions) < 2:
            return []
        
        gaps = []
        sorted_positions = sorted(y_positions)
        
        for i in range(1, len(sorted_positions)):
            gap_size = sorted_positions[i] - sorted_positions[i-1]
            if gap_size > 15:  # Significant gap threshold
                gaps.append((sorted_positions[i-1], sorted_positions[i]))
        
        return gaps
    
    def _find_header_boundary(self, spans: List[Dict], threshold: float, gaps: List[Tuple[float, float]]) -> float:
        """Find precise header bottom boundary - only if substantial header content exists"""
        # Only look in top 15% of page for actual header content
        strict_header_threshold = self.page_height * 0.15
        header_spans = [s for s in spans if s['center_y'] < strict_header_threshold]
        
        # Require minimum header content to create header zone
        if not header_spans or len(header_spans) < 3:
            return 0  # No meaningful header content
        
        # Find the lowest actual header span
        lowest_header_y = max(span['bbox'].y1 for span in header_spans)
        
        # Look for a significant gap after the header
        for gap_start, gap_end in gaps:
            if gap_start > lowest_header_y and gap_end < threshold and (gap_end - gap_start) > 10:
                return gap_start
        
        return lowest_header_y
    
    def _find_footer_boundary(self, spans: List[Dict], threshold: float, gaps: List[Tuple[float, float]]) -> float:
        """Find precise footer top boundary - only if substantial footer content exists"""
        # Only look in bottom 10% of page for actual footer content (much stricter)
        strict_footer_threshold = self.page_height * 0.9
        potential_footer_spans = [s for s in spans if s['center_y'] > strict_footer_threshold]
        
        # Additional filtering: footer content should be short text (page numbers, etc.)
        # Filter out long text that's likely body content misplaced in footer area
        footer_spans = []
        for span in potential_footer_spans:
            text = span.get('text', '').strip()
            # Footer content should be short (page numbers, chapter refs, etc.)
            if len(text) <= 20 and (text.isdigit() or len(text.split()) <= 3):
                footer_spans.append(span)
        
        # Require minimum footer content to create footer zone
        if not footer_spans or len(footer_spans) < 2:  # Even stricter - just 2+ short spans
            return self.page_height  # No meaningful footer content
        
        # Find the highest actual footer span
        highest_footer_y = min(span['bbox'].y0 for span in footer_spans)
        
        # Look for a significant gap before the footer
        for gap_start, gap_end in gaps:
            if gap_end < highest_footer_y and gap_start > threshold * 0.5 and (gap_end - gap_start) > 15:
                return gap_end
        
        return highest_footer_y
    
    def _is_intervening_element(self, span: Dict, all_spans: List[Dict]) -> bool:
        """Check if a span is an intervening element that breaks column flow"""
        # Only detect truly significant intervening elements, not small text breaks
        span_width = span['bbox'].x1 - span['bbox'].x0
        page_center = self.page_width / 2
        span_center = (span['bbox'].x0 + span['bbox'].x1) / 2
        
        # Must be significantly wide (crosses significant portion of page)
        if span_width < self.page_width * 0.4:  # Must be at least 40% of page width
            return False
        
        # Get nearby spans for comparison
        y_tolerance = 30  # Look within 30pts vertically
        nearby_spans = [s for s in all_spans if abs(s['center_y'] - span['center_y']) > y_tolerance 
                       and abs(s['center_y'] - span['center_y']) < y_tolerance * 3]
        
        if nearby_spans:
            avg_width = sum(s['bbox'].x1 - s['bbox'].x0 for s in nearby_spans) / len(nearby_spans)
            
            # Intervening only if significantly wider, centered, and crosses columns
            if (span_width > avg_width * 2.0 and  # Much more restrictive
                abs(span_center - page_center) < self.page_width * 0.15 and  # Better centered
                span_width > self.page_width * 0.5):  # Crosses most of page
                return True
        
        # Check for title-like characteristics - but be very restrictive
        text = span.get('text', '').strip()
        if len(text) > 10 and span_width > self.page_width * 0.5:  # Must be substantial text and width
            # All caps titles or clear section headers
            if (text.isupper() and len(text.split()) <= 4) or \
               (text.istitle() and len(text.split()) <= 3 and any(word in text.upper() for word in ['CHAPTER', 'SECTION', 'PART'])):
                return True
        
        return False
    
    def _detect_full_width_elements(self, zones: Dict[str, List[Dict]]) -> List[Dict]:
        """Detect elements that span the full width of the page"""
        full_width_elements = []
        
        # Analyze intervening elements to see which are full-width
        for span in zones.get('intervening', []):
            span_width = span['bbox'].x1 - span['bbox'].x0
            span_left = span['bbox'].x0
            span_right = span['bbox'].x1
            
            # Check if element spans most of the page width
            page_margin = 50  # Consider 50pt margins
            effective_page_width = self.page_width - (2 * page_margin)
            
            if (span_width > effective_page_width * 0.7 and  # At least 70% of page width
                span_left < page_margin * 2 and  # Starts near left margin
                span_right > self.page_width - page_margin * 2):  # Ends near right margin
                
                full_width_elements.append({
                    'span': span,
                    'type': self._classify_full_width_element(span),
                    'y_position': span['center_y']
                })
        
        return full_width_elements
    
    def _classify_full_width_element(self, span: Dict) -> str:
        """Classify the type of full-width element"""
        text = span.get('text', '').strip()
        
        # Check text characteristics
        if text.isupper() and len(text.split()) <= 4:
            return 'section_title'
        elif text.istitle() and len(text.split()) <= 6:
            return 'chapter_title'  
        elif text.count('.') > 2 or text.count('_') > 5:
            return 'separator_line'
        else:
            return 'full_width_text'
    
    def _detect_intervening_elements(self, zones: Dict[str, List[Dict]]) -> List[Dict]:
        """Detect elements that intervene in the normal column flow"""
        intervening = []
        
        # Get body text sorted by vertical position
        body_spans = sorted(zones.get('body', []), key=lambda s: s['center_y'])
        intervening_spans = zones.get('intervening', [])
        
        for int_span in intervening_spans:
            int_y = int_span['center_y']
            
            # Find body text above and below this element
            above_spans = [s for s in body_spans if s['center_y'] < int_y - 10]
            below_spans = [s for s in body_spans if s['center_y'] > int_y + 10]
            
            if above_spans and below_spans:
                # This element interrupts the flow
                intervening.append({
                    'span': int_span,
                    'interrupts_flow': True,
                    'above_count': len(above_spans),
                    'below_count': len(below_spans),
                    'element_type': self._classify_intervening_element(int_span)
                })
            else:
                # This element is at the beginning or end
                intervening.append({
                    'span': int_span,
                    'interrupts_flow': False,
                    'element_type': self._classify_intervening_element(int_span)
                })
        
        return intervening
    
    def _classify_intervening_element(self, span: Dict) -> str:
        """Classify the type of intervening element"""
        text = span.get('text', '').strip()
        span_width = span['bbox'].x1 - span['bbox'].x0
        
        # Font size analysis (if available)
        font_size = span.get('size', 12)
        
        if font_size > 14:
            return 'heading'
        elif text.isupper():
            return 'section_header'
        elif span_width > self.page_width * 0.6:
            return 'wide_element'
        else:
            return 'text_break'
    
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
    
    def _create_enhanced_boundaries(self, layout: str, divider: Optional[float], 
                                   zones: Dict[str, List[Dict]], 
                                   full_width_elements: List[Dict],
                                   intervening_elements: List[Dict]) -> List[TextRegion]:
        """Create enhanced text region boundaries with precise zones and intervening elements"""
        regions = []
        
        # Create precise header regions (only where substantial text actually exists)
        if zones['header'] and len(zones['header']) >= 3:
            header_spans = zones['header']
            header_x = [coord for span in header_spans for coord in [span['bbox'].x0, span['bbox'].x1]]
            header_y = [coord for span in header_spans for coord in [span['bbox'].y0, span['bbox'].y1]]
            
            boundary_margin = 2.0
            header_region = TextRegion(
                bbox=fitz.Rect(
                    max(0, min(header_x) - boundary_margin),
                    min(header_y) - boundary_margin,
                    min(self.page_width, max(header_x) + boundary_margin),
                    max(header_y) + boundary_margin
                ),
                text_type='header',
                spans_count=len(header_spans)
            )
            regions.append(header_region)
        
        # Create precise footer regions (only where substantial text actually exists)
        if zones['footer'] and len(zones['footer']) >= 2:  # Match the stricter footer criteria
            footer_spans = zones['footer']
            footer_x = [coord for span in footer_spans for coord in [span['bbox'].x0, span['bbox'].x1]]
            footer_y = [coord for span in footer_spans for coord in [span['bbox'].y0, span['bbox'].y1]]
            
            boundary_margin = 2.0
            footer_region = TextRegion(
                bbox=fitz.Rect(
                    max(0, min(footer_x) - boundary_margin),
                    min(footer_y) - boundary_margin,
                    min(self.page_width, max(footer_x) + boundary_margin),
                    max(footer_y) + boundary_margin
                ),
                text_type='footer',
                spans_count=len(footer_spans)
            )
            regions.append(footer_region)
        
        # Create body regions (excluding intervening elements)
        body_spans = zones['body']
        if body_spans:
            regions.extend(self._create_body_regions(layout, divider, body_spans))
        
        # Create regions for full-width elements
        for fw_element in full_width_elements:
            span = fw_element['span']
            boundary_margin = 3.0  # Slightly larger margin for full-width elements
            
            fw_region = TextRegion(
                bbox=fitz.Rect(
                    span['bbox'].x0 - boundary_margin,
                    span['bbox'].y0 - boundary_margin,
                    span['bbox'].x1 + boundary_margin,
                    span['bbox'].y1 + boundary_margin
                ),
                text_type=fw_element['type'],
                spans_count=1
            )
            regions.append(fw_region)
        
        # Create regions for intervening elements (that aren't full-width)
        for int_element in intervening_elements:
            span = int_element['span']
            
            # Skip if already covered by full-width elements
            is_full_width = any(abs(fw['y_position'] - span['center_y']) < 5 for fw in full_width_elements)
            if is_full_width:
                continue
            
            boundary_margin = 2.0
            int_region = TextRegion(
                bbox=fitz.Rect(
                    span['bbox'].x0 - boundary_margin,
                    span['bbox'].y0 - boundary_margin,
                    span['bbox'].x1 + boundary_margin,
                    span['bbox'].y1 + boundary_margin
                ),
                text_type='intervening_' + int_element['element_type'],
                spans_count=1
            )
            regions.append(int_region)
        
        return regions
    
    def _create_body_regions(self, layout: str, divider: Optional[float], body_spans: List[Dict]) -> List[TextRegion]:
        """Create body text regions with precise boundaries"""
        if not body_spans:
            return []
        
        regions = []
        boundary_margin = 2.0
        
        # Get body text bounds
        all_body_x = [coord for span in body_spans for coord in [span['bbox'].x0, span['bbox'].x1]]
        all_body_y = [coord for span in body_spans for coord in [span['bbox'].y0, span['bbox'].y1]]
        
        min_y = min(all_body_y) - boundary_margin
        max_y = max(all_body_y) + boundary_margin
        
        # Determine horizontal boundaries
        if self.document_layout and self.document_layout.has_facing_pages:
            text_left = min(all_body_x)
            text_right = max(all_body_x)
            min_x = max(0, text_left - boundary_margin)
            max_x = min(self.page_width, text_right + boundary_margin)
        else:
            min_x = max(36.0, min(all_body_x) - boundary_margin)
            max_x = min(self.page_width - 36.0, max(all_body_x) + boundary_margin)
        
        if layout == 'double' and divider:
            # Create left and right column regions with enhanced logic
            visual_divider = self._detect_visual_separator()
            
            if visual_divider and abs(visual_divider - divider) < 1:
                # Enhanced visual separator logic
                tolerance = 10.0
                left_column_text = [span for span in body_spans if span['bbox'].x1 < divider - tolerance]
                right_column_text = [span for span in body_spans if span['bbox'].x0 > divider + tolerance]
                
                if left_column_text and right_column_text:
                    rightmost_left_text = max(span['bbox'].x1 for span in left_column_text)
                    leftmost_right_text = min(span['bbox'].x0 for span in right_column_text)
                    
                    text_clearance = 2.0
                    max_movement = 4.0
                    
                    left_inner_boundary = max(divider - max_movement, rightmost_left_text + text_clearance)
                    right_inner_boundary = min(divider + max_movement, leftmost_right_text - text_clearance)
                    
                    left_boundary_distance = divider - min_x
                    final_right_boundary = min(divider + left_boundary_distance, self.page_width)
                    
                    left_region = TextRegion(
                        bbox=fitz.Rect(min_x, min_y, left_inner_boundary, max_y),
                        text_type='body_left',
                        spans_count=len(left_column_text)
                    )
                    right_region = TextRegion(
                        bbox=fitz.Rect(right_inner_boundary, min_y, final_right_boundary, max_y),
                        text_type='body_right', 
                        spans_count=len(right_column_text)
                    )
                    regions.extend([left_region, right_region])
                else:
                    # Fallback to simple divider
                    left_spans = [s for s in body_spans if s['bbox'].x1 < divider]
                    right_spans = [s for s in body_spans if s['bbox'].x0 > divider]
                    
                    left_region = TextRegion(
                        bbox=fitz.Rect(min_x, min_y, divider, max_y),
                        text_type='body_left',
                        spans_count=len(left_spans)
                    )
                    right_region = TextRegion(
                        bbox=fitz.Rect(divider, min_y, max_x, max_y),
                        text_type='body_right',
                        spans_count=len(right_spans)
                    )
                    regions.extend([left_region, right_region])
            else:
                # Text-based divider
                left_spans = [s for s in body_spans if (s['bbox'].x0 + s['bbox'].x1) / 2 < divider]
                right_spans = [s for s in body_spans if (s['bbox'].x0 + s['bbox'].x1) / 2 > divider]
                
                left_region = TextRegion(
                    bbox=fitz.Rect(min_x, min_y, divider, max_y),
                    text_type='body_left',
                    spans_count=len(left_spans)
                )
                right_region = TextRegion(
                    bbox=fitz.Rect(divider, min_y, max_x, max_y),
                    text_type='body_right',
                    spans_count=len(right_spans)
                )
                regions.extend([left_region, right_region])
        else:
            # Single column
            single_region = TextRegion(
                bbox=fitz.Rect(min_x, min_y, max_x, max_y),
                text_type='body_single',
                spans_count=len(body_spans)
            )
            regions.append(single_region)
        
        return regions


def annotate_page_boundaries(page: fitz.Page, regions: List[TextRegion]) -> None:
    """Add colored boundary annotations to a page"""
    colors = {
        'body_single': [0, 0.8, 0],          # Green
        'body_left': [0, 0.8, 0],            # Green  
        'body_right': [0, 0.8, 0],           # Green
        'header': [0, 0.6, 0.8],             # Teal
        'footer': [0, 0.6, 0.8],             # Teal
        'section_title': [0.8, 0.4, 0],      # Orange
        'chapter_title': [0.9, 0.3, 0.1],    # Red-orange
        'full_width_text': [0.7, 0.5, 0.2],  # Golden
        'separator_line': [0.5, 0.5, 0.5],   # Gray
        'intervening_heading': [0.8, 0, 0.8], # Magenta
        'intervening_section_header': [0.6, 0, 0.6], # Purple
        'intervening_wide_element': [0.4, 0.7, 0.9], # Light blue
        'intervening_text_break': [0.9, 0.7, 0.4],   # Yellow
        'title': [0.8, 0.4, 0]               # Orange (legacy)
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
            
            # Group regions by type for cleaner output
            region_groups = {}
            for region in regions:
                if region.text_type not in region_groups:
                    region_groups[region.text_type] = []
                region_groups[region.text_type].append(region)
            
            # Show detected regions with counts
            for region_type, region_list in region_groups.items():
                if len(region_list) == 1:
                    region = region_list[0] 
                    spans_info = f" ({region.spans_count} spans)" if region.spans_count > 0 else ""
                    print(f"  {region_type}{spans_info}: {region.bbox}")
                else:
                    total_spans = sum(r.spans_count for r in region_list)
                    spans_info = f" ({total_spans} spans total)" if total_spans > 0 else ""
                    print(f"  {region_type} ({len(region_list)} regions{spans_info}):")
                    for i, region in enumerate(region_list):
                        region_spans_info = f" ({region.spans_count} spans)" if region.spans_count > 0 else ""
                        print(f"    [{i+1}]{region_spans_info}: {region.bbox}")
            
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