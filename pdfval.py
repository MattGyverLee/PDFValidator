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

import json, math, argparse, sys, re
import fitz  # PyMuPDF
import numpy as np
from collections import Counter, defaultdict
from difflib import SequenceMatcher

def rects_intersect(a, b, iou_thresh=0.05):
    # a,b are fitz.Rect; compute IoU-ish to avoid tiny touches
    inter = a & b
    if inter.is_empty: return False
    inter_area = inter.get_area()
    union_area = a.get_area() + b.get_area() - inter_area
    return (inter_area / union_area) >= iou_thresh

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
        raw = page.get_text("rawdict")
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
        raw = page.get_text("rawdict")
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
        raw = page.get_text("rawdict")
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
        raw = page.get_text("rawdict")
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
    raw = page.get_text("rawdict")
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

    # 1) Margin intrusions (text)
    for r, txt in text_rects:
        if r.intersects(left_m) or r.intersects(right_m) or r.intersects(top_m) or r.intersects(bottom_m):
            issues.append({"type":"margin_intrusion_text", "detail":txt[:60], "bbox":list(r)})

    # 1b) Margin intrusions (images)
    for r in image_rects:
        if r.intersects(left_m) or r.intersects(right_m) or r.intersects(top_m) or r.intersects(bottom_m):
            issues.append({"type":"margin_intrusion_image", "bbox":list(r)})

    # 2) Text overlap (span-over-span IoU)
    # O(n^2) over spans per page; typically fine. For big pages, you can grid-index.
    for i in range(len(text_rects)):
        ri, ti = text_rects[i]
        for j in range(i+1, len(text_rects)):
            rj, tj = text_rects[j]
            if rects_intersect(ri, rj, iou_thresh=0.08):
                # Ignore identical boxes with identical text (could be ligature splits).
                if not (ri == rj and ti == tj):
                    issues.append({"type":"overlap_text_text", "detail": f"{ti[:30]} | {tj[:30]}", "bboxes":[list(ri), list(rj)]})

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
        raw = page.get_text("rawdict")
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

def check_page_enhanced(page, margin_pts, dpi_threshold, page_number=None, total_pages=None, enable_advanced=True, doc=None):
    """Enhanced page checking with all validation features"""
    # Run original checks
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

    # Margin zones
    left_m   = fitz.Rect(pr.x0, pr.y0, pr.x0 + margin_pts, pr.y1)
    right_m  = fitz.Rect(pr.x1 - margin_pts, pr.y0, pr.x1, pr.y1)
    top_m    = fitz.Rect(pr.x0, pr.y0, pr.x1, pr.y0 + margin_pts)
    bottom_m = fitz.Rect(pr.x0, pr.y1 - margin_pts, pr.x1, pr.y1)

    # Gather text blocks / spans / glyphs and images
    raw = page.get_text("rawdict")
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

    # 1) Margin intrusions (text)
    for r, txt in text_rects:
        if r.intersects(left_m) or r.intersects(right_m) or r.intersects(top_m) or r.intersects(bottom_m):
            issues.append({"type":"margin_intrusion_text", "detail":txt[:60], "bbox":list(r)})

    # 1b) Margin intrusions (images)
    for r in image_rects:
        if r.intersects(left_m) or r.intersects(right_m) or r.intersects(top_m) or r.intersects(bottom_m):
            issues.append({"type":"margin_intrusion_image", "bbox":list(r)})

    # 2) Text overlap (span-over-span IoU)
    # O(n^2) over spans per page; typically fine. For big pages, you can grid-index.
    for i in range(len(text_rects)):
        ri, ti = text_rects[i]
        for j in range(i+1, len(text_rects)):
            rj, tj = text_rects[j]
            if rects_intersect(ri, rj, iou_thresh=0.08):
                # Ignore identical boxes with identical text (could be ligature splits).
                if not (ri == rj and ti == tj):
                    issues.append({"type":"overlap_text_text", "detail": f"{ti[:30]} | {tj[:30]}", "bboxes":[list(ri), list(rj)]})

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
def check_page(page, margin_pts, dpi_threshold, page_number=None, total_pages=None, enable_advanced=True, doc=None):
    """Main page checking function with enhanced validation"""
    return check_page_enhanced(page, margin_pts, dpi_threshold, page_number, total_pages, enable_advanced, doc)

def annotate_pdf(input_path, output_path, per_page_issues, document_issues=None, report_summary=None):
    """Enhanced PDF annotation with comprehensive issue highlighting and summary pages"""
    doc = fitz.open(input_path)
    
    # Enhanced color mapping for all validation types
    color_by_type = {
        # Original checks - Red family
        "margin_intrusion_text": (1,0,0),
        "margin_intrusion_image": (1,0,0),
        "overlap_text_text": (0,0,1),
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
    ap.add_argument("pdf", help="Input PDF")
    ap.add_argument("--margin-pts", type=float, default=36.0, help="Margin threshold in points (72pt = 1 inch)")
    ap.add_argument("--dpi-threshold", type=int, default=300, help="Min acceptable placed PPI for images")
    ap.add_argument("--annotate", help="Write annotated PDF here")
    ap.add_argument("--json", help="Write JSON report here")
    ap.add_argument("--basic-only", action="store_true", help="Run only basic checks (disable advanced validation)")
    ap.add_argument("--check-text-similarity", action="store_true", help="Enable text consistency checks between pages")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    enable_advanced = not args.basic_only
    
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
        issues = check_page_enhanced(page, args.margin_pts, args.dpi_threshold, pno+1, len(doc), enable_advanced, doc)
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

    # Enhanced console summary
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
    
    print(json.dumps(report["summary"], indent=2))

def check_hyphenation_at_breaks(page, next_page=None):
    """Check for hyphenated words at end of page breaks"""
    issues = []
    
    try:
        raw = page.get_text("rawdict")
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
        raw = page.get_text("rawdict")
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
        raw = page.get_text("rawdict")
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
                
                # Check for common header patterns
                if page_number and page_number > 1:
                    # Look for page numbers in headers
                    page_num_found = any(str(i) in text for i in range(max(1, page_number-2), min(total_pages+1, page_number+3)))
                    chapter_pattern = re.search(r'chapter\s+\d+', text, re.IGNORECASE)
                    
                    if not page_num_found and not chapter_pattern and len(text.split()) < 2:
                        issues.append({
                            "type": "minimal_header_content",
                            "detail": f"Header may be incomplete: '{text}'",
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
