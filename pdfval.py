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

import json, math, argparse, sys
import fitz  # PyMuPDF

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

def check_page(page, margin_pts, dpi_threshold):
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

def annotate_pdf(input_path, output_path, per_page_issues):
    doc = fitz.open(input_path)
    color_by_type = {
        "margin_intrusion_text": (1,0,0),
        "margin_intrusion_image": (1,0,0),
        "overlap_text_text": (0,0,1),
        "overlap_text_image": (0.5,0,0.5),
        "low_dpi_image": (1,0.5,0),
        "dpi_unknown_image": (0.5,0.5,0.5),
        "font_not_embedded": (0,0,0),
        "page_box_error": (0,0,0),
    }
    for pno, issues in per_page_issues.items():
        page = doc[pno]
        for iss in issues:
            col = color_by_type.get(iss["type"], (0,0,0))
            if "bbox" in iss:
                r = fitz.Rect(iss["bbox"])
                #page.draw_rect(r, color=col, width=1)
                page.draw_circle(r.tl, radius=5, color=col, width=1)  # small circle at top-left
                page.draw_circle(r.br, radius=5, color=col, width=1)  # bottom-right
                page.draw_rect(r, color=col, width=1)  # keep the box

            elif "bboxes" in iss:
                for bb in iss["bboxes"]:
                    r = fitz.Rect(bb)
                    page.draw_rect(r, color=col, width=1)
            # label
            label = iss["type"]
            if "detail" in iss and iss["detail"]:
                label += f": {iss['detail']}"
            # Put small sticky note at top-left of first bbox if present
            if "bbox" in iss:
                r = fitz.Rect(iss["bbox"])
                page.add_text_annot(r.tl, label)
            elif "bboxes" in iss:
                r = fitz.Rect(iss["bboxes"][0])
                page.add_text_annot(r.tl, label)
    doc.save(output_path)
    doc.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Input PDF")
    ap.add_argument("--margin-pts", type=float, default=36.0, help="Margin threshold in points (72pt = 1 inch)")
    ap.add_argument("--dpi-threshold", type=int, default=300, help="Min acceptable placed PPI for images")
    ap.add_argument("--annotate", help="Write annotated PDF here")
    ap.add_argument("--json", help="Write JSON report here")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    report = {
        "file": args.pdf,
        "pages": len(doc),
        "checks": {
            "margin_pts": args.margin_pts,
            "dpi_threshold": args.dpi_threshold,
        },
        "issues": {
            # page index (0-based) -> list of issues
        },
        "summary": {}
    }

    total_issues = 0
    for pno in range(len(doc)):
        page = doc[pno]
        issues = check_page(page, args.margin_pts, args.dpi_threshold)
        if issues:
            report["issues"][str(pno)] = issues
            total_issues += len(issues)

    report["summary"]["total_issues"] = total_issues
    report["summary"]["pages_with_issues"] = len(report["issues"])
    report["summary"]["ok"] = (total_issues == 0)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if args.annotate and report["issues"]:
        annotate_pdf(args.pdf, args.annotate, {int(k): v for k,v in report["issues"].items()})

    # Also print a brief console summary
    print(json.dumps(report["summary"], indent=2))

if __name__ == "__main__":
    sys.exit(main())
