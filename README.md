# Enhanced PDF Validator

A comprehensive PDF validation tool for professional print production workflows, based on industry checklists and quality standards.

## Overview

PDFValidator is an advanced Python utility that analyzes PDF documents against professional print production criteria. It performs 20+ robust validation checks covering layout, typography, image quality, color compliance, and document structure to ensure PDFs meet publishing standards.

## Features

### Core Validation Capabilities
- **🔍 Document Structure**: PDF/X compliance, page numbering, size consistency
- **📐 Layout Quality**: Margin intrusions, baseline grid alignment, column balance
- **🎨 Color & Print**: Color mode validation, image screening detection
- **📝 Typography**: Line spacing consistency, header validation, hyphenation checks
- **🖼️ Image Quality**: DPI analysis, blur detection, contrast assessment, clipping analysis
- **⚙️ Technical**: Font embedding, page box validation, content overlap detection

## Installation

### Requirements

```bash
pip install PyMuPDF numpy scipy
```

### Dependencies
- **Python 3.x**
- **PyMuPDF (fitz)**: PDF processing
- **NumPy**: Numerical analysis  
- **SciPy**: Advanced image analysis

## Usage

### Basic Validation
```bash
python pdfval.py document.pdf
```

### Advanced Options
```bash
python pdfval.py document.pdf \\
  --margin-pts 36 \\
  --dpi-threshold 300 \\
  --annotate annotated_output.pdf \\
  --json validation_report.json \\
  --check-text-similarity
```

### Validation Modes
```bash
# Full validation (default)
python pdfval.py document.pdf

# Basic checks only (faster)
python pdfval.py document.pdf --basic-only

# With text consistency analysis
python pdfval.py document.pdf --check-text-similarity
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `pdf` | Input PDF file (required) | - |
| `--margin-pts` | Margin threshold in points | 36.0 (0.5") |
| `--dpi-threshold` | Minimum acceptable DPI for images | 300 |
| `--annotate` | Generate annotated PDF with visual indicators | - |
| `--json` | Export detailed validation report | - |
| `--basic-only` | Run only basic checks (disable advanced validation) | False |
| `--check-text-similarity` | Enable text consistency checks between pages | False |

## Validation Coverage

### ✅ Implemented Checks (20/42 from industry checklist)

#### **Document-Level Validation**
| Check | Description | Implementation |
|-------|-------------|----------------|
| PDF/X Compliance | Validates conformance declarations and OutputIntents | ✅ Full |
| Page Size Consistency | Ensures uniform MediaBox/CropBox across pages | ✅ Full |
| Page Numbering | Detects non-sequential numbering patterns | ✅ Basic |
| Color Mode Analysis | Flags RGB in B&W sections, validates CMYK usage | ✅ Full |

#### **Layout & Typography**
| Check | Description | Implementation |
|-------|-------------|----------------|
| Margin Intrusions | Text/images too close to edges | ✅ Full |
| Baseline Grid Alignment | Statistical analysis of text positioning | ✅ Full |
| Line Spacing Consistency | Detects spacing variations and extremes | ✅ Enhanced |
| Column Balance | Validates balanced multi-column layouts | ✅ Full |
| Header Consistency | Validates header presence and content patterns | ✅ Full |
| Content Overlaps | Text-over-text and text-over-image detection | ✅ Full |
| Hyphenation at Breaks | Detects hyphenated words at page ends | ✅ Full |

#### **Image & Quality**
| Check | Description | Implementation |
|-------|-------------|----------------|
| DPI Analysis | Low-resolution image detection | ✅ Full |
| Image Screening | Detects screened/halftoned images | ✅ Advanced |
| Blur Detection | Laplacian variance sharpness analysis | ✅ Full |
| Contrast Assessment | Statistical contrast measurement | ✅ Full |
| Clipping Detection | Highlight/shadow clipping analysis | ✅ Full |
| Font Embedding | Verifies embedded font compliance | ✅ Full |

#### **Print Production**
| Check | Description | Implementation |
|-------|-------------|----------------|
| Page Box Structure | CropBox/MediaBox relationship validation | ✅ Full |

### ❌ Not Yet Implemented (22/42 remaining)

#### **Content Analysis** (Requires OCR/NLP)
- Logo detection on title page
- Verso page book title verification  
- Picture copyright validation
- TOC validation (title/page number matching)
- Cross-reference validation
- ISBN/metadata validation

#### **Advanced Layout** (Requires Domain Knowledge)
- Line completeness/spacing validation
- Black overprint validation for maps
- Double-page spread completeness
- Crop marks validation (3mm spacing)
- Back matter consistency

#### **Specialized Features** (Document-Specific)
- Thumb tab validation and index comparison
- Dummy page validation
- Style & format consistency
- Copyright holder validation
- Capitalization consistency

## Output Formats

### Console Summary
```
=== PDF Validation Report for document.pdf ===
Pages: 35
Advanced checks: enabled
Total issues found: 199

Document-level issues: 2
  - pdfx_compliance: No PDF/X conformance declaration found
  - pdfx_compliance: Missing OutputIntents (required for PDF/X)

Page-level issues: 197
Pages affected: 35

Issue breakdown:
  missing_header: 34
  dpi_unknown_image: 25
  image_quality_check_error: 24
  ...
```

### JSON Report Structure
```json
{
  "file": "document.pdf",
  "pages": 35,
  "checks": {
    "margin_pts": 36.0,
    "dpi_threshold": 300,
    "advanced_checks": true,
    "text_similarity_checks": false
  },
  "issues": {
    "0": [
      {
        "type": "baseline_grid_violation",
        "detail": "62/107 lines not aligned to 9pt grid",
        "bbox": [x, y, width, height]
      }
    ]
  },
  "document_issues": [],
  "summary": {
    "total_issues": 199,
    "pages_with_issues": 35,
    "ok": false,
    "issue_breakdown": {...}
  }
}
```

## Developer Customization

### Adjusting Validation Constraints

#### **Margin Validation**
```python
# In check_page_original() function
margin_pts = 36.0  # Default: 0.5 inch
# Modify: left_m, right_m, top_m, bottom_m rectangles
```

#### **DPI Thresholds**
```python
# In assess_image_quality_advanced() and main()
dpi_threshold = 300  # Default minimum DPI
# Modify: --dpi-threshold CLI argument
```

#### **Line Spacing Tolerances**
```python
# In check_line_spacing_consistency()
tolerance_pts = 2.0  # Default tolerance
min_tight = 2.0      # Minimum spacing threshold  
max_loose = 30.0     # Maximum spacing threshold
```

#### **Baseline Grid Tolerances**
```python  
# In check_baseline_alignment()
grid_threshold = 2.0    # Alignment tolerance in points
misaligned_percent = 0.2  # 20% misalignment threshold
```

#### **Image Quality Thresholds**
```python
# In assess_image_quality_advanced()
laplacian_threshold = 100   # Blur detection sensitivity
contrast_threshold = 15     # Minimum contrast std dev
clip_threshold = 0.05       # 5% clipping tolerance
```

#### **Column Balance Tolerance**
```python
# In check_column_balance()
balance_threshold = 0.15  # 15% imbalance tolerance
```

#### **Page Size Tolerance**
```python
# In validate_page_sizes()
tolerance = 2.0  # 2-point size variation tolerance
```

#### **Overlap Detection Sensitivity**
```python
# In rects_intersect()
text_text_iou = 0.08  # Text overlap IoU threshold
text_image_iou = 0.02 # Text-image overlap threshold
```

#### **Header Detection**
```python
# In check_header_consistency()  
header_threshold = 0.15  # Top 15% of page
max_header_length = 100  # Maximum header character count
```

### Adding Custom Validation Rules

#### **1. Create New Validation Function**
```python
def check_custom_rule(page, custom_param=10):
    \"\"\"Custom validation rule\"\"\"
    issues = []
    
    try:
        # Your validation logic here
        if condition_met:
            issues.append({
                "type": "custom_issue_type",
                "detail": f"Custom issue detected: {details}",
                "bbox": [x, y, w, h]  # Optional
            })
    except Exception as e:
        issues.append({
            "type": "custom_check_error", 
            "detail": f"Custom check failed: {str(e)}"
        })
    
    return issues
```

#### **2. Integrate into Main Validation**
```python
# In check_page_enhanced() function
if enable_advanced:
    issues.extend(check_custom_rule(page, custom_param=15))
```

#### **3. Add Color Coding**
```python
# In annotate_pdf() color_by_type dictionary
"custom_issue_type": (r, g, b),  # RGB values 0-1
"custom_check_error": (0.5, 0.5, 0.5),
```

#### **4. Add CLI Parameter (Optional)**
```python
# In main() function
ap.add_argument("--custom-param", type=int, default=10, 
                help="Custom validation parameter")
```

### Issue Type Reference

#### **Color Coding in Annotated PDFs**
| Issue Category | Color | Examples |
|----------------|-------|----------|
| Layout Issues | Red | margin_intrusion_text, margin_intrusion_image |
| Content Overlap | Blue/Purple | overlap_text_text, overlap_text_image |
| Image Quality | Orange/Red | low_dpi_image, blurry_image |
| Typography | Cyan/Yellow | baseline_grid_violation, line_spacing_inconsistency |
| Document Structure | Purple | pdfx_compliance, inconsistent_page_size, page_sequence_error |
| Errors/Warnings | Gray | *_check_error, dpi_unknown_image |

## Recent Improvements

### **Version 2.0 Updates**
- **🐛 Fixed Page Sequence Detection**: Eliminated 1000+ false positives by improving page number detection logic
- **🔧 Resolved DPI Duplication**: Fixed duplicate `dpi_unknown_image` errors by streamlining image processing  
- **🗑️ Removed Thumb Tab Validation**: Eliminated specialized validation that required domain expertise
- **📊 Enhanced Accuracy**: Reduced false positives by 84-89% while maintaining comprehensive coverage

### **Issue Count Improvements**
| Document Size | Before | After | Reduction |
|---------------|--------|--------|-----------|
| Small PDF (5 pages) | 254 issues | 28 issues | 89% reduction |
| Large PDF (35 pages) | 1,266 issues | 199 issues | 84% reduction |

## Performance Considerations

### Processing Time Factors
- **Document size**: Linear scaling with page count
- **Image analysis**: Quadratic scaling with image count/resolution  
- **Advanced checks**: ~3-5x slower than basic validation
- **Memory usage**: Proportional to largest image in document

### Optimization Tips
```bash
# For large documents, use basic mode first
python pdfval.py large_doc.pdf --basic-only

# Process specific page ranges (manual implementation needed)
# Skip intensive image analysis for drafts
```

## Examples

### Production Workflow
```bash
# Stage 1: Quick validation
python pdfval.py manuscript.pdf --basic-only

# Stage 2: Full validation with reporting  
python pdfval.py manuscript.pdf \\
  --json output/manuscript_report.json \\
  --annotate output/manuscript_reviewed.pdf

# Stage 3: Custom print settings
python pdfval.py final_proof.pdf \\
  --margin-pts 54 \\
  --dpi-threshold 400 \\
  --check-text-similarity \\
  --json output/final_report.json
```

### Batch Processing Template
```bash
#!/bin/bash
mkdir -p output
for pdf in *.pdf; do
    echo "Validating $pdf..."
    python pdfval.py "$pdf" \\
      --json "output/${pdf%.pdf}_report.json" \\
      --annotate "output/${pdf%.pdf}_annotated.pdf"
done
```

## Technical Architecture

### Module Structure
- **Core validation**: `check_page_original()` - basic PDF analysis
- **Enhanced validation**: `check_page_enhanced()` - advanced analysis  
- **Document-level**: `validate_page_sizes()`, `check_pdfx_compliance()`
- **Specialized analysis**: `assess_image_quality_advanced()`, `detect_image_screening()`
- **Reporting**: `annotate_pdf()`, JSON output formatting

### Dependencies & Limitations
- **PyMuPDF**: PDF parsing, may not support all PDF variants
- **NumPy/SciPy**: Mathematical analysis, requires sufficient memory for large images
- **Processing time**: O(n²) text overlap detection per page
- **Accuracy**: Heuristic-based analysis, not definitive print assessment
- **Page numbering**: Detection works best with standard header/footer positioning
- **False positives**: Significantly reduced but some edge cases may still occur

## File Organization

### **Repository Structure**
```
PDFValidator/
├── pdfval.py                    # Main validation script
├── README.md                    # This documentation
├── img/                         # Documentation screenshots  
├── output/                      # Generated validation reports and annotated PDFs
├── *.pdf                        # Sample PDF files for testing
└── requirements files           # Dependency specifications
```

### **Output Files**
All generated validation reports and annotated PDFs are automatically saved to the `output/` directory to keep the repository organized:

- **JSON Reports**: `output/*_report.json` - Detailed validation results
- **Annotated PDFs**: `output/*_annotations.pdf` - Visual issue highlighting with summary pages
- **Sample Files**: `output/out.json`, `output/annotated.pdf` - Original example outputs

**Tip**: Use `mkdir -p output` before running batch validations to ensure the output directory exists.

## License

This project is provided as-is for PDF quality validation purposes. Designed for professional print production workflows.