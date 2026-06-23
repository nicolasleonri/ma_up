import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Set

# ---------------------------------------------------------------------------
# Label filtering — editorial content only, per detector
# ---------------------------------------------------------------------------

# NewspaperNavigator labels:
#   photograph, illustration, map, comic/cartoon, editorial cartoon,
#   headline, advertisement, article
LP_KEEP_LABELS: Set[str] = {"article", "headline"}

# DocStructBench labels:
#   Title, Plain text, Abandon, Figure, Figure caption,
#   Table, Table caption, Reference, Equation
# "Abandon" = headers/footers/page numbers — discard.
YOLO_KEEP_LABELS: Set[str] = {"title", "plain text", "figure caption"}

# PP-DocLayout-L labels (23 categories):
#   doc_title, paragraph_title, text, page_number, abstract, content,
#   reference, footnote, header, footer, algorithm, formula,
#   formula_number, image, figure_title, table, table_title, seal,
#   chart_title, chart, header_image, footer_image, aside_text
PP_KEEP_LABELS: Set[str] = {"doc_title", "paragraph_title", "text", "figure_title"}

# Surya layout labels (canonical, post-relabel):
#   Caption, Footnote, Equation, ListGroup, PageHeader, PageFooter,
#   Picture, SectionHeader, Table, Text, Figure, Code, Form,
#   TableOfContents, ChemicalBlock, Diagram, Bibliography, BlankPage
SURYA_KEEP_LABELS: Set[str] = {"SectionHeader", "Text", "Caption"}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LayoutRegion:
    """A single detected region (headline, body text, caption…)."""
    label: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int
    grid_row: int = -1
    grid_col: int = -1

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "score": round(self.score, 4),
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "grid_row": self.grid_row,
            "grid_col": self.grid_col,
        }


@dataclass
class Article:
    """
    A grouped set of regions that belong to the same article.
    The merged bounding box covers all member regions.
    """
    regions: List[LayoutRegion] = field(default_factory=list)

    @property
    def x1(self) -> int:
        return min(r.x1 for r in self.regions)

    @property
    def y1(self) -> int:
        return min(r.y1 for r in self.regions)

    @property
    def x2(self) -> int:
        return max(r.x2 for r in self.regions)

    @property
    def y2(self) -> int:
        return max(r.y2 for r in self.regions)

    @property
    def grid_row(self) -> int:
        return self.regions[0].grid_row if self.regions else -1

    @property
    def grid_col(self) -> int:
        return self.regions[0].grid_col if self.regions else -1

    def to_dict(self) -> dict:
        return {
            "x1": self.x1, "y1": self.y1,
            "x2": self.x2, "y2": self.y2,
            "grid_row": self.grid_row,
            "grid_col": self.grid_col,
            "num_regions": len(self.regions),
            "regions": [r.to_dict() for r in self.regions],
        }


# ---------------------------------------------------------------------------
# Grid index assignment
# ---------------------------------------------------------------------------

def assign_grid_index(
    region: LayoutRegion,
    image_height: int,
    image_width: int,
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> LayoutRegion:
    """Assign a (row, col) grid cell based on the region's center point."""
    col = min(int(region.center_x / image_width * grid_cols), grid_cols - 1)
    row = min(int(region.center_y / image_height * grid_rows), grid_rows - 1)
    region.grid_row = row
    region.grid_col = col
    return region


# ---------------------------------------------------------------------------
# Article grouping
# ---------------------------------------------------------------------------

def _horizontal_overlap_ratio(a: LayoutRegion, b: LayoutRegion) -> float:
    """
    Fraction of horizontal overlap relative to the narrower region.
    Returns 0 if no overlap.
    """
    overlap = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    narrower = min(a.x2 - a.x1, b.x2 - b.x1)
    if narrower == 0:
        return 0.0
    return overlap / narrower


def group_regions_into_articles(
    regions: List[LayoutRegion],
    image_height: int,
    col_overlap_threshold: float = 0.5,
    vertical_gap_ratio: float = 0.04,
) -> List[Article]:
    """
    Group detected regions into articles using two criteria:
      1. Column alignment  — two regions must share at least
         `col_overlap_threshold` of horizontal overlap to belong together.
      2. Vertical proximity — the vertical gap between them must be no more
         than `vertical_gap_ratio * image_height` (default 4% of page height).

    Grouping is done with union-find so transitive chains are captured
    (title → subheadline → body all merge even if title and body are far apart,
    as long as the chain of neighbours connects them).

    Articles are returned sorted top-to-bottom by their merged y1.
    """
    n = len(regions)
    if n == 0:
        return []

    max_gap = vertical_gap_ratio * image_height

    # Union-Find
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    # Compare every pair — O(n²), fine for typical page region counts (<100)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = regions[i], regions[j]

            # Must share a column
            if _horizontal_overlap_ratio(a, b) < col_overlap_threshold:
                continue

            # Must be vertically adjacent (gap between bottom of one and top of other)
            top, bottom = (a, b) if a.y1 <= b.y1 else (b, a)
            gap = bottom.y1 - top.y2
            if gap < 0 or gap <= max_gap:
                union(i, j)

    # Collect groups
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for i, r in enumerate(regions):
        groups[find(i)].append(r)

    # Build Article objects, sorted top-to-bottom
    articles = [Article(regions=members) for members in groups.values()]
    articles.sort(key=lambda a: a.y1)
    return articles


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

class LayoutParserDetector:
    """
    Newspaper layout detector using LayoutParser + Newspaper Navigator model.
    Keeps only editorial regions (article, headline) and groups them into
    Article objects.
    """

    MODEL_PATH = "lp://NewspaperNavigator/faster_rcnn_R_50_FPN_3x/config"

    def __init__(self, score_threshold: float = 0.5):
        import layoutparser as lp

        self.model = lp.models.Detectron2LayoutModel(
            self.MODEL_PATH,
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", score_threshold],
            label_map={
                0: "photograph",
                1: "illustration",
                2: "map",
                3: "comic/cartoon",
                4: "editorial cartoon",
                5: "headline",
                6: "advertisement",
                7: "article",
            },
        )

    def detect(self, image: np.ndarray) -> List[LayoutRegion]:
        """Return editorial regions only (no grouping — done in pipeline)."""
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image)
        layout = self.model.detect(pil_image)

        regions = []
        for block in layout:
            label = block.type
            if label not in LP_KEEP_LABELS:
                continue
            regions.append(LayoutRegion(
                label=label,
                score=block.score,
                x1=int(block.block.x_1),
                y1=int(block.block.y_1),
                x2=int(block.block.x_2),
                y2=int(block.block.y_2),
            ))
        return regions


class DocLayoutYOLODetector:
    """
    Newspaper layout detector using DocLayout-YOLO (DocStructBench weights).
    Keeps only editorial regions (title, plain text, figure caption) and
    groups them into Article objects.
    """

    HF_MODEL_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
    WEIGHT_FILE = "doclayout_yolo_docstructbench_imgsz1024.pt"

    def __init__(self, score_threshold: float = 0.25, imgsz: int = 1024):
        from doclayout_yolo import YOLOv10
        from huggingface_hub import hf_hub_download

        weight_path = hf_hub_download(
            repo_id=self.HF_MODEL_ID,
            filename=self.WEIGHT_FILE,
        )
        self.model = YOLOv10(weight_path)
        self.score_threshold = score_threshold
        self.imgsz = imgsz

    def detect(self, image: np.ndarray) -> List[LayoutRegion]:
        """Return editorial regions only (no grouping — done in pipeline)."""
        results = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.score_threshold,
            verbose=False,
        )

        regions = []
        for result in results:
            for box in result.boxes:
                label = result.names[int(box.cls)]
                if label.lower() not in YOLO_KEEP_LABELS:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                regions.append(LayoutRegion(
                    label=label,
                    score=float(box.conf),
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
        return regions
    

class PPDocLayoutDetector:
    """
    Newspaper layout detector using PaddleOCR's PP-DocLayout-L model.
    Keeps only editorial regions (doc_title, paragraph_title, text,
    figure_title) and groups them into Article objects.
    """

    MODEL_NAME = "PP-DocLayout-L"

    def __init__(self, score_threshold: float = 0.5):
        from paddleocr import LayoutDetection

        self.model = LayoutDetection(model_name=self.MODEL_NAME)
        self.score_threshold = score_threshold

    def detect(self, image: np.ndarray) -> List[LayoutRegion]:
        """Return editorial regions only (no grouping — done in pipeline)."""
        results = self.model.predict(image, batch_size=1)

        regions = []
        for result in results:
            # paddleocr result objects expose the dict either directly
            # or nested under "res", depending on version.
            data = result["res"] if "res" in result else result

            for box in data.get("boxes", []):
                label = box["label"]
                score = float(box["score"])
                if label not in PP_KEEP_LABELS or score < self.score_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box["coordinate"])
                regions.append(LayoutRegion(
                    label=label,
                    score=score,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
        return regions

class SuryaLayoutDetector:
    """
    Layout-only Surya detector.
    Avoids Surya 2 VLM/llama.cpp backend when possible.
    """

    def __init__(self, score_threshold: float = 0.5):
        from surya.layout import LayoutPredictor

        self.score_threshold = score_threshold
        self.model = LayoutPredictor()


    def detect(self, image: np.ndarray) -> List[LayoutRegion]:
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image)

        predictions = self.model([pil_image])
        prediction = predictions[0]

        regions = []

        for box in prediction.bboxes:
            label = box.label
            score = float(box.confidence)

            if (
                label not in SURYA_KEEP_LABELS
                or score < self.score_threshold
            ):
                continue

            x1, y1, x2, y2 = map(int, box.bbox)

            regions.append(
                LayoutRegion(
                    label=label,
                    score=score,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

        return regions


class HistogramColumnDetector:
    """
    Histogram-based layout detector for binary newspaper images.
 
    Segments the page into regions by finding ink-free vertical and horizontal
    gaps in the pixel projection histograms — no ML model required.
 
    Detection strategy
    ------------------
    1. Binarise the input image (Otsu threshold on the grayscale version).
    2. Project pixel ink counts onto the X axis to find vertical column
       boundaries, then onto the Y axis within each column to find row
       boundaries.
    3. Each (column, row) cell that contains enough ink becomes a
       LayoutRegion labelled "plain text".
 
    The detector always achieves complete coverage: every pixel belongs to
    exactly one region, so the union of all bounding boxes tiles the page
    without gaps or overlaps.
 
    Parameters
    ----------
    min_col_width : int
        Minimum pixel width of a vertical text column (filters ruled lines
        and thin artefacts).
    min_row_height : int
        Minimum pixel height of a horizontal text row inside a column.
    gap_threshold_ratio : float
        A column (or row) of pixels is considered a gap when its ink count
        falls below ``gap_threshold_ratio * max_ink_count``.  Default 0.05
        means "less than 5 % of the most ink-dense column".
    score : float
        Confidence score assigned to every region (no ML model, so fixed).
    """

    def __init__(
        self,
        min_col_width: int = 75,
        min_row_height: int = 60,
        gap_threshold_ratio: float = 0.05,
        score_threshold: float = 1.0,
    ):
        self.min_col_width = min_col_width
        self.min_row_height = min_row_height
        self.gap_threshold_ratio = gap_threshold_ratio
        self.score = score_threshold
 
    # ------------------------------------------------------------------
    # Public interface — matches LayoutParserDetector / DocLayoutYOLODetector
    # ------------------------------------------------------------------
 
    def detect(self, image: np.ndarray) -> List[LayoutRegion]:
        """
        Detect layout regions in a BGR or grayscale newspaper image.
 
        Parameters
        ----------
        image : np.ndarray
            BGR (H×W×3) or grayscale (H×W) uint8 image.
 
        Returns
        -------
        List[LayoutRegion]
            One region per detected (column, row) cell, labelled "plain text".
        """
        binary = self._binarise(image)
        col_bounds = self._find_spans(
            projection=np.sum(binary == 0, axis=0),
            min_size=self.min_col_width,
            total_size=binary.shape[1],
            gap_threshold_ratio=self.gap_threshold_ratio,
        )
 
        regions: List[LayoutRegion] = []
        for x1, x2 in col_bounds:
            col_strip = binary[:, x1:x2]
            row_bounds = self._find_spans(
                projection=np.sum(col_strip == 0, axis=1),
                min_size=self.min_row_height,
                total_size=binary.shape[0],
                gap_threshold_ratio=self.gap_threshold_ratio,
            )
            for y1, y2 in row_bounds:
                regions.append(LayoutRegion(
                    label="plain text",
                    score=self.score,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                ))
 
        return regions
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _binarise(image: np.ndarray) -> np.ndarray:
        """
        Convert any input image to a binary array (0 = ink, 255 = background).
 
        Accepts BGR colour or single-channel grayscale.  Uses Otsu's method
        so it adapts to each page's contrast without manual tuning.
        """
        import cv2
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
 
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
 
        # Normalise polarity: ink = 0, background = 255
        if np.mean(binary) < 128:
            binary = 255 - binary
 
        return binary
 
    @staticmethod
    def _find_spans(
        projection: np.ndarray,
        min_size: int,
        total_size: int,
        gap_threshold_ratio: float,
    ) -> List[Tuple[int, int]]:
        """
        Split a 1-D ink-count projection into contiguous text spans with full
        coverage (gaps are absorbed into the nearest span boundary).
 
        Parameters
        ----------
        projection : np.ndarray
            1-D array of ink pixel counts (one entry per column or row).
        min_size : int
            Minimum span width/height to keep; narrower runs are merged into
            their neighbour rather than dropped.
        total_size : int
            Total width (for columns) or height (for rows) of the image.
        gap_threshold_ratio : float
            Fraction of the peak ink count below which a position is a gap.
 
        Returns
        -------
        List[Tuple[int, int]]
            (start, end) pairs in pixel coordinates, non-overlapping and
            collectively covering [0, total_size].
        """
        if projection.max() == 0:
            return [(0, total_size)]
 
        threshold = projection.max() * gap_threshold_ratio
        is_gap = projection < threshold
 
        # Locate raw text runs (ignoring min_size for now)
        text_runs: List[Tuple[int, int]] = []
        in_run = False
        start = 0
        for i, gap in enumerate(is_gap):
            if not gap and not in_run:
                start = i
                in_run = True
            elif gap and in_run:
                if i - start >= min_size:
                    text_runs.append((start, i))
                in_run = False
        # Handle run that reaches the end of the image
        if in_run and total_size - start >= min_size:
            text_runs.append((start, total_size))
 
        if not text_runs:
            return [(0, total_size)]
 
        # Build coverage boundaries: each span extends to the midpoint of the
        # surrounding gap so that every pixel is assigned to exactly one span.
        bounds: List[Tuple[int, int]] = []
        current = 0
        for i, (t_start, t_end) in enumerate(text_runs):
            if i == len(text_runs) - 1:
                # Last span: extend to the image edge
                bounds.append((current, total_size))
            else:
                next_t_start = text_runs[i + 1][0]
                boundary = (t_end + next_t_start) // 2
                bounds.append((current, boundary))
                current = boundary
 
        return bounds
