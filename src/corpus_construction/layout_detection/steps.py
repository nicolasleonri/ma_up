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