from pathlib import Path
import zipfile
import tempfile
import re
import numpy as np
import pandas as pd
import nibabel as nib


# ============================================================
# SETTINGS
# ============================================================

ROOT = Path(
    r"C:\Users\24esu\Downloads\Processed_data_nii"
)

OUTPUT = Path(
    r"C:\Users\24esu\Downloads\processed_nii_metadata.csv"
)

SITES = [
    "UCL",
    "RUNMC",
    "I2CVB",
    "HK",
    "BMC",
    "BIDMC",
]

PERCENTILES = [1, 5, 25, 50, 75, 95, 99]


# ============================================================
# OPTIONAL LABEL MAPPING
# ============================================================

# IMPORTANT:
# Leave these as None unless you KNOW that your segmentation
# labels correspond to CG and PZ.
#
# Example only:
# CG_LABEL = 1
# PZ_LABEL = 2

CG_LABEL = None
PZ_LABEL = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def is_segmentation_name(name):
    """
    Determine whether a filename represents a segmentation.
    """
    name = name.lower()

    return (
        "_segmentation" in name
        or "_seg." in name
        or name.endswith("_seg.nii")
        or name.endswith("_seg.nii.gz")
        or "mask" in name
        or "label" in name
    )


def strip_nii_extension(name):
    """
    Remove .nii or .nii.gz.
    """
    name = Path(name).name

    if name.lower().endswith(".nii.gz"):
        return name[:-7]

    if name.lower().endswith(".nii"):
        return name[:-4]

    return name


def normalize_case_id(name):
    """
    Convert image/mask filenames into a common case identifier.

    Examples:
        prostate_01.nii
            -> prostate_01

        prostate_01_segmentation.nii
            -> prostate_01

        prostate_01_seg.nii
            -> prostate_01
    """

    name = strip_nii_extension(name)

    suffixes = [
        "_segmentation",
        "-segmentation",
        "_segment",
        "-segment",
        "_seg",
        "-seg",
        "_mask",
        "-mask",
        "_label",
        "-label",
        "_labels",
        "-labels",
    ]

    lower = name.lower()

    for suffix in suffixes:
        if lower.endswith(suffix):
            name = name[:-len(suffix)]
            break

    return name


# ============================================================
# NIFTI ORIENTATION / DIRECTION
# ============================================================

def get_geometry(img):
    """
    Extract dimensions, voxel spacing, affine matrix,
    direction matrix, orientation codes, and origin.

    NIfTI affine:
        physical coordinate = affine @ voxel coordinate

    The 3x3 affine component contains both voxel scaling
    and direction. We normalize its columns to obtain a
    direction-cosine matrix.
    """

    shape = img.shape

    if len(shape) < 3:
        raise ValueError(
            f"Expected at least 3 dimensions, got {shape}"
        )

    X = int(shape[0])
    Y = int(shape[1])
    Z = int(shape[2])

    zooms = img.header.get_zooms()

    sx = float(zooms[0])
    sy = float(zooms[1])
    sz = float(zooms[2])

    affine = np.asarray(
        img.affine,
        dtype=float
    )

    # First 3 columns encode spatial axes
    spatial = affine[:3, :3]

    # Normalize each column to obtain direction cosines
    norms = np.linalg.norm(
        spatial,
        axis=0
    )

    direction = np.zeros(
        (3, 3),
        dtype=float
    )

    for j in range(3):
        if norms[j] != 0:
            direction[:, j] = (
                spatial[:, j] / norms[j]
            )

    orientation = nib.aff2axcodes(
        affine
    )

    origin = affine[:3, 3]

    result = {
        "X": X,
        "Y": Y,
        "Z": Z,

        "sx_mm": sx,
        "sy_mm": sy,
        "sz_mm": sz,

        "orientation_codes":
            "".join(orientation),

        "origin_x": float(origin[0]),
        "origin_y": float(origin[1]),
        "origin_z": float(origin[2]),
    }

    # Direction matrix
    for r in range(3):
        for c in range(3):
            result[
                f"direction_{r}{c}"
            ] = float(
                direction[r, c]
            )

    # Full NIfTI affine
    for r in range(4):
        for c in range(4):
            result[
                f"affine_{r}{c}"
            ] = float(
                affine[r, c]
            )

    return result


# ============================================================
# RAW INTENSITY STATISTICS
# ============================================================

def get_raw_array(img):
    """
    Load stored NIfTI voxel values WITHOUT applying
    NIfTI slope/intercept scaling when possible.

    This is useful because the requested statistic is
    'raw intensity'.
    """

    dataobj = img.dataobj

    # get_unscaled() is available for standard nibabel
    # NIfTI ArrayProxy objects.
    if hasattr(dataobj, "get_unscaled"):
        data = np.asanyarray(
            dataobj.get_unscaled()
        )
    else:
        data = np.asanyarray(
            dataobj
        )

    return data


def intensity_stats(img):

    data = get_raw_array(img)

    x = np.asarray(
        data,
        dtype=np.float64
    ).ravel()

    # Remove NaN/Inf if present
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {
            "raw_min": np.nan,
            "raw_max": np.nan,
            "raw_mean": np.nan,
            "raw_median": np.nan,
            **{
                f"raw_p{p}": np.nan
                for p in PERCENTILES
            }
        }

    result = {
        "raw_min": float(np.min(x)),
        "raw_max": float(np.max(x)),
        "raw_mean": float(np.mean(x)),
        "raw_median": float(
            np.median(x)
        ),
    }

    for p in PERCENTILES:
        result[
            f"raw_p{p}"
        ] = float(
            np.percentile(x, p)
        )

    return result


# ============================================================
# SEGMENTATION STATISTICS
# ============================================================

def segmentation_stats(seg_img):

    data = np.asanyarray(
        seg_img.dataobj
    )

    # Segmentation labels should be discrete.
    #
    # np.rint protects against labels stored as floats,
    # e.g. 0.0 and 1.0.
    labels_data = np.rint(
        data
    ).astype(np.int64)

    labels = np.unique(
        labels_data
    )

    total = int(
        labels_data.size
    )

    background = int(
        np.count_nonzero(
            labels_data == 0
        )
    )

    foreground = int(
        np.count_nonzero(
            labels_data != 0
        )
    )

    foreground_fraction = (
        foreground / total
        if total > 0
        else np.nan
    )

    result = {
        "label_values_present":
            ",".join(
                str(int(x))
                for x in labels
            ),

        "background_voxels":
            background,

        "foreground_voxels":
            foreground,

        "foreground_fraction":
            foreground_fraction,

        # Generic whole-prostate interpretation:
        # all nonzero segmentation voxels.
        "whole_prostate_voxels":
            foreground,
    }

    # Count every label separately.
    for label in labels:

        label = int(label)

        result[
            f"label_{label}_voxels"
        ] = int(
            np.count_nonzero(
                labels_data == label
            )
        )

    # Only calculate CG/PZ if mapping is known.
    if CG_LABEL is not None:

        result["cg_voxels"] = int(
            np.count_nonzero(
                labels_data == CG_LABEL
            )
        )

    else:

        result["cg_voxels"] = np.nan

    if PZ_LABEL is not None:

        result["pz_voxels"] = int(
            np.count_nonzero(
                labels_data == PZ_LABEL
            )
        )

    else:

        result["pz_voxels"] = np.nan

    return result


# ============================================================
# PHYSICAL FOREGROUND VOLUME
# ============================================================

def physical_volume_ml(
    foreground_voxels,
    sx,
    sy,
    sz
):

    if pd.isna(foreground_voxels):
        return np.nan

    return (
        foreground_voxels
        * sx
        * sy
        * sz
        / 1000.0
    )


# ============================================================
# EXTRACT ZIP FILES
# ============================================================

def collect_nii_files(site_dir, temp_root):
    """
    Find ordinary .nii/.nii.gz files and extract NIfTI
    files contained inside ZIP archives.

    Returns:
        list of dictionaries:
            {
                "path": actual readable path,
                "source": original source description
            }
    """

    files = []

    # --------------------------------------------------------
    # Normal NIfTI files
    # --------------------------------------------------------

    for path in site_dir.rglob("*"):

        if not path.is_file():
            continue

        lower = path.name.lower()

        if (
            lower.endswith(".nii")
            or lower.endswith(".nii.gz")
        ):

            files.append({
                "path": path,
                "source": str(path)
            })

    # --------------------------------------------------------
    # ZIP archives
    # --------------------------------------------------------

    for zip_path in site_dir.rglob("*.zip"):

        print(
            f"    Opening ZIP: "
            f"{zip_path.name}"
        )

        zip_output = (
            temp_root
            / site_dir.name
            / zip_path.stem
        )

        zip_output.mkdir(
            parents=True,
            exist_ok=True
        )

        try:

            with zipfile.ZipFile(
                zip_path,
                "r"
            ) as z:

                for member in z.namelist():

                    lower = member.lower()

                    if not (
                        lower.endswith(".nii")
                        or lower.endswith(".nii.gz")
                    ):
                        continue

                    z.extract(
                        member,
                        zip_output
                    )

                    extracted = (
                        zip_output
                        / member
                    )

                    files.append({
                        "path": extracted,

                        "source":
                            f"{zip_path} :: {member}"
                    })

        except Exception as e:

            print(
                f"    ZIP ERROR: "
                f"{zip_path}: {e}"
            )

    return files


# ============================================================
# MATCH IMAGES TO SEGMENTATIONS
# ============================================================

def pair_files(files):

    images = {}
    segmentations = {}

    for item in files:

        path = item["path"]

        case_id = normalize_case_id(
            path.name
        )

        if is_segmentation_name(
            path.name
        ):

            segmentations.setdefault(
                case_id,
                []
            ).append(item)

        else:

            images.setdefault(
                case_id,
                []
            ).append(item)

    return images, segmentations


# ============================================================
# MAIN
# ============================================================

rows = []


with tempfile.TemporaryDirectory() as temp:

    temp_root = Path(temp)

    for site in SITES:

        site_dir = ROOT / site

        print()
        print("=" * 70)
        print(f"SITE: {site}")
        print("=" * 70)

        if not site_dir.exists():

            print(
                f"WARNING: folder not found: "
                f"{site_dir}"
            )

            continue

        # ----------------------------------------------------
        # Find/extract all NIfTI files
        # ----------------------------------------------------

        files = collect_nii_files(
            site_dir,
            temp_root
        )

        print(
            f"Found {len(files)} "
            f"NIfTI files."
        )

        images, segmentations = pair_files(
            files
        )

        print(
            f"Image case IDs: "
            f"{len(images)}"
        )

        print(
            f"Segmentation case IDs: "
            f"{len(segmentations)}"
        )

        # ----------------------------------------------------
        # Process each image
        # ----------------------------------------------------

        for case_id in sorted(images):

            image_items = images[
                case_id
            ]

            # There may be multiple images for a case.
            for image_number, image_item in enumerate(
                image_items,
                start=1
            ):

                image_path = image_item[
                    "path"
                ]

                print(
                    f"  {site} / {case_id}"
                )

                try:

                    img = nib.load(
                        str(image_path)
                    )

                    geometry = get_geometry(
                        img
                    )

                    intensity = intensity_stats(
                        img
                    )

                    row = {
                        "case_id":
                            case_id,

                        "site":
                            site,

                        "dataset":
                            "Processed_data_nii",

                        "image_source":
                            image_item["source"],

                        "segmentation_available":
                            False,

                        "segmentation_source":
                            "",

                        "label_values_present":
                            "",

                        "background_voxels":
                            np.nan,

                        "foreground_voxels":
                            np.nan,

                        "foreground_fraction":
                            np.nan,

                        "whole_prostate_voxels":
                            np.nan,

                        "cg_voxels":
                            np.nan,

                        "pz_voxels":
                            np.nan,
                    }

                    row.update(
                        geometry
                    )

                    row.update(
                        intensity
                    )

                    # ----------------------------------------
                    # Find corresponding segmentation
                    # ----------------------------------------

                    seg_items = (
                        segmentations.get(
                            case_id,
                            []
                        )
                    )

                    if len(seg_items) > 0:

                        # Usually exactly one segmentation.
                        seg_item = seg_items[0]

                        seg_img = nib.load(
                            str(
                                seg_item["path"]
                            )
                        )

                        seg_stats = (
                            segmentation_stats(
                                seg_img
                            )
                        )

                        row.update(
                            seg_stats
                        )

                        row[
                            "segmentation_available"
                        ] = True

                        row[
                            "segmentation_source"
                        ] = seg_item[
                            "source"
                        ]

                        # ------------------------------------
                        # Check image/mask geometry
                        # ------------------------------------

                        row[
                            "dimensions_match_segmentation"
                        ] = (
                            tuple(
                                img.shape[:3]
                            )
                            ==
                            tuple(
                                seg_img.shape[:3]
                            )
                        )

                        row[
                            "affine_matches_segmentation"
                        ] = bool(
                            np.allclose(
                                img.affine,
                                seg_img.affine,
                                atol=1e-4
                            )
                        )

                        # ------------------------------------
                        # Physical prostate volume
                        # ------------------------------------

                        row[
                            "prostate_volume_ml"
                        ] = physical_volume_ml(
                            row[
                                "foreground_voxels"
                            ],
                            row["sx_mm"],
                            row["sy_mm"],
                            row["sz_mm"],
                        )

                    else:

                        row[
                            "dimensions_match_segmentation"
                        ] = False

                        row[
                            "affine_matches_segmentation"
                        ] = False

                        row[
                            "prostate_volume_ml"
                        ] = np.nan

                    rows.append(
                        row
                    )

                except Exception as e:

                    print(
                        f"    ERROR: {e}"
                    )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(
    rows
)


# ============================================================
# MULTIPARAMETRIC COMPLETENESS
# ============================================================

# At this point we DO NOT know from the filenames alone whether
# these NIfTI files represent T2, ADC, DWI, DCE, etc.
#
# Therefore we do not fabricate sequence completeness.
#
# These columns are included as NaN = "not determined".
#
# Once sequence names are known, these can be populated.

for column in [
    "has_axial_T2",
    "has_DWI",
    "has_ADC",
    "has_DCE",
    "complete_T2_DWI_ADC",
    "complete_T2_DWI_ADC_DCE",
]:

    df[column] = np.nan


# ============================================================
# ORDER COLUMNS
# ============================================================

first_columns = [

    "case_id",
    "site",
    "dataset",

    "X",
    "Y",
    "Z",

    "sx_mm",
    "sy_mm",
    "sz_mm",

    "orientation_codes",

    "direction_00",
    "direction_01",
    "direction_02",
    "direction_10",
    "direction_11",
    "direction_12",
    "direction_20",
    "direction_21",
    "direction_22",

    "raw_min",
    "raw_max",
    "raw_mean",
    "raw_median",

    "raw_p1",
    "raw_p5",
    "raw_p25",
    "raw_p50",
    "raw_p75",
    "raw_p95",
    "raw_p99",

    "label_values_present",

    "background_voxels",
    "foreground_voxels",
    "foreground_fraction",

    "whole_prostate_voxels",

    "cg_voxels",
    "pz_voxels",

    "prostate_volume_ml",

    "segmentation_available",
    "dimensions_match_segmentation",
    "affine_matches_segmentation",

    "has_axial_T2",
    "has_DWI",
    "has_ADC",
    "has_DCE",

    "complete_T2_DWI_ADC",
    "complete_T2_DWI_ADC_DCE",

    "image_source",
    "segmentation_source",
]

existing_first = [
    c
    for c in first_columns
    if c in df.columns
]

remaining = [
    c
    for c in df.columns
    if c not in existing_first
]

df = df[
    existing_first + remaining
]


# ============================================================
# SAVE CSV
# ============================================================

df.to_csv(
    OUTPUT,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    f"\nSaved:\n{OUTPUT}"
)

print(
    f"\nTotal image volumes: "
    f"{len(df)}"
)

print(
    "\nCases by site:"
)

print(
    df.groupby("site")
      .size()
      .to_string()
)

print(
    "\nSegmentation availability by site:"
)

print(
    df.groupby("site")[
        "segmentation_available"
    ]
    .agg(["count", "sum"])
    .to_string()
)

print(
    "\nOrientation codes:"
)

print(
    df.groupby(
        ["site", "orientation_codes"]
    )
    .size()
    .to_string()
)

print(
    "\nLabel values:"
)

print(
    df.groupby(
        ["site", "label_values_present"]
    )
    .size()
    .to_string()
)

print(
    "\nPreview:"
)

preview = [
    "case_id",
    "site",
    "X",
    "Y",
    "Z",
    "sx_mm",
    "sy_mm",
    "sz_mm",
    "orientation_codes",
    "raw_median",
    "label_values_present",
    "foreground_voxels",
    "foreground_fraction",
    "prostate_volume_ml",
]

print(
    df[
        [
            c
            for c in preview
            if c in df.columns
        ]
    ]
    .head(20)
    .to_string(index=False)
)
