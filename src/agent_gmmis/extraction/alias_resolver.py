"""Dataset Alias Resolver: Resolves scientific dataset synonyms, abbreviations, and challenge names to canonical GMMIS benchmarks."""

import re
from typing import Dict, List, Optional
from agent_gmmis.data.db_manager import DatabaseManager


# Canonical benchmark mapping based on Moglia et al., Information Fusion 2026
DEFAULT_ALIASES: Dict[str, List[str]] = {
    "FLARE": [
        "flare",
        "flare21",
        "flare22",
        "flare23",
        "flare24",
        "flare 2021",
        "flare 2022",
        "flare 2023",
        "flare 2024",
        "miccai flare",
        "miccai 2022 flare",
        "organs in abdomen ct",
        "abdomen ct",
        "abdomen ct dataset",
        "miccai 2022 flare challenge",
    ],
    "AMOS": [
        "amos",
        "amos22",
        "amos 2022",
        "miccai amos",
        "miccai 2022 amos",
        "organs in abdomen mr",
        "abdomen mr",
        "abdomen mr dataset",
        "amos ct",
        "amos mri",
        "amos mr",
        "amos external test set (mri only)",
        "amos external test set",
        "amos22 grand challenge",
    ],
    "BTCV": [
        "btcv",
        "synapse",
        "synapse multi-organ",
        "synapse multiorgan",
        "beyond the cranial vault",
        "multi-atlas labeling beyond the cranial vault",
        "btcv abdomen",
    ],
    "BraTS": [
        "brats",
        "brats20",
        "brats21",
        "brats22",
        "brats23",
        "brats24",
        "brats 2020",
        "brats 2021",
        "brats 2022",
        "brats 2023",
        "brain tumor segmentation",
        "brain tumor",
        "brats challenge",
        "brats dataset",
    ],
    "ATLAS v2.0": [
        "atlas v2.0",
        "atlas v2",
        "atlas 2.0",
        "atlas r2.0",
        "anatomical tracings of lesions after stroke",
        "atlas (v2.0)",
        "atlas lesion",
        "atlas v2.0 dataset",
    ],
    "ATLAS 2023": [
        "atlas 2023",
        "atlas2023",
        "atlas23",
        "a tumour and liver automatic segmentation",
        "atlas liver",
    ],
    "KiTS": [
        "kits",
        "kits19",
        "kits21",
        "kits23",
        "fed-kits",
        "fed-kits2019",
        "kidney tumor segmentation",
        "kidney and kidney tumor",
    ],
    "TotalSegmentator": [
        "totalsegmentator",
        "totalseg",
        "totalsegmentator mri",
        "totalsegmentator ct",
        "totalsegmentator dataset",
    ],
    "CHAOS": [
        "chaos",
        "chaos 2019",
        "chaos external test set (mri only)",
        "chaos external test set",
        "combined healthy abdominal organ segmentation",
    ],
    "PROMISE12": [
        "promise12",
        "prost.mri",
        "prostate mri",
        "promise 2012",
        "promise",
    ],
    "ACDC": [
        "acdc",
        "acdc 2017",
        "automatic cardiac diagnosis challenge",
        "automated cardiac diagnosis challenge",
    ],
    "ToothFairy": [
        "toothfairy",
        "toothfairy 2023",
        "tooth fairy",
    ],
    "CrossMoDA": [
        "crossmoda",
        "crossmoda 2021",
        "crossmoda 2022",
        "crossmoda 2023",
        "crossmoda23",
    ],
    "ISLES": [
        "isles",
        "isles22",
        "isles 2022",
        "isles 2015",
        "ischemic stroke lesion segmentation",
    ],
    "SegTHOR": [
        "segthor",
        "segthor 2019",
        "thoracic organs at risk",
    ],
    "WMH": [
        "wmh",
        "wmh 2017",
        "wmh2017",
        "white matter hyperintensities",
        "white matter hyperintensity",
    ],
    "FeTA": [
        "feta",
        "feta 2021",
        "feta 2022",
        "fetal tissue annotation",
    ],
    "MSSEG": [
        "msseg",
        "msseg 2016",
        "multiple sclerosis lesion segmentation",
    ],
    "OASIS-1": [
        "oasis-1",
        "oasis 1",
        "oasis1",
        "oasis",
    ],
    "OASIS-3": [
        "oasis-3",
        "oasis 3",
        "oasis3",
    ],
    "LiTS / MSD Liver": [
        "lits",
        "msd liver",
        "lits / msd liver",
        "lits/msd liver",
        "liver tumor segmentation",
        "task03_liver",
    ],
    "MSD Lung Tumors": [
        "msd lung",
        "msd lung tumors",
        "lung tumors",
        "task06_lung",
        "msd lung tumours",
    ],
    "MSD Spleen": [
        "msd spleen",
        "task09_spleen",
    ],
    "MSD Pancreas Tumour": [
        "msd pancreas",
        "msd pancreas tumour",
        "task07_pancreas",
        "pancreas tumour",
    ],
    "MSD Colon Cancer": [
        "msd colon",
        "msd colon cancer",
        "task10_colon",
    ],
    "MSD Hepatic Vessels": [
        "msd hepatic vessels",
        "task08_hepaticvessel",
    ],
    "MSD Hippocampus": [
        "msd hippocampus",
        "task04_hippocampus",
    ],
    "MSD Cardiac": [
        "msd cardiac",
        "task02_heart",
    ],
    "AbdomenCT-1K": [
        "abdomenct-1k",
        "abdomenct 1k",
        "abdomenct1k",
    ],
    "AbdomenAtlas": [
        "abdomenatlas",
        "abdomenatlas 1.0",
        "abdomenatlas-8k",
    ],
    "Prostate158": [
        "prostate158",
        "prostate 158",
    ],
    "WORD": [
        "word",
        "whole abdominal organ dataset",
    ],
    "3D-IRCADb": [
        "3d-ircadb",
        "3d ircadb",
        "ircadb",
    ],
    "CTSpine1K": [
        "ctspine1k",
        "ctspine 1k",
        "ct spine 1k",
    ],
    "VerSe": [
        "verse",
        "verse19",
        "verse20",
        "vertebrae segmentation",
    ],
    "Pancreas-CT": [
        "pancreas-ct",
        "nih pancreas",
        "nih pancreas-ct",
    ],
}


class DatasetAliasResolver:
    """Harmonizes non-standard, abbreviated, or organ-specific dataset mentions into canonical GMMIS benchmark IDs."""

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.aliases: Dict[str, List[str]] = dict(DEFAULT_ALIASES)
        canonical_names: Dict[str, str] = {c.lower(): c for c in DEFAULT_ALIASES.keys()}

        if db_manager:
            try:
                datasets = db_manager.get_all_datasets()
                for d in datasets:
                    canonical_names[d.name.lower()] = d.name
            except Exception:
                pass

        # Build reverse index: canonical names always map to themselves
        self.exact_map: Dict[str, str] = {}
        for lower_c, canon in canonical_names.items():
            self.exact_map[lower_c] = canon

        # Map explicit curated synonyms from DEFAULT_ALIASES
        for canon, syns in self.aliases.items():
            for s in syns:
                s_low = s.lower().strip()
                if s_low not in canonical_names or canonical_names[s_low] == canon:
                    self.exact_map[s_low] = canon

        # Augment with full names and sub-datasets from DB if not already mapped
        if db_manager:
            try:
                datasets = db_manager.get_all_datasets()
                for d in datasets:
                    if d.full_name:
                        fn_low = d.full_name.lower().strip()
                        if fn_low not in self.exact_map:
                            self.exact_map[fn_low] = d.name
                    if d.sub_dataset:
                        for sub in d.sub_dataset.split(","):
                            s_clean = sub.strip().lower()
                            # Never let a sub-dataset override an existing canonical dataset
                            if s_clean and s_clean not in canonical_names and s_clean not in self.exact_map:
                                self.exact_map[s_clean] = d.name
            except Exception:
                pass

    def resolve(self, raw_name: str) -> Optional[str]:
        """Resolves a raw extracted dataset name to its canonical GMMIS benchmark ID.
        
        Returns canonical name (e.g. 'FLARE', 'AMOS', 'BTCV') or None if unmapped.
        """
        if not raw_name:
            return None

        clean = raw_name.strip().lower()
        # Direct lookup
        if clean in self.exact_map:
            return self.exact_map[clean]

        # Strip common trailing filler words: dataset, challenge, benchmark, test set, mri only, ct only
        simplified = re.sub(
            r"\b(dataset|challenge|benchmark|test set|validation set|mri only|ct only|only|evaluation|val)\b",
            "",
            clean,
        ).strip(" -_()[]:,")
        if simplified in self.exact_map:
            return self.exact_map[simplified]

        # Substring / Boundary matching (longest match first to avoid partial collision)
        sorted_synonyms = sorted(self.exact_map.keys(), key=len, reverse=True)
        for syn in sorted_synonyms:
            if len(syn) < 3:
                continue
            # Check whole word boundary match in raw string
            pattern = r"\b" + re.escape(syn) + r"\b"
            if re.search(pattern, clean) or re.search(pattern, simplified):
                return self.exact_map[syn]

        return None


ORGAN_ALIASES: Dict[str, List[str]] = {
    "Dataset-Level": [
        "overall",
        "mean",
        "avg",
        "average",
        "macro-average",
        "macro average",
        "macro avg",
        "multi-organ mean",
        "multiorgan mean",
        "all",
        "all organs",
        "total",
        "mean dsc",
        "overall mean",
        "overall average",
        "mdice",
        "mean dice",
        "avg dice",
        "dataset-level",
        "dataset level",
        "all classes",
    ],
    # Brain Tumor / Neuro sub-tasks
    "Whole Tumor (WT)": [
        "wt",
        "whole tumor",
        "whole tumour",
        "whole tumor (wt)",
        "wt (whole tumor)",
    ],
    "Enhancing Tumor (ET)": [
        "et",
        "enhancing tumor",
        "enhancing tumour",
        "enhancing tumor (et)",
        "et (enhancing tumor)",
        "tumor enhancing",
    ],
    "Tumor Core (TC)": [
        "tc",
        "tumor core",
        "tumour core",
        "tumor core (tc)",
        "tc (tumor core)",
    ],
    "Brain": [
        "brain",
        "brain parenchyma",
        "brainstem",
    ],
    "Brain Tumor": [
        "brain tumor",
        "brain tumour",
        "glioma",
    ],
    "Hippocampus": [
        "hippocampus",
        "hippocampi",
    ],
    # Abdominal / BTCV / AMOS / TotalSegmentator / FLARE / KiTS / LiTS organs
    "Liver": [
        "liver",
        "liv",
        "hepatic",
    ],
    "Spleen": [
        "spleen",
        "spl",
    ],
    "Pancreas": [
        "pancreas",
        "pan",
    ],
    "Kidney": [
        "kidney",
        "kidneys",
        "kid",
        "renal",
    ],
    "Left Kidney": [
        "left kidney",
        "kidney (l)",
        "kidney l",
        "l kidney",
        "lkid",
        "lkidney",
        "kid. l",
        "kidney_l",
    ],
    "Right Kidney": [
        "right kidney",
        "kidney (r)",
        "kidney r",
        "r kidney",
        "rkid",
        "rkidney",
        "kid. r",
        "kidney_r",
    ],
    "Stomach": [
        "stomach",
        "sto",
        "gastric",
    ],
    "Gallbladder": [
        "gallbladder",
        "gall bladder",
        "gal",
        "gb",
    ],
    "Aorta": [
        "aorta",
        "aor",
    ],
    "IVC": [
        "ivc",
        "inferior vena cava",
        "vena cava",
    ],
    "Portal Vein": [
        "portal vein",
        "portal & splenic vein",
        "portal and splenic vein",
        "portal vein and splenic vein",
        "veins",
        "pv",
        "psv",
    ],
    "Esophagus": [
        "esophagus",
        "oesophagus",
        "eso",
    ],
    "Duodenum": [
        "duodenum",
        "duo",
    ],
    "Adrenal Gland": [
        "adrenal gland",
        "adrenal glands",
        "adrenal",
        "adr",
    ],
    "Right Adrenal Gland": [
        "right adrenal gland",
        "adrenal gland (r)",
        "adrenal r",
        "rag",
    ],
    "Left Adrenal Gland": [
        "left adrenal gland",
        "adrenal gland (l)",
        "adrenal l",
        "lag",
    ],
    "Bladder": [
        "bladder",
        "urinary bladder",
        "bla",
    ],
    "Prostate": [
        "prostate",
        "prostate gland",
        "pro",
    ],
    "Uterus": [
        "uterus",
    ],
    "Colon": [
        "colon",
    ],
    "Rectum": [
        "rectum",
    ],
    "Tumor": [
        "tumor",
        "tumour",
        "lesion",
        "liver tumor",
        "kidney tumor",
        "pancreas tumor",
        "pancreatic tumor",
        "lung tumor",
    ],
    # Cardiac / ACDC
    "Myocardium": [
        "myo",
        "myocardium",
        "lv myocardium",
        "left ventricular myocardium",
    ],
    "Left Ventricle": [
        "lv",
        "left ventricle",
        "left ventricular cavity",
        "lv cavity",
    ],
    "Right Ventricle": [
        "rv",
        "right ventricle",
        "right ventricular cavity",
        "rv cavity",
    ],
    "Heart": [
        "heart",
        "cardiac",
        "whole heart",
        "left atrium",
        "la",
    ],
    # Thoracic & Skeletal
    "Lung": [
        "lung",
        "lungs",
        "pulmonary",
    ],
    "Left Lung": [
        "left lung",
        "lung (l)",
        "lung l",
        "l lung",
    ],
    "Right Lung": [
        "right lung",
        "lung (r)",
        "lung r",
        "r lung",
    ],
    "Spinal Cord": [
        "spinal cord",
        "spinal canal",
    ],
    "Vertebrae": [
        "vertebrae",
        "spine",
    ],
    "Trachea": [
        "trachea",
    ],
}


def _normalize_organ_text(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s]", " ", name.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


_ORGAN_EXACT_MAP: Dict[str, str] = {}
for _canon_organ, _syns in ORGAN_ALIASES.items():
    _ORGAN_EXACT_MAP[_canon_organ.lower()] = _canon_organ
    _ORGAN_EXACT_MAP[_normalize_organ_text(_canon_organ)] = _canon_organ
    for _syn in _syns:
        _ORGAN_EXACT_MAP[_syn.lower()] = _canon_organ
        _ORGAN_EXACT_MAP[_normalize_organ_text(_syn)] = _canon_organ


class OrganAliasResolver:
    """Canonicalizes extracted organ / sub-task labels across extraction, auditing, and DB persistence."""

    aliases: Dict[str, List[str]] = ORGAN_ALIASES
    exact_map: Dict[str, str] = _ORGAN_EXACT_MAP

    @classmethod
    def resolve(cls, raw_organ_name: Optional[str]) -> str:
        """Resolves a raw organ or macro-average label into its canonical form."""
        if raw_organ_name is None:
            return "Dataset-Level"

        stripped = str(raw_organ_name).strip()
        if not stripped:
            return "Dataset-Level"

        low = stripped.lower()
        if low in cls.exact_map:
            return cls.exact_map[low]

        norm = _normalize_organ_text(stripped)
        if not norm:
            return "Dataset-Level"
        if norm in cls.exact_map:
            return cls.exact_map[norm]

        # Check multi-word boundary matches sorted longest-first
        for syn in sorted(cls.exact_map.keys(), key=len, reverse=True):
            if len(syn) < 3:
                continue
            if re.search(r"\b" + re.escape(syn) + r"\b", norm):
                return cls.exact_map[syn]

        return stripped.title()

    @classmethod
    def is_dataset_level(cls, organ_name: Optional[str]) -> bool:
        """Returns True if the given organ label resolves to aggregated Dataset-Level."""
        return cls.resolve(organ_name) == "Dataset-Level"


# Known 3D volumetric radiological datasets & modalities (for inclusive screening & 2D filtering)
KNOWN_3D_DATASETS_AND_MODALITIES = {
    "brats", "amos", "btcv", "synapse", "kits", "lits", "totalsegmentator", "acdc",
    "flare", "msd", "atlas", "word", "chaos", "promise12", "autopet", "hecktor",
    "luna16", "lidc", "verse", "segthor", "feta", "mm-whs", "mmwhs", "crossmoda",
    "toothfairy", "isles", "abdomenct", "pancreas-ct", "spleen", "hippocampus",
    "prostate158", "picai", "mnm", "m&ms", "ct", "mri", "pet", "cbct", "mra", "3d",
}

NON_3D_PHOTOGRAPHIC_MODALITIES = {
    "dermoscopy", "dermatoscopy", "skin lesion", "isic", "ph2",
    "fundus", "retinal fundus", "retinal", "drive", "stare", "chase", "refuge", "hrf", "oct",
    "histopathology", "pathology", "microscopy", "whole slide", "wsi", "cell", "nuclei", "glas", "monuseg", "pannuke", "consep", "dsb2018",
    "endoscopy", "colonoscopy", "polyp", "kvasir", "cvc-clinicdb", "cvc-colondb", "etis", "bkai",
    "x-ray", "xray", "cxr", "chest x-ray", "chest xray", "radiograph", "jsrt", "montgomery", "shenzhen",
    "2d ultrasound", "ultrasound", "busi", "tn3k", "ddti", "echocardiography", "camus",
    "laparoscopy", "surgical", "endovis", "cholec",
}


def is_non_3d_photographic_score(
    dataset_name: Optional[str] = None,
    modality: Optional[str] = None,
    organ_or_task: Optional[str] = None,
) -> bool:
    """Returns True if the score clearly evaluates a 2D non-volumetric photographic/planar domain."""
    ds_low = (dataset_name or "").lower().strip()
    mod_low = (modality or "").lower().strip()
    org_low = (organ_or_task or "").lower().strip()
    combined = f"{ds_low} {mod_low} {org_low}"

    # Protect known 3D volumetric radiological modalities and benchmarks
    for kw in ("ct", "mri", "mr ", " mr", "pet", "3d", "cbct", "mra", "brats", "btcv", "amos", "flare", "kits", "lits", "acdc", "totalsegmentator", "synapse", "msd", "atlas", "verse", "autopet", "hecktor", "crossmoda", "isles", "toothfairy", "promise12", "chaos", "word", "segthor"):
        if re.search(r"\b" + re.escape(kw.strip()) + r"\b", combined):
            return False

    # Special rule for Ultrasound/Echocardiography: only non-3D if not explicitly 3D
    for photo_kw in NON_3D_PHOTOGRAPHIC_MODALITIES:
        if re.search(r"\b" + re.escape(photo_kw) + r"\b", combined):
            return True

    return False


