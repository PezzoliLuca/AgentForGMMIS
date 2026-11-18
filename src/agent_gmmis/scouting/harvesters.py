"""Stage 1 Multi-Source Bibliographic Harvesters (arXiv, OpenAlex, EuropePMC/PubMed, Google Scholar)."""

import hashlib
import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import certifi

from agent_gmmis.config import DATA_DIR
from agent_gmmis.scouting.schemas import RawPaperCandidate, ScoutSource


DEFAULT_SCOUT_CACHE_DIR = DATA_DIR / "cache_scout"

POLITE_USER_AGENT = (
    "AgentForGMMIS/1.0 (Politecnico di Milano DEIB Medical AI Living Survey; "
    "mailto:luca2.pezzoli@mail.polimi.it)"
)

# Systematic PRISMA search queries covering both Tier 1 (Generalist/Foundation) and Tier 2 (3D Benchmark SOTA)
SYSTEMATIC_ARXIV_QUERIES: List[str] = [
    '(ti:SAM OR ti:SAM2 OR ti:MedSAM OR ti:"Segment Anything" OR ti:Foundation OR ti:Generalist OR ti:Universal OR ti:Promptable) AND (ti:Segmentation OR abs:segmentation) AND (abs:medical OR abs:CT OR abs:MRI OR abs:PET OR abs:3D OR abs:volumetric)',
    '(ti:segmentation OR abs:segmentation) AND (ti:universal OR abs:universal OR abs:"text-driven" OR abs:"text-prompted" OR abs:"vision-language" OR abs:"few-shot" OR abs:"semi-supervised") AND (abs:3D OR abs:volumetric OR abs:CT OR abs:MRI OR abs:PET)',
    'ti:"Medical Image Segmentation" AND (abs:SAM OR abs:"foundation" OR abs:generalist OR abs:universal OR abs:text OR abs:guidance OR abs:3D OR abs:volumetric)',
    '(ti:CT OR ti:MRI OR ti:PET OR ti:3D OR ti:Volumetric) AND (ti:Segmentation OR abs:segmentation) AND (abs:universal OR abs:promptable OR abs:"text-driven" OR abs:generalist OR abs:"foundation" OR abs:SAM OR abs:"self-supervised")',
    '(ti:segmentation OR abs:segmentation) AND (abs:TotalSegmentator OR abs:AMOS OR abs:BTCV OR abs:BraTS OR abs:FLARE OR abs:KiTS) AND (abs:3D OR abs:volumetric)',
]

SYSTEMATIC_KEYWORD_QUERIES: List[str] = [
    "Segment Anything medical image segmentation",
    "generalist foundation model medical segmentation MRI CT",
    "volumetric medical image segmentation SAM 3D",
    "universal promptable 3D medical segmentation TotalSegmentator AMOS BraTS",
    "text guidance vision-language medical image segmentation",
]

# Seed papers for Forward Citation Snowballing (DOI -> Short Label)
DEFAULT_SNOWBALL_SEED_DOIS: Dict[str, str] = {
    "10.1038/s41467-024-44824-z": "MedSAM (Nature Communications 2024)",
    "10.1148/ryai.230024": "TotalSegmentator (Radiology: AI 2023)",
    "10.48550/arXiv.2310.15161": "SAM-Med3D (arXiv 2023)",
    "10.48550/arXiv.2406.08418": "VISTA3D (CVPR/arXiv 2024)",
    "10.1016/j.inffus.2025.103709": "Moglia et al. GMMIS Survey (Information Fusion 2026)",
}


def _extract_arxiv_id(text: Optional[str]) -> Optional[str]:
    """
    Extracts canonical versionless arXiv ID (e.g. '2504.03600') from a URL, DOI, or string.
    Enforces valid YYMM format (MM in 01..12) and prevents false matches on publisher DOIs
    like '10.1016/j.bspc.2025.108086'.
    """
    if not text:
        return None
    s = text.strip()
    # If string is a URL or DOI, require explicit 'arxiv' prefix
    if "/" in s or "10." in s.lower():
        m = re.search(
            r"arxiv(?:\.org/(?:abs|pdf)/|\.|:)((?:1[4-9]|2[0-9])(?:0[1-9]|1[0-2])\.[0-9]{4,5})(?:v[0-9]+)?",
            s,
            re.IGNORECASE,
        )
        return m.group(1) if m else None

    # Bare arXiv ID or 'arXiv:YYMM.NNNNN'
    m_bare = re.match(
        r"^(?:arXiv:)?((?:1[4-9]|2[0-9])(?:0[1-9]|1[0-2])\.[0-9]{4,5})(?:v[0-9]+)?$",
        s,
        re.IGNORECASE,
    )
    return m_bare.group(1) if m_bare else None


def _infer_date_from_arxiv_id(arxiv_id: Optional[str]) -> Optional[str]:
    """Infers '20YY-MM-01' from a canonical YYMM.NNNNN arXiv identifier."""
    if not arxiv_id:
        return None
    m = re.match(r"^([12][0-9])(0[1-9]|1[0-2])\.[0-9]{4,5}$", arxiv_id)
    if m:
        return f"20{m.group(1)}-{m.group(2)}-01"
    return None


def _normalize_date(date_str: Optional[str]) -> Optional[str]:
    """Normalizes ISO timestamps or partial dates to YYYY-MM-DD."""
    if not date_str:
        return None
    clean = date_str.strip()[:10]
    if re.match(r"^\d{4}-\d{2}-\d{2}$", clean):
        return clean
    if re.match(r"^\d{4}-\d{2}$", clean):
        return f"{clean}-01"
    if re.match(r"^\d{4}$", clean):
        return f"{clean}-01-01"
    return None


def _is_within_date_window(
    pub_date: Optional[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> bool:
    """Checks if a normalized YYYY-MM-DD date falls within [start_date, end_date]."""
    if not pub_date:
        return True
    norm = _normalize_date(pub_date)
    if not norm:
        return True
    if start_date:
        s_norm = _normalize_date(start_date)
        if s_norm and norm < s_norm:
            return False
    if end_date:
        e_norm = _normalize_date(end_date)
        if e_norm and norm > e_norm:
            return False
    return True


class BaseHarvester:
    """Base class for bibliographic harvesters with disk caching and SSL verification."""

    source_type: ScoutSource = ScoutSource.ARXIV

    def __init__(self, cache_dir: Optional[Path] = None, use_cache: bool = True):
        self.cache_dir = cache_dir or DEFAULT_SCOUT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache
        self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    def _cache_path(self, prefix: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return self.cache_dir / f"{prefix}_{digest}.json"

    def _load_cached(self, prefix: str, key: str) -> Optional[List[RawPaperCandidate]]:
        if not self.use_cache:
            return None
        path = self._cache_path(prefix, key)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return [RawPaperCandidate.model_validate(item) for item in data]
            except Exception:
                return None
        return None

    def _save_cache(self, prefix: str, key: str, items: List[RawPaperCandidate]) -> None:
        if not self.use_cache:
            return
        path = self._cache_path(prefix, key)
        try:
            payload = [item.model_dump(mode="json") for item in items]
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def _http_get(self, url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 25) -> bytes:
        req_headers = {"User-Agent": POLITE_USER_AGENT}
        if headers:
            req_headers.update(headers)
        req = urllib.request.Request(url, headers=req_headers)
        with urllib.request.urlopen(req, context=self._ssl_ctx, timeout=timeout) as resp:
            return resp.read()


class ArxivHarvester(BaseHarvester):
    """Harvests preprints from the official arXiv Atom API (cs.CV, eess.IV, cs.AI, q-bio.QM)."""

    source_type = ScoutSource.ARXIV
    BASE_URL = "https://export.arxiv.org/api/query"

    def search(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 50,
    ) -> List[RawPaperCandidate]:
        cache_key = f"arxiv|{query}|{start_date}|{end_date}|{max_results}"
        cached = self._load_cached("arxiv", cache_key)
        if cached is not None:
            return cached

        # If query doesn't use arXiv field prefixes, wrap terms into all:(...) restricted to relevant categories
        formatted_q = query.strip()
        if not any(prefix in formatted_q for prefix in ("all:", "ti:", "abs:", "cat:")):
            tokens = [t.strip('"') for t in re.split(r"\s+", formatted_q) if len(t.strip('"')) >= 2]
            and_terms = " AND ".join(f'all:"{t}"' if "-" in t else f"all:{t}" for t in tokens[:7])
            formatted_q = f"({and_terms}) AND (cat:cs.CV OR cat:eess.IV OR cat:cs.AI OR cat:q-bio.QM)"

        if start_date or end_date:
            s_clean = (_normalize_date(start_date) or "2015-01-01").replace("-", "") + "0000"
            e_clean = (_normalize_date(end_date) or datetime.now().strftime("%Y-%m-%d")).replace("-", "") + "2359"
            formatted_q = f"({formatted_q}) AND submittedDate:[{s_clean} TO {e_clean}]"

        params = {
            "search_query": formatted_q,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"

        try:
            raw_xml = self._http_get(url)
            candidates = self.parse_atom_xml(
                raw_xml.decode("utf-8", errors="replace"),
                query_label=query,
                start_date=start_date,
                end_date=end_date,
            )
            self._save_cache("arxiv", cache_key, candidates)
            return candidates
        except Exception as exc:
            print(f"   ⚠️ [ArxivHarvester] Warning during query '{query[:50]}...': {exc}")
            return []

    @classmethod
    def parse_atom_xml(
        cls,
        xml_text: str,
        query_label: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[RawPaperCandidate]:
        """Parses arXiv Atom XML response into normalized RawPaperCandidate objects."""
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        results: List[RawPaperCandidate] = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return results

        for entry in root.findall("atom:entry", ns):
            id_el = entry.find("atom:id", ns)
            title_el = entry.find("atom:title", ns)
            summary_el = entry.find("atom:summary", ns)
            pub_el = entry.find("atom:published", ns)
            doi_el = entry.find("arxiv:doi", ns)

            if title_el is None or id_el is None:
                continue

            raw_id = (id_el.text or "").strip()
            arxiv_id = _extract_arxiv_id(raw_id)
            title = re.sub(r"\s+", " ", (title_el.text or "").strip())
            abstract = re.sub(r"\s+", " ", (summary_el.text or "").strip()) if summary_el is not None else ""
            pub_date = _normalize_date(pub_el.text if pub_el is not None else None)

            if not _is_within_date_window(pub_date, start_date, end_date):
                continue

            authors: List[str] = []
            for author_el in entry.findall("atom:author", ns):
                name_el = author_el.find("atom:name", ns)
                if name_el is not None and name_el.text:
                    authors.append(name_el.text.strip())

            doi = (doi_el.text or "").strip() if doi_el is not None and doi_el.text else None
            if not doi and arxiv_id:
                doi = f"10.48550/arXiv.{arxiv_id}"

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None
            abs_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else raw_id

            results.append(
                RawPaperCandidate(
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    publication_date=pub_date,
                    venue="arXiv",
                    arxiv_id=arxiv_id,
                    doi=doi,
                    url=abs_url,
                    pdf_url=pdf_url,
                    sources=[ScoutSource.ARXIV],
                    discovery_query=query_label,
                )
            )
        return results


class OpenAlexHarvester(BaseHarvester):
    """
    Harvests scholarly works from OpenAlex REST API (0.00 €, Polite Pool).
    Supports both keyword/date search and Forward Citation Snowballing.
    """

    source_type = ScoutSource.OPENALEX
    BASE_URL = "https://api.openalex.org/works"
    MAILTO = "luca2.pezzoli@mail.polimi.it"

    @staticmethod
    def reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
        """Reconstructs plain-text abstract from OpenAlex's `abstract_inverted_index`."""
        if not inverted_index or not isinstance(inverted_index, dict):
            return ""
        position_map: Dict[int, str] = {}
        for word, positions in inverted_index.items():
            if isinstance(positions, list):
                for pos in positions:
                    if isinstance(pos, int):
                        position_map[pos] = word
        if not position_map:
            return ""
        ordered_words = [position_map[idx] for idx in sorted(position_map.keys())]
        return " ".join(ordered_words)

    def search(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 50,
    ) -> List[RawPaperCandidate]:
        cache_key = f"openalex|{query}|{start_date}|{end_date}|{max_results}"
        cached = self._load_cached("openalex", cache_key)
        if cached is not None:
            return cached

        filters = ["has_abstract:true"]
        if start_date:
            filters.append(f"from_publication_date:{_normalize_date(start_date)}")
        if end_date:
            filters.append(f"to_publication_date:{_normalize_date(end_date)}")

        params = {
            "search": query,
            "filter": ",".join(filters),
            "per-page": min(max_results, 100),
            "sort": "publication_date:desc",
            "mailto": self.MAILTO,
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        try:
            raw_bytes = self._http_get(url)
            payload = json.loads(raw_bytes.decode("utf-8"))
            candidates = self.parse_works_json(
                payload,
                query_label=query,
                source_tag=ScoutSource.OPENALEX,
                start_date=start_date,
                end_date=end_date,
            )
            self._save_cache("openalex", cache_key, candidates)
            return candidates
        except Exception as exc:
            print(f"   ⚠️ [OpenAlexHarvester] Warning during query '{query[:50]}...': {exc}")
            return []

    def harvest_citations(
        self,
        seed_dois: Optional[Dict[str, str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_per_seed: int = 30,
    ) -> List[RawPaperCandidate]:
        """
        Forward Citation Snowballing: finds newly published papers citing foundational
        3D medical segmentation models (e.g., MedSAM, SAM-Med3D, TotalSegmentator, Moglia et al.).
        """
        seeds = seed_dois or DEFAULT_SNOWBALL_SEED_DOIS
        all_citing: List[RawPaperCandidate] = []

        for doi, label in seeds.items():
            cache_key = f"snowball|{doi}|{start_date}|{end_date}|{max_per_seed}"
            cached = self._load_cached("snowball", cache_key)
            if cached is not None:
                all_citing.extend(cached)
                continue

            try:
                # 1. Resolve OpenAlex Work ID from DOI
                work_url = f"{self.BASE_URL}/https://doi.org/{urllib.parse.quote(doi)}?mailto={self.MAILTO}"
                work_bytes = self._http_get(work_url)
                work_data = json.loads(work_bytes.decode("utf-8"))
                openalex_id = work_data.get("id", "").split("/")[-1]
                if not openalex_id:
                    continue

                # 2. Fetch works citing this seed within date window AND mentioning segmentation
                filters = [f"cites:{openalex_id}", "has_abstract:true"]
                if start_date:
                    filters.append(f"from_publication_date:{_normalize_date(start_date)}")
                if end_date:
                    filters.append(f"to_publication_date:{_normalize_date(end_date)}")

                params = {
                    "search": "3D medical image segmentation volumetric CT MRI",
                    "filter": ",".join(filters),
                    "per-page": min(max_per_seed, 100),
                    "sort": "publication_date:desc",
                    "mailto": self.MAILTO,
                }
                cite_url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
                cite_bytes = self._http_get(cite_url)
                cite_payload = json.loads(cite_bytes.decode("utf-8"))
                citing_works = self.parse_works_json(
                    cite_payload,
                    query_label=f"Cites: {label}",
                    source_tag=ScoutSource.CITATION_SNOWBALL,
                    start_date=start_date,
                    end_date=end_date,
                )
                self._save_cache("snowball", cache_key, citing_works)
                all_citing.extend(citing_works)
            except Exception as exc:
                print(f"   ⚠️ [OpenAlex Snowball] Could not fetch citations for {label} ({doi}): {exc}")

        return all_citing

    @classmethod
    def parse_works_json(
        cls,
        payload: Dict[str, Any],
        query_label: str = "",
        source_tag: ScoutSource = ScoutSource.OPENALEX,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[RawPaperCandidate]:
        results: List[RawPaperCandidate] = []
        for item in payload.get("results", []):
            title = (item.get("title") or item.get("display_name") or "").strip()
            if not title:
                continue
            pub_date = _normalize_date(item.get("publication_date"))
            if not _is_within_date_window(pub_date, start_date, end_date):
                continue

            abstract = cls.reconstruct_abstract(item.get("abstract_inverted_index"))
            authors: List[str] = []
            for auth in item.get("authorships", []):
                author_obj = auth.get("author") or {}
                if author_obj.get("display_name"):
                    authors.append(author_obj["display_name"])

            raw_doi = item.get("doi") or ""
            doi = raw_doi.replace("https://doi.org/", "").strip() if raw_doi else None

            ids_dict = item.get("ids") or {}
            pmid_raw = ids_dict.get("pmid") or ""
            pmid = pmid_raw.replace("https://pubmed.ncbi.nlm.nih.gov/", "").strip() if pmid_raw else None

            # Check locations for arXiv ID and PDF URL
            arxiv_id = _extract_arxiv_id(doi)
            pdf_url: Optional[str] = None
            venue: Optional[str] = None

            primary_loc = item.get("primary_location") or {}
            source_info = primary_loc.get("source") or {}
            venue = source_info.get("display_name")
            if primary_loc.get("pdf_url"):
                pdf_url = primary_loc["pdf_url"]

            oa_info = item.get("open_access") or {}
            if not pdf_url and oa_info.get("oa_url"):
                oa_url = oa_info["oa_url"]
                if oa_url.endswith(".pdf") or "arxiv.org" in oa_url:
                    pdf_url = oa_url

            for loc in item.get("locations", []):
                landing = loc.get("landing_page_url") or ""
                loc_pdf = loc.get("pdf_url") or ""
                if not arxiv_id:
                    arxiv_id = _extract_arxiv_id(landing) or _extract_arxiv_id(loc_pdf)
                if not pdf_url and loc_pdf:
                    pdf_url = loc_pdf

            if arxiv_id and not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            if arxiv_id and not venue:
                venue = "arXiv"

            landing_url = primary_loc.get("landing_page_url") or (f"https://doi.org/{doi}" if doi else item.get("id"))
            cited_by = int(item.get("cited_by_count") or 0)

            results.append(
                RawPaperCandidate(
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    publication_date=pub_date,
                    venue=venue,
                    arxiv_id=arxiv_id,
                    doi=doi,
                    pmid=pmid,
                    url=landing_url,
                    pdf_url=pdf_url,
                    cited_by_count=cited_by,
                    sources=[source_tag],
                    discovery_query=query_label,
                )
            )
        return results


class EuropePMCHarvester(BaseHarvester):
    """Harvests biomedical and engineering papers from EuropePMC / PubMed REST API."""

    source_type = ScoutSource.EUROPE_PMC
    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def search(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 40,
    ) -> List[RawPaperCandidate]:
        cache_key = f"epmc|{query}|{start_date}|{end_date}|{max_results}"
        cached = self._load_cached("epmc", cache_key)
        if cached is not None:
            return cached

        q_parts = [f"({query})"]
        if start_date or end_date:
            s_str = _normalize_date(start_date) or "2015-01-01"
            e_str = _normalize_date(end_date) or datetime.now().strftime("%Y-%m-%d")
            q_parts.append(f"FIRST_PDATE:[{s_str} TO {e_str}]")

        params = {
            "query": " AND ".join(q_parts),
            "resultType": "core",
            "format": "json",
            "pageSize": min(max_results, 100),
        }
        url = f"{self.BASE_URL}?{urllib.parse.urlencode(params)}"
        try:
            raw_bytes = self._http_get(url)
            payload = json.loads(raw_bytes.decode("utf-8"))
            candidates = self.parse_epmc_json(
                payload,
                query_label=query,
                start_date=start_date,
                end_date=end_date,
            )
            self._save_cache("epmc", cache_key, candidates)
            return candidates
        except Exception as exc:
            print(f"   ⚠️ [EuropePMCHarvester] Warning during query '{query[:50]}...': {exc}")
            return []

    @classmethod
    def parse_epmc_json(
        cls,
        payload: Dict[str, Any],
        query_label: str = "",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[RawPaperCandidate]:
        results: List[RawPaperCandidate] = []
        result_list = (payload.get("resultList") or {}).get("result", [])
        for item in result_list:
            title = re.sub(r"\s+", " ", (item.get("title") or "").strip()).rstrip(".")
            if not title:
                continue
            pub_date = _normalize_date(item.get("firstPublicationDate") or item.get("pubYear"))
            if not _is_within_date_window(pub_date, start_date, end_date):
                continue

            abstract = re.sub(r"<[^>]+>", " ", (item.get("abstractText") or "").strip())
            abstract = re.sub(r"\s+", " ", abstract)

            author_str = item.get("authorString") or ""
            authors = [a.strip() for a in author_str.split(",") if a.strip()]

            doi = (item.get("doi") or "").strip() or None
            pmid = (item.get("pmid") or "").strip() or None
            venue = (item.get("journalTitle") or item.get("bookOrReportDetails", {}).get("publisher") or "").strip() or None
            arxiv_id = _extract_arxiv_id(doi)

            pdf_url: Optional[str] = None
            landing_url: Optional[str] = f"https://doi.org/{doi}" if doi else None
            full_text_urls = (item.get("fullTextUrlList") or {}).get("fullTextUrl", [])
            for ft in full_text_urls:
                style = (ft.get("documentStyle") or "").lower()
                ft_url = ft.get("url") or ""
                if style == "pdf" and ft_url:
                    pdf_url = ft_url
                    break
                if not landing_url and ft_url:
                    landing_url = ft_url

            if arxiv_id and not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            cited_by = int(item.get("citedByCount") or 0)
            results.append(
                RawPaperCandidate(
                    title=title,
                    abstract=abstract,
                    authors=authors,
                    publication_date=pub_date,
                    venue=venue,
                    arxiv_id=arxiv_id,
                    doi=doi,
                    pmid=pmid,
                    url=landing_url,
                    pdf_url=pdf_url,
                    cited_by_count=cited_by,
                    sources=[ScoutSource.EUROPE_PMC],
                    discovery_query=query_label,
                )
            )
        return results


class GoogleScholarHarvester(BaseHarvester):
    """
    Google Scholar Harvester aligned with Moglia et al. protocol.
    Supports:
    1. Optional `SERPAPI_KEY` (if configured in environment) for structured JSON results.
    2. Direct polite HTML parsing with disk caching and an automatic Anti-CAPTCHA Circuit Breaker:
       if Google Scholar blocks or rate-limits the IP (HTTP 429/403/CAPTCHA), the circuit breaker
       trips gracefully so the pipeline never crashes or hangs.
    """

    source_type = ScoutSource.GOOGLE_SCHOLAR

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        use_cache: bool = True,
        serpapi_key: Optional[str] = None,
    ):
        super().__init__(cache_dir=cache_dir, use_cache=use_cache)
        self.serpapi_key = serpapi_key or os.getenv("SERPAPI_KEY")
        self.circuit_breaker_tripped: bool = False

    def search(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_results: int = 20,
    ) -> List[RawPaperCandidate]:
        cache_key = f"gscholar|{query}|{start_date}|{end_date}|{max_results}"
        cached = self._load_cached("gscholar", cache_key)
        if cached is not None:
            return cached

        if self.circuit_breaker_tripped:
            return []

        year_low = int(start_date[:4]) if start_date and len(start_date) >= 4 else None
        year_high = int(end_date[:4]) if end_date and len(end_date) >= 4 else None

        if self.serpapi_key:
            try:
                params: Dict[str, Any] = {
                    "engine": "google_scholar",
                    "q": query,
                    "num": min(max_results, 20),
                    "api_key": self.serpapi_key,
                }
                if year_low:
                    params["as_ylo"] = year_low
                if year_high:
                    params["as_yhi"] = year_high
                url = f"https://serpapi.com/search.json?{urllib.parse.urlencode(params)}"
                raw_bytes = self._http_get(url)
                payload = json.loads(raw_bytes.decode("utf-8"))
                candidates = self.parse_serpapi_json(payload, query_label=query)
                self._save_cache("gscholar", cache_key, candidates)
                return candidates
            except Exception as exc:
                print(f"   ⚠️ [GoogleScholarHarvester] SerpAPI warning: {exc}")
                return []

        # Direct polite request with automatic circuit breaker
        try:
            params_html: Dict[str, Any] = {"q": query, "hl": "en"}
            if year_low:
                params_html["as_ylo"] = year_low
            if year_high:
                params_html["as_yhi"] = year_high
            url = f"https://scholar.google.com/scholar?{urllib.parse.urlencode(params_html)}"
            raw_html = self._http_get(url, timeout=12).decode("utf-8", errors="replace")
            if "gs_captcha" in raw_html or "unusual traffic" in raw_html.lower():
                self.circuit_breaker_tripped = True
                print(
                    "   ℹ️ [GoogleScholarHarvester] Anti-CAPTCHA circuit breaker activated "
                    "(falling back seamlessly to OpenAlex + arXiv + EuropePMC)."
                )
                return []
            candidates = self.parse_scholar_html(
                raw_html,
                query_label=query,
                default_year=year_low,
                start_date=start_date,
                end_date=end_date,
            )
            self._save_cache("gscholar", cache_key, candidates)
            return candidates
        except Exception as exc:
            self.circuit_breaker_tripped = True
            print(
                f"   ℹ️ [GoogleScholarHarvester] Circuit breaker activated ({exc}); "
                "continuing with OpenAlex + arXiv + EuropePMC."
            )
            return []

    @classmethod
    def parse_serpapi_json(cls, payload: Dict[str, Any], query_label: str = "") -> List[RawPaperCandidate]:
        results: List[RawPaperCandidate] = []
        for item in payload.get("organic_results", []):
            title = (item.get("title") or "").strip()
            if not title:
                continue
            snippet = (item.get("snippet") or "").strip()
            link = item.get("link") or ""
            pub_info = (item.get("publication_info") or {}).get("summary") or ""
            year_match = re.search(r"\b(20[12]\d)\b", pub_info)
            pub_date = f"{year_match.group(1)}-01-01" if year_match else None
            arxiv_id = _extract_arxiv_id(link)

            pdf_url: Optional[str] = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else None
            for res in item.get("resources", []):
                if (res.get("file_format") or "").upper() == "PDF" and res.get("link"):
                    pdf_url = res["link"]
                    break

            cited_by = int(((item.get("inline_links") or {}).get("cited_by") or {}).get("total") or 0)
            results.append(
                RawPaperCandidate(
                    title=title,
                    abstract=snippet,
                    publication_date=pub_date,
                    venue="Google Scholar",
                    arxiv_id=arxiv_id,
                    url=link or None,
                    pdf_url=pdf_url,
                    cited_by_count=cited_by,
                    sources=[ScoutSource.GOOGLE_SCHOLAR],
                    discovery_query=query_label,
                )
            )
        return results

    @classmethod
    def parse_scholar_html(
        cls,
        html_text: str,
        query_label: str = "",
        default_year: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[RawPaperCandidate]:
        """Lightweight parser for standard Google Scholar result blocks (`gs_r gs_or gs_scl`)."""
        results: List[RawPaperCandidate] = []
        blocks = re.split(r'<div class="gs_r gs_or gs_scl"', html_text)[1:]
        for block in blocks:
            title_m = re.search(r'<h3 class="gs_rt"[^>]*>.*?(?:<a[^>]+href="([^"]+)"[^>]*>)?(.*?)(?:</a>)?</h3>', block, re.DOTALL)
            if not title_m:
                continue
            link = (title_m.group(1) or "").strip()
            title = re.sub(r"<[^>]+>", "", title_m.group(2) or "").strip()
            title = re.sub(r"^\[(?:PDF|HTML|BOOK|CITATION)\]\s*", "", title, flags=re.IGNORECASE).strip()
            if not title:
                continue

            snippet_m = re.search(r'<div class="gs_rs"[^>]*>(.*?)</div>', block, re.DOTALL)
            snippet = re.sub(r"<[^>]+>", " ", snippet_m.group(1) if snippet_m else "").strip()
            snippet = re.sub(r"\s+", " ", snippet)

            meta_m = re.search(r'<div class="gs_a"[^>]*>(.*?)</div>', block, re.DOTALL)
            meta_str = re.sub(r"<[^>]+>", " ", meta_m.group(1) if meta_m else "").strip()
            year_m = re.search(r"\b(20[12]\d)\b", meta_str)

            pdf_m = re.search(r'<div class="gs_or_ggsm"[^>]*>.*?<a[^>]+href="([^"]+)"', block, re.DOTALL)
            pdf_url = pdf_m.group(1).strip() if pdf_m else None
            arxiv_id = _extract_arxiv_id(link) or _extract_arxiv_id(pdf_url)
            if arxiv_id and not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            inferred_arxiv_date = _infer_date_from_arxiv_id(arxiv_id)
            pub_date = inferred_arxiv_date or (
                f"{year_m.group(1)}-01-01" if year_m else (f"{default_year}-01-01" if default_year else None)
            )
            if inferred_arxiv_date and not _is_within_date_window(pub_date, start_date, end_date):
                continue

            cite_m = re.search(r"Cited by (\d+)", block)
            cited_by = int(cite_m.group(1)) if cite_m else 0

            results.append(
                RawPaperCandidate(
                    title=title,
                    abstract=snippet,
                    publication_date=pub_date,
                    venue="Google Scholar",
                    arxiv_id=arxiv_id,
                    url=link or None,
                    pdf_url=pdf_url,
                    cited_by_count=cited_by,
                    sources=[ScoutSource.GOOGLE_SCHOLAR],
                    discovery_query=query_label,
                )
            )
        return results
