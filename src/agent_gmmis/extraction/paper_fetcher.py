"""Automated paper retrieval and local caching for literature extraction."""

from pathlib import Path
import re
import ssl
from typing import Optional
import urllib.request
import certifi

from agent_gmmis.config import DEFAULT_DB_PATH
from agent_gmmis.data.db_manager import DatabaseManager


DEFAULT_CACHE_DIR = Path("data/cache_papers")


class PaperFetcher:
    """Retrieves preprint/open-access PDF manuscripts directly by arXiv ID, URL, or model name."""

    def __init__(self, cache_dir: Optional[Path] = None, db_manager: Optional[DatabaseManager] = None):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = db_manager or DatabaseManager(DEFAULT_DB_PATH)

    @staticmethod
    def extract_arxiv_id(text: str) -> Optional[str]:
        """Extracts canonical arXiv identifier from a DOI, URL, or plain string."""
        match = re.search(r"(?:arxiv(?:\.org/(?:abs|pdf)/|\.)|arXiv:)?([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def fetch_arxiv(self, arxiv_identifier: str, force_download: bool = False) -> Path:
        """
        Downloads or retrieves a cached arXiv paper PDF.

        Args:
            arxiv_identifier: arXiv ID (e.g. '2408.00874') or full link/DOI.
            force_download: If True, re-downloads even if cached.
        """
        arxiv_id = self.extract_arxiv_id(arxiv_identifier)
        if not arxiv_id:
            raise ValueError(f"Could not parse valid arXiv ID from: {arxiv_identifier}")

        # Strip optional version suffix (e.g. 'v1') to avoid arXiv 404s on versioned pdf endpoints
        base_id = re.sub(r"v[0-9]+$", "", arxiv_id)
        target_file = self.cache_dir / f"{base_id}.pdf"
        if target_file.exists() and not force_download and target_file.stat().st_size > 1000:
            return target_file

        pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"
        ctx = ssl.create_default_context(cafile=certifi.where())
        headers = {
            "User-Agent": "AgentForGMMIS/1.0 (Politecnico di Milano Medical AI Survey Agent; mailto:luca2.pezzoli@mail.polimi.it)",
            "Accept": "application/pdf",
        }
        req = urllib.request.Request(pdf_url, headers=headers)

        print(f"📥 Downloading manuscript from arXiv: {pdf_url}...")
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
                content = response.read()
                if len(content) < 1000 or not content.startswith(b"%PDF"):
                    raise ValueError(f"Received non-PDF response from {pdf_url} (size={len(content)} bytes)")
                with open(target_file, "wb") as f:
                    f.write(content)
            print(f"✅ Successfully cached {target_file.name} ({target_file.stat().st_size / (1024*1024):.2f} MB)")
            return target_file
        except Exception as e:
            if target_file.exists():
                target_file.unlink()
            raise RuntimeError(f"Failed to download arXiv paper {arxiv_id}: {e}")

    def fetch_for_model(self, model_id_or_name: str, force_download: bool = False) -> Optional[Path]:
        """
        Looks up a model's publication link in database.sqlite and automatically fetches its PDF.

        Args:
            model_id_or_name: Model ID (e.g. 'SAM-Med3D') or partial name.
        """
        model = self.db.get_model(model_id_or_name)
        if not model:
            # Try fuzzy search in models
            all_models = self.db.get_all_models()
            for m in all_models:
                if model_id_or_name.lower() in m.id.lower() or (m.title and model_id_or_name.lower() in m.title.lower()):
                    model = m
                    break

        if not model:
            raise ValueError(f"Model '{model_id_or_name}' not found in database.sqlite")

        link = model.first_publication_link or model.last_publication_link
        if not link:
            raise ValueError(f"Model '{model.id}' does not have a publication link recorded in database.sqlite")

        arxiv_id = self.extract_arxiv_id(link)
        if not arxiv_id:
            raise ValueError(
                f"Model '{model.id}' links to '{link}', which is not an arXiv preprint (direct publisher download requires institutional access)."
            )

        return self.fetch_arxiv(arxiv_id, force_download=force_download)
