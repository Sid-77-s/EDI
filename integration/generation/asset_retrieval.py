"""
integration/generation/asset_retrieval.py — Local 3D Asset Retrieval & License Enforcement.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from integration.generation.asset_catalogue import LocalAssetPackResolver, search_catalogue
from integration.generation.model3d import Model3D, load_glb


@dataclass
class LicensePolicy:
    allowed_licenses: Optional[Set[str]] = None
    allow_unknown_license: bool = False

    def is_permitted(self, license_name: Optional[str]) -> bool:
        if not license_name:
            return self.allow_unknown_license
        if self.allowed_licenses is None:
            return True
        return license_name.strip().casefold() in {lic.casefold() for lic in self.allowed_licenses}


@dataclass
class RetrievalResult:
    concept: str
    model: Optional[Model3D]
    metadata: Dict[str, Any]
    matched: bool
    rank: int = 99


class Local3DAssetRetriever:
    """Retrieves 3D models from local database/directory while enforcing license compliance."""

    def __init__(
        self,
        database_path: Optional[Union[str, Path]] = None,
        models_dir: Optional[Union[str, Path]] = None,
        allowed_licenses: Optional[Set[str]] = None,
        allow_unknown_license: bool = False,
        synonyms: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        self.database_path = Path(database_path) if database_path else None
        self.models_dir = Path(models_dir) if models_dir else Path("assets/models")
        self.policy = LicensePolicy(allowed_licenses, allow_unknown_license)
        self.synonyms = synonyms or {}
        self.pack_resolver = LocalAssetPackResolver(
            self.models_dir,
            allowed_licenses=allowed_licenses,
            allow_unknown_license=allow_unknown_license,
        )

    def resolve(self, concept: str) -> Optional[Tuple[Model3D, Dict[str, Any]]]:
        """Resolves a concept to a (Model3D, metadata) pair or None."""
        # 1. Try curated sidecar / folder search first
        resolved = self.pack_resolver.resolve(concept)
        if resolved is not None:
            model, meta = resolved
            meta["match_rank"] = 0
            return model, meta

        # 2. Check synonyms
        concept_clean = concept.strip().lower()
        for canon, syn_list in self.synonyms.items():
            if concept_clean in [s.lower() for s in syn_list]:
                resolved = self.pack_resolver.resolve(canon)
                if resolved is not None:
                    model, meta = resolved
                    meta["match_rank"] = 1
                    return model, meta

        # 3. Database catalog search if available
        if self.database_path and self.database_path.is_file():
            results = search_catalogue(self.database_path, query=concept, limit=5)
            for res in results:
                lic = res.get("license")
                if not self.policy.is_permitted(lic):
                    continue
                # Look for matching local GLB
                uid = res.get("uid", "")
                name = res.get("name", "")
                for candidate_name in (f"{uid}.glb", f"{name}.glb"):
                    candidate_file = self.models_dir / candidate_name
                    if candidate_file.is_file():
                        model = load_glb(candidate_file)
                        model.concept = concept
                        meta = dict(res)
                        meta["asset_source"] = str(candidate_file)
                        meta["match_rank"] = 0 if name.lower() == concept_clean else 2
                        return model, meta

        return None
