"""Wiki configuration for facilities.

Loads wiki site configuration from facility YAML files.
Supports MediaWiki, Confluence, and generic site types.
"""

from dataclasses import dataclass


@dataclass
class WikiConfig:
    """Wiki configuration for a facility site.

    Supports multiple site types:
    - mediawiki: MediaWiki sites (SSH proxy or direct)
    - confluence: Atlassian Confluence (REST API)
    - twiki: TWiki sites (SSH proxy, live server)
    - twiki_static: Static TWiki HTML export (direct HTTP)
    - static_html: Generic static HTML site
    - generic: Generic HTML scraping
    """

    base_url: str
    portal_page: str
    facility_id: str
    site_type: str = (
        "mediawiki"  # mediawiki, confluence, twiki, twiki_static, static_html
    )
    auth_type: str = "none"  # none, tequila, session, basic
    access_method: str = "direct"  # direct, vpn (preferred network route)
    ssh_available: bool = False  # SSH access possible (for scp, tunneling)
    ssh_host: str | None = None  # SSH host for tunnel/proxy access
    credential_service: str | None = None  # keyring service name

    @classmethod
    def from_facility(cls, facility: str, site_index: int = 0) -> "WikiConfig":
        """Load wiki config from facility configuration.

        Args:
            facility: Facility identifier
            site_index: Index of wiki site in facility's wiki_sites list

        Returns:
            WikiConfig for the specified site
        """
        from imas_codex.discovery.base.facility import get_facility

        # Try to load from facility YAML first
        try:
            config = get_facility(facility)
            wiki_sites = config.get("wiki_sites", [])

            if wiki_sites and site_index < len(wiki_sites):
                site = wiki_sites[site_index]
                return cls(
                    base_url=site["url"],
                    portal_page=site.get("portal_page", ""),
                    facility_id=facility,
                    site_type=site.get("site_type", "mediawiki"),
                    auth_type=site.get("auth_type", "none"),
                    access_method=site.get("access_method", "direct"),
                    ssh_available=site.get("ssh_available", False),
                    ssh_host=site.get("ssh_host") or config.get("ssh_host"),
                    credential_service=site.get("credential_service"),
                )
        except Exception:
            pass  # Fall back to hardcoded defaults

        # Default configurations per facility (legacy fallback)
        configs = {
            "tcv": {
                "base_url": "https://spcwiki.epfl.ch/wiki",
                "portal_page": "Portal:TCV",
                "site_type": "mediawiki",
                "auth_type": "tequila",
                "access_method": "direct",
                "ssh_available": True,
                "credential_service": "tcv-wiki",
            },
            "iter": {
                "base_url": "https://confluence.iter.org",
                "portal_page": "IMP",
                "site_type": "confluence",
                "auth_type": "session",
                "access_method": "direct",
                "ssh_available": False,
                "credential_service": "iter-confluence",
            },
        }

        if facility not in configs:
            raise ValueError(
                f"Unknown facility: {facility}. Known: {list(configs.keys())}"
            )

        cfg = configs[facility]
        return cls(
            base_url=cfg["base_url"],
            portal_page=cfg["portal_page"],
            facility_id=facility,
            site_type=cfg.get("site_type", "mediawiki"),
            auth_type=cfg.get("auth_type", "none"),
            access_method=cfg.get("access_method", "direct"),
            ssh_available=cfg.get("ssh_available", False),
            ssh_host=cfg.get("ssh_host"),
            credential_service=cfg.get("credential_service"),
        )

    @classmethod
    def list_sites(cls, facility: str) -> list["WikiConfig"]:
        """List all wiki sites configured for a facility.

        Args:
            facility: Facility identifier

        Returns:
            List of WikiConfig for all configured sites
        """
        from imas_codex.discovery.base.facility import get_facility

        sites = []

        try:
            config = get_facility(facility)
            wiki_sites = config.get("wiki_sites", [])

            if wiki_sites:
                for i, _site in enumerate(wiki_sites):
                    sites.append(cls.from_facility(facility, site_index=i))
                return sites
        except Exception:
            pass

        # Fall back to single default site from hardcoded configs
        try:
            sites.append(cls.from_facility(facility))
        except ValueError:
            pass

        return sites

    @property
    def requires_auth(self) -> bool:
        """Check if this site requires authentication."""
        return self.auth_type not in ("none",)

    @property
    def requires_ssh(self) -> bool:
        """Check if SSH is available for this site."""
        return self.ssh_available


__all__ = ["WikiConfig"]
