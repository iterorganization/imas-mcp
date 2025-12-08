"""
Generic documentation search models.

This module contains Pydantic models for documentation search functionality
across multiple libraries with version support.
"""

from typing import Any

from pydantic import BaseModel, Field


class DocsSearchRequest(BaseModel):
    """Request model for documentation search"""

    query: str = Field(..., description="Search query string")
    library: str | None = Field(
        default=None, description="Documentation library to search"
    )
    limit: int | None = Field(
        default=None, ge=1, le=20, description="Maximum number of results"
    )
    version: str | None = Field(default=None, description="Specific version to search")


class DocsSearchPathsResult(BaseModel):
    """Individual search result from documentation"""

    url: str = Field(..., description="URL of the documentation page")
    content: str = Field(..., description="Content snippet or summary")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    mime_type: str | None = Field(default=None, description="Content type")
    library: str = Field(..., description="Source library name")
    version: str = Field(..., description="Source library version")


class DocsSearchMetadata(BaseModel):
    """Metadata for documentation search results"""

    total_results: int = Field(..., ge=0, description="Total number of results found")
    search_time_ms: int | None = Field(
        default=None, ge=0, description="Search execution time in milliseconds"
    )
    library_version: str = Field(..., description="Version of library searched")
    library_name: str = Field(..., description="Name of library searched")


class DocsSearchResponse(BaseModel):
    """Response model for documentation search"""

    results: list[DocsSearchPathsResult] = Field(
        default_factory=list, description="Search results"
    )
    query: str = Field(..., description="Original search query")
    library: str | None = Field(default=None, description="Library that was searched")
    version: str | None = Field(default=None, description="Version that was searched")
    search_metadata: DocsSearchMetadata = Field(..., description="Search metadata")
    error: str | None = Field(
        default=None, description="Error message if search failed"
    )
    available_libraries: list[str] = Field(
        default_factory=list, description="All available libraries"
    )


class LibrariesListResponse(BaseModel):
    """Response model for listing available libraries"""

    libraries: list[str] = Field(
        default_factory=list, description="List of available library names"
    )
    count: int = Field(..., ge=0, description="Number of available libraries")
    success: bool = Field(
        default=True, description="Whether the operation was successful"
    )
    error: str | None = Field(
        default=None, description="Error message if operation failed"
    )


class LibraryVersionsResponse(BaseModel):
    """Response model for library version information"""

    library: str = Field(..., description="Library name")
    versions: list[str] = Field(default_factory=list, description="Available versions")
    latest: str | None = Field(default=None, description="Latest version")
    count: int = Field(..., ge=0, description="Number of available versions")
    success: bool = Field(
        default=True, description="Whether the operation was successful"
    )
    error: str | None = Field(
        default=None, description="Error message if operation failed"
    )


class ScrapeStatusResponse(BaseModel):
    """Response model for scraping job status"""

    job_id: str = Field(..., description="Job ID")
    status: dict[str, Any] = Field(..., description="Status information")
    success: bool = Field(
        default=True, description="Whether status was successfully retrieved"
    )
    error: str | None = Field(
        default=None, description="Error message if status check failed"
    )
