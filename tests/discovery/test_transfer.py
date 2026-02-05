"""Tests for base transfer module."""

import pytest

from imas_codex.discovery.base.transfer import (
    TransferResult,
    decode_url,
    detect_content_type,
)


class TestDetectContentType:
    """Test magic bytes detection."""

    def test_detect_pdf(self):
        header = b"%PDF-1.4 rest of content"
        assert detect_content_type(header) == "application/pdf"

    def test_detect_zip(self):
        header = b"PK\x03\x04 rest of zip"
        assert detect_content_type(header) == "application/zip"

    def test_detect_html_doctype(self):
        header = b"<!DOCTYPE html><html>"
        assert detect_content_type(header) == "text/html"

    def test_detect_html_tag(self):
        header = b"<html><head>"
        assert detect_content_type(header) == "text/html"

    def test_detect_jpeg(self):
        header = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        assert detect_content_type(header) == "image/jpeg"

    def test_detect_png(self):
        header = b"\x89PNG\r\n\x1a\n"
        assert detect_content_type(header) == "image/png"

    def test_detect_unknown(self):
        header = b"unknown format data"
        assert detect_content_type(header) is None


class TestDecodeUrl:
    """Test URL decoding."""

    def test_decode_encoded_path(self):
        url = "https://wiki.example.com/images/File%3ATest%20Doc.pdf"
        decoded = decode_url(url)
        assert decoded == "https://wiki.example.com/images/File:Test Doc.pdf"

    def test_decode_already_decoded(self):
        url = "https://wiki.example.com/images/test.pdf"
        decoded = decode_url(url)
        assert decoded == url


class TestTransferResult:
    """Test TransferResult validation."""

    def test_success_property(self):
        result = TransferResult(url="test", content=b"data")
        assert result.success is True

    def test_failure_with_error(self):
        result = TransferResult(url="test", error="failed")
        assert result.success is False

    def test_failure_no_content(self):
        result = TransferResult(url="test")
        assert result.success is False

    def test_validate_pdf_valid(self):
        result = TransferResult(
            url="test.pdf",
            content=b"%PDF-1.4\n some content here",
        )
        assert result.validate_expected_type("pdf") is True

    def test_validate_pdf_invalid_html(self):
        result = TransferResult(
            url="test.pdf",
            content=b"<!DOCTYPE html><html>Error page</html>",
        )
        assert result.validate_expected_type("pdf") is False

    def test_validate_docx_valid(self):
        # docx files are ZIP format
        result = TransferResult(
            url="test.docx",
            content=b"PK\x03\x04\x14\x00 rest of zip content",
        )
        assert result.validate_expected_type("docx") is True

    def test_validate_ipynb_valid(self):
        result = TransferResult(
            url="test.ipynb",
            content=b'{"cells": [], "metadata": {}}',
        )
        assert result.validate_expected_type("ipynb") is True
