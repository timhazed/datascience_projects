import hashlib
from src.utils.hash import get_chunk_id


class TestGetChunkId:
    """Tests for the get_chunk_id function."""

    def test_returns_sha256_hash(self):
        """Verify function returns SHA-256 hexdigest."""
        content = "test content"
        expected = hashlib.sha256(content.encode('utf-8')).hexdigest()
        assert get_chunk_id(content) == expected

    def test_deterministic_output(self):
        """Same input always produces same hash."""
        content = "Section 4.2: PTO policy"
        result1 = get_chunk_id(content)
        result2 = get_chunk_id(content)
        assert result1 == result2

    def test_different_content_different_hash(self):
        """Different inputs produce different hashes."""
        content1 = "First policy document"
        content2 = "Second policy document"
        assert get_chunk_id(content1) != get_chunk_id(content2)

    def test_empty_string(self):
        """Empty string produces valid hash."""
        result = get_chunk_id("")
        expected = hashlib.sha256("".encode('utf-8')).hexdigest()
        assert result == expected
        assert len(result) == 64  # SHA-256 hex length

    def test_unicode_content(self):
        """Unicode content is properly handled."""
        content = "Politique RH: conges payes"
        result = get_chunk_id(content)
        assert len(result) == 64

    def test_whitespace_sensitivity(self):
        """Whitespace differences produce different hashes."""
        content1 = "test content"
        content2 = "test  content"  # Extra space
        assert get_chunk_id(content1) != get_chunk_id(content2)

    def test_hash_length_is_64_chars(self):
        """SHA-256 hex digest is always 64 characters."""
        assert len(get_chunk_id("short")) == 64
        assert len(get_chunk_id("a" * 10000)) == 64
