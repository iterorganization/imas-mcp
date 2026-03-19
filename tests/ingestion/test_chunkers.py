"""Tests for direct tree-sitter and text chunking."""

import pytest

from imas_codex.ingestion.chunkers import Chunk, chunk_code, chunk_text


class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_chunk_fields(self):
        c = Chunk(text="hello", start_line=0, end_line=5)
        assert c.text == "hello"
        assert c.start_line == 0
        assert c.end_line == 5


class TestChunkText:
    """Tests for text chunking with sliding window."""

    def test_single_chunk_small_text(self):
        text = "line1\nline2\nline3"
        chunks = chunk_text(text, chunk_size=1000)
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].start_line == 0

    def test_empty_text(self):
        chunks = chunk_text("")
        assert len(chunks) == 1
        assert chunks[0].text == ""

    def test_multiple_chunks(self):
        # Create text that will exceed chunk_size
        lines = [f"line {i} with some content to fill space" for i in range(100)]
        text = "\n".join(lines)
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1
        # All original text should be recoverable
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_chunk_overlap(self):
        lines = [f"line_{i}" for i in range(20)]
        text = "\n".join(lines)
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=20)
        # With overlap, later chunks should start before the previous chunk ended
        if len(chunks) > 1:
            # The second chunk should contain some content from the first
            first_lines = set(chunks[0].text.split("\n"))
            second_lines = set(chunks[1].text.split("\n"))
            assert first_lines & second_lines  # Some overlap

    def test_custom_separator(self):
        text = "part1;;part2;;part3"
        chunks = chunk_text(text, chunk_size=1000, separator=";;")
        assert len(chunks) == 1
        assert chunks[0].text == text

    def test_start_line_tracking(self):
        lines = [f"line_{i}" for i in range(50)]
        text = "\n".join(lines)
        chunks = chunk_text(text, chunk_size=60, chunk_overlap=0)
        assert chunks[0].start_line == 0
        # Each subsequent chunk should have a higher start_line
        for i in range(1, len(chunks)):
            assert chunks[i].start_line > chunks[i - 1].start_line


class TestChunkCode:
    """Tests for tree-sitter based code chunking."""

    def test_simple_python(self):
        code = """def hello():
    print("hello")

def world():
    print("world")
"""
        chunks = chunk_code(code, "python")
        assert len(chunks) >= 1
        # All functions should be in the chunks
        all_text = "\n".join(c.text for c in chunks)
        assert "hello" in all_text
        assert "world" in all_text

    def test_single_function(self):
        code = "def foo():\n    return 42\n"
        chunks = chunk_code(code, "python")
        assert len(chunks) == 1
        assert "foo" in chunks[0].text

    def test_large_code_splits(self):
        # Create code with many functions that exceeds max_chars
        functions = []
        for i in range(50):
            functions.append(f"def func_{i}():\n    x = {i}\n    return x\n")
        code = "\n".join(functions)
        chunks = chunk_code(code, "python", max_chars=200)
        assert len(chunks) > 1

    def test_fortran(self):
        code = """program test
    implicit none
    integer :: x
    x = 42
    print *, x
end program test
"""
        chunks = chunk_code(code, "fortran")
        assert len(chunks) >= 1
        assert "test" in chunks[0].text

    def test_unsupported_language_raises(self):
        with pytest.raises((ValueError, LookupError)):
            chunk_code("print('x')", "nonexistent_language_xyz")

    def test_chunk_has_line_info(self):
        code = """def a():
    pass

def b():
    pass
"""
        chunks = chunk_code(code, "python")
        for chunk in chunks:
            assert chunk.start_line >= 0
            assert chunk.end_line >= chunk.start_line

    def test_idl_single_procedure(self):
        """Parse a single IDL procedure with MDSplus access patterns."""
        code = """pro liuqe_params
; Sets up Liuqe parameters
on_error, 2
mds$open, 'tcv_shot', -1
nequil = mds$value('\\pcs::mgams.data:numeq')
rlia1 = mds$value('\\pcs::mgams.data:rlia1')
mds$close
end
"""
        chunks = chunk_code(code, "idl")
        assert len(chunks) >= 1
        all_text = "\n".join(c.text for c in chunks)
        assert "liuqe_params" in all_text
        assert "mds$value" in all_text

    def test_idl_multiple_routines(self):
        """Parse multiple IDL procedures and functions."""
        code = """function get_ip, shot
  mds$open, 'tcv_shot', shot
  ip = mds$value('\\results::i_p')
  mds$close
  return, ip
end

pro plot_equilibrium, shot
  compile_opt idl2
  psi = tdi('\\results::psi')
  contour, psi
end

pro main_analysis
  shots = [48952, 55608]
  for i = 0, n_elements(shots)-1 do begin
    ip = get_ip(shots[i])
    plot_equilibrium, shots[i]
  endfor
end
"""
        chunks = chunk_code(code, "idl")
        assert len(chunks) >= 1
        all_text = "\n".join(c.text for c in chunks)
        assert "get_ip" in all_text
        assert "plot_equilibrium" in all_text
        assert "main_analysis" in all_text

    def test_idl_trcf_mdsplus(self):
        """Parse real TCV IDL code with mds$open/mds$value patterns."""
        code = """pro fix_trcf_br, shot
mds$open, 'thoms', shot
if mds$value('getnci(".TRCF_001:CHANNEL_001","RLENGTH")') eq 24 then $
   ll = 0 $
else $
   ll = mds$value('size(.TRCF_001:CHANNEL_001)')
mds$close
mds$open, 'diagz', shot
mds$put, '\\TRCF_BREMS_IR_001:CHANNEL_001:ENDIDX', '$', ll-1
mds$close
print, shot
end
"""
        chunks = chunk_code(code, "idl")
        assert len(chunks) >= 1
        assert "fix_trcf_br" in chunks[0].text
