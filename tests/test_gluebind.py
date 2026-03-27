"""Tests for GlueBind module"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gluebind import hello, check_gpu, get_gpu_info, __version__


def test_version():
    """Test version is defined."""
    assert __version__ == "0.1.0"


def test_hello():
    """Test hello function."""
    result = hello()
    assert result == "Hello from GlueBind!"
    assert isinstance(result, str)


def test_check_gpu():
    """Test GPU check returns boolean."""
    result = check_gpu()
    assert isinstance(result, bool)
    print(f"GPU available: {result}")


def test_get_gpu_info():
    """Test GPU info retrieval."""
    info = get_gpu_info()
    assert isinstance(info, dict)
    
    if check_gpu():
        assert "gpus" in info
        assert "count" in info
        assert info["count"] > 0
        assert len(info["gpus"]) == info["count"]
    else:
        assert info == {}


def test_main():
    """Test main function runs without error."""
    from gluebind import main
    try:
        main()
    except SystemExit as e:
        assert e.code == 0
