#!/usr/bin/env python3
"""
Basic test script to verify the peer reviewer application structure.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all main modules can be imported."""
    try:
        print("Testing imports...")
        
        # Test core imports
        from peer_reviewer.core.pdf_processor import PDFProcessor
        print("✓ PDFProcessor imported successfully")
        
        from peer_reviewer.core.model_interface import ModelInterface, ModelConfig
        print("✓ ModelInterface imported successfully")
        
        from peer_reviewer.core.latex_generator import LaTeXGenerator
        print("✓ LaTeXGenerator imported successfully")
        
        from peer_reviewer.core.reviewer import PeerReviewer
        print("✓ PeerReviewer imported successfully")
        
        # Test utilities
        from peer_reviewer.utils.config import Config
        print("✓ Config imported successfully")
        
        from peer_reviewer.utils.helpers import validate_pdf_file
        print("✓ Helper functions imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_pdf_validation():
    """Test PDF validation functionality."""
    try:
        print("\nTesting PDF validation...")
        from peer_reviewer.utils.helpers import validate_pdf_file
        
        # Test with the existing PDF
        pdf_path = Path("papers/AI Agents vs. Agentic AI.pdf")
        
        if pdf_path.exists():
            result = validate_pdf_file(pdf_path)
            print(f"PDF validation result: {result}")
            
            if result["valid"]:
                print("✅ PDF validation successful!")
                return True
            else:
                print(f"❌ PDF validation failed: {result['error']}")
                return False
        else:
            print(f"❌ PDF file not found: {pdf_path}")
            return False
            
    except Exception as e:
        print(f"❌ PDF validation error: {e}")
        return False


def test_pdf_processing():
    """Test basic PDF processing functionality."""
    try:
        print("\nTesting PDF processing...")
        from peer_reviewer.core.pdf_processor import PDFProcessor
        
        pdf_path = Path("papers/AI Agents vs. Agentic AI.pdf")
        
        if not pdf_path.exists():
            print(f"❌ PDF file not found: {pdf_path}")
            return False
        
        processor = PDFProcessor(min_text_length=100)
        result = processor.extract_text(pdf_path)
        
        if result and result.get("success"):
            text_length = len(result["text"])
            print(f"✅ PDF processing successful! Extracted {text_length} characters")
            print(f"   Extraction method: {result.get('extraction_method', 'unknown')}")
            
            # Show first 200 characters
            preview = result["text"][:200] + "..." if len(result["text"]) > 200 else result["text"]
            print(f"   Text preview: {preview}")
            return True
        else:
            print(f"❌ PDF processing failed: {result}")
            return False
            
    except Exception as e:
        print(f"❌ PDF processing error: {e}")
        return False


def test_config():
    """Test configuration system."""
    try:
        print("\nTesting configuration...")
        from peer_reviewer.utils.config import Config
        
        config = Config()
        config_dict = config.get_config()
        
        print(f"✅ Configuration loaded successfully!")
        print(f"   Model name: {config_dict.get('model', {}).get('model_name', 'unknown')}")
        print(f"   Reasoning level: {config_dict.get('model', {}).get('reasoning_level', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False


def main():
    """Run all basic tests."""
    print("🚀 Starting Peer Reviewer Basic Tests\n")
    
    tests = [
        ("Import Tests", test_imports),
        ("PDF Validation", test_pdf_validation),
        ("PDF Processing", test_pdf_processing),
        ("Configuration", test_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"Running: {test_name}")
        print('='*50)
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"SUMMARY: {passed}/{total} tests passed")
    print('='*50)
    
    if passed == total:
        print("🎉 All tests passed! The application structure is working correctly.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()