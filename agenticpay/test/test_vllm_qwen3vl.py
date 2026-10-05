"""Test script for local Qwen3-VL."""

import os
import sys
from pathlib import Path

# Add project path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from agenticpay.models.qwen3_vl import Qwen3VL


def test_qwen3vl():
    """Test local Qwen3-VL text generation."""
    
    print("=" * 60)
    print("Testing Qwen3-VL locally")
    print("=" * 60)
    
    model_path = os.path.join(
        project_root,
        "agenticpay",
        "models",
        "download_models",
        "Qwen3-VL-2B-Instruct",
    )
    
    print(f"\n1. Loading model from local path: {model_path}")
    print("   Loading local model...")

    try:
        model = Qwen3VL(
            model_path=model_path,
            device_map="auto",
            dtype="auto",
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    print("\n2. Testing text-only generation...")
    text_prompt = "你好，请介绍一下你自己。你是什么模型？"
    
    print(f"   Prompt: {text_prompt}")
    print("   Generating response...")
    
    try:
        response = model.generate(
            prompt=text_prompt,
            temperature=0.7,
            max_tokens=256,
            top_p=0.9,
        )
        print(f"   Response: {response}")
        print("✓ Text-only generation test completed")
    except Exception as e:
        print(f"✗ Error in text generation: {e}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_qwen3vl()

