"""Simple local Qwen3-VL test."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from agenticpay.models.qwen3_vl import Qwen3VL


def test_qwen3vl_simple():
    """Simple test: single question-answer using local Qwen3-VL."""
    
    print("=" * 60)
    print("Qwen3-VL Simple Test - Single Q&A")
    print("=" * 60)
    
    model_path = os.path.join(
        project_root,
        "agenticpay",
        "models",
        "download_models",
        "Qwen3-VL-2B-Instruct",
    )

    if not os.path.exists(model_path):
        print(f"\n✗ Error: Model path not found: {model_path}")
        return
    
    print(f"\nLoading model from: {model_path}")
    
    try:
        model = Qwen3VL(
            model_path=model_path,
            device_map="auto",
            dtype="auto",
        )
        print("✓ Model loaded successfully")

        print("\nTesting single question-answer...")
        question = "你好，请介绍一下你自己。"
        print(f"Question: {question}")
        answer = model.generate(
            prompt=question,
            temperature=0.7,
            max_tokens=256,
        )
        print(f"Answer: {answer}")
        print("\n✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_qwen3vl_simple()
