"""
Standalone Checkpoint Verification Script for V7 Audio Deepfake System.

Calculates the SHA-256 cryptographic checksum of the target model checkpoint
and strictly validates it against the verified baseline hash.
Does NOT modify, replace, or alter the checkpoint or any research assets.
"""

import os
import sys
import hashlib

EXPECTED_SHA256 = "b356dacc24dfc1ff463b1f1c5f4488cf374330f6c2edae6c5dd5abe79b76397b"
DEFAULT_CHECKPOINT_REL = os.path.join("..", "v7", "best_lcnn2d_v7.pt")


def verify_checkpoint(checkpoint_path: str) -> bool:
    print("=" * 72)
    print("  V7 MODEL CHECKPOINT INTEGRITY VERIFICATION")
    print("=" * 72)

    abs_path = os.path.abspath(checkpoint_path)
    print(f"Target Checkpoint : {abs_path}")

    if not os.path.exists(abs_path):
        print(f"\n[ERROR] Checkpoint file not found: {abs_path}")
        print("Please ensure 'best_lcnn2d_v7.pt' is present in the expected directory.")
        return False

    file_size = os.path.getsize(abs_path)
    print(f"File Size         : {file_size:,} bytes")

    hasher = hashlib.sha256()
    try:
        with open(abs_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
    except Exception as e:
        print(f"\n[ERROR] Unable to read checkpoint file: {str(e)}")
        return False

    actual_sha = hasher.hexdigest().lower()
    print(f"Calculated SHA-256: {actual_sha}")
    print(f"Expected SHA-256  : {EXPECTED_SHA256}")

    if actual_sha == EXPECTED_SHA256:
        print("\n>>> VERIFICATION SUCCESS: Checkpoint integrity confirmed. <<<")
        print("=" * 72)
        return True
    else:
        print("\n>>> VERIFICATION FAILED: Checkpoint hash does not match baseline! <<<")
        print("Do not use an unverified or corrupted model checkpoint.")
        print("=" * 72)
        return False


def main() -> None:
    app_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(app_root, DEFAULT_CHECKPOINT_REL)

    success = verify_checkpoint(ckpt_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
