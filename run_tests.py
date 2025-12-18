"""Test runner script for running all tests and generating a report."""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


def run_all_tests(save_results=True, save_format="text"):
    """Run all tests and display results.
    
    Args:
        save_results: Whether to save test results to file
        save_format: Format to save results in ('text', 'json', or 'both')
    """
    
    print("=" * 80)
    print("AMBEDKARGPT - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Test files
    test_files = [
        "tests/test_chunking.py",
        "tests/test_retrieval.py",
        "tests/test_integration.py"
    ]
    
    # Check if test files exist
    for test_file in test_files:
        if not Path(test_file).exists():
            print(f"⚠️  Warning: {test_file} not found")
    
    print("Running test suite...")
    print("-" * 80)
    print()
    
    # Create results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Run pytest with all tests and capture output
    output_file = results_dir / f"test_output_{timestamp}.txt"
    json_file = results_dir / f"test_results_{timestamp}.json"
    
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "-ra",  # Show summary of all test outcomes
        f"--junit-xml={results_dir}/test_report_{timestamp}.xml",  # XML report
    ]
    
    # Capture output while streaming to console
    stdout_lines = []
    stderr_lines = []
    
    print("🧪 Tests in progress:")
    print("-" * 80)
    
    # Run pytest with streaming output
    process = subprocess.Popen(
        cmd,
        cwd=".",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Print output line by line as it comes
    for line in process.stdout:
        print(line, end="")
        stdout_lines.append(line)
    
    for line in process.stderr:
        print(line, end="")
        stderr_lines.append(line)
    
    # Wait for process to complete
    return_code = process.wait()
    
    stdout_text = "".join(stdout_lines)
    stderr_text = "".join(stderr_lines)
    
    # Save results if requested
    if save_results:
        if save_format in ["text", "both"]:
            # Save text output
            with open(output_file, "w", encoding="utf-8") as f:
                f.write("=" * 80 + "\n")
                f.write("AMBEDKARGPT - TEST RESULTS\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Return Code: {return_code}\n")
                f.write(f"Status: {'✓ PASSED' if return_code == 0 else '✗ FAILED'}\n\n")
                f.write("-" * 80 + "\n")
                f.write("OUTPUT\n")
                f.write("-" * 80 + "\n")
                f.write(stdout_text)
                if stderr_text:
                    f.write("\n" + "-" * 80 + "\n")
                    f.write("ERRORS\n")
                    f.write("-" * 80 + "\n")
                    f.write(stderr_text)
            
            print()
            print(f"✓ Test results saved to: {output_file}")
        
        if save_format in ["json", "both"]:
            # Parse results and save as JSON
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "return_code": return_code,
                "status": "PASSED" if return_code == 0 else "FAILED",
                "test_files": test_files,
                "stdout": stdout_text,
                "stderr": stderr_text
            }
            
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2)
            
            print(f"✓ JSON results saved to: {json_file}")
    
    print()
    print("=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)
    
    if save_results:
        print(f"\nResults saved to: {results_dir}/")
    
    return return_code


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run AmbedkarGPT test suite and optionally save results"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save test results to file (default: True)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save test results"
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "both"],
        default="both",
        help="Format to save results in (default: both)"
    )
    
    args = parser.parse_args()
    
    save_results = not args.no_save
    exit_code = run_all_tests(save_results=save_results, save_format=args.format)
    sys.exit(exit_code)
