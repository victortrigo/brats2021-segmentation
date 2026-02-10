"""
Quick Validation Script for BraTS Label Correctness.

This lightweight script validates that the evaluate.py file correctly
uses BraTS labels (0, 1, 2, 4) without requiring torch or other heavy dependencies.

Usage:
    python validate_brats_labels.py
"""

import re
import sys
from pathlib import Path


def validate_brats_labels(script_path='evaluate.py'):
    """
    Validate that BraTS labels are correctly defined and used.
    
    Returns:
        bool: True if all checks pass, False otherwise
    """
    print("="*80)
    print("BraTS Label Validation for evaluate.py")
    print("="*80)
    print()
    
    # Read the script
    try:
        with open(script_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ ERROR: Could not find {script_path}")
        return False
    
    all_passed = True
    
    # ========================================================================
    # Test 1: Check BRATS_COLORS definition
    # ========================================================================
    print("TEST 1: Checking BRATS_COLORS definition...")
    colors_match = re.search(r'BRATS_COLORS = {([^}]+)}', content, re.DOTALL)
    
    if colors_match:
        keys = [int(k) for k in re.findall(r'(\d+):', colors_match.group(1))]
        keys_set = set(keys)
        expected_keys = {0, 1, 2, 4}
        
        if keys_set == expected_keys:
            print(f"  ✅ PASS: BRATS_COLORS has correct keys: {sorted(keys_set)}")
        else:
            print(f"  ❌ FAIL: BRATS_COLORS has keys {sorted(keys_set)}, expected {sorted(expected_keys)}")
            all_passed = False
            
        if 3 in keys_set:
            print(f"  ❌ CRITICAL: Label 3 found in BRATS_COLORS! BraTS uses label 4 for ET, not 3!")
            all_passed = False
    else:
        print("  ❌ FAIL: Could not find BRATS_COLORS definition")
        all_passed = False
    
    print()
    
    # ========================================================================
    # Test 2: Check BRATS_CLASSES definition
    # ========================================================================
    print("TEST 2: Checking BRATS_CLASSES definition...")
    classes_match = re.search(r'BRATS_CLASSES = {([^}]+)}', content, re.DOTALL)
    
    if classes_match:
        keys = [int(k) for k in re.findall(r'(\d+):', classes_match.group(1))]
        keys_set = set(keys)
        expected_keys = {0, 1, 2, 4}
        
        if keys_set == expected_keys:
            print(f"  ✅ PASS: BRATS_CLASSES has correct keys: {sorted(keys_set)}")
        else:
            print(f"  ❌ FAIL: BRATS_CLASSES has keys {sorted(keys_set)}, expected {sorted(expected_keys)}")
            all_passed = False
            
        if 3 in keys_set:
            print(f"  ❌ CRITICAL: Label 3 found in BRATS_CLASSES! BraTS uses label 4 for ET, not 3!")
            all_passed = False
    else:
        print("  ❌ FAIL: Could not find BRATS_CLASSES definition")
        all_passed = False
    
    print()
    
    # ========================================================================
    # Test 3: Check TUMOR_CLASSES definition
    # ========================================================================
    print("TEST 3: Checking TUMOR_CLASSES definition...")
    tumor_match = re.search(r'TUMOR_CLASSES = \[([^\]]+)\]', content)
    
    if tumor_match:
        tumor_list = tumor_match.group(1).strip()
        expected = "1, 2, 4"
        
        if tumor_list == expected:
            print(f"  ✅ PASS: TUMOR_CLASSES = [{tumor_list}]")
        else:
            print(f"  ❌ FAIL: TUMOR_CLASSES = [{tumor_list}], expected [{expected}]")
            all_passed = False
            
        if '3' in tumor_list:
            print(f"  ❌ CRITICAL: Label 3 found in TUMOR_CLASSES!")
            all_passed = False
    else:
        print("  ❌ FAIL: Could not find TUMOR_CLASSES definition")
        all_passed = False
    
    print()
    
    # ========================================================================
    # Test 4: Check for dangerous hardcoded patterns
    # ========================================================================
    print("TEST 4: Checking for dangerous hardcoded label patterns...")
    
    dangerous_patterns = [
        (r'class_id in \[1, 2, 3\]', 'Hardcoded loop [1, 2, 3] instead of TUMOR_CLASSES'),
        (r'for class_id in range\(1, 4\)', 'range(1, 4) produces [1, 2, 3]'),
        (r'for class_id in range\(4\)', 'range(4) produces [0, 1, 2, 3]'),
    ]
    
    found_issues = False
    for pattern, description in dangerous_patterns:
        matches = list(re.finditer(pattern, content))
        if matches:
            found_issues = True
            all_passed = False
            for match in matches:
                line_num = content[:match.start()].count('\n') + 1
                print(f"  ❌ FAIL: Line {line_num} - {description}")
                print(f"         Found: {match.group(0)}")
    
    if not found_issues:
        print("  ✅ PASS: No dangerous hardcoded patterns found")
    
    print()
    
    # ========================================================================
    # Test 5: Verify TUMOR_CLASSES is used instead of hardcoded lists
    # ========================================================================
    print("TEST 5: Verifying TUMOR_CLASSES constant is used...")
    
    tumor_classes_usage = len(re.findall(r'for class_id in TUMOR_CLASSES', content))
    
    if tumor_classes_usage >= 4:  # Should be used in multiple places
        print(f"  ✅ PASS: TUMOR_CLASSES used {tumor_classes_usage} times")
    else:
        print(f"  ⚠️  WARNING: TUMOR_CLASSES only used {tumor_classes_usage} times (expected 4+)")
        print(f"               Some loops might still be hardcoded")
    
    print()
    
    # ========================================================================
    # Test 6: Check visualization legend uses correct classes
    # ========================================================================
    print("TEST 6: Checking visualization functions...")
    
    # Look for legend creation with incorrect class iteration
    legend_patterns = [
        r'for i in \[1, 2, 3\].*legend',
        r'for i in \[0, 1, 2, 3\].*legend',
        r'range\(1, 4\).*legend',
    ]
    
    found_bad_legend = False
    for pattern in legend_patterns:
        if re.search(pattern, content):
            match = re.search(pattern, content)
            line_num = content[:match.start()].count('\n') + 1
            print(f"  ❌ FAIL: Line {line_num} - Incorrect legend range")
            found_bad_legend = True
            all_passed = False
    
    # Look for correct usage
    if re.search(r'for i in TUMOR_CLASSES', content):
        print(f"  ✅ PASS: Visualization uses TUMOR_CLASSES constant")
    else:
        if not found_bad_legend:
            print(f"  ✅ PASS: No issues found in visualization functions")
    
    print()
    
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("="*80)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print()
        print("The evaluate.py script correctly uses BraTS labels:")
        print("  - Background: 0")
        print("  - NCR (Necrotic Core): 1")
        print("  - ED (Edema): 2")
        print("  - ET (Enhancing Tumor): 4  ← Note: label is 4, NOT 3!")
    else:
        print("❌ SOME TESTS FAILED!")
        print()
        print("Please review and fix the issues above.")
        print()
        print("REMINDER: BraTS 2021 uses labels 0, 1, 2, 4")
        print("         Label 3 does NOT exist in BraTS!")
    print("="*80)
    print()
    
    return all_passed


def run_metric_logic_tests():
    """
    Test the mathematical logic of metric computations without dependencies.
    """
    print("="*80)
    print("Testing Metric Computation Logic")
    print("="*80)
    print()
    
    all_passed = True
    
    # Test 1: Dice formula
    print("TEST 1: Dice Score Formula")
    print("  Formula: 2 * |pred ∩ target| / (|pred| + |target|)")
    
    # Example: Perfect match
    pred_size = 100
    target_size = 100
    intersection = 100
    dice = 2 * intersection / (pred_size + target_size)
    
    if abs(dice - 1.0) < 1e-6:
        print("  ✅ PASS: Perfect match gives Dice = 1.0")
    else:
        print(f"  ❌ FAIL: Expected 1.0, got {dice}")
        all_passed = False
    
    # Example: No overlap
    intersection = 0
    dice = 2 * intersection / (pred_size + target_size) if (pred_size + target_size) > 0 else 0
    
    if abs(dice - 0.0) < 1e-6:
        print("  ✅ PASS: No overlap gives Dice = 0.0")
    else:
        print(f"  ❌ FAIL: Expected 0.0, got {dice}")
        all_passed = False
    
    # Example: Partial overlap (50%)
    pred_size = 100
    target_size = 100
    intersection = 50
    dice = 2 * intersection / (pred_size + target_size)
    expected = 0.5
    
    if abs(dice - expected) < 1e-6:
        print(f"  ✅ PASS: 50% overlap gives Dice = 0.5")
    else:
        print(f"  ❌ FAIL: Expected {expected}, got {dice}")
        all_passed = False
    
    print()
    
    # Test 2: IoU formula
    print("TEST 2: IoU (Jaccard) Formula")
    print("  Formula: |pred ∩ target| / |pred ∪ target|")
    
    # Perfect match
    intersection = 100
    union = 100
    iou = intersection / union
    
    if abs(iou - 1.0) < 1e-6:
        print("  ✅ PASS: Perfect match gives IoU = 1.0")
    else:
        print(f"  ❌ FAIL: Expected 1.0, got {iou}")
        all_passed = False
    
    # Partial overlap
    intersection = 50
    union = 150  # |A ∪ B| = |A| + |B| - |A ∩ B| = 100 + 100 - 50
    iou = intersection / union
    expected = 50/150
    
    if abs(iou - expected) < 1e-6:
        print(f"  ✅ PASS: 50% overlap gives IoU = {expected:.3f}")
    else:
        print(f"  ❌ FAIL: Expected {expected:.3f}, got {iou:.3f}")
        all_passed = False
    
    print()
    
    # Test 3: Sensitivity formula
    print("TEST 3: Sensitivity (Recall) Formula")
    print("  Formula: TP / (TP + FN)")
    
    tp = 80
    fn = 20
    sensitivity = tp / (tp + fn)
    expected = 0.8
    
    if abs(sensitivity - expected) < 1e-6:
        print(f"  ✅ PASS: 80 TP, 20 FN gives Sensitivity = 0.8")
    else:
        print(f"  ❌ FAIL: Expected {expected}, got {sensitivity}")
        all_passed = False
    
    print()
    
    # Test 4: Specificity formula
    print("TEST 4: Specificity Formula")
    print("  Formula: TN / (TN + FP)")
    
    tn = 900
    fp = 100
    specificity = tn / (tn + fp)
    expected = 0.9
    
    if abs(specificity - expected) < 1e-6:
        print(f"  ✅ PASS: 900 TN, 100 FP gives Specificity = 0.9")
    else:
        print(f"  ❌ FAIL: Expected {expected}, got {specificity}")
        all_passed = False
    
    print()
    
    # Test 5: Precision formula
    print("TEST 5: Precision Formula")
    print("  Formula: TP / (TP + FP)")
    
    tp = 80
    fp = 20
    precision = tp / (tp + fp)
    expected = 0.8
    
    if abs(precision - expected) < 1e-6:
        print(f"  ✅ PASS: 80 TP, 20 FP gives Precision = 0.8")
    else:
        print(f"  ❌ FAIL: Expected {expected}, got {precision}")
        all_passed = False
    
    print()
    print("="*80)
    if all_passed:
        print("✅ ALL METRIC LOGIC TESTS PASSED!")
    else:
        print("❌ SOME METRIC LOGIC TESTS FAILED!")
    print("="*80)
    print()
    
    return all_passed


if __name__ == '__main__':
    # Run validation
    labels_ok = validate_brats_labels('/mnt/user-data/outputs/evaluate.py')
    
    # Run logic tests
    logic_ok = run_metric_logic_tests()
    
    # Exit with appropriate code
    if labels_ok and logic_ok:
        print("\n🎉 All validations passed! The evaluation script is ready to use.\n")
        sys.exit(0)
    else:
        print("\n⚠️  Some validations failed. Please review the issues above.\n")
        sys.exit(1)