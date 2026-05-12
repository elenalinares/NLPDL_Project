import subprocess
import re

def run_f1(gold, pred):
    """Calls the official span_f1.py and parses its output."""
    try:
        result = subprocess.run(
            ['python', 'data/span_f1.py', gold, pred],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        
        # Parse metrics using regex
        metrics = {}
        
        # Exact Match (Slot-F1)
        f1_match = re.search(r'slot-f1:\s+([\d.]+)', output)
        prec_match = re.search(r'precision:\s+([\d.]+)', output)
        rec_match = re.search(r'recall:\s+([\d.]+)', output)
        
        if f1_match:
            metrics['exact_f1'] = float(f1_match.group(1))
            metrics['exact_p'] = float(prec_match.group(1))
            metrics['exact_r'] = float(rec_match.group(1))
            
        # Unlabeled
        ul_f1 = re.search(r'ul_slot-f1:\s+([\d.]+)', output)
        if ul_f1:
            metrics['ul_f1'] = float(ul_f1.group(1))
            
        # Loose
        # Fixed regex to not match 'ul_slot-f1'
        l_f1 = re.search(r'\s+l_slot-f1:\s+([\d.]+)', output)
        if l_f1:
            metrics['l_f1'] = float(l_f1.group(1))
            
        return metrics
    except Exception as e:
        print(f"Error running evaluation for {gold}: {e}")
        return None

def print_comparison():
    print("Running Official Evaluations...")
    
    formal = run_f1("outputs/results/formal_gold.iob2", "outputs/results/formal_pred.iob2")
    informal = run_f1("outputs/results/informal_gold.iob2", "outputs/results/informal_pred.iob2")
    
    if not formal or not informal:
        print("Could not complete comparison. Check if split files exist in outputs/results/")
        return

    print("\n" + "="*70)
    print(f"{'Metric (Official span_f1.py)':<30} | {'Formal':<12} | {'Informal':<12} | {'Diff':<8}")
    print("-" * 70)
    
    display_metrics = [
        ("Exact Match F1", "exact_f1"),
        ("Exact Precision", "exact_p"),
        ("Exact Recall", "exact_r"),
        ("Unlabeled F1", "ul_f1"),
        ("Loose Match F1", "l_f1"),
    ]
    
    for label, key in display_metrics:
        f_val = formal.get(key, 0)
        i_val = informal.get(key, 0)
        diff = f_val - i_val
        print(f"{label:<30} | {f_val:<12.4f} | {i_val:<12.4f} | {diff:>+8.4f}")
    
    print("="*70)
    print("Conclusion: Positive 'Diff' means the model performs better on Formal text.")

if __name__ == "__main__":
    print_comparison()
