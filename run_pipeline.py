import subprocess
import os
import sys
import time

def run_step(name, command):
    print(f"\n>>> STARTING STEP: {name}")
    print(f"Executing: {command}")
    
    # Ensure current directory is in PYTHONPATH for internal imports
    env = os.environ.copy()
    project_root = os.getcwd()
    env["PYTHONPATH"] = project_root + os.pathsep + env.get("PYTHONPATH", "")
    
    start_time = time.time()
    
    try:
        # We use shell=True for Windows compatibility with environment variables if needed
        # but here we pass the env dict directly.
        process = subprocess.Popen(
            command,
            env=env,
            shell=True,
            stdout=sys.stdout,
            stderr=sys.stderr
        )
        process.communicate()
        
        if process.returncode != 0:
            print(f"\n[!] ERROR in {name}. Pipeline stopped.")
            sys.exit(process.returncode)
            
        elapsed = time.time() - start_time
        print(f">>> COMPLETED: {name} in {elapsed:.2f} seconds")
        
    except Exception as e:
        print(f"\n[!] FAILED to run {name}: {e}")
        sys.exit(1)

def main():
    print("="*60)
    print("NLPDL PROJECT: FULL PIPELINE (EXCLUDING EVALUATION)")
    print("="*60)
    
    python_exe = sys.executable

    # 0. Train Baseline NER Model
    # This generates the 'ner_model' and the initial predictions files.
    run_step(
        "Baseline NER Training", 
        f"{python_exe} src/ner/train_ner.py"
    )

    # 1. Data Preparation
    run_step(
        "Data Preparation", 
        f"{python_exe} src/preprocessing/build_formality_dataset.py"
    )
    
    # 2. Train Formality Classifier
    # Note: This includes the optimized NER feature extraction we built
    run_step(
        "Classifier Training", 
        f"{python_exe} src/classification/train_classifier.py"
    )
    
    # 3. Classify NER Sentences
    run_step(
        "NER Sentence Classification", 
        f"{python_exe} src/classification/classify_ner_sentences.py"
    )
    
    # 4. Split Datasets by Formality
    run_step(
        "Dataset Splitting (Pre-Eval)", 
        f"{python_exe} src/ner/evaluate_by_formality.py"
    )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("All formal/informal subsets are ready in outputs/results/")
    print("Run 'python comprehensive_eval.py' to see final F1 scores.")
    print("="*60)

if __name__ == "__main__":
    main()
