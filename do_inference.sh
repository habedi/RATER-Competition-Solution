#!/usr/bin/bash

# Help message function
display_help() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --clean     Clean up csv files and temporary directories"
    echo "  --fast      Enable fast mode for faster inference"
    echo "  --help      Display this help message"
    exit 0
}

# Initialize variables for script paths
WBF_SCRIPT="bin/wbf.py"
INFERENCE_SCRIPT="bin/run_inference.sh"
EFFECTIVENESS_SCRIPT="bin/predict_effectiveness.py"
INTERIM_SUBMISSION="tmp/fixed_interim_submission.csv"
FINAL_SUBMISSION="final_submission.csv"

# Check if --fast flag is set for faster inference
FAST=false
while true; do
  case "$1" in
    --clean)
      echo "Cleaning up csv files and temporary directories..."
      rm -f *.csv
      rm -rf tmp
      shift
      ;;
    --fast)
      FAST=true
      WBF_SCRIPT="bin/wbf_fast_v2.py"
      INFERENCE_SCRIPT="bin/run_inference_fast_v2.sh"
      echo "Fast inference enabled."
      shift
      ;;
    --help)
      display_help
      ;;
    --)
      shift
      break
      ;;
    *)
      #echo "Error: Invalid option '$1'"
      break
      ;;
  esac
done

# Capture the overall start time
overall_start_time=$(date +%s)

# Run the script and calculate its runtime
start_time=$(date +%s)
bash "$INFERENCE_SCRIPT"
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Runtime of $INFERENCE_SCRIPT: $((runtime / 60)) minutes and $((runtime % 60)) seconds"

# Run the script and calculate its runtime
start_time=$(date +%s)
python "$WBF_SCRIPT"
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Runtime of $WBF_SCRIPT: $((runtime / 60)) minutes and $((runtime % 60)) seconds"

# Run the script and calculate its runtime
start_time=$(date +%s)
python bin/fix_interim_submission.py
end_time=$(date +%s)
runtime=$((end_time - start_time))
echo "Runtime of fix_interim_submission.py: $((runtime / 60)) minutes and $((runtime % 60)) seconds"

# Run the script only if --fast flag is not set
if [ "$FAST" = false ]; then
  start_time=$(date +%s)
  python "$EFFECTIVENESS_SCRIPT"
  end_time=$(date +%s)
  runtime=$((end_time - start_time))
  echo "Runtime of predict_effectiveness.py: $((runtime / 60)) minutes and $((runtime % 60)) seconds"
else
  cp "$INTERIM_SUBMISSION" "$FINAL_SUBMISSION"
  if [ $? -eq 0 ]; then
    echo "Copied $INTERIM_SUBMISSION to ./$FINAL_SUBMISSION successfully."
  else
    echo "Error: Failed to copy $INTERIM_SUBMISSION to ./$FINAL_SUBMISSION."
  fi
fi

# Calculate the overall runtime
overall_end_time=$(date +%s)
overall_runtime=$((overall_end_time - overall_start_time))
echo "Overall runtime: $((overall_runtime / 3600)) hours and $(((overall_runtime / 60) % 60)) minutes"
