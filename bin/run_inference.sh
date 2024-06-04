#!/bin/bash

# Directory containing the necessary binary files
bin_dir=bin

# Temporary directory for storing intermediate results
tmp_dir=tmp/raw_preds

# Create temporary directory if it doesn't exist
mkdir -p "$tmp_dir"

# Function to perform inference using the specified model and parameters
do_inference() {

    start_time=$(date +%s)  # Capture start time

    echo "---------------------------------------------------------------------------------------------------------------------------------"
    echo "Performing inference using '$2' model:"
    echo "python $bin_dir/generate_predictions.py --model_paths $1 --save_name $2 --max_len $3 --batch_size $4"
    python "$bin_dir/generate_predictions.py" --model_paths $1 --save_name "$tmp_dir/$2" --max_len $3 --batch_size $4
    echo "Inference is done!"

   end_time=$(date +%s)  # Capture end time
   execution_time=$((end_time - start_time))  # Calculate execution time in seconds
   execution_time_minutes=$((execution_time / 60))  # Convert execution time to minutes

   echo "Execution time: $execution_time_minutes minutes"

}

# Execute the function with the necessary arguments for each model
# The arguments are as follows:
# 1. Path to the model checkpoint(s)
# 2. Name of the model
# 3. Maximum sequence length
# 4. Batch size

do_inference \
"models/model_pack_1/auglsgrobertalarge/lsg-roberta-large-0/checkpoint-6750 \
models/model_pack_1/auglsgrobertalarge/lsg-roberta-large-2/checkpoint-7000 \
models/model_pack_1/auglsgrobertalarge/lsg-roberta-large-5/checkpoint-6500" \
"lsg" "1536" "40"

do_inference \
"models/model_pack_1/bird-base/fold1 \
models/model_pack_1/bird-base/fold3 \
models/model_pack_1/bird-base/fold5" \
"bigbird_base_chris" "1024" "96"

do_inference \
"models/model_pack_1/feedbackyoso/yoso-4096-0/checkpoint-12500 \
models/model_pack_1/feedbackyoso/yoso-4096-2/checkpoint-11000 \
models/model_pack_1/feedbackyoso/yoso-4096-4/checkpoint-12500" \
"yoso" "1536" "96"

do_inference \
"models/model_pack_1/funnel-large-6folds/large-v628-f1/checkpoint-11500 \
models/model_pack_1/funnel-large-6folds/large-v627-f3/checkpoint-11000 \
models/model_pack_1/funnel-large-6folds/large-v623-f4/checkpoint-10500" \
"funnel" "1536" "40"

do_inference \
"models/model_pack_1/deberta-lstm-jaccard/jcl-deberta-large-f1/checkpoint-4500 \
models/model_pack_1/deberta-lstm-jaccard/jcl-deberta-large-f2/checkpoint-5000 \
models/model_pack_1/deberta-lstm-jaccard/jcl-deberta-large-f3/checkpoint-3500" \
"debertawithlstm" "1536" "32"

do_inference \
"models/model_pack_1/deberta-v2-xlarge/deberta-v2-xlarge-v6000-f0/checkpoint-7500 \
models/model_pack_1/deberta-v2-xlarge/deberta-v2-xlarge-v6003-f3/checkpoint-9000" \
"deberta_v2" "1536" "16"

do_inference \
"models/model_pack_1/deberta-large-100/fold0 \
models/model_pack_1/deberta-large-100/fold1 \
models/model_pack_1/deberta-large-100/fold2" \
"debertal_chris" "1536" "32"

do_inference \
"models/model_pack_1/deberta-large-v2/deberta-large-v2100-f0/checkpoint-10500 \
models/model_pack_1/deberta-large-v2/deberta-large-v2101-f1/checkpoint-11500 \
models/model_pack_1/deberta-large-v2/deberta-large-v2102-f2/checkpoint-8500" \
"debertal" "1536" "32"

do_inference \
"models/model_pack_1/deberta-xlarge-1536/deberta-xlarge-v8004-f4/checkpoint-14000 \
models/model_pack_1/deberta-xlarge-1536/deberta-xlarge-v4005-f5/checkpoint-13000" \
"debertaxl" "1536" "32"

do_inference \
"models/model_pack_1/longformerwithbilstmhead/aug-longformer-large-4096-f0/checkpoint-5500 \
models/model_pack_1/longformerwithbilstmhead/aug-longformer-large-4096-f2/checkpoint-7500 \
models/model_pack_1/longformerwithbilstmhead/aug-longformer-large-4096-f5/checkpoint-6000" \
"longformerwithlstm" "1536" "64"
