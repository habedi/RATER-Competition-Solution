# My Solution for the RATER Competition

This repository contains my solution for
the [RATER competition](https://the-learning-agency.com/robust-algorithms-for-thorough-essay-rating/overview/).

## Installing the Dependencies

To run the code in this repository, you must have `Python 3.8` or newer, `PyTorch`, `Transformers`, and `LightGBM` among
other things like `Pandas` and `Scikit-learn` installed on your machine.
The [requirements.txt](requirements.txt) file lists the required Python packages with the correct versions that you can
install using the pip package manager.

```bash
pip install -r requirements.txt
```

Also, you may want to install a few additional CLI tools to make it easier to work with the code in this repository.
You can install these tools using the [install_useful_cli_tools.sh](bin/install_useful_cli_tools.sh) script.

```bash
sudo bash bin/install_useful_cli_tools.sh
```

The installation script works on Ubuntu and other Debian-based systems. If you are using a different operating system,
you may need to install the tools manually.

## The Models

The solution consists of two main parts: segmenting the essay into discourse elements and detecting the effectiveness of
the discourse elements.

The models used in the solution are stored in the [models](models) directory. The figure shows how the models are used
in the inference pipeline:

![pipeline_architecture](data/assets/pipeline_architecture.png)

### Segmentation Models

The models used for segmenting the essay into discourse elements are primarily from the work of the team that ranked
second in the (final) private leaderboard of
the [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/competitions/feedback-prize-2021/) Kaggle
competition.
Their solution is available [here](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313389), including
the solution details,
model weights, and the code for training and inference.

### Effectiveness Detection Model

The model for detecting the effectiveness of the discourse elements is an ensemble of `LightGBM` models trained on the
text of the discourse elements and the essays.
The models are stored in the [effectiveness_models.pkl](models/effectiveness_models.pkl) file. The code for training the
models and inference is in the [predict_effectiveness.py](bin/predict_effectiveness.py) file.

### Training Data and Code, and Inference Code

The table below includes the references to the training data and the code for training and inference for the models used
in
the solution:

| Index | Training Data                                                                                                                | Training Code                                                                            | Inference Code                                           | Description                                                                               |
|-------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------------------|-------------------------------------------------------------------------------------------|
| 1     | [https://www.kaggle.com/competitions/feedback-prize-2021/data](https://www.kaggle.com/competitions/feedback-prize-2021/data) | [https://github.com/ubamba98/feedback-prize](https://github.com/ubamba98/feedback-prize) | [generate_predictions.py](bin/generate_predictions.py)   | The training data, and training and inference code for the segmentation models.           |
| 2     | [competition_data](data/competition_data)                                                                                    | [predict_effectiveness.py](bin/predict_effectiveness.py)                                 | [predict_effectiveness.py](bin/predict_effectiveness.py) | The training data, and training and inference code for the effectiveness detection model. |

As mentioned earlier, the pretrained models from the solution of the team that ranked second in the private leaderboard
of
the [Feedback Prize - Evaluating Student Writing](https://www.kaggle.com/competitions/feedback-prize-2021/) Kaggle
competition was used for the segmentation. Please see this post for more details about their solution:
[https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313389](https://www.kaggle.com/competitions/feedback-prize-2021/discussion/313389).

The table below shows the individual inference score of each segmentation model on RATER's public leaderboard. Measurements were
taken on a machine with an NVIDIA RTX A6000 GPU.

| Model                                                              | Score  |
|--------------------------------------------------------------------|--------|
| [lsg](models/model_pack_1/auglsgrobertalarge)                      | N/A    |
| [bigbird_base_chris](models/model_pack_1/bird-base)                | N/A    |
| [yoso](models/model_pack_1/feedbackyoso)                           | N/A    |
| [funnel](models/model_pack_1/funnel-large-6folds)                  | 0.6044 |
| [debertawithlstm](models/model_pack_1/models/model_pack_1)         | 0.6288 |
| [deberta_v2](models/model_pack_1/deberta-v2-xlarge)                | 0.6105 |
| [debertal_chris](models/model_pack_1/deberta-large-100)            | 0.6229 |
| [debertal](models/model_pack_1/deberta-large-v2)                   | 0.6150 |
| [debertaxl](models/model_pack_1/deberta-xlarge-1536)               | 0.6154 |
| [longformerwithlstm](models/model_pack_1/longformerwithbilstmhead) | 0.6185 |

### Downloading the Segmentation Models

You can download the segmentation models, if you haven't already, by running the following commands in the root directory
of this repository:

```bash
bash bin/download_models.sh && bash bin/make_model_pack.sh
```

Note that the size of the models is around 80 GB, so the download may take a while. 

## Running the Inference Pipeline

You can execute the end-to-end inference pipeline by running the [do_inference.sh](do_inference.sh) script.
The script will perform inference on the test data stored in the [test](data/competition_data/test) data directory and
create a CSV file named `final_submission.csv` in the current directory.

To run the pipeline, you can use the following command:

```bash
bash do_inference.sh --clean --fast
```

Or, if you want to run the pipeline in the background, you can use the following command:

```bash
nohup bash do_inference.sh --clean --fast > inference.log 2>&1 &
```

The `--clean` flag is optional. If you use it, the pipeline will remove the old CSV files and temporary files and
folders created during the previous runs of the pipeline. If you use the `--fast` flag, only the
best segmentation model will be used for the inference, and the effectiveness detection model will be skipped.
This will make the pipeline run much faster.

The `inference.log` file will contain the output of the pipeline. You can monitor the progress of the pipeline by
running the following command:

```bash
ccze -m ansi < inference.log
```

I'm assuming that you have the `ccze` tool installed on your machine.
(If not, running the [install_useful_cli_tools.sh](bin/install_useful_cli_tools.sh) script will install it for you.)

Below, you can see the playback of an example inference log (inference is done with the best segmentation model only, which is
`debertawithlstm`):

![playback](data/runs/28apr-3/inference_log.gif)

Please note that if the `--fast` flag is not set, the pipeline may take a long time to finish.
If you want to make the inference on machines with GPUs with less memory, you may want to reduce the batch sizes (to
around 4 or 8) in the [run_inference.sh](bin/run_inference.sh) script
to avoid getting out-of-memory errors.

## The Predictions File

When the pipeline finishes successfully, a file named [final_submission.csv](final_submission.csv) should be created in
the current directory.
The file contains the predictions for the test set in the required format for the competition and should be ready for
submission.

## License

Most of the files in this repository are licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
The exceptions are the files in the [competition_data](data/competition_data) directory and those in
the [model_pack_1](models/model_pack_1) directory.

