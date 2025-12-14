# Deep Learning Class (VITMMA19) Project Work

## Project Details

### Project Information

- **Selected Topic**: Legal Text Decoder
- **Student Name**: Hegedüs Nóra
- **Aiming for +1 Mark**: No

### Solution Description

This project aims to classify the complexity/readability of legal text snippets (specifically from terms of service or contracts) into 5 distinct levels (from "Very difficult to understand" to "Easy to understand"). The solution is implemented as a complete Deep Learning pipeline consisting of data acquisition, preprocessing, baseline modeling, k-fold cross-validation development, and ensemble inference.

**Methodology:**
1.  **Data Processing**: The pipeline automatically downloads raw JSON data from Google Drive. Texts are tokenized using a simple regex-based tokenizer, and a vocabulary is built. Class imbalance is handled by calculating and applying class weights during training (`CrossEntropyLoss`).
2.  **Baseline Model**: A simple Recurrent Neural Network (GRU) with an Embedding layer is used as a reference point to establish minimum performance metrics.
3.  **Model Architecture**: The main model utilizes a Bidirectional GRU (BiGRU) to capture context from both directions of the text. It consists of:
    * Embedding Layer (32 dim)
    * Bi-directional GRU Layer (64 hidden dim)
    * Concatenation of the final hidden states
    * Fully Connected (Linear) Layer mapping to the 5 output classes.
4.  **Training & Validation**: A 5-Fold Cross-Validation strategy is employed to ensure the model's robustness and to detect overfitting on the small dataset (approx. 479 samples). Early stopping is implemented in the baseline.
5.  **Inference**: An ensemble approach is used for the final inference, averaging the probability outputs of the 5 models trained during the cross-validation phase to produce stable predictions.

**Results**:
The baseline model achieved an accuracy of approx. 33%, heavily biased towards the majority class. The final Ensemble model improved generalization with a test accuracy around 35-40% and demonstrated better confusion matrix distribution across classes compared to the baseline, despite the extremely limited dataset size.

### Data Preparation

The data preparation process is fully automated within the pipeline.

1.  **Automatic Download**: The script `src/01-data-preprocessing.py` uses `gdown` to download the raw JSON annotation files directly from Google Drive into the `data/` directory. No manual download is required.
2.  **Parsing & Cleaning**: The script parses the proprietary JSON format, extracts the text and the corresponding label (1-5), and normalizes the labels.
3.  **Preprocessing**: 
    * A vocabulary is built from the training texts (`<pad>` and `<unk>` tokens included).
    * Texts are tokenized and converted to integer sequences.
    * Class weights are computed to handle the imbalanced dataset.
    * The processed data is split into stratified Train/Test sets and saved as `processed_data.pt`.

To run this, simply execute the Docker container or run `python src/01-data-preprocessing.py`.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
````

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run -v ${PWD}/data:/app/data dl-project > log/run.log 2>&1
```

  * **Note**: On Windows PowerShell, use `${PWD}\data` instead of `${PWD}/data`.
  * The container is configured to run the full pipeline sequentially: Preprocessing -\> Baseline Training -\> Model Development (K-Fold) -\> Evaluation -\> Inference on samples.
  * The logs will be saved to `log/run.log`.

### File Structure and Functions

The repository is structured as follows:

  - **`src/`**: Contains the source code for the machine learning pipeline.

      - `01-data-preprocessing.py`: Downloads raw data from GDrive, parses JSONs, builds vocabulary, calculates class weights, and saves train/test splits.
      - `02-baseline-training.py`: Trains a simple GRU baseline model on the full training set with class weighting and early stopping.
      - `03-model-development.py`: Performs 5-Fold Cross-Validation using a Bidirectional GRU model to evaluate architecture performance.
      - `04-evaluation.py`: Evaluates both the Baseline and the Ensemble of CV models on the held-out test set, generating confusion matrices and metrics (Acc, F1, MAE).
      - `05-inference.py`: Loads the 5 trained CV models and performs ensemble prediction on new sample texts.
      - `config.py`: Configuration file containing hyperparameters (paths, device, num\_classes, label map).
      - `utils.py`: Helper functions for logging, tokenization, and model summary.

  - **`data/`**: Directory where raw JSONs are downloaded and processed.

  - **`output/`**: Directory where model artifacts (.pth), processed data (.pt), and vocabulary are saved.

  - **`log/`**: Contains log files.

      - `run.log`: The output log file showing the full execution pipeline, hyperparameters, and metrics.

  - **Root Directory**:

      - `Dockerfile`: Configuration file for building the Docker image.
      - `requirements.txt`: List of Python dependencies.
      - `README.md`: Project documentation.
      - `run.sh`: Run the pipeline.
