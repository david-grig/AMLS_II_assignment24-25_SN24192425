# Contrail Detection using U-Net in PyTorch

## Project Goal

This project implements a U-Net based deep learning model using PyTorch to identify aviation contrails in satellite imagery. The goal is based on the Kaggle competition "Google Research - Identify Contrails to Reduce Global Warming", aiming to contribute to methods for mitigating the climate impact of aviation.

## Dataset: Google Research Contrails

The dataset for this project consists of GOES-16 ABI satellite image sequences (infrared bands 8-16) and corresponding human-annotated contrail masks.

**IMPORTANT: Dataset Size**

The **full training dataset provided is extremely large (~450 GB)**. Downloading and storing this dataset locally can be challenging due to bandwidth and disk space requirements.

## Recommended Environment: Kaggle Notebooks

Due to the dataset size, it is **highly recommended** to run this project within the **Kaggle Notebook environment**. Kaggle provides free access to computational resources and mounts the competition datasets directly, avoiding the need for local downloads.

### Data Paths in Kaggle

When working in a Kaggle Notebook associated with the competition, the data is typically available under the following paths:

* **Base Input Path:**
    ```
    /kaggle/input/google-research-identify-contrails-reduce-global-warming/
    ```
* **Training Data:**
    ```
    /kaggle/input/google-research-identify-contrails-reduce-global-warming/train/
    ```
* **Validation Data:**
    ```
    /kaggle/input/google-research-identify-contrails-reduce-global-warming/validation/
    ```

The code in the associated notebook(s) uses these paths (or variables derived from them like `BASE_DATA_PATH`, `TRAIN_DATA_PATH`, `VALIDATION_DATA_PATH` as defined within main.py) to access the data files (`.npy` format).

*Note: The validation dataset (`validation.zip`) is significantly smaller (~8.5 GB) and could potentially be downloaded and used for local development if required, although the Kaggle environment remains the recommended approach.*

### Enabling GPU Acceleration

Training deep learning models like U-Net is computationally intensive. To significantly speed up the training process, **ensure you enable GPU acceleration** in your Kaggle Notebook:

1.  Open your notebook on Kaggle.
2.  Navigate to the **Settings** panel (often on the right-hand side).
3.  Under the **Accelerator** section, select **GPU** from the dropdown menu.
4.  A common powerful option available is the **"GPU T4 x2"**. Select this or another available GPU option.
5.  Confirm the changes. Your notebook session will restart with GPU capabilities enabled.

The PyTorch code in this project includes logic to automatically detect and utilize the available GPU (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`).

## Implementation Details

* **Framework:** PyTorch
* **Model:** U-Net architecture for semantic segmentation.
* **Key Libraries:** `torch`, `torchvision`, `numpy`, `albumentations` (for data augmentation).

## Running the Code

1.  Ensure you are running within a Kaggle Notebook environment attached to the "Google Research - Identify Contrails" competition dataset.
2.  Enable GPU acceleration as described above.
3.  Run the cells in the notebook sequentially to:
    * Load and preprocess data using the custom `Dataset` and `DataLoader`.
    * Define the U-Net model, loss function (Dice Loss), and optimizer (Adam).
    * Execute the training loop, which includes validation, metric calculation (Dice Coefficient), learning rate scheduling, model checkpointing, and early stopping.
    * Evaluate the final model.