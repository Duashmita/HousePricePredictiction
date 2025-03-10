# Housing Price Prediction with PyTorch

This project demonstrates how to build and train a simple neural network using [PyTorch](https://pytorch.org/) to predict housing prices based on California housing data.

## Overview
1. **Data Loading & Preprocessing**  
   - Cleans the dataset (drops duplicates and missing values).  
   - One-hot encodes the `ocean_proximity` feature.  
   - Converts the DataFrame to tensors suitable for PyTorch.

2. **Model Architecture**  
   - A feed-forward neural network with two hidden layers (13 → 32 → 10 → 1).  
   - Uses ReLU activation between layers.

3. **Training & Testing**  
   - Uses the L1 loss function (`nn.L1Loss()`) and the Adam optimizer.  
   - Batches the data using `DataLoader` with a batch size of 45.  
   - Saves training loss in `loss_train` and a random sampling of test loss in `loss_test`.

4. **Saving the Model**  
   - Trained model parameters are saved to `Model.pt`.

## Requirements
- Python 3.7+
- PyTorch
- NumPy
- pandas
- matplotlib (for plotting, if you use the plotting script)

Install the dependencies using:
```bash
pip install torch numpy pandas matplotlib
```

## Usage
1. Prepare the Dataset
    - Place your housing_dataset.csv file in the same directory as the scripts.
2. Run the Training Script
    - Run the main script (e.g., python main.py) to train the model.
    - This will produce two lists of losses (loss_train and loss_test) and save the trained model weights to Model.pt.
3. Plot the Results (Optional)
    - Use the second script `plottingthemodel.py` to plot the training and test losses over time:

## Notes
- Hyperparameters: Feel free to adjust the batch size, number of epochs, and learning rate to see if performance improves.
- Loss Function: Currently set to nn.L1Loss() (mean absolute error). You can switch to nn.MSELoss() if mean squared error is preferred.
- Train/Test Split: For a more conventional approach, split your data into training and testing sets before training and evaluating.

Enjoy experimenting with the model and feel free to customize the architecture or data preprocessing to improve performance!