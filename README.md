# CS271 Project - Machine Learning

### Recommended configuration:
- Python Version: 3.13
- NVIDIA GPU with CUDA 13.0 compatibility

This project may work on different Python versions, but it is not guaranteed. 
Without a compatible CUDA-enabled GPU, the Conv2DLSTM model will run much slower.
The code for the Conv2DLSTM model comes from [this repository](https://github.com/georgeyiasemis/2D-Convolutional-Recurrent-Neural-Networks-with-PyTorch) with minor changes.

### Instructions

#### **Step 1: Clone the Repository**

```
git clone https://github.com/rtest42/CS271-Project
```

#### **Step 2: Download the database files**

Download the zip file [here](https://dataserv.ub.tum.de/s/?dir=/5.625deg/2m_temperature), unzip the file, and move the folder to the project directory.

#### **Step 3: Configure Python Environment**

Configure a virtual Python environment (optional).

Ensure packages are installed via ```pip install -r requirements.txt```.

#### **Step 4: Configure Hyperparameters and Database Files**

In `kernel_ridge_residual.py` and `conv2dlstm_model.py`, you can adjust the hyperparameters and database files used to train, validate, and test the machine learning models.

#### **Step 5: Run the Machine Learning Models**

You might need to mark the project directory as root.

```
python kernel_ridge_residual.py
python conv2dlstm_model.py
```
#### **Step 6: View Model Evaluation Statistics**
You can also run this command while the Python programs are running:
```
mlflow ui
```
