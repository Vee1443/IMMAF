# Inter-Modal Multi-Head Attention-based Spatiotemporal Deep Fusion Network (IMMAF)

## Project Overview

This project implements a novel deep learning architecture for spatiotemporal data prediction. The Inter-Modal Multi-Head Attention-based Spatiotemporal Deep Fusion Network (IMMAF-SDFN) effectively captures complex spatiotemporal patterns in time-series data, focusing on applications in environmental monitoring, energy systems, and industrial processes.

The architecture combines Spatial-Temporal Graph Convolutional Networks (STGCN) with various recurrent neural networks (LSTM, BiLSTM, GRU) to model spatial dependencies alongside temporal dynamics in multivariate time series data.

## Key Features

- Fusion of multiple deep learning architectures (STGCN, LSTM, BiLSTM, GRU)
- Sequence modeling with configurable lookback window
- Support for different batch sizes and training configurations
- Comprehensive evaluation metrics (MAE, RMSE, R², MAPE)
- Performance visualization tools for model comparison
- Different model variations for ablation studies

## Architecture

The project implements several model variations:

1. **Base Fusion Model**: Combines ST-GCN and LSTM networks
2. **BiLSTM Fusion Model**: Utilizes bidirectional LSTM for enhanced sequential pattern recognition
3. **GRU Fusion Model**: Employs Gated Recurrent Units for efficient temporal modeling

Each fusion architecture follows a similar pattern:
- ST-GCN path: Processes spatiotemporal features through convolutional and recurrent layers
- RNN path: Processes sequential data through various recurrent network architectures
- Fusion mechanism: Concatenates outputs from both paths and passes through dense layers

## Requirements

```
tensorflow>=2.4.0
keras>=2.4.0
numpy>=1.19.2
pandas>=1.1.3
scikit-learn>=0.23.2
matplotlib>=3.3.2
```

## Data Preparation

The model expects time series data with spatial and temporal components. The preprocessing pipeline includes:

1. Loading data from CSV files
2. Feature selection and normalization using StandardScaler
3. Sequence creation with configurable lookback window
4. Train/validation/test splitting

## Usage

### Data Preprocessing

```python
# Load and prepare data
df = pd.read_csv(<Path_to_dataset>)
selected_columns = # Select relevant columns
target_column = # Set target column

# Normalize data
scaler = StandardScaler()
df[selected_columns] = scaler.fit_transform(df[selected_columns])

# Create sequences with lookback
lookback = 5
X, y = create_sequences(df[selected_columns], df[target_column], lookback)
```

### Model Training

```python
# Define hyperparameters
batch_size = 32
epochs = 10

# Initialize model
fusion_model = create_fusion_model(X_shape_stgcn, X_shape_lstm)
fusion_model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['mae', rmse, r2, mape], run_eagerly=True)
                     
# Train model
history = fusion_model.fit(
    [X_train_stgcn, X_train_lstm],
    y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=([X_val_stgcn, X_val_lstm], y_val),
    verbose=1
)
```

### Window Size Experiments

The code includes experiments with different window sizes (32, 64, 128, 256) to determine optimal context length:

```python
window_sizes = [32, 64, 128, 256]

all_metrics = {}
models=[]

for window_size in window_sizes:
    fusion_model = create_fusion_model(X_shape_stgcn, X_shape_lstm, window_size)
    # Training and evaluation
    # ...
    
    # Store metrics
    all_metrics[window_size] = {
        'mae': test_metrics[1],
        'rmse': test_metrics[2],
        'r2': test_metrics[3],
        'mape': test_metrics[4]
    }
```

## Model Variations

### 1. ST-GCN with LSTM Fusion

```python
def create_st_gcn_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(filters=64, kernel_size=(1, 3), kernel_initializer=he_normal())(input_layer)
    x = LeakyReLU(alpha=0.1)(x)
    x = BatchNormalization()(x)
    x = TimeDistributed(Flatten())(x)
    x = LSTM(64, activation='relu', kernel_initializer=he_normal())(x)
    # ...

def create_lstm_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = LSTM(32, activation='relu', kernel_initializer=he_normal())(input_layer)
    # ...
```

### 2. ST-GCN with BiLSTM Fusion

```python
def create_st_gcn_model2(input_shape):
    # ...
    x = Bidirectional(LSTM(64, activation='relu', kernel_initializer='he_normal'))(x)
    # ...

def create_bilstm_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = Bidirectional(LSTM(32, activation='relu', kernel_initializer='he_normal'))(input_layer)
    # ...
```

### 3. ST-GCN with GRU Fusion

```python
def create_st_gcn_model3(input_shape):
    # ...
    x = GRU(64, activation='relu', kernel_initializer='he_normal')(x)
    # ...

def create_gru_model(input_shape):
    input_layer = Input(shape=input_shape)
    x = GRU(32, activation='relu', kernel_initializer=he_normal())(input_layer)
    # ...
```

## Performance Evaluation

The project evaluates model performance using multiple metrics:

1. Mean Absolute Error (MAE)
2. Root Mean Square Error (RMSE)
3. R-squared (R²)
4. Mean Absolute Percentage Error (MAPE)

Custom metric functions are implemented for TensorFlow:

```python
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

def mape(y_true, y_pred):
    return 100 * K.mean(K.abs((y_true - y_pred) / y_true))
```

## Visualization

The project includes comprehensive visualization tools to compare model performance:

- Training vs. validation metrics across epochs
- Performance comparison between fusion models and individual components
- Error distribution analysis
- Prediction visualization

```python
# Example visualization code
epochs_range = range(1, epochs + 1)
plt.plot(epochs_range, history_fusion.history['val_mae'], label='Fusion (Validation)')
plt.plot(epochs_range, history_lstm.history['val_mae'], label='LSTM (Validation)')
plt.plot(epochs_range, history_stgcn.history['val_mae'], label='ST-GCN (Validation)')
plt.title('Validation Mean Absolute Error Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Validation MAE')
plt.legend()
plt.show()
```

## Model Checkpointing

The code includes model checkpointing to save model states at different epochs:

```python
checkpoint_path = f"model_checkpoint_window_{window_size}_epoch_{{epoch:02d}}.h5"
model_checkpoint = ModelCheckpoint(
    checkpoint_path,
    save_best_only=False,
    verbose=1
)
```
