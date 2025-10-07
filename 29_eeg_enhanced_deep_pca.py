import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, LSTM, GRU
from tensorflow.keras.layers import Bidirectional, Reshape, Flatten, TimeDistributed, Concatenate
from tensorflow.keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1_l2
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set style for plots
plt.style.use('ggplot')
sns.set(font_scale=1.2)
sns.set_style("whitegrid")

print("Loading EEG dataset...")
file_path = "combined_reduced_epochs (1).csv"
df = pd.read_csv(file_path)

print(f"Dataset shape: {df.shape}")
print(f"Number of unique trials: {df['trial_id'].nunique()}")
print(f"Number of unique labels: {df['label'].nunique()}")
print(f"Number of unique channels: {df['channel'].nunique()}")
print(f"Number of unique epochs: {df['epoch'].nunique()}")

# Identify feature columns (excluding metadata)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
metadata_cols = ['epoch', 'channel', 'label', 'trial_id']
feature_cols = [col for col in numerical_cols if col not in metadata_cols]
print(f"Number of features: {len(feature_cols)}")

# Define important channels based on previous analysis
important_channels = [61, 58, 2, 31, 23, 4, 10, 15, 30, 40]
print(f"Using top {len(important_channels)} important channels")

# Data augmentation functions
def add_gaussian_noise(X, noise_factor=0.05):
    """Add Gaussian noise to the data"""
    noise = np.random.normal(0, noise_factor, X.shape)
    return X + noise

def time_warp(X, sigma=0.2, knot=4):
    """Apply time warping to the data"""
    # Create a copy to avoid modifying the original data
    X_warped = X.copy()
    
    if len(X.shape) == 3:  # For 3D data (trials, time, features)
        # For each trial
        for i in range(X.shape[0]):
            # Create random warping points
            knots = np.linspace(0, 1, knot+2)
            warper = np.zeros_like(knots)
            warper[0] = 0  # Start point fixed
            warper[-1] = 1  # End point fixed
            warper[1:-1] = np.sort(np.random.uniform(0, 1, knot))
            
            # Create the mapping for interpolation
            time_points = np.linspace(0, 1, X.shape[1])
            warped_points = np.interp(time_points, knots, warper)
            
            # Apply warping to each feature dimension
            for dim in range(X.shape[2]):
                X_warped[i, :, dim] = np.interp(warped_points, time_points, X[i, :, dim])
    else:  # For 2D data (samples, features)
        # Just add small random noise instead of warping for 2D data
        X_warped = X + np.random.normal(0, 0.01, X.shape)
    
    return X_warped

def spectral_augment(X, max_mask_pct=0.1, n_freq_masks=2):
    """Apply spectral augmentation (frequency masking)"""
    aug_X = X.copy()
    
    if len(X.shape) == 3:  # For 3D data
        freq_width = int(X.shape[2] * max_mask_pct)
        for _ in range(n_freq_masks):
            for i in range(X.shape[0]):
                f0 = np.random.randint(0, X.shape[2] - freq_width)
                aug_X[i, :, f0:f0+freq_width] = 0
    else:  # For 2D data
        n_features = X.shape[1]
        freq_width = int(n_features * max_mask_pct)
        
        for _ in range(n_freq_masks):
            for i in range(X.shape[0]):
                f0 = np.random.randint(0, n_features - freq_width)
                aug_X[i, f0:f0+freq_width] = 0
    
    return aug_X

def mixup(X, y, alpha=0.2):
    """Apply mixup augmentation"""
    batch_size = X.shape[0]
    weights = np.random.beta(alpha, alpha, batch_size)
    
    # Ensure weights are column vectors for proper broadcasting
    if len(X.shape) == 2:
        weights = weights.reshape(batch_size, 1)
    elif len(X.shape) == 3:
        weights = weights.reshape(batch_size, 1, 1)
    
    # Create random index permutation
    index = np.random.permutation(batch_size)
    
    # Create mixup samples
    X_mixed = weights * X + (1 - weights) * X[index]
    
    # Convert y to one-hot if it's not already
    if len(y.shape) == 1:
        y_onehot = to_categorical(y)
    else:
        y_onehot = y
        
    # Reshape weights for proper broadcasting with y_onehot
    if len(y_onehot.shape) > 1:
        weights = weights.reshape(batch_size, 1)
        
    # Mix the labels
    y_mixed = weights * y_onehot + (1 - weights) * y_onehot[index]
    
    return X_mixed, y_mixed

# Feature extraction functions
def extract_pca_features(df, n_components=30):
    """
    Extract PCA features from important channels
    """
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting PCA features for {n_trials} trials")
    
    # Initialize arrays
    X_trials = []
    y_trials = []
    
    # Extract features for each trial
    for trial in unique_trials:
        # Get data for this trial
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label (should be the same for all rows in the trial)
        label = trial_data['label'].iloc[0]
        
        # Filter for important channels only
        trial_data = trial_data[trial_data['channel'].isin(important_channels)]
        
        # Extract features by averaging across epochs for each channel
        channel_features = trial_data.groupby('channel')[feature_cols].mean()
        
        # Flatten the channel features into a single vector
        features = channel_features.values.flatten()
        
        X_trials.append(features)
        y_trials.append(label)
    
    # Convert to numpy arrays
    X_trials = np.array(X_trials)
    y_trials = np.array(y_trials)
    
    print(f"Raw feature shape: {X_trials.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_trials)
    
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"Explained variance with {n_components} components: {pca.explained_variance_ratio_.sum():.4f}")
    
    return X_pca, y_trials, pca, scaler

def extract_temporal_features(df):
    """
    Extract features preserving temporal structure for recurrent models
    """
    # Get unique trials and epochs
    unique_trials = sorted(df['trial_id'].unique())
    unique_epochs = sorted(df['epoch'].unique())
    n_trials = len(unique_trials)
    n_epochs = len(unique_epochs)
    
    print(f"Extracting temporal features for {n_trials} trials across {n_epochs} epochs")
    
    # Initialize arrays
    X_temporal = np.zeros((n_trials, n_epochs, len(important_channels)))
    y_temporal = np.zeros(n_trials)
    
    # Extract features for each trial
    for i, trial in enumerate(unique_trials):
        # Get data for this trial
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label (should be the same for all rows in the trial)
        y_temporal[i] = trial_data['label'].iloc[0]
        
        # Extract features for each epoch
        for j, epoch in enumerate(unique_epochs):
            epoch_data = trial_data[trial_data['epoch'] == epoch]
            
            # Skip if epoch doesn't have data
            if len(epoch_data) == 0:
                continue
            
            # Extract mean feature value for each channel
            for k, channel in enumerate(important_channels):
                channel_data = epoch_data[epoch_data['channel'] == channel]
                
                # Skip if channel doesn't have data
                if len(channel_data) == 0:
                    continue
                
                # Use mean of all features
                X_temporal[i, j, k] = channel_data[feature_cols].mean().mean()
    
    # Standardize features
    # Reshape to 2D for standardization
    orig_shape = X_temporal.shape
    X_reshaped = X_temporal.reshape(-1, X_temporal.shape[2])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    # Reshape back to 3D
    X_temporal_scaled = X_scaled.reshape(orig_shape)
    
    return X_temporal_scaled, y_temporal.astype(int)

def extract_spatial_features(df):
    """
    Extract features preserving spatial structure for CNN models
    """
    # Get unique trials
    unique_trials = sorted(df['trial_id'].unique())
    n_trials = len(unique_trials)
    
    print(f"Extracting spatial features for {n_trials} trials")
    
    # Initialize arrays - creating a 2D spatial grid (8x8) from the channels
    X_spatial = np.zeros((n_trials, 8, 8, 1))
    y_spatial = np.zeros(n_trials)
    
    # Channel to grid mapping (approximate based on standard 10-20 system)
    # This maps each channel to a position in the 8x8 grid
    channel_grid_map = {
        61: (0, 3), 58: (0, 4),  # Front
        2: (1, 2), 31: (1, 5),
        23: (2, 1), 4: (2, 6),
        10: (3, 0), 15: (3, 7),
        30: (4, 0), 40: (4, 7),  # Back
        # Fill remaining important channels if needed
    }
    
    # Extract features for each trial
    for i, trial in enumerate(unique_trials):
        # Get data for this trial
        trial_data = df[df['trial_id'] == trial]
        
        # Skip if trial doesn't have data
        if len(trial_data) == 0:
            continue
        
        # Get the label (should be the same for all rows in the trial)
        y_spatial[i] = trial_data['label'].iloc[0]
        
        # Extract features for each channel and place in the grid
        for channel, (row, col) in channel_grid_map.items():
            channel_data = trial_data[trial_data['channel'] == channel]
            
            # Skip if channel doesn't have data
            if len(channel_data) == 0:
                continue
            
            # Use mean of all features across all epochs
            X_spatial[i, row, col, 0] = channel_data[feature_cols].mean().mean()
    
    # Normalize the spatial features
    X_spatial = (X_spatial - X_spatial.mean()) / (X_spatial.std() + 1e-8)
    
    return X_spatial, y_spatial.astype(int)

# Extract PCA features only
print("\nExtracting features for Deep PCA model...")
X_pca, y_pca, pca_model, pca_scaler = extract_pca_features(df, n_components=30)
print(f"PCA feature shape: {X_pca.shape}")

# Split data with stratification
print("\nSplitting data into training and testing sets...")
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y_pca, test_size=0.3, random_state=42, stratify=y_pca
)

# Convert to categorical for deep learning
y_train_cat = to_categorical(y_train_pca, num_classes=4)
y_test_cat = to_categorical(y_test_pca, num_classes=4)

# Data augmentation for training sets
print("\nAugmenting training data...")
# Augment PCA features
X_train_pca_noise = add_gaussian_noise(X_train_pca)
X_train_pca_warp = time_warp(X_train_pca)
X_train_pca_spectral = spectral_augment(X_train_pca)
X_train_pca_aug, y_train_pca_aug = mixup(X_train_pca, y_train_cat)

# Combine all augmented data
X_train_pca_combined = np.vstack([X_train_pca, X_train_pca_noise, X_train_pca_warp, X_train_pca_spectral])
y_train_pca_combined = np.vstack([y_train_cat, y_train_cat, y_train_cat, y_train_cat])

print(f"Augmented PCA feature shape: {X_train_pca_combined.shape}")

# Define callbacks for deep learning models
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=10,
    min_lr=0.0001
)

model_checkpoint = ModelCheckpoint(
    'deep_pca_best_model_v2.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Create Deep Neural Network for PCA features
def create_deep_pca_model(input_shape):
    model = Sequential([
        # Input layer
        Dense(512, activation='relu', input_shape=(input_shape,), 
              kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        
        # Hidden layers
        Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        
        # Output layer
        Dense(4, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall', 'F1Score']
    )
    
    return model



# Train Deep PCA model
print("\nTraining Deep PCA model...")
deep_pca_model = create_deep_pca_model(X_train_pca_combined.shape[1])
deep_pca_history = deep_pca_model.fit(
    X_train_pca_combined, y_train_pca_combined,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# Evaluate model
print("\nEvaluating Deep PCA model...")
deep_pca_preds = deep_pca_model.predict(X_test_pca)
deep_pca_classes = np.argmax(deep_pca_preds, axis=1)
deep_pca_accuracy = accuracy_score(y_test_pca, deep_pca_classes)
print(f"Deep PCA Model Accuracy: {deep_pca_accuracy:.4f}")

# Print classification report
print("\nClassification Report (Deep PCA Model):")
print(classification_report(y_test_pca, deep_pca_classes))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_pca, deep_pca_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Deep PCA Model (Accuracy: {deep_pca_accuracy:.4f})')

plt.tight_layout()
plt.savefig('model_confusion_matrices.png')
plt.close()

# Plot learning curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(deep_pca_history.history['accuracy'], label='Train')
plt.plot(deep_pca_history.history['val_accuracy'], label='Validation')
plt.title('Deep PCA Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
hist_df = pd.DataFrame(deep_pca_history.history)
hist_df.to_csv("model_learning_curves_data.csv", index_label="epoch")
plt.subplot(1, 2, 2)
plt.plot(deep_pca_history.history['loss'], label='Train Loss')
plt.plot(deep_pca_history.history['val_loss'], label='Validation Loss')
plt.title('Deep PCA Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('model_learning_curves.png')
plt.close()

# Plot per-class performance
class_report_deep_pca = classification_report(y_test_pca, deep_pca_classes, output_dict=True)
class_f1_deep_pca = [class_report_deep_pca[str(i)]['f1-score'] for i in range(4)]
class_precision = [class_report_deep_pca[str(i)]['precision'] for i in range(4)]
class_recall = [class_report_deep_pca[str(i)]['recall'] for i in range(4)]

plt.figure(figsize=(12, 6))
x = np.arange(4)
width = 0.25

plt.bar(x - width, class_f1_deep_pca, width, label='F1-Score', color='skyblue')
plt.bar(x, class_precision, width, label='Precision', color='lightgreen')
plt.bar(x + width, class_recall, width, label='Recall', color='salmon')

plt.axhline(y=0.5, color='r', linestyle='--', label='Baseline')
plt.xlabel('Class')
plt.ylabel('Score')
plt.title('Per-Class Performance Metrics')
plt.xticks(x, ['Class 0', 'Class 1', 'Class 2', 'Class 3'])
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('per_class_performance.png')
plt.close()

# Save the model and parameters
print("\nSaving model and parameters...")
deep_pca_model.save('deep_pca_final_model.h5')

import pickle
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca_model, f)
    
with open('pca_scaler.pkl', 'wb') as f:
    pickle.dump(pca_scaler, f)

print("Models and parameters saved successfully.")

# Compare with previous approaches
print("\nComparing with previous approaches:")
print(f"PCA + Ensemble (20 components): 68.06%")
print(f"Optimized Stacking Ensemble: 72.22%")
print(f"Deep PCA Model: {deep_pca_accuracy:.4f}")

print("\nEnhanced Deep PCA model training completed.")
print(f"Accuracy achieved: {deep_pca_accuracy:.4f}")
print("Visualization images saved.")

# Print model summary
print("\nDeep PCA Model Summary:")
deep_pca_model.summary()
