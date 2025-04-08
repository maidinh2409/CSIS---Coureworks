import numpy as np

# Giả sử X_train, y_train, X_test và y_test đã được định nghĩa trước
num_samples, num_features = X_train.shape
learning_rate = [0.009, 0.01, 0.05, 0.08]
num_epochs = 100

def loss_function(y_actual, y_pred, n_samples):
    mse = np.sqrt(np.sum((y_actual - y_pred)**2)) / n_samples
    return mse

def gradient(y_actual, y_pred, n_samples, x):
    return -2 / n_samples * np.dot(x.T, (y_actual - y_pred))

def gradient_intercept(y_actual, y_pred, n_samples):
    return -2 / n_samples * np.sum((y_actual - y_pred))

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true))**2)
    ss_residual = np.sum((y_true - y_pred)**2)
    return 1 - ss_residual / ss_total

# Để lưu RMSE


rmse_train_all = []
rmse_test_all = []
r2_test_all = []

for i in learning_rate:
    weight = np.random.rand(num_features)  # Reset weight mỗi lần
    intercept = 0
    rmse_train_list = []
    rmse_test_list = []
    r2_test_list = []

    for epoch in range(num_epochs):
        y_pred_train = np.dot(X_train, weight) + intercept

        grad_x = gradient(y_train, y_pred_train, num_samples, X_train)
        grad_b = gradient_intercept(y_train, y_pred_train, num_samples)

        weight -= i * grad_x
        intercept -= i * grad_b

        epoch_rmse_train = loss_function(y_train, y_pred_train, num_samples)
        y_pred_test = np.dot(X_test, weight) + intercept
        epoch_rmse_test = loss_function(y_test, y_pred_test, len(y_test))
        epoch_r2_test = r2_score(y_test, y_pred_test)

        rmse_train_list.append(epoch_rmse_train)
        rmse_test_list.append(epoch_rmse_test)
        r2_test_list.append(epoch_r2_test)

        print(f"Epoch {epoch} - RMSE Train: {epoch_rmse_train:.4f}, RMSE Test: {epoch_rmse_test:.4f}, R² Test: {epoch_r2_test:.4f}")

    # Lưu từng learning rate vào danh sách tổng
    rmse_train_all.append((i, rmse_train_list))
    rmse_test_all.append((i, rmse_test_list))
    r2_test_all.append((i, r2_test_list))



plt.figure(figsize=(14, 6))

# --- RMSE ---
plt.subplot(1, 2, 1)
for lr, rmse_list in rmse_test_all:
    plt.plot(rmse_list, label=f'Test RMSE (lr={lr})')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.title('Test RMSE over Epochs for Different Learning Rates')
plt.legend()
plt.grid(True)

# --- R² ---
plt.subplot(1, 2, 2)
for lr, r2_list in r2_test_all:
    plt.plot(r2_list, label=f'Test R² (lr={lr})')
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title('Test R² over Epochs for Different Learning Rates')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

