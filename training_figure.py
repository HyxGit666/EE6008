import preprocessing as pp
import matplotlib.pyplot as plt
import numpy as np
import model_repo
from keras import callbacks
from keras.callbacks import History

import keras


def plot_graphs(model_history, metric, model_name):
    plt.plot(model_history.history[metric])
    plt.plot(model_history.history["val_"+metric], '')
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.title("Training Process of "+model_name)
    plt.legend([metric, 'val_'+metric])
    plt.show()


def plot_prediction(actual, predict, dataset):
    plt.plot(actual, label='target')
    plt.plot(predict, label='prediction')
    plt.xlabel("days")
    plt.ylabel("Results")
    plt.title("Predictions on " + dataset)
    plt.legend()
    plt.show()


def accuracy_within10(test_predictions, test_labels):
    count = 0
    accurate = []
    for i in range(len(test_predictions)):
        if abs(test_predictions[i] - test_labels[i]) <= 10:
            count += 1
            accurate.append([i, abs(test_predictions[i] - test_labels[i])])

    print("Accuracy_Error within 10 min:", count / len(test_labels))
    return count / len(test_labels)
    #print(accurate)


def accuracy_within15(test_predictions, test_labels):
    count = 0
    for i in range(len(test_predictions)):
        if abs(test_predictions[i] - test_labels[i]) <= 15:
            count += 1

    print("Accuracy_Error within 15 min:", count / len(test_labels))
    return count / len(test_labels)


def accuracy_within20(test_predictions, test_labels):
    count = 0
    for i in range(len(test_predictions)):
        if abs(test_predictions[i] - test_labels[i]) <= 20:
            count += 1

    print("Accuracy_Error within 20 min:", count / len(test_labels))
    return count / len(test_labels)


def mape_calculation(test_prediction, test_labels):
    mape = np.mean(np.abs((test_prediction - test_labels) / test_labels))

    print("Mean Absolute Percentage Error is", float(mape))
    return float(mape)


def mse_calculation(test_prediction, test_labels):
    mse = np.mean((test_prediction - test_labels) ** 2)

    print("Mean Squared Error is", float(mse))
    return float(mse)


def mae_calculation(test_prediction, test_labels):
    mae = np.mean(np.abs(test_prediction - test_labels))

    print("Mean Absolute Error is", float(mae))
    return float(mae)

def rmse_calculation(test_prediction, test_labels):
    mse = np.mean((test_prediction - test_labels) ** 2)
    rmse = np.sqrt(mse)

    print("Root Mean Squared Error is", float(rmse))
    return float(rmse)

if __name__ == "__main__":
    GRU = model_repo.gru(pp.inputs.shape[1], pp.inputs.shape[2])
    DCNN=model_repo.create_DCNN_model(pp.inputs.shape[1], pp.inputs.shape[2])
    LSTM_att = model_repo.create_LSTM_model4(pp.inputs.shape[1], pp.inputs.shape[2])
    CNN_LSTM = model_repo.create_cnn_lstm_model(pp.inputs.shape[1], pp.inputs.shape[2])
    RNN=model_repo.rnn(pp.inputs.shape[1], pp.inputs.shape[2])
    LSTM_ED = model_repo.create_LSTM_model_ED(pp.inputs.shape[1], pp.inputs.shape[2])
    LSTM = model_repo.create_LSTM_model_ED(pp.inputs.shape[1], pp.inputs.shape[2])

    all_models = {'DCNN': DCNN,'LSTM_att': LSTM_att, 'CNN_LSTM': CNN_LSTM}


    all_models_history = {}
    accuracy_10=set()
    accuracy_15=set()
    accuracy_20=set()
    mse_1=set()
    mae_1=set()
    rmse_1=set()

    # 模型训练过程
    for model_name, model in all_models.items():  # 假设 all_models 是一个包含所有模型的字典，键是模型名称，值是模型对象

        history = History()

        # gru = model_repo_reshape.gru(pp.inputs.shape[1], pp.inputs.shape[2])
        # keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
        early_stopping = callbacks.EarlyStopping(monitor='mse', mode='min', patience=30, restore_best_weights=True)
        print(pp.train_samples.shape)
        all_models_history[model_name] = model.fit(pp.train_samples,
                            np.array(pp.train_labels),
                            epochs=1500,
                            batch_size=32,
                            validation_data=(pp.val_samples, np.array(pp.val_labels)),
                            validation_steps=30,
                            callbacks=[early_stopping])

        train_predicted = model.predict(pp.train_samples)
        val_predicted = model.predict(pp.val_samples)
        test_predicted = model.predict(pp.test_samples)

        plot_prediction(pp.train_labels, train_predicted, "training data")
        plot_prediction(pp.val_labels, val_predicted, "validation data")
        plot_prediction(pp.test_labels, test_predicted, "test data")

        accuracy_10.add(accuracy_within10(test_predicted, pp.test_labels))
        accuracy_15.add(accuracy_within15(test_predicted, pp.test_labels))
        accuracy_20.add(accuracy_within20(test_predicted, pp.test_labels))
        print("--------------------------------------------------------------")
        print("Accuracy on the training data")
        accuracy_within10(train_predicted, pp.train_labels)

        print("--------------------------------------------------------------")
        # mape_calculation(test_predicted, pp.test_labels)
        mse_1.add(mse_calculation(test_predicted, pp.test_labels))
        mae_1.add(mae_calculation(test_predicted, pp.test_labels))
        rmse_1.add(rmse_calculation(test_predicted, pp.test_labels))

    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 遍历每个模型的训练历史记录
    for model_name, model_history in all_models_history.items():
        # 提取损失记录
        loss = model_history.history['loss']
        # 绘制损失曲线
        plt.plot(loss, label=model_name)

    # 添加图例、标题和轴标签
    plt.legend()  # 添加图例
    plt.title('Training Loss of Models')  # 设置标题
    plt.xlabel('Epoch')  # 设置 x 轴标签
    plt.ylabel('Loss')  # 设置 y 轴标签

    plt.grid(True)
    plt.show()
    plt.clf()


    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 遍历所有模型的训练历史记录
    for model_name, history in all_models_history.items():
        mae = history.history['mae']  # 提取 MAE
        plt.plot(mae, label=f'{model_name} - MAE')

    # 添加图例、标题和轴标签
    plt.legend()  # 添加图例
    plt.title('MAE of Models')  # 设置标题
    plt.xlabel('Epoch')  # 设置 x 轴标签
    plt.ylabel('Value')  # 设置 y 轴标签

    # 显示网格线
    plt.grid(True)
    plt.show()
    plt.clf()

    plt.figure(figsize=(10, 6))  # 设置图形大小

    # 遍历所有模型的训练历史记录
    for model_name, history in all_models_history.items():
        mse = history.history['mse']  # 提取 MSE
        plt.plot(mse, label=f'{model_name} - MSE')

    # 添加图例、标题和轴标签
    plt.legend()  # 添加图例
    plt.title('MSE of Models')  # 设置标题
    plt.xlabel('Epoch')  # 设置 x 轴标签
    plt.ylabel('Value')  # 设置 y 轴标签

    # 显示网格线
    plt.grid(True)
    plt.show()
    plt.clf()

print(accuracy_10)
print(accuracy_15)
print(accuracy_20)
print(mse_1)
print(mae_1)
print(rmse_1)
