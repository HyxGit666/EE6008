import preprocessing as pp
import matplotlib.pyplot as plt
import numpy as np
import models_HEYU
from keras import callbacks
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
    print(accurate)


def accuracy_within15(test_predictions, test_labels):
    count = 0
    for i in range(len(test_predictions)):
        if abs(test_predictions[i] - test_labels[i]) <= 15:
            count += 1

    print("Accuracy_Error within 15 min:", count / len(test_labels))


def accuracy_within20(test_predictions, test_labels):
    count = 0
    for i in range(len(test_predictions)):
        if abs(test_predictions[i] - test_labels[i]) <= 20:
            count += 1

    print("Accuracy_Error within 20 min:", count / len(test_labels))


def mape_calculation(test_prediction, test_labels):
    mape = np.mean(np.abs((test_prediction - test_labels) / test_labels))

    print("Mean Absolute Percentage Error is", float(mape))


def mse_calculation(test_prediction, test_labels):
    mse = np.mean((test_prediction - test_labels) ** 2)

    print("Mean Squared Error is", float(mse))


def mae_calculation(test_prediction, test_labels):
    mae = np.mean(np.abs(test_prediction - test_labels))

    print("Mean Absolute Error is", float(mae))


if __name__ == "__main__":
    model = model_repo_reshape.create_LSTM_model1(pp.inputs.shape[1], pp.inputs.shape[2])
    # model = model_repo_reshape.gru(pp.inputs.shape[1], pp.inputs.shape[2])
    # model = model_repo_reshape.rnn(pp.inputs.shape[1], pp.inputs.shape[2])
    # keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
    early_stopping = callbacks.EarlyStopping(monitor='mse', mode='min', patience=30, restore_best_weights=True)
    history = model.fit(pp.train_samples,
                        np.array(pp.train_labels),
                        epochs=1200,
                        batch_size=32,
                        validation_data=(pp.val_samples, np.array(pp.val_labels)),
                        validation_steps=30,
                        callbacks=[early_stopping])

    train_predicted = model.predict(pp.train_samples)
    val_predicted = model.predict(pp.val_samples)
    test_predicted = model.predict(pp.test_samples)
    print(test_predicted)

    plot_graphs(history, "mse", "LSTM_model1")

    plot_prediction(pp.train_labels, train_predicted, "training data")
    plot_prediction(pp.val_labels, val_predicted, "validation data")
    plot_prediction(pp.test_labels, test_predicted, "test data")

    accuracy_within10(test_predicted, pp.test_labels)
    accuracy_within15(test_predicted, pp.test_labels)
    accuracy_within20(test_predicted, pp.test_labels)
    print("--------------------------------------------------------------")
    print("Accuracy on the training data")
    accuracy_within10(train_predicted, pp.train_labels)

    print("--------------------------------------------------------------")
    # mape_calculation(test_predicted, pp.test_labels)
    mse_calculation(test_predicted, pp.test_labels)
    mae_calculation(test_predicted, pp.test_labels)
