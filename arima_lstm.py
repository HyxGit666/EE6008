# author: bbx
# brief: Pridect flight delay by using LTSM(learning nn)
# last Modified: 
import numpy as np
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt 
from statsmodels.tsa.arima.model import ARIMA
from multiprocessing import Pool
import math

class DataHandler:
    def __init__(self, path, use_random_split = False):
        self.path = path
        self.data = pd.read_csv(path)
        # self.data = self.data.fillna(0.0) # drop nan values
        # self.data = self.data.astype('float32')
        print('Data set size:', self.data.shape)
        self.feature_threshold = 0.0
        self.weather_feature = []
        self.delay_time = [] # target
        self.labels = []
        self.sliding_window_step = 7
        self.use_random_split = use_random_split
        self.split_rate = 0.7
        self.varified_rate = 0.2
        
        self.scalar = 0.0
        self.smoothing_alpha = 0.2
        self.arima_residual = []

    def __feature_encoder(self, value):
        if math.isnan(value): return value
        if value > self.feature_threshold: return 1
        else: return 0

    # single input normalization
    # def __normalize(self, list_data):
    #     max_value = max(list_data)
    #     min_value = min(list_data)
    #     scalar = max_value - min_value
    #     self.scalar = scalar
    #     return list(map(lambda x: x / scalar, list_data)) 

    def __normalize(self, data):
        array_data = np.asarray(data)
        max_value_column = np.max(array_data, axis=0)
        min_value_column = np.min(array_data, axis=0)
        array_scalar = max_value_column - min_value_column
        print('multi normalize:\n', max_value_column, '\n', min_value_column, '\n', array_scalar)
        self.scalar = float(array_scalar[len(data[0]) - 1])
        i = 0
        j = 0
        normalized_data = data
        for i in range(0, len(data)):
            for j in range(0, len(data[0])):
                normalized_data[i][j] = data[i][j] / float(array_scalar[j])
        return normalized_data

    def __arima_forecast(self, data, p, d, q, step):
        tmp_data = [24.0, -2.0, 18.0, 3.0, 134.0]
        model = ARIMA(data, order=(p, d, q))
        # model = ARIMA(tmp_data, order=(p, d, q))
        self.arima_model_fit = model_fit = model.fit()
        self.arima_residual = model_fit.resid.tolist()

    def __check_list_nan(self, list_vector):
        # print(list_vector)
        for element in list_vector:
            if(math.isnan(element)):
                return True
        return False
    
    def __change_WT_series(self, value):
        if(math.isnan(value)): return 0.0
        else: return value
        

    def __read_data(self):
        for idx, row in self.data.iterrows():            
            # weather extract
            # 懒得改了
            wt1 = row['WT01']
            wt1 = self.__change_WT_series(wt1)

            wt2 = row['WT02']
            wt2 = self.__change_WT_series(wt2)

            wt3 = row['WT03']
            wt3 = self.__change_WT_series(wt3)

            wt4 = row['WT04']
            wt4 = self.__change_WT_series(wt4)

            wt5 = row['WT05']
            wt5 = self.__change_WT_series(wt5)

            wt6 = row['WT06']
            wt6 = self.__change_WT_series(wt6)

            wt8 = row['WT08']
            wt8 = self.__change_WT_series(wt8)

            wt9 = row['WT09']
            wt9 = self.__change_WT_series(wt9)

            wsf2 = row['WSF2']
            wsf5 = row['WSF5']
            wdf2 = row['WDF2']
            wdf5 = row['WDF5']
            tmin = row['TMIN']
            tmax = row['TMAX']
            tavg = row['TAVG']

            # change specific feature
            prcp = row['PRCP']
            prcp = self.__feature_encoder(prcp)
            snow = row['SNOW']
            snow = self.__feature_encoder(snow)
            
            tmp_weather_feature = [wt1, wt2, wt3, wt4, wt5, wt6, wt8, wt9, wsf2, wsf5, wdf2, wdf5, tmin, tmax, tavg, prcp, snow]
            if(self.__check_list_nan(tmp_weather_feature)): 
                continue
            self.weather_feature.append(tmp_weather_feature)

            # label extract
            delay_time = row['ARR_DELAY']
            # print('Check delay_time:', delay_time)
            if delay_time > 50.0 or math.isnan(delay_time):
                continue
            delay_time = float(delay_time)
            tmp_delay_time = [delay_time]
            self.delay_time.append(delay_time)
            
            # date extract -- useless
            # date = pd.to_datetime(row['FL_DATE'])
            # self.data['FL_DATE'] = pd.to_datetime(self.data['FL_DATE'])
        
            # splicing data
            self.labels.append(tmp_weather_feature + tmp_delay_time)
            # print('delay_time', delay_time, 'single labels', (tmp_weather_feature + tmp_delay_time))
        
        print('Checking data valid size', len(self.labels))
        # normalize specific data
        self.labels = self.__normalize(self.labels)
        # self.labels = self.__exponential_smoothing(self.labels, self.smoothing_alpha)
        # print('normalize', self.labels)
        # print(len(self.labels), len(self.labels[0]))

    def __create_dataset(self):
        data_x, data_y = [], []
        for i in range(len(self.labels) - self.sliding_window_step):
            x = self.labels[i:(i + self.sliding_window_step)]
            data_x.append(x)
            # data_y.append(self.labels[i + self.sliding_window_step]) # single label
            data_y.append(self.labels[i + self.sliding_window_step][len(self.labels[0]) - 1])
        # print('data_y', len(data_y))
        self.__arima_forecast(data_y, 4, 1, 4, self.sliding_window_step)
        return np.array(data_x), np.array(data_y)
    
    def __split_dataset(self):
        data_x, data_y = self.__create_dataset()
        # step 1
        train_size = int(len(data_x) * self.split_rate)
        var_index = int(len(data_x) * (self.split_rate + self.varified_rate))
        print('Checking index: train_idx', train_size, 'var_idx', var_index)

        # step 2
        train_x = data_x[:train_size]
        # train_y = data_y[:train_size]
        train_y = np.array(self.arima_residual[:train_size]) # using ARIMA model
        var_x = data_x[train_size:var_index]
        var_y = data_y[train_size:var_index]
        test_x = data_x[var_index:]
        test_y = data_y[var_index:]
        self.test_y = test_y
        # self.var_y = var_y
        # print('Checking type train_x', type(train_x), 'test_y', type(test_y))
        print('Checking data train_y', train_y[-1], f'var_y[{var_y[0]}, {var_y[-1]}]', 'test_y', test_y[0])

        # step 3
        # reshape the dataset
        train_x = train_x.reshape(-1, len(self.labels[0]), self.sliding_window_step)
        train_y = train_y.reshape(-1, 1, 1)
        var_x = var_x.reshape(-1, len(self.labels[0]), self.sliding_window_step)
        var_y = var_y.reshape(-1, 1, 1)
        test_x = test_x.reshape(-1, len(self.labels[0]), self.sliding_window_step)
        # print('check shape train_x, train_y, test_x', train_x.shape, train_y.shape, test_x.shape)
        print(f'Checking shape train_x:{train_x.shape}, train_y:{train_y.shape}, var_x:{var_x.shape}, test_x:{test_x.shape}')

        # transform to torch tensor
        train_x_torch = torch.from_numpy(train_x).float()
        train_y_torch = torch.from_numpy(train_y).float()
        var_x_torch = torch.from_numpy(var_x).float()
        var_y_torch = torch.from_numpy(var_y).float()
        test_x_torch = torch.from_numpy(test_x).float()
        # print('split dataset')
        # print(train_x_torch.shape)
        return train_x_torch, train_y_torch, var_x_torch, var_y_torch, test_x_torch
    
    def data_processing(self):
        self.__read_data()
        train_x, train_y, var_x, var_y, test_x = self.__split_dataset()
        forecast = self.arima_model_fit.forecast(steps = len(test_x))
        # self.arima_forecast = forecast.tolist()
        print('train_y shape', train_y.shape, 'train_x shape', train_x.shape, 'test_x_shape', test_x.shape)
        # print('forecast', forecast)
        return train_x, train_y, test_x, var_x, var_y, forecast

# LSTM网络
class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2): # 构造函数
        #inpu_size 是输入的样本的特征维度， hidden_size 是LSTM层的神经元个数，
        #output_size是输出的特征维度
        super(LSTMModule, self).__init__()# super用于多层继承使用，必须要有的操作
        # 两层LSTM网络
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        #把上一层总共hidden_size个的神经元的输出向量作为输入向量，然后回归到output_size维度的输出向量中
        self.reg = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)# 单个下划线表示不在意的变量，这里是LSTM网络输出的两个隐藏层状态
        s, b, h = x.shape
        x = x.view(s * b, h)
        x = self.reg(x)
        # print(x.shape)
        x = x.view(s, b, -1)# 使用-1表示第三个维度自动根据原来的shape 和已经定了的s,b来确定
        x = x[:, -1:, :]
        return x

# 早停机制
class EarlyStopping():
    def __init__(self, patience = 7, min_delta = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
    
    def __call__(self, val_loss):
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

class LSTMHandler():
    def __init__(self):
        self.dh = DataHandler(file_path)
        self.train_x, self.train_y, self.test_x, self.varified_x, self.varified_y, self.arima_forecast = self.dh.data_processing()
        self.net = LSTMModule(self.dh.sliding_window_step, 4)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-2)
        self.train_epoch = 1600
        self.early_stop = EarlyStopping(patience = 50, min_delta = 0.01)
    
    def train_process(self):
        # train_x, train_y, test_x = self.dh.data_processing()
        # print(train_x.shape, train_y.shape)
        # print(train_x.dtype, train_y.dtype)
        # print(train_x[0], train_y[0])
        e_fig = []
        loss_fig = []
        for epoch in range(self.train_epoch):
            # break
            var_x = self.train_x
            var_y = self.train_y
            out = self.net(var_x)
            loss = self.criterion(out, var_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # out = out[:, -1:, :]
            # print(out.shape)

            # 早停机制
            varified_x = self.varified_x
            varified_y = self.varified_y
            varified_out = self.net(varified_x)
            varified_loss = self.criterion(varified_out, varified_y)
            # self.early_stop(varified_loss.item())
            # if(self.early_stop.early_stop):
            #     print(f'Trigger early stop: final epoch:{epoch}')
            #     break


            e_fig.append(epoch)
            loss_fig.append(loss.item())
            if (epoch + 1) % 100 == 0:
                print('Epoch: {}, Loss:{:.5f}'.format(epoch + 1, loss.item()))
                print(f'Check varified loss {varified_loss.item()}') # 早停机制

        torch.save(self.net.state_dict(), model_save_path)
        # draw figure
        plt.figure()
        plt.plot(e_fig, loss_fig)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('train_loss')
        plt.savefig(train_loss_fig_path)
        plt.close()
        
    def test_process(self):
        self.net.load_state_dict(torch.load(model_save_path, map_location='cpu'))

        # train pred
        var_train_data = self.train_x
        pred_train_data = self.net(var_train_data)
        pred_train_data = pred_train_data.cpu().view(-1).data.numpy()
        pred_train_data = pred_train_data * self.dh.scalar
        original_train_y_data = self.train_y.cpu().view(-1).data.numpy() * self.dh.scalar

        # draw figure: train picture
        plt.figure()
        plt.plot(original_train_y_data, 'b', label='real')
        plt.plot(pred_train_data, 'r', label='prediction')
        # plt.plot(arima_test_y, 'r', label='arima_prediction')
        plt.legend(loc='best')
        plt.title('train prediction')
        # plt.show()
        plt.savefig(train_varified_fig_path)
        plt.close()

        # var pred
        var_varified_data = self.varified_x
        pred_varified_data = self.net(var_varified_data)
        pred_varified_data = pred_varified_data.cpu().view(-1).data.numpy()
        pred_varified_data = pred_varified_data * self.dh.scalar
        original_varified_data = self.varified_y.cpu().view(-1).data.numpy() * self.dh.scalar

        # draw figure: varified picture
        plt.figure()
        plt.plot(original_varified_data, 'b', label='real')
        plt.plot(pred_varified_data, 'r', label='prediction')
        # plt.plot(arima_test_y, 'r', label='arima_prediction')
        plt.legend(loc='best')
        plt.title('varified prediction')
        # plt.show()
        plt.savefig(varified_varified_fig_path)
        plt.close()


        # test pred
        var_data = self.test_x
        print('Check test_x data type', type(var_data))
        pred_test = self.net(var_data)
        pred_test = pred_test.cpu().view(-1).data.numpy()
        original_test_y = self.dh.test_y * self.dh.scalar
        print('Check original database test y', type(self.dh.test_y))
        pred_test_y = pred_test * self.dh.scalar

        # arima
        arima_test_y = self.arima_forecast * self.dh.scalar
        # print(arima_test_y)


        # draw figure: test picture
        plt.figure()
        plt.plot(original_test_y, 'b', label='real')
        plt.plot(pred_test_y, 'r', label='prediction')
        # plt.plot(arima_test_y, 'r', label='arima_prediction')
        plt.legend(loc='best')
        plt.title('test prediction')
        # plt.show()
        plt.savefig(test_varified_fig_path)
        plt.close()
        
        # calculate test prediction        
        true_data = original_test_y
        true_data = np.array(true_data)
        true_data = np.squeeze(true_data)  # 从二维变成一维
        # print(true_data.shape)
        # print(pred_test_y.shape)
        # print(true_data, pred_test_y)
        # MSE = true_data - pred_test_y
        # print(MSE)
        # MSE = MSE * MSE
        # MSE_loss = sum(MSE) / len(MSE)
        # print(MSE_loss)
        
        # 计算MAE
        mae = np.mean(np.abs(true_data - pred_test_y))
        print("MAE:", mae)

        # 计算RMSE
        rmse = np.sqrt(np.mean((true_data - pred_test_y) ** 2))
        print("RMSE:", rmse)

        self.calculate_delay_acc(pred_test_y, true_data)

        # 计算MAPE 不适合用于评估这个模型
        epsilon = 1e-8
        mape = np.mean(np.abs((true_data - pred_test_y) / (true_data + epsilon))) * 100
        print("MAPE:", mape, "%")
        
    def calculate_delay_acc(self, pred_data, true_data):
        pred_data.tolist()
        true_data.tolist()
        print('Calculate: check type & len', type(pred_data), type(true_data), len(pred_data), len(true_data))
        minus = []
        tmp_minus = 0.0
        for i in range(0, len(pred_data)):
            tmp_minus = abs(pred_data[i] - true_data[i])
            minus.append(tmp_minus)
        
        valid_10_cnt = 0
        valid_15_cnt = 0
        valid_20_cnt = 0
        
        for j in range(0, len(minus)):
            if(minus[j] < 10): valid_10_cnt += 1
            elif(minus[j] < 15): valid_15_cnt += 1
            elif(minus[j] < 20): valid_20_cnt += 1
        final_15_cnt = valid_15_cnt + valid_10_cnt
        final_20_cnt = valid_10_cnt + valid_15_cnt + valid_20_cnt
        print('Calculate valid: 10, 15, 20:', valid_10_cnt, final_15_cnt, final_20_cnt)
        
        valid_10_per = valid_10_cnt / len(minus) * 100
        valid_15_per = final_15_cnt / len(minus) * 100
        valid_20_per = final_20_cnt / len(minus) * 100
        print('Calculate valid: 10, 15, 20(percentage):', valid_10_per, '%', valid_15_per, '%', valid_20_per, '%')
        
if __name__ == '__main__':
    file_path = 'NewDataBack.csv'
    model_save_path = 'flight_delay_model.pkl'
    train_loss_fig_path = 'train_loss.png'
    train_varified_fig_path = 'train_varified.png'
    test_varified_fig_path = 'test_varified.png'
    varified_varified_fig_path = 'varified_varified.png'
    
    lh = LSTMHandler()
    lh.train_process() # do training
    lh.test_process()