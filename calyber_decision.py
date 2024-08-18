import torch
import torch.nn as nn

# class SinActivation(torch.nn.Module):
#     def forward(self, x):
#         return torch.sin(x)

class decision(nn.Module):
    def __init__(self, max_queue_length, queue_features, price_options_size):
        super(decision, self).__init__()
        self.queue_features = queue_features
        self.dense1 = nn.Linear(self.queue_features, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.dense2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dense_6_1 = nn.Linear(64, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dense_7_1 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dense_7_2 = nn.Linear(128, 256)
        self.bn5 = nn.BatchNorm1d(256)
        # self.dense3 = nn.Linear(256, 256)
        # self.bn6 = nn.BatchNorm1d(256)
        # self.dense_8_1 = nn.Linear(256, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dense_8_2 = nn.Linear(256, 256)
        # self.bn8 = nn.BatchNorm1d(256)

        self.queue_lstm1 = nn.LSTM(input_size = queue_features, 
                                   hidden_size = 64, num_layers = 1,
                                   batch_first = True)
        self.queue_lstm2 = nn.LSTM(input_size = 64, 
                                   hidden_size = 128, num_layers = 1, 
                                   batch_first=True, dropout = 0.25)
        self.queue_lstm3 = nn.LSTM(input_size = 128, 
                                   hidden_size = 256, num_layers = 1, 
                                   batch_first = True)
        self.queue_lstm4 = nn.LSTM(input_size = 256, 
                                   hidden_size = 256, num_layers = 1, 
                                   batch_first = True)
        
        self.fc1 = nn.Linear(512, 256)
        self.bn9 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.bn10 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.bn11 = nn.BatchNorm1d(256)
        self.fc_8_1 = nn.Linear(256, 128)
        self.bn12 = nn.BatchNorm1d(128)
        # self.fc_8_2 = nn.Linear(128, 128)
        # self.bn13 = nn.BatchNorm1d(128)
        # self.fc_8_3 = nn.Linear(128, 128)
        # self.bn14 = nn.BatchNorm1d(128)

        self.match_fc_1 = nn.Linear(128, 128)
        self.bn15 = nn.BatchNorm1d(128)
        self.match_fc_2_1 = nn.Linear(128, 128)
        self.bn16 = nn.BatchNorm1d(128)
        self.match_fc_5_1 = nn.Linear(128, 64)
        self.bn17 = nn.BatchNorm1d(64)
        self.match_fc_2 = nn.Linear(64, 32)
        self.bn18 = nn.BatchNorm1d(32)
        self.match_output = nn.Linear(32, max_queue_length + 1)
        self.match_values_fc = nn.Linear(32, 16)
        self.bn19 = nn.BatchNorm1d(16)
        self.match_values = nn.Linear(16, 1)
        # self.match_linear = nn.Linear(max_queue_length + 1, max_queue_length + 1)

        self.price_fc_1 = nn.Linear(128, 64)
        self.bn20 = nn.BatchNorm1d(64)
        self.price_fc_2_1 = nn.Linear(64, 64)
        self.bn21 = nn.BatchNorm1d(64)
        self.price_fc_3_1 = nn.Linear(64, 64)
        self.bn22 = nn.BatchNorm1d(64)
        self.price_fc_4_1 = nn.Linear(64, 32)
        self.bn23 = nn.BatchNorm1d(32)
        self.price_output = nn.Linear(32, price_options_size)
        self.price_values = nn.Linear(32, 1)
        # self.price_liner = nn.Linear(price_options_size, price_options_size)

        self.dropout = nn.Dropout(p = 0.2)
        self.dropout2 = nn.Dropout(p = 0.1)

        self.match2price_fc_1 = nn.Linear(max_queue_length + 1, 64)
        self.bn24 = nn.BatchNorm1d(64)
        self.match2price_fc_2 = nn.Linear(64, 64)
        self.bn25 = nn.BatchNorm1d(64)
        self.match2price_fc_3 = nn.Linear(64, price_options_size)
        self.match_final_1 = nn.Linear(max_queue_length + 1, max_queue_length + 1)
        self.bn28 = nn.BatchNorm1d(max_queue_length + 1)
        self.match_final_2 = nn.Linear(max_queue_length + 1, max_queue_length + 1)

        self.price2match_fc_1 = nn.Linear(price_options_size, 64)
        self.bn26 = nn.BatchNorm1d(64)
        self.price2match_fc_2 = nn.Linear(64, 64)
        self.bn27 = nn.BatchNorm1d(64)
        self.price2match_fc_3 = nn.Linear(64, max_queue_length + 1)
        self.price_final_1 = nn.Linear(price_options_size, price_options_size)
        self.bn29 = nn.BatchNorm1d(price_options_size)
        self.price_final_2 = nn.Linear(price_options_size, price_options_size)

        # self.softmax = nn.Softmax(dim = 1)
        # self.sin = SinActivation()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)


    def forward(self, current_rider_input, queue_input):
        current_rider = torch.max_pool1d(current_rider_input.permute(0, 2, 1), kernel_size=current_rider_input.size(1)).squeeze(2)
        rider = nn.ReLU()(self.bn1(self.dense1(current_rider)))
        rider = self.dropout(nn.ReLU()(self.bn2(self.dense2(rider))))
        rider = nn.ReLU()(self.bn3(self.dense_6_1(rider)))
        rider = self.dropout(nn.ReLU()(self.bn4(self.dense_7_1(rider))))
        rider = nn.ReLU()(self.bn5(self.dense_7_2(rider)))
        # rider = self.dropout2(nn.ReLU()(self.bn6(self.dense3(rider))))
        # rider = self.dropout2(nn.ReLU()(self.bn7(self.dense_8_1(rider))))
        # rider = self.dropout2(nn.ReLU()(self.bn8(self.dense_8_2(rider))))

        queue, _ = self.queue_lstm1(queue_input)
        queue, _ = self.queue_lstm2(queue)
        queue, _ = self.queue_lstm3(queue)
        queue, _ = self.queue_lstm4(queue)
        queue = queue[:, -1, :]  # Taking the last output of the LSTM

        concat = torch.cat((rider, queue), dim=1)

        fc = nn.ReLU()(self.bn9(self.fc1(concat)))
        fc = self.dropout(nn.ReLU()(self.bn10(self.fc2(fc))))
        fc = nn.ReLU()(self.bn11(self.fc3(fc)))
        fc = self.dropout2(nn.ReLU()(self.bn12((self.fc_8_1(fc)))))
        # fc = self.dropout2(nn.ReLU()(self.bn13(self.fc_8_2(fc))))
        # fc = self.dropout2(nn.ReLU()(self.bn14(self.fc_8_3(fc))))

        match = nn.ReLU()(self.bn15(self.match_fc_1(fc)))
        match = nn.ReLU()(self.bn16(self.match_fc_2_1(match)))
        match = self.dropout(nn.ReLU()(self.bn17(self.match_fc_5_1(match))))
        match = nn.ReLU()(self.bn18(self.match_fc_2(match)))
        match_output_intermediate = self.match_output(match)
        # match_output = self.sin(match_output)
        match_values = self.dropout(nn.ReLU()(self.bn19(self.match_values_fc(match))))
        match_values = self.match_values(match_values)
        
        # match_output = self.softmax(match_output)
        # match_output = self.match_linear(match_output)

        price = nn.ReLU()(self.bn20(self.price_fc_1(fc)))
        price = self.dropout2(nn.ReLU()(self.bn21(self.price_fc_2_1(price))))
        price = self.dropout2(nn.ReLU()(self.bn22(self.price_fc_3_1(price))))
        price = self.dropout2(nn.ReLU()(self.bn23(self.price_fc_4_1(price))))
        price_output_intermediate = self.price_output(price)

        price_values = self.price_values(price)


        price_output = nn.ReLU()(self.bn24(self.match2price_fc_1(match_output_intermediate)))
        price_output = nn.ReLU()(self.bn25(self.match2price_fc_2(price_output)))
        price_output = nn.ReLU()((self.match2price_fc_3(price_output)))
        price_output = price_output + price_output_intermediate
        price_output = nn.ReLU()(self.bn29(self.price_final_1(price_output)))
        price_output = self.price_final_2(price_output)

        match_output = nn.ReLU()(self.bn26(self.price2match_fc_1(price_output_intermediate)))
        match_output = nn.ReLU()(self.bn27(self.price2match_fc_2(match_output)))
        match_output = nn.ReLU()((self.price2match_fc_3(match_output)))
        match_output = match_output + match_output_intermediate
        match_output = nn.ReLU()(self.bn28(self.match_final_1(match_output)))
        match_output = self.match_final_2(match_output)

        match_output = match_output - torch.max(match_output, dim = 1, keepdim = True)[0]  # Subtracting the max value to avoid overflow
        match_output = match_output + match_values
        price_output = price_output - torch.max(price_output, dim=1, keepdim=True)[0]
        price_output = price_output + price_values
        # price_output = self.softmax(price_output)
        # price_output = self.price_liner(price_output)
        return price_output, match_output
