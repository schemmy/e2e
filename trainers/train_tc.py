# -*- coding: utf-8 -*-
# @Author: chenxinma
# @Date:   2018-10-01 16:04:49
# @Last Modified by:   chenxinma
# @Last Modified at:   2018-10-29 13:31:32

from models.model import *
from models.loss import *
from data_loader.data_loader import *
import os
from subprocess import call
from tensorboardX import SummaryWriter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# import seaborn as sns

class Trainer(object):

    def __init__(self, model, args, device):
        
        self.model = model

        self.batch_size = args.bs
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.model_path = args.model_path
        self.log_path = args.log_path
        self.num_epochs = args.num_epochs
        self.model_to_load = args.model_to_load
        self.train_check = args.train_check
        self.log_step = args.log_step
        self.save_step = args.save_step
        self.learning_rate = args.learning_rate
        self.test_sku = args.test_sku
        self.b_value = args.b_value

        self.device = device


    def train_v5_tc(self):

        # self.model = End2End_v5_tc(self.device).to(self.device)

        if os.path.exists('%s/%s' %(self.log_path, self.model_name)):
            call(['rm', '-r', '%s/%s' %(self.log_path, self.model_name)])
        call(['mkdir', '%s/%s' %(self.log_path, self.model_name)])

        train_writer = SummaryWriter('%s/%s/train/' %(self.log_path, self.model_name))
        test_writer = SummaryWriter('%s/%s/test/' %(self.log_path, self.model_name))

        # enc_X = torch.Tensor(128,rnn_hist_long)
        # enc_y = torch.Tensor(128,rnn_hist_long)
        # dec_X = torch.Tensor(128,31,num_quantiles)
        # x_vlt = torch.Tensor(128,len(VLT_FEA))
        # x_cat = torch.Tensor(128,len(CAT_FEA_HOT))
        # x_oth = torch.Tensor(128,len(MORE_FEA))
        # x_is = torch.Tensor(128,len(IS_FEA))
        # rubish = torch.Tensor(128,30)
        # writer.add_graph(model, (enc_X, enc_y, dec_X, x_vlt, x_cat, x_oth, x_is), verbose=False)


        # Train the Models
        # total_step = len(data_loader)

        data_loader, test_loader = get_loader(self.batch_size, self.device, self.model_name, b=self.b_value)
        
        e2e_loss = E2E_loss(self.device)

        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        curr_epoch = 0

        if self.train_check != 'None':
            # model.load_state_dict(torch.load('../logs/torch/e2e_v6_30.pkl'))
            checkpoint = torch.load(self.model_path+self.train_check, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            curr_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print('load model!')

        print(p1,p2,p3,p4,p5,p6,p7,p8)
        for epoch in range(curr_epoch, self.num_epochs):
            train_loss0, train_loss1 = 0, 0
            for i, X in enumerate(data_loader):
                out, out_vlt, out_sf = self.model(X[:,:p1], X[:,p1:p2], X[:,p2:p3], X[:,p3:p4], X[:,p4:p5])
                batch_loss1, batch_loss0 = e2e_loss(out_vlt, X[:,p6:p7], out_sf, X[:,p7:p8], out, X[:,p5:p6])

                # if epoch>0:
                optimizer.zero_grad()
                batch_loss0.backward()
                optimizer.step()

                train_loss1 += batch_loss1.item()
                train_loss0 += batch_loss0.item()

                if (i+1) % self.log_step == 0:

                    test_loss0, test_loss1 = 0, 0
                    for _, X in enumerate(test_loader):
                        out, out_vlt, out_sf = self.model(X[:,:p1], X[:,p1:p2], X[:,p2:p3], X[:,p3:p4], X[:,p4:p5])
                        loss1, loss0 = e2e_loss(out_vlt, X[:,p6:p7], out_sf, X[:,p7:p8], out, X[:,p5:p6])
                        test_loss1 += loss1.item()
                        test_loss0 += loss0.item()

                    print('Epoch %d pct %.3f, loss_1 %.5f, loss_ttl %.5f, test_loss_1 %.5f, test_loss_ttl %.5f' % 
                                    (epoch,(i+1)/len(data_loader),train_loss1/self.log_step,train_loss0/self.log_step,
                                        test_loss1/len(test_loader), test_loss0/len(test_loader)))

                    for name, param in self.model.named_parameters():
                        train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch*len(data_loader)+i)
                    train_writer.add_scalar('train_loss_out',train_loss1/self.log_step, epoch*len(data_loader)+i)
                    train_writer.add_scalar('train_loss_ttl',train_loss0/self.log_step, epoch*len(data_loader)+i)
                    test_writer.add_scalar('test_loss_out', test_loss1/len(test_loader), epoch*len(data_loader)+i)
                    test_writer.add_scalar('test_loss_ttl', test_loss0/len(test_loader), epoch*len(data_loader)+i)
                    train_writer.export_scalars_to_json('%s/%s/train/scalars_train.json' %(self.log_path, self.model_name))
                    test_writer.export_scalars_to_json('%s/%s/test/scalars_test.json' %(self.log_path, self.model_name))
                    train_loss0, train_loss1 = 0, 0

            if (epoch+1) % self.save_step == 0:
                # torch.save(model.state_dict(), os.path.join('../logs/torch/', 'e2e_v6_%d.pkl' %(epoch+1)))
                torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': batch_loss1,
                            }, os.path.join('../logs/torch/', 'e2e_%s_%d.pkl' %(self.model_name,epoch+1)))

        train_writer.close()
        test_writer.close()



    def eval_v5_tc(self):

        # self.model = End2End_v5_tc(self.device).to(self.device)
        _, test_loader = get_loader(self.batch_size, self.device, self.model_name, eval=1, test_sku=self.test_sku, b=self.b_value)

        # model.load_state_dict(torch.load('../logs/torch/e2e_v6_20.pkl'))
        checkpoint = torch.load(self.model_path+self.model_to_load, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('load model!')
        
        for _, X in enumerate(test_loader):
            out, out_vlt, out_sf = self.model(X[:,:p1], X[:,p1:p2], X[:,p2:p3], X[:,p3:p4], X[:,p4:p5])

        pd_scaler = pd.read_csv('../data/1320_feature/scaler.csv')
        df_idx = pd.read_csv('../logs/torch/pred_sku.csv')
        # if self.b_value != None:
            # LABEL = ['demand_RV_%i' %self.b_value]
        out = out.detach().cpu().numpy() / pd_scaler.loc[1, LABEL[0]] + pd_scaler.loc[0, LABEL[0]]

        out_sf = out_sf.detach().cpu().numpy() / pd_scaler.loc[1, LABEL_sf[0]] + pd_scaler.loc[0, LABEL_sf[0]]
        out_sf = np.exp(out_sf) - 1
        out_vlt = out_vlt.detach().cpu().numpy() / pd_scaler.loc[1, LABEL_vlt[0]] + pd_scaler.loc[0, LABEL_vlt[0]]
        pred = pd.DataFrame(out, columns=['E2E_MLP_pred']) 
        pred_sf = pd.DataFrame(out_sf, columns=['E2E_NN_SF_mean_pred'])
        pred_vlt = pd.DataFrame(out_vlt, columns=['E2E_NN_vlt_pred'])
        pred = pd.concat([df_idx, pred, pred_sf, pred_vlt], axis=1)
        pred.to_csv('%s/pred_v5.csv' %self.model_path, index=False)



    def sens_v5_tc(self):

        _, test_loader = get_loader(self.batch_size, self.device, self.model_name, eval=1, 
                                    test_sku=self.test_sku, b=self.b_value)

        checkpoint = torch.load(self.model_path+self.model_to_load, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('load model!')
        pd_scaler = pd.read_csv('../data/1320_feature/scaler.csv')
       
        for _, X in enumerate(test_loader):
            X_clone = torch.mean(X, dim=0).unsqueeze(0)
        tune = np.arange(0, 1.05, 0.05)

        for i in range(X_clone.shape[1] - 2):
            y_values = []
            x_values = []
            for t in tune:
                X = X_clone.clone()
                X[0, i] = X[0, i] + t
                if X[0, i] >= 0:
                    out, out_vlt, out_sf = self.model(X[:,:p1], X[:,p1:p2], X[:,p2:p3], \
                                                        X[:,p3:p4], X[:,p4:p5])
                    x_values.append(X[0, i].detach().cpu().numpy())
                    y_values.append(out[0].detach().cpu().numpy())

            x_values = x_values / pd_scaler.iloc[1, i+1] + pd_scaler.iloc[0, i+1]
            y_values = y_values / pd_scaler.loc[1, LABEL[0]] + pd_scaler.loc[0, LABEL[0]] 

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(x_values, y_values);
            ax.set_xlabel(pd_scaler.columns[i+1])
            ax.set_ylabel('Replenishment prediction')
            ax.set_ylim((0, 1000))
            ax.set_title('Sensitivity analysis')
            plt.savefig('../figures/sens/sens_%i.png' %i);
            plt.close()

        

    def sens_naive(self):

        _, test_loader = get_loader(self.batch_size, self.device, self.model_name, eval=1, 
                                    test_sku=self.test_sku, b=self.b_value)

        checkpoint = torch.load(self.model_path+self.model_to_load, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('load model!')
        pd_scaler = pd.read_csv('../data/1320_feature/scaler.csv')
       
        for _, X in enumerate(test_loader):
            X_clone = torch.mean(X, dim=0).unsqueeze(0)

        y_values = np.zeros((16, p5))
        for i in range(p5):
            X = X_clone.clone()
            X[0, i] = 0
            out_, out_vlt, out_sf, l4_,l3_ = self.model(X[:,:p1], X[:,p1:p2], X[:,p2:p3], \
                                                    X[:,p3:p4], X[:,p4:p5])
            X[0, i] = 1
            out, out_vlt, out_sf, l4,l3 = self.model(X[:,:p1], X[:,p1:p2], X[:,p2:p3], \
                                                    X[:,p3:p4], X[:,p4:p5])
            
            y_values[0, i] = out[0] - out_[0]
            y_values[3, i] = l4[0][0] - l4_[0][0]
            y_values[2, i] = l4[0][1] - l4_[0][1]
            y_values[1, i] = l4[0][2] - l4_[0][2]
            y_values[4, i] = l4[0][3] - l4_[0][3]
            y_values[5, i] = l4[0][4] - l4_[0][4]
            for j in range(10):
                y_values[j+6, i] = l3[0][j] - l3_[0][j]

        df = pd.DataFrame(y_values, index=['pred','l4h1','l4h2','l4h3','l4h4','l4h5',
                                            'l3h1','l3h2','l3h3','l3h4','l3h5',
                                            'l3h6','l3h7','l3h8','l3h9','l3h10'
                                            ])
        df.insert(123, column='s', value=df.iloc[:,112])
        df.insert(123, column='s1', value=df.iloc[:,113])
        df.insert(123, column='s2', value=df.iloc[:,114])
        df.insert(123, column='s3', value=df.iloc[:,115])
        df.insert(123, column='s4', value=df.iloc[:,116])
        df.insert(123, column='s5', value=df.iloc[:,117])
        df.insert(123, column='s6', value=df.iloc[:,118])
        df.insert(123, column='s7', value=df.iloc[:,119])
        df.columns = np.arange(0,133,1)
        df.iloc[:6,:] = (df.iloc[:6,:] - df.iloc[:6,:].mean()) / (df.iloc[:6,:].max() - df.iloc[:6,:].min())
        df.iloc[6:,:] = (df.iloc[6:,:] - df.iloc[6:,:].mean()) / (df.iloc[6:,:].max() - df.iloc[6:,:].min())
        fig, ax = plt.subplots(figsize=(20,9))
        sns.heatmap(ax=ax, data=df, cmap='RdYlGn_r')
        plt.savefig('../figures/sens_naive/heat.eps', dpi=200)

        

    def train_v6_tc(self):


        if os.path.exists('%s/%s' %(self.log_path, self.model_name)):
            call(['rm', '-r', '%s/%s' %(self.log_path, self.model_name)])
        call(['mkdir', '%s/%s' %(self.log_path, self.model_name)])

        train_writer = SummaryWriter('%s/%s/train/' %(self.log_path, self.model_name))
        test_writer = SummaryWriter('%s/%s/test/' %(self.log_path, self.model_name))


        data_loader, test_loader = get_loader(self.batch_size, self.device, self.model_name, b=self.b_value)
        
        e2e_loss = E2E_v6_loss(self.device)

        params = list(self.model.parameters())
        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        curr_epoch = 0

        if self.train_check != 'None':
            # model.load_state_dict(torch.load('../logs/torch/e2e_v6_30.pkl'))
            checkpoint = torch.load(self.model_path+self.train_check, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            curr_epoch = checkpoint['epoch']
            loss = checkpoint['loss']
            print('load model!')

        for epoch in range(curr_epoch, self.num_epochs):
            train_loss0, train_loss1 = 0, 0

            for i, (X, S1, S2) in enumerate(data_loader):
                out, out_vlt, out_sf = self.model(S1[:,:,:2], S1[:,:,2:], S2[:,:,:2], X[:,:p1], X[:,p2:p3], X[:,p3:p4], X[:,p4:p5])
                batch_loss1, batch_loss0 = e2e_loss(out_vlt, X[:,p6:p7], out_sf, S2[:,:,2], out, X[:,p5:p6])

                # if epoch>0:
                optimizer.zero_grad()
                batch_loss0.backward()
                optimizer.step()

                train_loss1 += batch_loss1.item()
                train_loss0 += batch_loss0.item()

                if (i+1) % self.log_step == 0:

                    test_loss0, test_loss1 = 0, 0
                    for _, (X, S1, S2) in enumerate(test_loader):
                        out, out_vlt, out_sf = self.model(S1[:,:,:2], S1[:,:,2:], S2[:,:,:2], X[:,:p1], X[:,p2:p3], X[:,p3:p4], X[:,p4:p5])
                        loss1, loss0 = e2e_loss(out_vlt, X[:,p6:p7], out_sf, S2[:,:,2], out, X[:,p5:p6])
                        test_loss1 += loss1.item()
                        test_loss0 += loss0.item()

                    print('Epoch %d pct %.3f, loss_1 %.5f, loss_ttl %.5f, test_loss_1 %.5f, test_loss_ttl %.5f' % 
                                    (epoch,(i+1)/len(data_loader),train_loss1/self.log_step,train_loss0/self.log_step,
                                        test_loss1/len(test_loader), test_loss0/len(test_loader)))

                    for name, param in self.model.named_parameters():
                        train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch*len(data_loader)+i)
                    train_writer.add_scalar('train_loss_out',train_loss1/self.log_step, epoch*len(data_loader)+i)
                    train_writer.add_scalar('train_loss_ttl',train_loss0/self.log_step, epoch*len(data_loader)+i)
                    test_writer.add_scalar('test_loss_out', test_loss1/len(test_loader), epoch*len(data_loader)+i)
                    test_writer.add_scalar('test_loss_ttl', test_loss0/len(test_loader), epoch*len(data_loader)+i)
                    train_writer.export_scalars_to_json('%s/%s/train/scalars_train.json' %(self.log_path, self.model_name))
                    test_writer.export_scalars_to_json('%s/%s/test/scalars_test.json' %(self.log_path, self.model_name))
                    train_loss0, train_loss1 = 0, 0

            if (epoch+1) % self.save_step == 0:
                # torch.save(model.state_dict(), os.path.join('../logs/torch/', 'e2e_v6_%d.pkl' %(epoch+1)))
                torch.save({
                            'epoch': epoch+1,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': batch_loss1,
                            }, os.path.join('../logs/torch/', 'e2e_%s_%d.pkl' %(self.model_name,epoch+1)))

        train_writer.close()
        test_writer.close()

    def eval_v6_tc(self):


        _, test_loader = get_loader(self.batch_size, self.device, self.model_name, eval=1, test_sku=self.test_sku, b=self.b_value)

        checkpoint = torch.load(self.model_path+self.model_to_load, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('load model!')
        
        for _, (X, S1, S2) in enumerate(test_loader):
            out, out_vlt, out_sf = self.model(S1[:,:,:2], S1[:,:,2:], S2[:,:,:2], X[:,:p1], X[:,p2:p3], X[:,p3:p4], X[:,p4:p5])

        pd_scaler = pd.read_csv('../data/1320_feature/scaler.csv')
        df_idx = pd.read_csv('../logs/torch/pred_sku.csv')
        out = out.detach().cpu().numpy() / pd_scaler.loc[1, LABEL[0]] + pd_scaler.loc[0, LABEL[0]]
        if 'v5' in self.model_name:
            out_sf = out_sf.detach().cpu().numpy() / pd_scaler.loc[1, LABEL_sf[0]] + pd_scaler.loc[0, LABEL_sf[0]]
            out_sf = np.exp(out_sf) - 1
            out_vlt = out_vlt.detach().cpu().numpy() / pd_scaler.loc[1, LABEL_vlt[0]] + pd_scaler.loc[0, LABEL_vlt[0]]
            pred = pd.DataFrame(out, columns=['E2E_MLP_pred']) 
            pred_sf = pd.DataFrame(out_sf, columns=['E2E_NN_SF_mean_pred'])
            pred_vlt = pd.DataFrame(out_vlt, columns=['E2E_NN_vlt_pred'])
            pred = pd.concat([df_idx, pred, pred_sf, pred_vlt], axis=1)
            pred.to_csv('%s/pred_v5.csv' %self.model_path, index=False)
        else:
            out_sf = out_sf.view(-1, rnn_pred_long, num_quantiles).detach().cpu().numpy()
            out_sf = np.exp(out_sf) - 1
            out_vlt = out_vlt.detach().cpu().numpy() / pd_scaler.loc[1, LABEL_vlt[0]] + pd_scaler.loc[0, LABEL_vlt[0]]
            pred = pd.DataFrame(out, columns=['E2E_RNN_pred']) 
            pred_vlt = pd.DataFrame(out_vlt, columns=['E2E_NN_vlt_pred'])
            pred = pd.concat([df_idx, pred, pred_vlt], axis=1)
            pred.to_csv('%s/pred_v6.csv'  %self.model_path, index=False)
            out_sf.dump('%s/pred_E2E_SF_RNN.pkl'  %self.model_path)



