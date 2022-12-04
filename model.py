import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

class ConvInputModel(nn.Module):
    def __init__(self):
        super(ConvInputModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(24)
        self.conv4 = nn.Conv2d(24, 24, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(24)

        
    def forward(self, img):
        """convolution"""
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

  
class FCOutputModel(nn.Module):
    def __init__(self):
        super(FCOutputModel, self).__init__()

        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class BasicModel(nn.Module):
    def __init__(self, args, name):
        super(BasicModel, self).__init__()
        self.name=name

    def train_(self, input_img, input_qst, label):
        self.optimizer.zero_grad()
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        loss.backward()
        self.optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss
        
    def test_(self, input_img, input_qst, label):
        output = self(input_img, input_qst)
        loss = F.nll_loss(output, label)
        pred = output.data.max(1)[1]
        correct = pred.eq(label.data).cpu().sum()
        accuracy = correct * 100. / len(label)
        return accuracy, loss

    def save_model(self, epoch):
        torch.save(self.state_dict(), 'model/epoch_{}_{:02d}.pth'.format(self.name, epoch))


class RN(BasicModel):
    def __init__(self, args):
        super(RN, self).__init__(args, 'RN')
        
        self.conv = ConvInputModel()
        
        self.relation_type = args.relation_type
        
        if self.relation_type == 'ternary':
            ##(number of filters per object+coordinate of object)*3+question vector
            self.g_fc1 = nn.Linear((24+2)*3+18, 256)
        else:
            ##(number of filters per object+coordinate of object)*2+question vector
            self.g_fc1 = nn.Linear((24+2)*2+18, 256)

        self.g_fc2 = nn.Linear(256, 256)
        self.g_fc3 = nn.Linear(256, 256)
        self.g_fc4 = nn.Linear(256, 256)

        self.f_fc1 = nn.Linear(256, 256)

        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))


        self.fcout = FCOutputModel()
        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
 
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)
        
        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]
        # x_flat = (64 x 25 x 24)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)
        
        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        

        if self.relation_type == 'ternary':
            # add question everywhere
            qst = torch.unsqueeze(qst, 1) # (64x1x18)
            qst = qst.repeat(1, 25, 1) # (64x25x18)
            qst = torch.unsqueeze(qst, 1)  # (64x1x25x18)
            qst = torch.unsqueeze(qst, 1)  # (64x1x1x25x18)

            # cast all triples against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26)
            x_i = torch.unsqueeze(x_i, 3)  # (64x1x25x1x26)
            x_i = x_i.repeat(1, 25, 1, 25, 1)  # (64x25x25x25x26)
            
            x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26)
            x_j = torch.unsqueeze(x_j, 2)  # (64x25x1x1x26)
            x_j = x_j.repeat(1, 1, 25, 25, 1)  # (64x25x25x25x26)

            x_k = torch.unsqueeze(x_flat, 1)  # (64x1x25x26)
            x_k = torch.unsqueeze(x_k, 1)  # (64x1x1x25x26)
            x_k = torch.cat([x_k, qst], 4)  # (64x1x1x25x26+18)
            x_k = x_k.repeat(1, 25, 25, 1, 1)  # (64x25x25x25x26+18)

            # concatenate all together
            x_full = torch.cat([x_i, x_j, x_k], 4)  # (64x25x25x25x3*26+18)

            # reshape for passing through network
            x_ = x_full.view(mb * (d * d) * (d * d) * (d * d), 96)  # (64*25*25*25x3*26+18) = (1.000.000, 96)
        else:
            # add question everywhere
            qst = torch.unsqueeze(qst, 1)
            qst = qst.repeat(1, 25, 1)
            qst = torch.unsqueeze(qst, 2)

            # cast all pairs against each other
            x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x26+18)
            x_i = x_i.repeat(1, 25, 1, 1)  # (64x25x25x26+18)
            x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x26+18)
            x_j = torch.cat([x_j, qst], 3)
            x_j = x_j.repeat(1, 1, 25, 1)  # (64x25x25x26+18)
            
            # concatenate all together
            x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*26+18)
        
            # reshape for passing through network
            x_ = x_full.view(mb * (d * d) * (d * d), 70)  # (64*25*25x2*26*18) = (40.000, 70)
            
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        # reshape again and sum
        if self.relation_type == 'ternary':
            x_g = x_.view(mb, (d * d) * (d * d) * (d * d), 256)
        else:
            x_g = x_.view(mb, (d * d) * (d * d), 256)

        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)

        return self.fcout(x_f)




class CNN_MLP(BasicModel):
    def __init__(self, args):
        super(CNN_MLP, self).__init__(args, 'CNNMLP')

        self.conv  = ConvInputModel_SOC()
        self.fc1   = nn.Linear(5*5*24 + 18, 256)  # question concatenated to all
        self.fcout = FCOutputModel()

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)
        #print([ a for a in self.parameters() ] )
  
    def forward(self, img, qst):
        x = self.conv(img) ## x = (64 x 24 x 5 x 5)

        """fully connected layers"""
        x = x.view(x.size(0), -1)
        
        x_ = torch.cat((x, qst), 1)  # Concat question
        
        x_ = self.fc1(x_)
        x_ = F.relu(x_)
        
        return self.fcout(x_)


    """
    Integrate features after multiple convolutions then perform normalization, 
    and output a probability for each classification situation; the subsequent 
    classifier can be classified according to the probability obtained by fully connected outputs.
    Adding a fully connected layer to learn nonlinear combination features in an easy way.
    """
#Fully connected output for Sort-of-CLEVR model
class FCOutputModel_SOC(nn.Module):
    def __init__(self):
        super(FCOutputModel_SOC, self).__init__()
        #latter 3 layer of 4-layer MLP for f
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, 10)

    def forward(self, x):
        """ 
        Full connection
        ReLU non-linearities
        """
        #ReLU non-linearities
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


#Convolutional Input for Sort-Of-CLEVR model
class ConvInputModel_SOC(nn.Module):
    def __init__(self):
        super(ConvInputModel_SOC, self).__init__()
        #Four convolutional layers with 32, 64, 128, and 256 kernels
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.batchNorm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.batchNorm4 = nn.BatchNorm2d(256)

        
    def forward(self, img):
        """
        Convolution
        ReLU non-linearities and batch normalization
        """
        x = self.conv1(img)
        x = F.relu(x)
        x = self.batchNorm1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchNorm2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchNorm3(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchNorm4(x)
        return x

#CNN Augmented with RN Model for Sort-of-CLEVR
class CNN_RN_SOC(BasicModel):
    def __init__(self, args):
        super(CNN_RN_SOC, self).__init__(args, 'CNN_RN_SOC')
        
        self.conv = ConvInputModel_SOC()
        
        self.relation_type = args.relation_type

        """ 
        g
        4-layer MLP of 2000 units each
        """
        if self.relation_type == 'ternary':
            ##(number of filters per object+coordinate of object)*3+question vector
            self.g_fc1 = nn.Linear((256+2)*3+18, 2000)
        else:
            ##(number of filters per object+coordinate of object)*2+question vector
            self.g_fc1 = nn.Linear((256+2)*2+18, 2000)

        self.g_fc2 = nn.Linear(2000, 2000)
        self.g_fc3 = nn.Linear(2000, 2000)
        self.g_fc4 = nn.Linear(2000, 2000)

        """
        f
        4-layer MLP of 2000, 1000, 500, and 100 units with ReLU non-linearities
        """
        self.f_fc1 = nn.Linear(2000, 1000)
        self.fcout = FCOutputModel_SOC()


        #coordinate tensors
        self.coord_oi = torch.FloatTensor(args.batch_size, 2)
        self.coord_oj = torch.FloatTensor(args.batch_size, 2)
        if args.cuda:
            self.coord_oi = self.coord_oi.cuda()
            self.coord_oj = self.coord_oj.cuda()
        self.coord_oi = Variable(self.coord_oi)
        self.coord_oj = Variable(self.coord_oj)

        # prepare coord tensor
        def cvt_coord(i):
            return [(i/5-2)/2., (i%5-2)/2.]
        
        self.coord_tensor = torch.FloatTensor(args.batch_size, 25, 2)
        if args.cuda:
            self.coord_tensor = self.coord_tensor.cuda()
        self.coord_tensor = Variable(self.coord_tensor)
        np_coord_tensor = np.zeros((args.batch_size, 25, 2))
        for i in range(25):
            np_coord_tensor[:,i,:] = np.array( cvt_coord(i) )
        self.coord_tensor.data.copy_(torch.from_numpy(np_coord_tensor))



        
        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, img, qst):
        
        x = self.conv(img) ## x = (64 x 256 x 5 x 5)

        """g"""
        mb = x.size()[0]
        n_channels = x.size()[1]
        d = x.size()[2]

        # x_flat = (64 x 25 x 256)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)

        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        

        qst = torch.unsqueeze(qst, 1) # (64x1x18)
        qst = qst.repeat(1, 25, 1) # (64x25x18)
        qst = torch.unsqueeze(qst, 2) # (64x25x1x18)

        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x256+2)
        x_i = x_i.repeat(1, 25, 1, 1)  # (64x25x25x256+2)
        x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x256+2)
        x_j = torch.cat([x_j, qst], 3) # (64x25x1x256+2+18)
        x_j = x_j.repeat(1, 1, 25, 1)  # (64x25x25x256+2+18)

        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*(256+2+18)) = (64x25x25x534)

        # reshape for passing through network
        x_ = x_full.view(mb * (d * d) * (d * d), 534)  # (64*25*25x2*256*18) = (40.000, 534)

        #g MLP
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        #reshape for passing into f
        x_g = x_.view(mb, (d * d) * (d * d), 2000)

        #element wise sum for passing into f
        x_g = x_g.sum(1).squeeze()
        
        """f MLP """
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)

        return self.fcout(x_f)

    def activate_cuda(self):
        self.on_gpu = True


#Relational network for training via state description
class RN_state_desc(BasicModel):
    def __init__(self, args):
        super(RN_state_desc, self).__init__(args, 'RN_state_desc')

        """
        g
        4 layer MLP of 512 units
        """
        self.g_fc1 = nn.Linear(32, 512)
        self.g_fc2 = nn.Linear(512, 512)
        self.g_fc3 = nn.Linear(512, 512)
        self.g_fc4 = nn.Linear(512, 512)

        """
        f 
        3 layer MLP of 512, 256, and 100 units
        """
        self.f_fc1 = nn.Linear(512, 512)
        self.f_fc2 = nn.Linear(512, 1024)
        self.f_fc3 = nn.Linear(1024, 10)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, mat, qst):


        x_flat = mat
        mb = mat.shape[0]
        d = mat.shape[1]
        
        
        # add question everywhere
        qst = torch.unsqueeze(qst, 1) # (64x1x18)
        qst = qst.repeat(1, 6, 1) # (64x6x18)
        qst = torch.unsqueeze(qst, 2) # (64x6x1x18)


        # cast all pairs against each other
        x_i = torch.unsqueeze(x_flat, 1)  # (64x1x6x7)
        x_i = x_i.repeat(1, 6, 1, 1)  # (64x6x6x7)
        x_j = torch.unsqueeze(x_flat, 2)  # (64x6x1x7) 
        x_j = torch.cat([x_j, qst], 3) # (64x6x1x7+18)
        x_j = x_j.repeat(1, 1, 6, 1)  # (64x6x6x7+18)

            
        # concatenate all together
        x_full = torch.cat([x_i,x_j],3) # (64x6x6x2*7+18)

        # reshape for passing through network
        x_ = x_full.view(mb*d*d,32)  # (64*(6*6)x2*7+18) = (2304, 32)

        """g"""
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        #reshape for network
        x_g = x_.view(mb,d*d,512)
        
        #element wise sum
        x_g = x_g.sum(1).squeeze()
        
        """f"""
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.relu(x_f)
        x_f = F.dropout(x_f, p = 0.02)
        x_f = self.f_fc3(x_f)


        return F.log_softmax(x_f, dim=1)
