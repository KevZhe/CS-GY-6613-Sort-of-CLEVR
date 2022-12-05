import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time


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
Integrate features after multiple convolutions and normalizations.
"""
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
        Convolutions
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

#CNN Augmented with RN Model for Sort-of-CLEVR
class CNN_RN_SOC(BasicModel):
    def __init__(self, args):
        super(CNN_RN_SOC, self).__init__(args, 'CNN_RN_SOC')
        
        self.conv = ConvInputModel_SOC()

        """ 
        g
        4-layer MLP of 2000 units each
        """

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

        """
        Using CNNs to process image to create sets objects procesable by RN
        """
        #Process input image through
        x = self.conv(img) ## x = (64 x 256 x 5 x 5)

        #mini-batch size
        mb = x.size()[0]
        #number of channels
        n_channels = x.size()[1]
        #dim
        d = x.size()[2]

        # x_flat = (64 x 25 x 256)
        x_flat = x.view(mb,n_channels,d*d).permute(0,2,1)

        # add coordinates
        x_flat = torch.cat([x_flat, self.coord_tensor],2)
        
        """
        Everything below is part of RN
        """

        """
        creating object pairs + question
        """

        """
        preparing question tensor
        """
        #expand our question tensor in the 2nd dimension to be repeatable
        qst = torch.unsqueeze(qst, 1) # (64x1x18)
        #repeat 25 times in the 2nd dimension so it can be added to all pairs (25 objects)
        qst = qst.repeat(1, 25, 1) # (64x25x18)
        #expand in 3rd dimension so it can be added to features later
        qst = torch.unsqueeze(qst, 2) # (64x25x1x18)

        """
        cast all pairs against each other
        """
        #expand our input object in 2nd dimension to be repeatable
        x_i = torch.unsqueeze(x_flat, 1)  # (64x1x25x256+2)
        #repeat 25 times in the 2nd dimension (objects xi)
        x_i = x_i.repeat(1, 25, 1, 1)  # (64x25x25x256+2)
        #expand our input object in 3rd dimension to be repeatable
        x_j = torch.unsqueeze(x_flat, 2)  # (64x25x1x256+2)
        #add our question to be asked to our objects xj's features
        x_j = torch.cat([x_j, qst], 3) # (64x25x1x256+2+18)
        #repeat 25 times in the 3rd dimension (objects xj with question)
        x_j = x_j.repeat(1, 1, 25, 1)  # (64x25x25x256+2+18)

        # concatenate objects xi with xj + question
        x_full = torch.cat([x_i,x_j],3) # (64x25x25x2*(256+2)+18)) = (64x25x25x534)

        # reshape for passing through network
        x_ = x_full.view(mb * (d * d) * (d * d), 534)  # (64*25*25x2*258+18) = (40.000, 534)
        
        """g"""

        #run through g MLP
        x_ = self.g_fc1(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc2(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc3(x_)
        x_ = F.relu(x_)
        x_ = self.g_fc4(x_)
        x_ = F.relu(x_)
        
        #reshape output for passing into f
        x_g = x_.view(mb, (d * d) * (d * d), 2000)

        #element wise sum for passing into f
        x_g = x_g.sum(1).squeeze()

        """f"""

        #run through f MLP
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
        3 layer MLP of 512, 1024, and 10 units
        """
        self.f_fc1 = nn.Linear(512, 512)
        self.f_fc2 = nn.Linear(512, 1024)
        self.f_fc3 = nn.Linear(1024, 10)

        self.optimizer = optim.Adam(self.parameters(), lr=args.lr)


    def forward(self, mat, qst):

        #no need for processing because state description rows are already objects processable by RN
        x_flat = mat
        #mini-batch size
        mb = mat.shape[0]
        #dim
        d = mat.shape[1]
        
        #add question everywhere
        #expand question tensor in 1st dimension to be repeatable
        qst = torch.unsqueeze(qst, 1) # (64x1x18)
        #repeat question tensor 6 times in 1st dimension for 6 objects
        qst = qst.repeat(1, 6, 1) # (64x6x18)
        #expand in 3rd dimension so it can be added to features later 
        qst = torch.unsqueeze(qst, 2) # (64x6x1x18)


        #cast all pairs against each other
        #expand in 2nd dimension to repeat objects xi
        x_i = torch.unsqueeze(x_flat, 1)  # (64x1x6x7)
        #repeat 6 times in 2nd dimension (creating objects xi)
        x_i = x_i.repeat(1, 6, 1, 1)  # (64x6x6x7)
        #expand in 3rd dimension to repeat objects xj
        x_j = torch.unsqueeze(x_flat, 2)  # (64x6x1x7)
        #add our question to be asked to our objects xj's features
        x_j = torch.cat([x_j, qst], 3) # (64x6x1x7+18)
        #repeat 6 times in the 3rd dimension (objects xj with question)
        x_j = x_j.repeat(1, 1, 6, 1)  # (64x6x6x7+18)

            
        # concatenate all together (have all possible object pairings of xi and xj with question)
        x_full = torch.cat([x_i,x_j],3) # (64x6x6x2*7+18)

        # reshape for passing through network
        x_ = x_full.view(mb*d*d,32)  # (64*(6*6)x2*7+18) = (2304, 32)

        """g"""

        #run through g MLP
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

        #run through f MLP
        x_f = self.f_fc1(x_g)
        x_f = F.relu(x_f)
        x_f = self.f_fc2(x_f)
        x_f = F.relu(x_f)
        x_f = F.dropout(x_f, p = 0.02)
        x_f = self.f_fc3(x_f)


        return F.log_softmax(x_f, dim=1)
