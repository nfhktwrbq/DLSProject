#from sklearn.model_selection import train_test_split
from GAN.utils import PicDataset
import torch
from GAN.models import generatorNet
from GAN.models  import discriminatorNet
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from GAN.utils import printPics
import copy
from PIL import Image
import torchvision.transforms as transforms
from GAN.utils import RESCALE_SIZE

class Pix2pix:
    def __init__(self, train_pathA, train_pathB, file_ext='jpg', val_part=0.25, batch_size=25, lr=0.0003, lambdal1 = 0.2, transform = ['RANDOM_HOR_FLIP','RANDOM_CROP']):
        self.train_pathA = train_pathA
        self.train_pathB = train_pathB
        self.val_part = val_part
        self.batch_size = batch_size
        self.lambdal1 = lambdal1
        self.lr = lr
        self.transform = transform
        self.file_ext = file_ext
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_files = (self.train_pathA, self.train_pathB)
        val_files = (self.train_pathA, self.train_pathB)

        #train_files, val_files = train_test_split(self.prepareFiles(self.train_pathA, self.train_pathB) , test_size=self.val_part)
        self.train_dataset = PicDataset(train_files, self.transform)
        self.val_dataset = PicDataset(val_files, self.transform)

        self.gen = generatorNet().to(self.device)
        self.dsc = discriminatorNet().to(self.device)

    def saveModel(self, path):
        model_weights = copy.deepcopy(self.gen.state_dict())
        torch.save(model_weights, path + "model_gen.pth")
        model_weights = copy.deepcopy(self.dsc.state_dict())
        torch.save(model_weights, path + "model_dsc.pth") 

    def loadModels(self, path):
        self.gen.load_state_dict(torch.load(path +  "model_gen.pth", map_location=torch.device('cpu')))
        self.dsc.load_state_dict(torch.load(path + "model_dsc.pth", map_location=torch.device('cpu')))

    def lossGan(self, yPred, yReal, eps = 1e-7):
        yReal = (yReal - yReal.min())/(yReal.max() - yReal.min())
        yPred = (yPred - yPred.min())/(yPred.max() - yPred.min())
        return  torch.mean(torch.log(eps+yReal)) + torch.mean(torch.log(eps + 1.0 - yPred))

    def prepareFiles(self, pathA, pathB):
        TRAIN_DIR_B = Path(pathA)        
        TRAIN_DIR_A = Path(pathB)
        trainA = sorted(list(TRAIN_DIR_A.rglob('*.'+self.file_ext)))
        trainB = sorted(list(TRAIN_DIR_B.rglob('*.'+self.file_ext)))
        return list(zip(trainA ,trainB ))

    def test(self, test_pathA, test_pathB):
        test_dataset = PicDataset(self.prepareFiles(test_pathA, test_pathB), self.transform)
        test_loader = DataLoader(test_dataset ,batch_size=self.batch_size, shuffle=True)
        with torch.no_grad():       
            outputs = []
            for inputs, real in test_loader:
                inputs = inputs.to(self.device)
                self.gen.eval()
                output = self.gen(inputs).cpu()
                outputs.append(output)
                printPics(inputs.cpu(), output, real)
        return outputs

    def train(self, epochs):
        optimG = torch.optim.Adam(filter(lambda p: p.requires_grad, self.gen.parameters()),lr=self.lr )
        optimD = torch.optim.Adam(filter(lambda p: p.requires_grad, self.dsc.parameters()),lr=self.lr )
        criterionGAN=self.lossGan
        criterionL1 = torch.nn.L1Loss()
        train_loader = DataLoader(self.train_dataset ,batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset ,batch_size=self.batch_size, shuffle=True)
        historyLossD = []
        historyLossG = []
        
        A_val, B_val = next(iter(val_loader))
        
        for epoch in range(epochs):  
            avg_lossD = 0
            avg_lossG = 0
            self.gen.train()
            for A, B in train_loader:
                
                A = A.to(self.device)
                B = B.to(self.device)            
                
                optimD.zero_grad()
                
                generatedB = self.gen(A)            
               
                for param in self.dsc.parameters():
                    param.requires_grad = True
                
                generatedAB = torch.cat((A, generatedB), 1)
                predGenerated = self.dsc(generatedAB.detach())
                
                lossDGenerated = criterionL1(predGenerated, torch.ones_like(predGenerated))
                
                realAB = torch.cat((A, B), 1)
                predReal = self.dsc(realAB.detach())
                
                lossDReal = criterionL1(predReal, torch.zeros_like(predReal))
                
                lossD = (lossDGenerated + lossDReal) / 2
                
                lossD.backward(retain_graph=True)
                
                optimD.step()
               
                
                optimG.zero_grad()
                
                for param in self.dsc.parameters():
                    param.requires_grad = False
                
                predGenerated = self.dsc(generatedAB.detach())
                
                realAB = torch.cat((A, B), 1)
                
                predReal = self.dsc(realAB.detach())
                
                lossG_GAN = criterionGAN(predGenerated, predReal)
                
                lossG_L1 = criterionL1(generatedB, B) * self.lambdal1
                
                lossG = lossG_GAN + lossG_L1
                
                lossG.backward()
                
                optimG.step()

                avg_lossD += float(lossD) / len(train_loader)
                avg_lossG += float(lossG) / len(train_loader)
                
            historyLossD.append(avg_lossD)
            historyLossG.append(avg_lossG)
            self.gen.eval()                  
            if epoch % 50 == 0:
                B_hat = self.gen(A_val.to(self.device)).detach().cpu()
                print('lossD: %f' % avg_lossD,";  ", 'lossG: %f' % avg_lossG )
                printPics(A_val, B_hat, B_val)
                
        return historyLossD, historyLossG

    def getOutput(self, input_image):
        transform = transforms.Compose([
            transforms.Resize((RESCALE_SIZE , RESCALE_SIZE )),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = Image.open(input_image)
        image.load()
        inputs = transform(image).unsqueeze(0)
        with torch.no_grad():
            inputs = inputs.to(self.device)
            self.gen.eval()
            output = self.gen(inputs).cpu()
        output= (output[0]-output[0].min())/(output[0].max()-output[0].min())
        return output
