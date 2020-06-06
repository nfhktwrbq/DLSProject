import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import torchvision.transforms as transforms
import copy



class SimpleCnn(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1ChanelNum = 64
        self.layer2ChanelNum = 128
        self.layer3ChanelNum = 256
        self.layer4ChanelNum = 512
        self.linLayerChanelNum = 4096

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.layer1ChanelNum, kernel_size=3),                 # 0
            nn.ReLU(inplace=True),                                                                      # 1
            nn.Conv2d(self.layer1ChanelNum, out_channels=self.layer1ChanelNum, kernel_size=3),          # 2
            nn.ReLU(inplace=True),                                                                      # 3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),              # 4
            nn.Conv2d(self.layer1ChanelNum, out_channels=self.layer2ChanelNum, kernel_size=3),          # 5
            nn.ReLU(inplace=True),                                                                      # 6
            nn.Conv2d(self.layer2ChanelNum, out_channels=self.layer2ChanelNum, kernel_size=3),          # 7
            nn.ReLU(inplace=True),                                                                      # 8
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),              # 9
            nn.Conv2d(self.layer2ChanelNum, out_channels=self.layer3ChanelNum, kernel_size=3),          # 10
            nn.ReLU(inplace=True),                                                                      # 11
            nn.Conv2d(self.layer3ChanelNum, out_channels=self.layer3ChanelNum, kernel_size=3),          # 12
            nn.ReLU(inplace=True),                                                                      # 13
            nn.Conv2d(self.layer3ChanelNum, out_channels=self.layer3ChanelNum, kernel_size=3),          # 14
            nn.ReLU(inplace=True),                                                                      # 15
            nn.Conv2d(self.layer3ChanelNum, out_channels=self.layer3ChanelNum, kernel_size=3),          # 16
            nn.ReLU(inplace=True),                                                                      # 17
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),              # 18
            nn.Conv2d(self.layer3ChanelNum, out_channels=self.layer4ChanelNum, kernel_size=3),          # 19
            nn.ReLU(inplace=True),                                                                      # 20
            nn.Conv2d(self.layer4ChanelNum, out_channels=self.layer4ChanelNum, kernel_size=3),          # 21
            nn.ReLU(inplace=True),                                                                      # 22
            nn.Conv2d(self.layer4ChanelNum, out_channels=self.layer4ChanelNum, kernel_size=3),          # 23
            nn.ReLU(inplace=True),                                                                      # 24
            nn.Conv2d(self.layer4ChanelNum, out_channels=self.layer4ChanelNum, kernel_size=3),          # 25
            nn.ReLU(inplace=True),                                                                      # 26
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),              # 27
            nn.Conv2d(self.layer4ChanelNum, out_channels=self.layer4ChanelNum, kernel_size=3),          # 28
            nn.ReLU(inplace=True),                                                                      # 29
            nn.Conv2d(self.layer4ChanelNum, out_channels=self.layer4ChanelNum, kernel_size=3),          # 30
            nn.ReLU(inplace=True),                                                                      # 31
            nn.Conv2d(self.layer4ChanelNum, out_channels=self.layer4ChanelNum, kernel_size=3),          # 32
            nn.ReLU(inplace=True),                                                                      # 33
            nn.Conv2d(self.layer4ChanelNum, out_channels=self.layer4ChanelNum, kernel_size=3),          # 34
            nn.ReLU(inplace=True),                                                                      # 35
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),              # 36
        )

    def forward(self, x):
        x = self.net(x)
        return x

class ContentLoss(nn.Module):
    def __init__(self, target, ):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input


class StyleLoss(nn.Module):
    def __init__(self, target_feature, instance):
        super(StyleLoss, self).__init__()
        self.instance = instance
        self.target = instance.gram_matrix(target_feature).detach()

    def forward(self, input):
        G = self.instance.gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

class StyleTransfer:
    def __init__(self, style_image, content_image):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu
        self.loader = transforms.Compose([
            transforms.Resize((self.imsize, self.imsize)),  # scale imported image
            transforms.ToTensor()])  # transform it into a torch tensor
        self.style_img = self.image_loader(style_image)
        self.content_img = self.image_loader(content_image)

        #self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
        self.cnn = SimpleCnn().to(self.device).eval()

        self.cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
        self.cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device)
        self.content_layers_default = ['conv_4']
        self.style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        print(self.style_img.size(), self.content_img.size())
        assert self.style_img.size() == self.content_img.size(), \
            "we need to import style and content images of the same size"

    def image_loader(self, image_name):
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = self.loader(image).unsqueeze(0)
        return image.to(self.device, torch.float)

    def gram_matrix(self, input):
        a, b, c, d = input.size()  # a=batch size(=1)
        # b=number of feature maps
        # (c,d)=dimensions of a f. map (N=c*d)

        features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

        G = torch.mm(features, features.t())  # compute the gram product

        # we 'normalize' the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(a * b * c * d)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std



    def get_style_model_and_losses(self, cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=None,
                                   style_layers=None):
        if content_layers is None:
            content_layers = self.content_layers_default
        if style_layers is None:
            style_layers = self.style_layers_default

        cnn = copy.deepcopy(cnn)

        # normalization module
        normalization = Normalization(normalization_mean, normalization_std).to(self.device)

        # just in order to have an iterable access to or list of content/syle
        # losses
        content_losses = []
        style_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv

        #for layer in cnn.children():
        for layer in cnn.net.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # and StyleLoss we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            elif isinstance(layer, nn.Sequential):
                continue
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))


            model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature, self)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        # now we trim off the layers after the last content and style losses
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def get_input_optimizer(self, input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def run_style_transfer(self, cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=300,
                           style_weight=1000000, content_weight=1):
        """Run the style transfer."""
        print('Building the style transfer model..')

        model, style_losses, content_losses = self.get_style_model_and_losses(cnn,
            normalization_mean, normalization_std, style_img, content_img)
        optimizer = self.get_input_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:

            def closure():
                # correct the values of updated input image
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img

    def getOutput(self):
        input_img = self.content_img.clone()
        return self.run_style_transfer(self.cnn, self.cnn_normalization_mean, self.cnn_normalization_std,
                            self.content_img, self.style_img, input_img)

