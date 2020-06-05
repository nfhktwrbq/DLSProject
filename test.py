class Parent:
    def __init__(self, target, ):
        self.target = target
        print('Init parent: ' + str(target))

    def Method(self):
        print('parent method')




class Outer:
    def __init__(self):
        self.ch2 = self.Child2(target='222')

    def Method(self, msg):
        print('parent method ', msg)

    class Child1(Parent):
        def __init__(self, target, ):
            super(Outer.Child1, self).__init__(target)
            self.target = target
            print('Init child1: ' + str(target))

        def Method(self):
            Outer.Method('qwerty')

    class Child2(Parent):
        def __init__(self, target, ):
            super().__init__(target)
            self.target = target
            self.ch1 = Outer.Child1('111')
            self.ch1.Method()
            print('Init child2: ' + str(target))

        def Method(self):
            print('child2 method')

Outer()

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from styleTransfer import StyleTransfer
from PIL import Image
import torch

style =  './style.jpg'
content = './orig.webp'

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

imsize = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

style_img = image_loader(style)
content_img = image_loader(content)

plt.figure()
imshow(style_img, title='Style Image')
plt.ioff()
plt.show()

plt.figure()
imshow(content_img, title='Content Image')
plt.ioff()
plt.show()



out = StyleTransfer(style, content).getOutput()
imshow(out, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
