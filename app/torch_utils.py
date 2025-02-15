import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from skimage.color import rgb2lab, lab2rgb
import numpy as np


def generate_l_ab(images):
    lab = rgb2lab(images.permute(0, 2, 3, 1).cpu().numpy())
    X = lab[:, :, :, 0]
    X = X.reshape(X.shape+(1,))
    Y = lab[:, :, :, 1:] / 128
    return torch.tensor(X, dtype=torch.float).permute(0, 3, 1, 2), torch.tensor(Y, dtype=torch.float).permute(0, 3, 1, 2)


def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class Base_Model(nn.Module):
    def training_batch(self, batch):
        images, _ = batch
        X, Y = generate_l_ab(images)
        outputs = self.forward(X)
        loss = F.mse_loss(outputs, Y)
        return loss

    def validation_batch(self, batch):
        images, _ = batch
        X, Y = generate_l_ab(images)
        outputs = self.forward(X)
        loss = F.mse_loss(outputs, Y)
        return {'val_loss': loss.item()}

    def validation_end_epoch(self, outputs):
        epoch_loss = sum([x['val_loss'] for x in outputs]) / len(outputs)
        return {'epoch_loss': epoch_loss}


class Encoder_Decoderv1(Base_Model):  # the autoencoder
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2,
                      padding=get_padding(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, kernel_size=3, stride=2,
                      padding=get_padding(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 256, kernel_size=3, stride=2,
                      padding=get_padding(3, 2)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 256, kernel_size=3, padding=get_padding(3)),
            nn.ReLU(),
            nn.BatchNorm2d(256),

            nn.Conv2d(256, 128, kernel_size=3, padding=get_padding(3)),
            nn.Upsample(size=(64, 64)),
            nn.Conv2d(128, 64, kernel_size=3, padding=get_padding(3)),
            nn.Upsample(size=(128, 128)),
            nn.Conv2d(64, 32, kernel_size=3, padding=get_padding(3)),
            nn.Conv2d(32, 16, kernel_size=3, padding=get_padding(3)),
            nn.Conv2d(16, 2, kernel_size=3, padding=get_padding(3)),
            nn.Tanh(),
            nn.Upsample(size=(256, 256))
        )

    def forward(self, images):
        return self.network(images)


class Encoder_Decoderv2(Base_Model):
    def __init__(self, input_size=128):
        super(Encoder_Decoderv2, self).__init__()
        MIDLEVEL_FEATURE_SIZE = 128

        resnet = models.resnet18(num_classes=365)
        resnet.conv1.weight = nn.Parameter(
            resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

        self.upsample = nn.Sequential(
            nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, input):

        # Pass input through ResNet-gray to extract features
        midlevel_features = self.midlevel_resnet(input)

        # Upsample to get colors
        output = self.upsample(midlevel_features)
        return output


def load_checkpoint(filepath, m='l'):  # loading the pretrained weights
    model = Encoder_Decoderv2() if m == 'a' or m == 'p' else Encoder_Decoderv1()
    model.load_state_dict(torch.load(filepath, map_location='cpu'))
    model.eval()

    return model


def transform_tensor_pil(tensor):
    return T.ToPILImage()(tensor.squeeze_(0))


def transform_image(image):  # convert all images into a similar size
    test_transforms = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    return test_transforms(image)


def to_rgb(grayscale_input, ab_output):
    color_image = torch.cat((grayscale_input, ab_output),
                            0).numpy()  # combine channels
    color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
    color_image[:, :, 0:1] = color_image[:, :, 0:1]
    color_image[:, :, 1:3] = (color_image[:, :, 1:3]) * 128
    color_image = lab2rgb(color_image.astype(np.float64))
    return color_image


# given the kind of input, chooses the model to predict on, returns a numpy array
def get_prediction(image, m):
    l_img = rgb2lab(image.permute(1, 2, 0))[:, :, 0]
    l_img = torch.tensor(l_img).type(
        torch.FloatTensor).unsqueeze(0).unsqueeze(0)
    if m == 'a':
        PATH = PATH_ANIMALS
    elif m == 'f':
        PATH = PATH_FRUITS
    elif m == 'p':
        PATH = PATH_PEOPLE
    else:
        PATH = PATH_LANDSCAPES
    model = load_checkpoint(PATH, m)
    ab_img = model(l_img)
    l_img = l_img.squeeze(0)
    ab_img = ab_img.squeeze(0)
    return to_rgb(l_img.detach(), ab_img.detach())


PATH_FRUITS = 'app/fruits.pth'
PATH_LANDSCAPES = 'app/landscapes.pth'
PATH_PEOPLE = 'app/people.pth'
PATH_ANIMALS = 'app/animals.pth'
