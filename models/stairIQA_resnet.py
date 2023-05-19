import torch
from torch import nn
import torchvision.models as models


class resnet50(torch.nn.Module):
    def __init__(self, pretrained = True):
        super(resnet50, self).__init__()
        if pretrained == True:
            resnet50_features = nn.Sequential(*list(models.resnet50(weights='DEFAULT').children()))
        else:
            resnet50_features = nn.Sequential(*list(models.resnet50().children()))

        self.feature_extraction_stem = torch.nn.Sequential()
        self.feature_extraction1 = torch.nn.Sequential()
        self.feature_extraction2 = torch.nn.Sequential()
        self.feature_extraction3 = torch.nn.Sequential()
        self.feature_extraction4 = torch.nn.Sequential()

        self.avg_pool = torch.nn.Sequential()

        for x in range(0,4):
            self.feature_extraction_stem.add_module(str(x), resnet50_features[x])

        for x in range(4,5):
            self.feature_extraction1.add_module(str(x), resnet50_features[x])
        
        for x in range(5,6):
            self.feature_extraction2.add_module(str(x), resnet50_features[x])

        for x in range(6,7):
            self.feature_extraction3.add_module(str(x), resnet50_features[x])

        for x in range(7,8):
            self.feature_extraction4.add_module(str(x), resnet50_features[x])


        self.hyper1_1 = self.hyper_structure1(64,256)
        self.hyper2_1 = self.hyper_structure2(256,512)
        self.hyper3_1 = self.hyper_structure2(512,1024)
        self.hyper4_1 = self.hyper_structure2(1024,2048)

        self.hyper2_2 = self.hyper_structure2(256,512)
        self.hyper3_2 = self.hyper_structure2(512,1024)
        self.hyper4_2 = self.hyper_structure2(1024,2048)

        self.hyper3_3 = self.hyper_structure2(512,1024)
        self.hyper4_3 = self.hyper_structure2(1024,2048)

        self.hyper4_4 = self.hyper_structure2(1024,2048)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.quality = self.quality_regression(2048, 128, 1)

    def hyper_structure1(self,in_channels,out_channels):
 
        hyper_block = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//4,kernel_size=1,stride=1, padding=0,bias=False),
            nn.Conv2d(in_channels//4,in_channels//4,kernel_size=3,stride=1, padding=1,bias=False),
            nn.Conv2d(in_channels//4,out_channels,kernel_size=1,stride=1, padding=0,bias=False),
        )

        return hyper_block

    def hyper_structure2(self,in_channels,out_channels):
        hyper_block = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//4,kernel_size=1,stride=1, padding=0,bias=False),
            nn.Conv2d(in_channels//4,in_channels//4,kernel_size=3,stride=2, padding=1,bias=False),
            nn.Conv2d(in_channels//4,out_channels,kernel_size=1,stride=1, padding=0,bias=False),
        )

        return hyper_block

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block
        

    def forward(self, x):

        x = self.feature_extraction_stem(x)

        x_hyper1 = self.hyper1_1(x)
        x = self.feature_extraction1(x)


        x_hyper1 = self.hyper2_1(x_hyper1+x)
        x_hyper2 = self.hyper2_2(x)
        x = self.feature_extraction2(x)


        x_hyper1 = self.hyper3_1(x_hyper1+x)
        x_hyper2 = self.hyper3_2(x_hyper2+x)
        x_hyper3 = self.hyper3_3(x)
        x = self.feature_extraction3(x)


        x_hyper1 = self.hyper4_1(x_hyper1+x)
        x_hyper2 = self.hyper4_2(x_hyper2+x)
        x_hyper3 = self.hyper4_3(x_hyper3+x)       
        x_hyper4 = self.hyper4_4(x)
        x = self.feature_extraction4(x)

        x = x+x_hyper1+x_hyper2+x_hyper3+x_hyper4

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.quality(x)

            
        return x



class resnet50_imdt(torch.nn.Module):
    def __init__(self, pretrained = True):
        super(resnet50_imdt, self).__init__()
        if pretrained == True:
            resnet50_features = nn.Sequential(*list(models.resnet50(weights='DEFAULT').children()))
        else:
            resnet50_features = nn.Sequential(*list(models.resnet50().children()))

        self.feature_extraction_stem = torch.nn.Sequential()
        self.feature_extraction1 = torch.nn.Sequential()
        self.feature_extraction2 = torch.nn.Sequential()
        self.feature_extraction3 = torch.nn.Sequential()
        self.feature_extraction4 = torch.nn.Sequential()

        self.avg_pool = torch.nn.Sequential()

        for x in range(0,4):
            self.feature_extraction_stem.add_module(str(x), resnet50_features[x])

        for x in range(4,5):
            self.feature_extraction1.add_module(str(x), resnet50_features[x])
        
        for x in range(5,6):
            self.feature_extraction2.add_module(str(x), resnet50_features[x])

        for x in range(6,7):
            self.feature_extraction3.add_module(str(x), resnet50_features[x])

        for x in range(7,8):
            self.feature_extraction4.add_module(str(x), resnet50_features[x])


        self.hyper1_1 = self.hyper_structure1(64,256)
        self.hyper2_1 = self.hyper_structure2(256,512)
        self.hyper3_1 = self.hyper_structure2(512,1024)
        self.hyper4_1 = self.hyper_structure2(1024,2048)

        self.hyper2_2 = self.hyper_structure2(256,512)
        self.hyper3_2 = self.hyper_structure2(512,1024)
        self.hyper4_2 = self.hyper_structure2(1024,2048)

        self.hyper3_3 = self.hyper_structure2(512,1024)
        self.hyper4_3 = self.hyper_structure2(1024,2048)

        self.hyper4_4 = self.hyper_structure2(1024,2048)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.quality1 = self.quality_regression(2048, 128, 1)
        self.quality2 = self.quality_regression(2048, 128, 1)
        self.quality3 = self.quality_regression(2048, 128, 1)
        self.quality4 = self.quality_regression(2048, 128, 1)
        self.quality5 = self.quality_regression(2048, 128, 1)
        self.quality6 = self.quality_regression(2048, 128, 1)

    def hyper_structure1(self,in_channels,out_channels):
 
        hyper_block = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//4,kernel_size=1,stride=1, padding=0,bias=False),
            nn.Conv2d(in_channels//4,in_channels//4,kernel_size=3,stride=1, padding=1,bias=False),
            nn.Conv2d(in_channels//4,out_channels,kernel_size=1,stride=1, padding=0,bias=False),
        )

        return hyper_block

    def hyper_structure2(self,in_channels,out_channels):
        hyper_block = nn.Sequential(
            nn.Conv2d(in_channels,in_channels//4,kernel_size=1,stride=1, padding=0,bias=False),
            nn.Conv2d(in_channels//4,in_channels//4,kernel_size=3,stride=2, padding=1,bias=False),
            nn.Conv2d(in_channels//4,out_channels,kernel_size=1,stride=1, padding=0,bias=False),
        )

        return hyper_block

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block
        

    def forward(self, x):

        x = self.feature_extraction_stem(x)

        x_hyper1 = self.hyper1_1(x)
        x = self.feature_extraction1(x)


        x_hyper1 = self.hyper2_1(x_hyper1+x)
        x_hyper2 = self.hyper2_2(x)
        x = self.feature_extraction2(x)


        x_hyper1 = self.hyper3_1(x_hyper1+x)
        x_hyper2 = self.hyper3_2(x_hyper2+x)
        x_hyper3 = self.hyper3_3(x)
        x = self.feature_extraction3(x)


        x_hyper1 = self.hyper4_1(x_hyper1+x)
        x_hyper2 = self.hyper4_2(x_hyper2+x)
        x_hyper3 = self.hyper4_3(x_hyper3+x)       
        x_hyper4 = self.hyper4_4(x)
        x = self.feature_extraction4(x)

        x = x+x_hyper1+x_hyper2+x_hyper3+x_hyper4

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x1 = self.quality1(x)
        x2 = self.quality2(x)
        x3 = self.quality3(x)
        x4 = self.quality4(x)
        x5 = self.quality5(x)
        x6 = self.quality6(x)

            
        return x1, x2, x3, x4, x5, x6

if __name__ == '__main__':

    model = resnet50()

    print(model)