import os, argparse

import numpy as np

import torch
from torchvision import transforms

import models.ResNet_staircase as ResNet_staircase
from PIL import Image

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat




def fit_function(y_output, popt):
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic



def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="Authentic Image Quality Assessment")
    parser.add_argument('--model_path', help='Path of model snapshot.', default='', type=str)
    parser.add_argument('--test_image_name', type=str)
    parser.add_argument('--test_method', default='five', type=str,
                        help='use the center crop or five crop to test the image (default: one)')
    parser.add_argument('--output_name', type=str)


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    test_image_name = args.test_image_name
    database = ['Koniq10k', 'SPAQ', 'LIVE_challenge', 'BID', 'FLIVE', 'FLIVE_patch']

    popt_all = [[120.41629963, -28.56005564, 46.48938183, 34.6190837],\
                [85.18902596, 11.6431685, 53.20173936, 17.09686183],\
                    [87.71193443, 13.06699313, 52.12460518, 20.14566219],\
                        [93.65937742, 0.51803345, 49.84010415, 28.07474279],\
                            [94.71514948, 21.6468321, 39.27372875, 15.37350998],\
                                [81.43059766, 23.51273452, 47.11992594, 9.2093784 ]]
    # model file
    model_path_all = ['ResNet_staircase_50-EXP1-Koniq10k.pkl',\
                      'ResNet_staircase_50-EXP1-SPAQ.pkl',\
                        'ResNet_staircase_50-EXP1-LIVE_challenge.pkl',\
                            'ResNet_staircase_50-EXP1-BID.pkl',\
                                'ResNet_staircase_50-EXP1-FLIVE.pkl',\
                                    'ResNet_staircase_50-EXP1-FLIVE_patch.pkl']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_scores_all = np.zeros([6])

    for i in range(6):
        model_path = model_path_all[i]
        popt = popt_all[i]

        output_name = args.output_name

        trained_database = database[i]
        
        # load the network
        model = ResNet_staircase.resnet50(pretrained = False)
        model = torch.nn.DataParallel(model)
        model = model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()


        if trained_database == 'FLIVE':
            if args.test_method == 'one':
                transformations_test = transforms.Compose([transforms.Resize(340),transforms.CenterCrop(320), \
                    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            elif args.test_method == 'five':
                transformations_test = transforms.Compose([transforms.Resize(340),transforms.FiveCrop(320), \
                    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), \
                        (lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                            std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])
        elif trained_database == 'FLIVE_patch':
            if args.test_method == 'one':
                transformations_test = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), \
                    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            elif args.test_method == 'five':
                transformations_test = transforms.Compose([transforms.Resize(256),transforms.FiveCrop(224), \
                    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), \
                        (lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                            std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])
        else:
            if args.test_method == 'one':
                transformations_test = transforms.Compose([transforms.Resize(384),transforms.CenterCrop(320), \
                    transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            elif args.test_method == 'five':
                transformations_test = transforms.Compose([transforms.Resize(384),transforms.FiveCrop(320), \
                    (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), \
                        (lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])

        test_image = Image.open(test_image_name)
        test_image = test_image.convert('RGB')
        test_image = transformations_test(test_image)
        test_image = test_image.unsqueeze(0)



        with torch.no_grad():
            if args.test_method == 'one':
                test_image = test_image.to(device)
                if trained_database == 'FLIVE':
                    outputs,_,_,_,_,_ = model(test_image)
                elif trained_database == 'FLIVE_patch':
                    _,outputs,_,_,_,_ = model(test_image)
                elif trained_database == 'LIVE_challenge':
                    _,_,outputs,_,_,_ = model(test_image)
                elif trained_database == 'Koniq10k':
                    _,_,_,outputs,_,_ = model(test_image)
                elif trained_database == 'SPAQ':
                    _,_,_,_,outputs,_ = model(test_image)
                elif trained_database == 'BID':
                    _,_,_,_,_,outputs = model(test_image)           
                test_scores = outputs.item()
                test_scores = fit_function(test_scores, popt)
                test_scores_all[i] = test_scores


            elif args.test_method == 'five':
                bs, ncrops, c, h, w = test_image.size()
                itest_imagemage = test_image.to(device)
                if trained_database == 'FLIVE':
                    outputs,_,_,_,_,_ = model(test_image.view(-1, c, h, w))
                elif trained_database == 'FLIVE_patch':
                    _,outputs,_,_,_,_ = model(test_image.view(-1, c, h, w))
                elif trained_database == 'LIVE_challenge':
                    _,_,outputs,_,_,_ = model(test_image.view(-1, c, h, w))
                elif trained_database == 'Koniq10k':
                    _,_,_,outputs,_,_ = model(test_image.view(-1, c, h, w))
                elif trained_database == 'SPAQ':
                    _,_,_,_,outputs,_ = model(test_image.view(-1, c, h, w))
                elif trained_database == 'BID':
                    _,_,_,_,_,outputs = model(test_image.view(-1, c, h, w))
                test_scores = outputs.view(bs, ncrops, -1).mean(1).item()
                test_scores = fit_function(test_scores, popt)
                test_scores_all[i] = test_scores



    test_scores = np.mean(test_scores_all)
    print(test_image_name)
    print(test_scores)

    if not os.path.exists(output_name):
        os.system(r"touch {}".format(output_name))

    f = open(output_name,'w')
    f.write(test_image_name)
    f.write(',')
    f.write(str(test_scores))
    f.write('\n')

    f.close()
