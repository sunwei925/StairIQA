import os, argparse, time

import numpy as np
import yaml

import torch
import torch.nn as nn
from torchvision import transforms
from utils import performance_fit

import IQADataset
# import scipy.io as scio
import models.stairIQA_resnet as stairIQA_resnet


def train_and_test(model, optimizer, criterion, trained_model_file, args, data_name):
    
    n_epoch = args.config[data_name]['n_epochs']
    train_loader = args.train_loader[data_name]
    test_loader = args.test_loader[data_name]
    print_samples = args.print_samples
    num_epoch = args.num_epochs
    n_train = args.n_train_sample[data_name]
    batch_size = args.batch_size
    n_test = args.n_test_sample[data_name]

    for i_epoch in range(n_epoch):
        print(data_name + ':')
        print("eval mode")
        # eval
        model.eval()
        batch_losses = []
        batch_losses_each_disp = []
        session_start_time = time.time()
        for i, (image, mos) in enumerate(train_loader):
            image = image.to(device)
            mos = mos[:,np.newaxis]
            mos = mos.to(device)

            if data_name == 'FLIVE':
                mos_output,_,_,_,_,_ = model(image)
            elif data_name == 'FLIVE_patch':
                _,mos_output,_,_,_,_ = model(image)
            elif data_name == 'LIVE_challenge':
                _,_,mos_output,_,_,_ = model(image)
            elif data_name == 'Koniq10k':
                _,_,_,mos_output,_,_ = model(image)
            elif data_name == 'SPAQ':
                _,_,_,_,mos_output,_ = model(image)
            elif data_name == 'BID':
                _,_,_,_,_,mos_output = model(image)

            # MSE loss
            loss = criterion(mos_output,mos)
            batch_losses.append(loss.item())
            batch_losses_each_disp.append(loss.item())

            optimizer.zero_grad()   # clear gradients for next train
            torch.autograd.backward(loss)
            optimizer.step()

            if (i+1) % print_samples == 0:
                session_end_time = time.time()
                avg_loss_epoch = sum(batch_losses_each_disp) / print_samples
                print ('Epoch [%d/%d], Iter [%d/%d] Losses: %.4f CostTime: %.4f' % \
                    (epoch*n_epoch+i_epoch+1, num_epoch*n_epoch, i+1, n_train//batch_size, \
                        avg_loss_epoch, session_end_time-session_start_time))    
                batch_losses_each_disp = []
                session_start_time = time.time()

        avg_loss = sum(batch_losses)/(i+1)
        print('Epoch [%d/%d], training loss is: %.4f' %(epoch*n_epoch+i_epoch+1, num_epoch*n_epoch, avg_loss))

        # Test 
        model.eval()
        y_output = np.zeros(n_test)
        y_test = np.zeros(n_test)

        with torch.no_grad():
            for i, (image, mos) in enumerate(test_loader):
                if args.test_method == 'one':
                    image = image.to(device)
                    y_test[i] = mos.item()
                    mos = mos.to(device)
                    if data_name == 'FLIVE':
                        outputs,_,_,_,_,_ = model(image)
                    elif data_name == 'FLIVE_patch':
                        _,outputs,_,_,_,_ = model(image)
                    elif data_name == 'LIVE_challenge':
                        _,_,outputs,_,_,_ = model(image)
                    elif data_name == 'Koniq10k':
                        _,_,_,outputs,_,_ = model(image)
                    elif data_name == 'SPAQ':
                        _,_,_,_,outputs,_ = model(image)
                    elif data_name == 'BID':
                        _,_,_,_,_,outputs = model(image)
                    y_output[i] = outputs.item()
                    
                elif args.test_method == 'five':
                    bs, ncrops, c, h, w = image.size()
                    y_test[i] = mos.item()
                    image = image.to(device)
                    mos = mos.to(device)

                    if data_name == 'FLIVE':
                        outputs,_,_,_,_,_ = model(image.view(-1, c, h, w))
                    elif data_name == 'FLIVE_patch':
                        _,outputs,_,_,_,_ = model(image.view(-1, c, h, w))
                    elif data_name == 'LIVE_challenge':
                        _,_,outputs,_,_,_ = model(image.view(-1, c, h, w))
                    elif data_name == 'Koniq10k':
                        _,_,_,outputs,_,_ = model(image.view(-1, c, h, w))
                    elif data_name == 'SPAQ':
                        _,_,_,_,outputs,_ = model(image.view(-1, c, h, w))
                    elif data_name == 'BID':
                        _,_,_,_,_,outputs = model(image.view(-1, c, h, w))

                    outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
                    y_output[i] = outputs_avg.item()

        test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(y_test, y_output)
        print("Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SRCC, test_KRCC, test_PLCC, test_RMSE))

        if test_SRCC > args.best_test_criterion[data_name]:
            print("Update best model using best_val_criterion ")

            torch.save(model.state_dict(), trained_model_file + data_name + '.pkl')

            args.best_performance[data_name][0:4] = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
            args.best_test_criterion[data_name] = test_SRCC  # update best val SROCC

            print("The best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SRCC, test_KRCC, test_PLCC, test_RMSE))

    scheduler.step()
    lr_current = scheduler.get_last_lr()
    print('The current learning rate is {:.06f}'.format(lr_current[0]))

    return


def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="In the wild Image Quality Assessment")
    parser.add_argument('--gpu', dest='gpu_id', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
          default=30, type=int)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=40, type=int)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--decay_ratio', type=float, default=0.9)
    parser.add_argument('--decay_interval', type=float, default=10)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--model', default='stairIQA_resnet', type=str,
                        help='model name (default: stairIQA_resnet)')
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--print_samples', type=int, default = 50)
    parser.add_argument('--test_method', default='five', type=str,
                        help='use the center crop or five crop to test the image (default: one)')
    parser.add_argument('--exp_id', default=0, type=int,
                        help='exp id for train-test splits (default: 0)')


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    with open('config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    args.config = config

    print('The current exp_id is ' + str(args.exp_id))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    args.train_loader = {}
    args.test_loader = {}
    args.best_test_criterion = {}
    args.best_performance = {}
    args.n_train_sample = {}
    args.n_test_sample = {}

    for i_database in args.config:
        
        train_filename_list = os.path.join(args.config[i_database]['filename_dir'], \
                                            args.config[i_database]['train_filename'] + '_' + str(args.exp_id)+'.csv')
        test_filename_list = os.path.join(args.config[i_database]['filename_dir'], \
                                            args.config[i_database]['test_filename'] + '_' + str(args.exp_id)+'.csv')
        

        transformations_train = transforms.Compose([transforms.Resize(args.config[i_database]['resize']),\
                                                    transforms.RandomCrop(args.config[i_database]['crop_size']), \
                                                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                                                                                   std=[0.229, 0.224, 0.225])])

        if args.test_method == 'one':
            transformations_test =  transforms.Compose([transforms.Resize(args.config[i_database]['resize']),\
                                                        transforms.CenterCrop(args.config[i_database]['crop_size']), \
                                                            transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                                                                                                       std=[0.229, 0.224, 0.225])])
        elif args.test_method == 'five':
            transformations_test =  transforms.Compose([transforms.Resize(args.config[i_database]['resize']),\
                                                        transforms.FiveCrop(args.config[i_database]['crop_size']), (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), (lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])


        train_dataset = IQADataset.IQA_dataloader(args.config[i_database]['database_dir'], train_filename_list, transformations_train, i_database)
        test_dataset = IQADataset.IQA_dataloader(args.config[i_database]['database_dir'], test_filename_list, transformations_test, i_database)

        args.train_loader[i_database] = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        args.test_loader[i_database] = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8)

        args.best_test_criterion[i_database] = -1
        args.best_performance[i_database] = np.zeros(4)

        args.n_train_sample[i_database] = len(train_dataset)
        args.n_test_sample[i_database] = len(test_dataset)


    trained_model_file = os.path.join(args.snapshot, '{}-EXP{}-'.format(args.model, args.exp_id))
    
    # load the network
    if args.model == 'stairIQA_resnet':
        model = stairIQA_resnet.resnet50_imdt(pretrained = True)


    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.to(device)

    criterion = nn.MSELoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0000001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)


    print("Ready to train network")


    for epoch in range(args.num_epochs):
        model.eval()
        
        # train and test FLIVE patch
        train_and_test(model, optimizer, criterion, trained_model_file, args, 'FLIVE_patch')
        train_and_test(model, optimizer, criterion, trained_model_file, args, 'LIVE_challenge')
        train_and_test(model, optimizer, criterion, trained_model_file, args, 'BID')

        train_and_test(model, optimizer, criterion, trained_model_file, args, 'FLIVE')
        train_and_test(model, optimizer, criterion, trained_model_file, args, 'LIVE_challenge')
        train_and_test(model, optimizer, criterion, trained_model_file, args, 'BID')
        
        train_and_test(model, optimizer, criterion, trained_model_file, args, 'Koniq10k')
        train_and_test(model, optimizer, criterion, trained_model_file, args, 'LIVE_challenge')
        train_and_test(model, optimizer, criterion, trained_model_file, args, 'BID')
        
        train_and_test(model, optimizer, criterion, trained_model_file, args, 'SPAQ')
        train_and_test(model, optimizer, criterion, trained_model_file, args, 'LIVE_challenge')
        train_and_test(model, optimizer, criterion, trained_model_file, args, 'BID')


    
    
    
    
    
    
    
    
    
    
    
    for i_database in args.config:
        print(i_database)
        print("The best Val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(args.best_performance[i_database][0], \
            args.best_performance[i_database][1], args.best_performance[i_database][2], args.best_performance[i_database][3]))
        np.save(os.path.join(args.results_path, args.model + '_' + i_database + '_' + str(args.exp_id)), args.best_performance[i_database])