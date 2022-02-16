#!/usr/bin/python
#-*- coding: utf-8 -*-
import sys, time, os, argparse, socket
import yaml
import numpy
import pdb
import torch
import glob
from tuneThreshold import *
from SpeakerNet import SpeakerNet
from DatasetLoader import get_data_loader
import scipy.io as io
parser = argparse.ArgumentParser(description = "SpeakerNet");
parser.add_argument('--config',         type=str,   default=None,   help='Config YAML file');
## Data loader
parser.add_argument('--max_frames',     type=int,   default=200,    help='Input length to the network for training');
parser.add_argument('--eval_frames',    type=int,   default=300,    help='Input length to the network for testing; 0 uses the whole files');
parser.add_argument('--batch_size',     type=int,   default=200,    help='Batch size, number of speakers per batch');
parser.add_argument('--max_seg_per_spk', type=int,  default=100,    help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=4,     help='Number of loader threads');
parser.add_argument('--augment',        type=bool,  default=False,  help='Augment input')
parser.add_argument('--spec_aug',        type=bool,  default=False,  help='Augment input')
## Training details
parser.add_argument('--test_interval',  type=int,   default=10,     help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=500,    help='Maximum number of epochs');
parser.add_argument('--trainfunc',      type=str,   default="",     help='Loss function');
## Optimizer
parser.add_argument('--optimizer',      type=str,   default="adam", help='sgd or adam');
parser.add_argument('--scheduler',      type=str,   default="steplr", help='Learning rate scheduler');
parser.add_argument('--lr',             type=float, default=0.001,  help='Learning rate');
parser.add_argument("--lr_decay",       type=float, default=0.75,   help='Learning rate decay every [test_interval] epochs');
parser.add_argument('--weight_decay',   type=float, default=2e-5,      help='Weight decay in the optimizer');
## Loss functions
parser.add_argument("--hard_prob",      type=float, default=0.5,    help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank",      type=int,   default=10,     help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin',         type=float, default=1,      help='Loss margin, only for some loss functions');
parser.add_argument('--scale',          type=float, default=15,     help='Loss scale, only for some loss functions');
parser.add_argument('--nPerSpeaker',    type=int,   default=1,      help='Number of utterances per speaker per batch, only for metric learning based losses');
parser.add_argument('--nClasses',       type=int,   default=10,   help='Number of speakers in the softmax layer, only for softmax-based losses');
## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
parser.add_argument('--save_path',      type=str,   default="./data/exp1", help='Path for model and logs');
## Training and test data
parser.add_argument('--train_list',     type=str,   default="",     help='Train list');
parser.add_argument('--test_list',      type=str,   default="",     help='Evaluation list');
parser.add_argument('--enroll_list', type=str, default="", help='enroll list')
parser.add_argument('--enroll_path', type=str, default='./train/wav', help='Absolute path to the enroll set')
parser.add_argument('--train_path',     type=str,   default="./train/wav", help='Absolute path to the train set');
parser.add_argument('--test_path',      type=str,   default="./test/wav", help='Absolute path to the test set');
parser.add_argument('--musan_path',     type=str,   default="./musan", help='Absolute path to the test set')
parser.add_argument('--rir_path',       type=str,   default="./simulated_rirs", help='Absolute path to the test set');
## Model definition
parser.add_argument('--n_mels',         type=int,   default=40,     help='Number of mel filterbanks');
parser.add_argument('--log_input',      type=bool,  default=False,  help='Log input features')
parser.add_argument('--model',          type=str,   default="",     help='Name of model definition');
parser.add_argument('--encoder_type',   type=str,   default="SAP",  help='Type of encoder');
parser.add_argument('--nOut',          type=int,   default=512,    help='Embedding size in the last FC layer');
parser.add_argument('--gpu', type=int, default=1, help='GPU device number.')
## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')
args = parser.parse_args();
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['CUDA_LAUNCH_BLOCKING'] = str(args.gpu)
## Parse YAML
def find_option_type(key, parser):
    for opt in parser._get_optional_actions():
        if ('--' + key) in opt.option_strings:
           return opt.type
    raise ValueError

if args.config is not None:
    with open(args.config, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        if k in args.__dict__:
            typ = find_option_type(k, parser)
            args.__dict__[k] = typ(v)
        else:
            sys.stderr.write("Ignored unknown parameter {} in yaml.\n".format(k))
## Initialise directories
model_save_path     = args.save_path+"/model"
result_save_path    = args.save_path+"/result"
if not(os.path.exists(model_save_path)):
    os.makedirs(model_save_path)     
if not(os.path.exists(result_save_path)):
    os.makedirs(result_save_path)
## Load models
if args.eval == True:
    with torch.no_grad():
        s = SpeakerNet(**vars(args));
else:
    s = SpeakerNet(**vars(args));
it          = 1;
prevloss    = float("inf");
sumloss     = 0;
min_eer     = [100];
min_acc = [0];
## Load model weights
modelfiles = glob.glob('%s/model0*.model'%model_save_path)
modelfiles.sort()
if len(modelfiles) >= 1:
    s.loadParameters(modelfiles[-1]);
    print("Model %s loaded from previous state!"%modelfiles[-1]);
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
if(args.initial_model != ""):
    s.loadParameters(args.initial_model); 
    print("Model %s loaded!"%args.initial_model);
print(it)
for ii in range(0,it-1):
    s.__scheduler__.step()    
## Evaluation code
if args.eval == True:
    sc, lab, trials = s.evaluate_enroll_test(args.test_list, test_path=args.test_path,enroll_list=args.enroll_list,enroll_path=args.enroll_path,nDataLoaderThread=10,print_interval=100,num_eval=10,eval_frames=args.eval_frames)
    result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
    p_target = 0.5
    c_miss = 1
    c_fa = 1
    fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
    mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)
    eer2 , m_dcf = compute_min_cost(sc,lab,p_target)
    print('EER %2.4f MinDCF %.5f'%(result[1],mindcf))
    print('EER %2.4f MinDCF %.5f'%(eer2,m_dcf))
    userinp = input()
    while True:
        if userinp == '':
            break;
        else:
            with open(userinp,'w') as outfile:
                for vi, val in enumerate(sc):
                    outfile.write('%s %.4f\n'%(trials[vi], val))
            break;
    print('trials')
    userinp = input()
    while True:
        if userinp == '':
            break;
        else:
            with open(userinp,'w') as outfile:
                for vi, val in enumerate(sc):
                    outfile.write('%s %.4f\n'%(trials[vi], lab[vi]))
            break;
    quit()
    print('test-train')
    userinp = input()
    while True:
        if userinp == '':
            break;
        else:
            with open(userinp,'w') as outfile:
                for vi, val in enumerate(all_scores_test_train):
                    outfile.write('%.4f %s\n'%(val, all_trials_test_train[vi]))
            break;
    print('enroll-train')
    userinp = input()
    while True:
        if userinp == '':
            quit();
        else:
            with open(userinp,'w') as outfile:
                for vi, val in enumerate(all_scores_enroll_train):
                    outfile.write('%.4f %s\n'%(val, all_trials_enroll_train[vi]))
            quit();
    quit();
## Write args to scorefile
scorefile = open(result_save_path+"/scores.txt", "a+");
for items in vars(args):
    print(items, vars(args)[items]);
    scorefile.write('%s %s\n'%(items, vars(args)[items]));
scorefile.flush()
## Initialise data loader
trainLoader = get_data_loader(args.train_list, **vars(args));
test_eer = []
train_acc = []
train_loss = []
train_eer = []
while(1):   
    clr = [x['lr'] for x in s.__optimizer__.param_groups]
    print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training %s with LR %f..."%(args.model,max(clr))); 
    if it == 0:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...");
        sc, lab, _ = s.evaluate_enroll_test(args.test_list, test_path=args.test_path,enroll_list=args.enroll_list, enroll_path=args.enroll_path,nDataLoaderThread=8, print_interval=100, num_eval=10, eval_frames=args.eval_frames)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
        min_eer.append(result[1])
        test_eer.append(result[1])
        scorefile.flush()
    loss, trainacc, traineer = s.train_network(loader=trainLoader);
    train_acc.append(trainacc)
    train_eer.append(traineer)
    train_loss.append(loss)
    ## Validate and save
    if it % args.test_interval == 0:
        s.saveParameters(model_save_path+"/model%09d.model"%it);  
        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...");
        sc, lab, _ = s.evaluate_enroll_test(args.test_list, test_path=args.test_path,enroll_list=args.enroll_list,enroll_path=args.enroll_path,nDataLoaderThread=10,print_interval=100,num_eval=10,eval_frames=args.eval_frames)
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
        min_eer.append(result[1])
        test_eer.append(result[1])
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TAcc %2.2f, TEER %2.4f, TLOSS %f, VEER %2.4f, MINEER %2.4f"%( max(clr), trainacc, traineer, loss, result[1], min(min_eer)));
        scorefile.write("IT %d, LR %f, TAcc %2.2f, TEER %2.4f, TLOSS %f, VEER %2.4f, MINEER %2.4f\n"%(it, max(clr), trainacc, traineer, loss, result[1], min(min_eer)));
        scorefile.flush()
        with open(model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
            eerfile.write('%.4f'%result[1])
    else:
        s.saveParameters(model_save_path+"/model%09d.model"%it);
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TAcc %2.2f, TEER %2.4f, TLOSS %f"%( max(clr), trainacc, traineer, loss));
        scorefile.write("IT %d, LR %f,  TAcc %2.2f, TEER %2.4f, TLOSS %f\n"%(it, max(clr), trainacc, traineer, loss));
    if it >= args.max_epoch:
        #quit();
        break
    it+=1;
    io_save_path = args.save_path+"/save_result.mat"
    io.savemat(io_save_path, {'train_acc':train_acc,"train_loss":train_loss, "train_eer" :train_eer, "test_eer":test_eer})
    print("");

io_save_path = args.save_path+"/save_result.mat"
io.savemat(io_save_path, {'train_acc':train_acc,"train_loss":train_loss, "train_eer" :train_eer, "test_eer":test_eer})
scorefile.close();