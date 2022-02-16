#!/usr/bin/python
#-*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy, math, pdb, sys, random
import time, os, itertools, shutil, importlib
from tuneThreshold import tuneThresholdfromScore
from DatasetLoader import *
import numpy as np
from scipy.spatial.distance import cdist
import csv
import scipy.io as io
class SpeakerNet(nn.Module):

    def __init__(self, model, optimizer, scheduler, trainfunc, **kwargs):
        super(SpeakerNet, self).__init__();

        SpeakerNetModel = importlib.import_module('models.'+model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs).cuda();

        LossFunction = importlib.import_module('loss.'+trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs).cuda();

        Optimizer = importlib.import_module('optimizer.'+optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.parameters(), **kwargs)

        Scheduler = importlib.import_module('scheduler.'+scheduler).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)
        assert self.lr_step in ['epoch', 'iteration']
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Train network
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def train_network(self, loader):

        self.train();

        stepsize = loader.batch_size;

        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0     # EER or accuracy
        eer1 = 0
        tstart = time.time()
        
        for data, data_label in loader:

            data = data.transpose(0,1)

            self.zero_grad();

            feat = []
            for inp in data:
                outp   = self.__S__.forward(inp.cuda())
                feat.append(outp)

            feat = torch.stack(feat,dim=1).squeeze()

            label   = torch.LongTensor(data_label).cuda()

            nloss, prec1, eer = self.__L__.forward(feat,label)

            loss    += nloss.detach().cpu();
            top1    += prec1
            eer1 += eer
            counter += 1;
            index   += stepsize;

            nloss.backward();
            self.__optimizer__.step();

            telapsed = time.time() - tstart
            tstart = time.time()

            sys.stdout.write("\rProcessing (%d) "%(index));
            sys.stdout.write("Loss %f TAcc %2.3f%% TEER %2.4f - %.2f Hz "%(loss/counter, top1/counter, eer1 * 1.0/ counter, stepsize/telapsed));
            sys.stdout.flush();

            if self.lr_step == 'iteration': self.__scheduler__.step()

        if self.lr_step == 'epoch': self.__scheduler__.step()

        sys.stdout.write("\n");
        
        return (loss/counter, top1/counter , eer1 * 1.0/ counter);
    def train_network_first(self, loader):
    
        self.eval();
        stepsize = loader.batch_size;
        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0     # EER or accuracy
        eer1 = 0
        tstart = time.time()
        for data, data_label in loader:
            data = data.transpose(0,1)
            #self.zero_grad();
            feat = []
            for inp in data:
                outp   = self.__S__.forward(inp.cuda())
                feat.append(outp)
            feat = torch.stack(feat,dim=1).squeeze()
            label   = torch.LongTensor(data_label).cuda()
            print(label)
            nloss, prec1, eer = self.__L__.forward(feat,label)
            loss    += nloss.detach().cpu();
            top1    += prec1
            counter += 1;
            eer1 += eer
            index   += stepsize;
            #nloss.backward();
            #self.__optimizer__.step();
            telapsed = time.time() - tstart
            tstart = time.time()
            sys.stdout.write("\rProcessing (%d) "%(index));
            sys.stdout.write("Loss %f TAcc %2.3f%% TEER %2.4f - %.2f Hz "%(loss/counter, top1/counter,eer1 * 1.0/counter, stepsize/telapsed));
            sys.stdout.flush();
            #if self.lr_step == 'iteration': self.__scheduler__.step()
        #if self.lr_step == 'epoch': self.__scheduler__.step()
        sys.stdout.write("\n");
        return (loss/counter, top1/counter, eer1 * 1.0/counter);


    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====
    def getTrainList(self, listfilename='', print_interval=100, train_path='', num_eval=1, eval_frames=None):
        self.eval();
        #trainLoader = get_data_loader(args.train_list, **vars(args));
        lines       = []
        files       = []
        utt = []

        feats       = {}
        tstart    = time.time()

        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line):
                    break;

                data = line.split();
                utt.append(data[0])
                files.append(data[1])
            #files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        # setfiles.sort()
        uttfiles = list(set(utt))
        uttfiles.sort()
        # dictkeys = list(set([x.split()[0] for x in lines]))
        # dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(uttfiles) }
        
        nDataLoaderThread = 8
        train_dataset = test_dataset_loader(setfiles, train_path, num_eval=num_eval, eval_frames=eval_frames)
        test_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )
        ## Save all features to file
        # for idx, file in enumerate(files):

        #     inp1 = torch.FloatTensor(loadWAV(os.path.join(train_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()
        #     #[num_eval, E]
        #     ref_feat = self.__S__.forward(inp1).detach()
        #     ref_feat = F.normalize(ref_feat, p=2, dim=1)
        #     ref_feat = ref_feat.cpu()
        #         # print(ref_feat.shape)
        #         # com_feat = F.normalize(com_feat, p=2, dim=1)  
        #     #filename = '%06d.wav'%idx

        #     feats[file]     = ref_feat

        #     telapsed = time.time() - tstart

        #     if idx % print_interval == 0:
        #         sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(files),idx/telapsed,ref_feat.size()[1]));
        # audio, test_list_index
        for idx, data in enumerate(test_loader):
            # audio
            inp1                = data[0][0].cuda()
            ref_feat            = self.__S__.forward(inp1).detach().cpu()
            # files
            feats[data[1][0]]   = ref_feat
            telapsed            = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

        print('')
        result_embedding, utt_id = [],[]
        ori_id = []
        for vi, val in enumerate(files):
            result_embedding.append(feats[val])
            utt_id.append(dictkeys[utt[vi]])
            ori_id.append(utt[vi])
        result_embedding = [t.numpy() for t in result_embedding]
        #[t.numpy() for t in utt_id]
        return (result_embedding, utt_id, ori_id)
        # io.savemat('ap17_test_all_SEresnet_W_v4.mat', {'embedding':result_embedding,"utt_id":utt_id,'ori_id':ori_id})
      
    def getTestEnrollList(self, listfilename='', print_interval=100, train_path='', num_eval=1, eval_frames=None):
        self.eval();
        #trainLoader = get_data_loader(args.train_list, **vars(args));
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()
        refs = []
        coms = []
        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line):
                    break;
                data = line.split();
                files.append(data[1])
                files.append(data[2])
                lines.append(line)
        setfiles = list(set(files))
        setfiles.sort()
        ## Save all features to file
        for idx, file in enumerate(setfiles):
            inp1 = torch.FloatTensor(loadWAV(os.path.join(train_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()
            #[num_eval, E]
            ref_feat = self.__S__.forward(inp1).detach().cpu()
            filename = '%06d.wav'%idx

            feats[file]     = ref_feat

            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));
        print('')
        #all_scores = [];
        all_labels = [];
        all_trials = [];
        tstart = time.time()
        ## Read files and compute all scores
        for idx, line in enumerate(lines):
            data = line.split();
            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data
            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()
            if self.__L__.test_normalize:
                print(ref_feat.shape)
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                print(ref_feat.shape)
                com_feat = F.normalize(com_feat, p=2, dim=1)  
            all_labels.append(int(data[0]));
            refs.append(ref_feat.cpu())
            coms.append(com_feat.cpu())
            all_trials.append(data[1]+" "+data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        print('\n')
        refs = [t.numpy() for t in refs]
        coms = [t.numpy() for t in coms]

        io.savemat('save_test_enroll.mat', {'enroll_embedding':refs,"tables":all_labels, "com_embedding":coms})
        #return (all_labels, all_trials)

    def prepare_plda(self, train_file_name, print_interval=100, train_path="", num_eval=1, eval_frames=None):
        #self.getTrainList(train_file_name, print_interval=print_interval, train_path=train_path, num_eval=1, eval_frames=eval_frames)
        
        #dev_id, dev_utt, dev_vector = load_ivector('train_plda.txt')
        print('Loading data')
        data = io.loadmat('save.mat')
        dev_id = data['utt_id']
        dev_vector = data['embedding']

        dev_vector = np.squeeze(dev_vector)
        dev_id = np.squeeze(dev_id)
        #print(dev_id)
        #print(dev_vector)
        # print(dev_vector)
        # print(dev_id)
        le = LabelEncoder()
        trn_id_enc = le.fit_transform(dev_id)
        #print(trn_id_enc)
        num_bl_spk = len(np.unique(trn_id_enc))
        #print(num_bl_spk)
        trn_id_cat = np_utils.to_categorical(trn_id_enc, num_classes=num_bl_spk)
        #print(trn_id_cat)
        # train plda model
        print('Training plda model')
        plda_model = plda.fit_plda_model_simplified(dev_vector, trn_id_cat, numiter=10, vdim=192, udim=192)

        #print(plda_model.shape)
        #print(m.shape)
        #print(w.shape)
      
        #io.savemat('save_plda_par.mat', {'model':plda_model,"m":m,"w":w})
        return plda_model

    def evaluateFromList_by_identification(self, listfilename, print_interval=100, test_path='', num_eval=1, eval_frames=None, batch_size=128):        
        self.eval();
        stepsize = batch_size;
        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0    # EER or accuracy
        top5 = 0
        lines       = []
        files       = []
        feats       = []
        table = []
        tables = []
        utt = []
        tstart = time.time()
        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line):
                    break;
                data = line.split();
                files.append(data[1])
                utt.append(data[0])
                lines.append(line)
        # uttfiles = list(set(utt))
        # uttfiles = uttfiles.sort()
        utts = list(set([x.split()[0] for x in lines]))
        utts.sort()
        dictkeys = { key : ii for ii, key in enumerate(utts) }

        ## Save all features to file
        feat = []
        #print(dictkeys)
        print(len(utt))
        print(len(dictkeys))
        for idx, file in enumerate(files):

            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()
            #[num_eval, E]
            print(inp1.shape)
            ref_feat = self.__S__.forward(inp1).detach().cpu()
            feat.append(ref_feat)
            print(utt[idx])
            print(dictkeys[utt[idx]])
            table.append(dictkeys[utt[idx]])
            telapsed = time.time() - tstart

            # if idx % print_interval == 0:
            #     sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));
        print(len(feat))
        print(len(table))
        tem_feat, tem_table = [], []
        for id, embding in enumerate(feat):
            if id > 0 and id % stepsize == 0:
                feats.append(tem_feat)
                tables.append(tem_table)
                tem_feat, tem_table = [], []
                tem_feat.append(feat[id])
                tem_table.append(table[id])
            else:
                tem_feat.append(feat[id])
                tem_table.append(table[id])
            if id == len(feat) - 1:
                feats.append(tem_feat)
                tables.append(tem_table)
        for id, embding in enumerate(feats):
            # data = data.transpose(0,1)
            # self.zero_grad();
            # feat = []
            # for inp in embding:
            #     outp   = self.__S__.forward(inp.cuda())
            #     feat.append(outp)
            #feat = torch.stack(feat,dim=1).squeeze()
            #label   = torch.LongTensor(data_label).cuda()
            # print(feats[id])
            # print(tables[id])
            feats[id] =[t.numpy() for t in feats[id]]
            #tables[id] = [t.numpy() for t in tables[id]]
            # print(feats[id])
            # print(tables[id])
            # print(feats[id].shape)
            tem_table = torch.LongTensor(tables[id]).cuda()
            tem_feat = torch.tensor(feats[id], dtype=torch.float32).cuda()
            tem_feat = torch.squeeze(tem_feat, 1)
            # print(tem_feat)
            # print(tem_table)
            # print(tem_feat.shape)
            nloss, prec1, prec5 = self.__L__.forward(tem_feat, tem_table)
            print(prec1)
            loss    += nloss.detach().cpu();
            top1    += prec1
            top5 += prec5
            counter += 1;
            if id < len(feats) - 1: 
                index  += stepsize;
            else:
                index += len(tables[id])
            #nloss.backward();
            #self.__optimizer__.step();

            telapsed = time.time() - tstart
            tstart = time.time()

            sys.stdout.write("\rProcessing (%d) "%(index));
            sys.stdout.write("Loss %f TEER/TAcc %2.3f%% - %.2f Hz "%(loss/counter, top1/counter, stepsize/telapsed));
            sys.stdout.flush()
        sys.stdout.write("\n");
        return (loss/counter, top1/counter, top5/counter);
    
    def evaluateFromList_by_plda(self, listfilename, train_file_name, print_interval=100, test_path='', train_path='',num_eval=10, eval_frames=None):
        self.eval();
        #self.getTestEnrollList(listfilename=listfilename, print_interval=100, train_path=test_path, num_eval=1, eval_frames=eval_frames)
        #quit()
        #self.getTrainList(listfilename=train_file_name, print_interval=100, train_path=train_path, num_eval=1, eval_frames=eval_frames)
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()
        print('Loading data')
        data = io.loadmat('save_ecapa.mat')
        dev_id = data['utt_id']
        dev_vector = data['embedding']
        dev_vector = np.squeeze(dev_vector)
        dev_id = np.squeeze(dev_id)
        le = LabelEncoder()
        trn_id_enc = le.fit_transform(dev_id)
        num_bl_spk = len(np.unique(trn_id_enc))
        trn_id_cat = np_utils.to_categorical(trn_id_enc, num_classes=num_bl_spk)
        print(num_bl_spk)
        print(len(dev_vector))
        # trn_id_cat = np_utils.to_categorical(trn_id_enc, num_classes=num_bl_spk)
        # clf_deep = LinearDiscriminantAnalysis(n_components=100)
        # clf_deep.fit(dev_vector, dev_id)

        # clf_deep = LinearDiscriminantAnalysis(n_components=192)
        # clf_deep.fit(dev_vector, dev_id)
        # dev_vector_lda = clf_deep.transform(dev_vector)

        #dev_vector_lda = evaluate_system.length_norm(dev_vector_lda)
        # evalute lda-plda
        
        print('training LDA-PLDA model')
        numiter = 1
        # vdim=192
        # udim=192
        # plda_model = plda.fit_plda_model_simplified(dev_vector_lda, trn_id_cat, numiter=numiter,vdim=vdim, udim=udim)
        
        plda_model = plda.fit_plda_model_two_cov(dev_vector, trn_id_cat, numiter=numiter)
        # dev_scores_lda_plda = plda.get_plda_scores_two_cov(lda_plda_model, spk_mean_lda, dev_ivector_lda)
        # dev_ensemble_scores_lda_plda = plda.get_plda_scores_two_cov(lda_plda_model, trn_bl_ivector_lda, dev_ivector_lda)

        # trn_bl_ivector_lda = clf_deep.transform(trn_bl_ivector_ln_la)
        # trn_bg_ivector_lda = clf_deep.transform(trn_bg_ivector_ln_la)
        # trn_ivector_lda = clf_deep.transform(trn_ivector_ln_la)
        # dev_ivector_lda = clf_deep.transform(dev_ivector_ln_la)
        # trn_bl_ivector_lda = length_norm(trn_bl_ivector_lda)
        # trn_bg_ivector_lda = length_norm(trn_bg_ivector_lda)

        # overfit_classifier = plda.Classifier()
        # overfit_classifier.fit_model(dev_vector, dev_id)

        # better_classifier = plda.Classifier()
        # better_classifier.fit_model(dev_vector, dev_id, n_principal_components=100)
        
        
        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line):
                    break;

                data = line.split();

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0,1)] + data

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        for idx, file in enumerate(setfiles):

            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()
            #[num_eval, E]
            ref_feat = self.__S__.forward(inp1).detach().cpu()

            filename = '%06d.wav'%idx

            feats[file]     = ref_feat

            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

        print('')
        all_scores = [];
        all_labels = [];
        all_trials = [];
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split();

            ## Append random label if missing
            # if len(data) == 2: data = [random.randint(0,1)] + data

            ref_feat = feats[data[1]]
            com_feat = feats[data[2]]
            # ref_feat = clf_deep.transform(ref_feat)
            # com_feat = clf_deep.transform(com_feat)

            ref_feat = torch.Tensor(ref_feat).cuda()
            com_feat = torch.Tensor(com_feat).cuda()
            print(idx)
            if self.__L__.test_normalize:
                print(ref_feat.shape)
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                print(ref_feat.shape)
                com_feat = F.normalize(com_feat, p=2, dim=1)
            ref_feat = ref_feat.cpu()
            com_feat = com_feat.cpu()
            ref_feat = ref_feat.numpy() 
            com_feat = com_feat.numpy() 

            #ref_feat = better_classifier.model.transform(ref_feat, from_space='D', to_space='U_model')
            #ref_feat = better_classifier.model.transform(ref_feat, from_space='D', to_space='X')
            #ref_feat = better_classifier.model.transform(ref_feat, from_space='D', to_space='U')
            #com_feat = better_classifier.model.transform(com_feat, from_space='D', to_space='U_model')
            #com_feat = better_classifier.model.transform(com_feat, from_space='D', to_space='X')
            #com_feat = better_classifier.model.transform(com_feat, from_space='D', to_space='U')

            #print(ref_feat.shape)
            # dist = better_classifier.model.calc_same_diff_log_likelihood_ratio(ref_feat, com_feat)

            
            # dist = better_classifier.model.calc_same_diff_log_likelihood_ratio(ref_feat, com_feat)
            # [I B, 1][I B ,1] --> [I B 1] [1 B I] = [I I]
            # [I I]
            #Model', ['v', 'u', 'mu', 'sigma'])
            # 'invb', 'invw', 'mu'
            #plda_model.v = plda_model.v.cpu()
            #plda_model.invb = plda_model.invb.cpu()
            #plda_model.invw = plda_model.invw.cpu()
            #plda_model.mu = plda_model.mu.cpu()
            #dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            #dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            
            dist = plda.get_plda_scores_two_cov(plda_model, ref_feat, com_feat)

            #dist = plda.get_plda_scores_simplified(plda_model, ref_feat, com_feat)
            
            print(dist)
            #biao
            score =  numpy.mean(dist);
            print(score)
            all_scores.append(score);  
            all_labels.append(int(data[0]));
            all_trials.append(data[1]+" "+data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        print('\n')

        return (all_scores, all_labels, all_trials);

    def evaluateFromList_by_lda(self, listfilename, train_file_name, print_interval=100, test_path='', train_path='',num_eval=10, eval_frames=None):
        self.eval();
        #self.getTrainList(listfilename=train_file_name, print_interval=100, train_path=train_path, num_eval=1, eval_frames=eval_frames)
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()
        print('Loading data')
        data = io.loadmat('save_ecapa.mat')
        dev_id = data['utt_id']
        dev_vector = data['embedding']
        dev_vector = np.squeeze(dev_vector)
        dev_id = np.squeeze(dev_id)
        le = LabelEncoder()
        trn_id_enc = le.fit_transform(dev_id)
        num_bl_spk = len(np.unique(trn_id_enc))
        print(num_bl_spk)
        trn_id_cat = np_utils.to_categorical(trn_id_enc, num_classes=num_bl_spk)
        clf_deep = LinearDiscriminantAnalysis(n_components=100)
        clf_deep.fit(dev_vector, dev_id)
        # trn_bl_ivector_lda = clf_deep.transform(trn_bl_ivector_ln_la)
        # trn_bg_ivector_lda = clf_deep.transform(trn_bg_ivector_ln_la)
        # trn_ivector_lda = clf_deep.transform(trn_ivector_ln_la)
        # dev_ivector_lda = clf_deep.transform(dev_ivector_ln_la)
        # trn_bl_ivector_lda = length_norm(trn_bl_ivector_lda)
        # trn_bg_ivector_lda = length_norm(trn_bg_ivector_lda)
        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line):
                    break;

                data = line.split();

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0,1)] + data

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        for idx, file in enumerate(setfiles):

            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()
            #[num_eval, E]
            ref_feat = self.__S__.forward(inp1).detach().cpu()

            filename = '%06d.wav'%idx

            feats[file]     = ref_feat

            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

        print('')
        all_scores = [];
        all_labels = [];
        all_trials = [];
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split();

            ## Append random label if missing
            # if len(data) == 2: data = [random.randint(0,1)] + data

            ref_feat = feats[data[1]]
            com_feat = feats[data[2]]
            ref_feat = clf_deep.transform(ref_feat)
            com_feat = clf_deep.transform(com_feat)

            ref_feat = torch.Tensor(ref_feat).cuda()
            com_feat = torch.Tensor(com_feat).cuda()
            if self.__L__.test_normalize:
                print(ref_feat.shape)
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                print(ref_feat.shape)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            # [I B, 1][I B ,1] --> [I B 1] [1 B I] = [I I]
            # [I I]
            #dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            print(dist)
            #biao
            score =  numpy.mean(dist);
            print(score)
            all_scores.append(score);  
            all_labels.append(int(data[0]));
            all_trials.append(data[1]+" "+data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        print('\n')

        return (all_scores, all_labels, all_trials);

    
     ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Evaluate from list
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def evaluateFromList2(self, test_list, test_path, nDataLoaderThread, print_interval=100, num_eval=10, eval_frames=400):
        
        self.eval();
        
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()

        ## Read all lines
        with open(test_list) as f:
            lines = f.readlines()

        ## Get a list of unique file names
        files = sum([x.strip().split()[-2:] for x in lines],[])
        setfiles = list(set(files))
        setfiles.sort()

        ## Define test data loader
        #nDataLoaderThread = 2
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, eval_frames=eval_frames)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )

        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp1                = data[0][0].cuda()
            ref_feat            = self.__S__.forward(inp1).detach().cpu()
            feats[data[1][0]]   = ref_feat
            telapsed            = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

        print('')
        all_scores = [];
        all_labels = [];
        all_trials = [];
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split()
            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()

            if self.__L__.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)
            #dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            #score = -1 * numpy.mean(dist);
            score = numpy.mean(dist)
            all_scores.append(score);  
            all_labels.append(int(data[0]));
            all_trials.append(data[1]+" "+data[2])
            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        print('')

        return (all_scores, all_labels, all_trials);

    def evaluateFromList(self, listfilename, print_interval=100, test_path='', num_eval=10, eval_frames=None):
        
        self.eval();
        #trainLoader = get_data_loader(args.train_list, **vars(args));
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()

        enroll_txt = []
        test_txt = []
        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line):
                    break;

                data = line.split();

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0,1)] + data

                files.append(data[1])
                files.append(data[2])
                enroll_txt.append(data[1])
                test_txt.append(data[2])
                lines.append(line)

        test_du = list(set(test_txt))
        enroll_du = list(set(enroll_txt))
        # print(len(test_du))
        # print(len(enroll_du))
        setfiles = list(set(files))
        setfiles.sort()

        ## Save all features to file
        for idx, file in enumerate(setfiles):

            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()
            #[num_eval, E]
            ref_feat = self.__S__.forward(inp1).detach().cpu()

            filename = '%06d.wav'%idx

            feats[file]     = ref_feat

            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

        # print('')
        all_scores = [];
        all_labels = [];
        all_trials = [];
        tstart = time.time()

        ## Read files and compute all scores
        for idx, line in enumerate(lines):

            data = line.split();

            ## Append random label if missing
            if len(data) == 2: data = [random.randint(0,1)] + data

            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()

            if self.__L__.test_normalize:
                # print(ref_feat.shape)
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                # print(ref_feat.shape)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            # [I B, 1][I B ,1] --> [I B 1] [1 B I] = [I I]
            # [I I]
            #dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            # print(dist)
            #biao
            score =  numpy.mean(dist);
            # print(score)
            all_scores.append(score);  
            all_labels.append(int(data[0]));
            all_trials.append(data[1]+" "+data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        # print('\n')

        return (all_scores, all_labels, all_trials);

    def evaluate_enroll_test(self, test_list, test_path,enroll_list, enroll_path,nDataLoaderThread, print_interval=100, num_eval=10, eval_frames=None):
        # test_list: id utt
        # enroll_list : id utt
        # produce trails enroll, test, 0/1
        #return (result_embedding, utt_id)
        result_embedding, utt_id, ori_id = self.getTrainList(listfilename=enroll_list, print_interval=100, train_path=enroll_path, num_eval=num_eval, eval_frames=eval_frames)
        dev_vector = np.squeeze(result_embedding)
        dev_id = np.squeeze(utt_id)
         # get 100 cohort avg embedding and sort 
        group_speaker = []
        tem_list = []
        train_sub = []
        speaker = []
        uttfiles = list(set(ori_id))
        uttfiles.sort()
        dictkeys_enroll = { key : ii for ii, key in enumerate(uttfiles) }
        for i, evl in enumerate(ori_id):
            tem_list.append(dev_vector[i])
            if i+1 < len(dev_id) and dev_id[i+1] != dev_id[i]:
                #train_sub.append(str('speaker_'+ str(dev_id[i])))
                train_sub.append(ori_id[i])
                group_speaker.append(tem_list)
                tem_list = []
            if i+1 == len(dev_id):
                #train_sub.append(str('speaker_'+ str(dev_id[i])))
                train_sub.append(ori_id[i])
                group_speaker.append(tem_list)
            # if len(group_speaker) == 1000:
            #     break
        assert len(train_sub) == len(group_speaker)
        train_dic = {}
        for i, evl in enumerate(group_speaker):
            speaker.append(np.sum(group_speaker[i], axis=0) / len(group_speaker[i]))
            train_dic[train_sub[i]] = np.sum(group_speaker[i], axis=0) / len(group_speaker[i])
        #print(train_dic)
        assert len(train_sub) == len(speaker)
        # print(len(train_sub))
        # print(train_sub)
        #quit()
        self.eval();
        
        utt = []
        test_utt = []
        lines       = []
        enroll_files  = []
        test_files = []
        feats       = {}
        tstart      = time.time()
        with open(test_list) as test_file:
            while True:
                line = test_file.readline()
                if (not line):
                    break
                data = line.split()
                test_utt.append(data[0])
                test_files.append(data[1])
                lines.append(line)
        uttfiles = list(set(test_utt))
        uttfiles.sort()
        dictkeys_test = { key : ii for ii, key in enumerate(uttfiles) }
        # assert dictkeys_enroll == dictkeys_test
        ## Get a list of unique file names
        # files = sum([x.strip().split()[-2:] for x in lines],[])
        set_test_files = list(set(test_files))
        #setfiles.sort()
        #set_enroll_files = list(set(enroll_files))
        ## Define test data loader
        #nDataLoaderThread = 2
        test_dataset = test_dataset_loader(set_test_files, test_path, num_eval=num_eval, eval_frames=eval_frames)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )
        # data : torch.FloatTensor(audio), self.test_list[index]
        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp1                = data[0][0].cuda()
            ref_feat            = self.__S__.forward(inp1).detach().cpu()
            feats[data[1][0]]   = ref_feat
            telapsed            = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(set_test_files),idx/telapsed,ref_feat.size()[1]));
        print('')
        all_scores = [];
        all_labels = [];
        all_trials = [];
        tstart = time.time()
        test_du = list(set(test_files))
        enroll_du = list(set(train_sub))
        #lines_enroll_test = []
        # produce enroll-test
        # for i in range(len(enroll_du)):
        #     for j in range(len(test_du)):
        #         tem.append(enroll_du[i])
        #         tem.append(test_du[j]) 
        #         lines_enroll_train.append(tem)
        #         tem=[]
        #print(len(lines_enroll_test))
        #quit
        ## Read files and compute all scores
        for idx, line in enumerate(lines):
            data = line.split();
            test_feat = feats[data[1]].cuda()
            test_feat = test_feat.squeeze(-1)
            #print(test_feat.shape)
            length = len(torch.tensor(train_dic[train_sub[0]]).shape)
            for ii, i in enumerate(train_sub):
                if length == 1:
                    enroll_feat = torch.tensor(train_dic[i]).cuda().unsqueeze(0)
                else:
                    enroll_feat = torch.tensor(train_dic[i]).cuda()
                #print(enroll_feat.shape)
                if i == data[0]:
                    lable = 1
                else:
                    lable = 0
                if self.__L__.test_normalize:
                    enroll_feat = F.normalize(enroll_feat, p=2, dim=1)
                    test_feat = F.normalize(test_feat, p=2, dim=1)
                #dist = F.pairwise_distance(enroll_feat.unsqueeze(-1), test_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
                # print(enroll_feat.shape)
                # print(test_feat.shape)
                dist = F.cosine_similarity(enroll_feat.unsqueeze(-1), test_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
                score = numpy.mean(dist)
                #score = -1 * numpy.mean(dist);
                all_scores.append(score);  
                all_labels.append(int(lable));
                all_trials.append(i+" "+data[1])
                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx * len(train_sub) + ii + 1,len(lines) * len(train_sub),(idx * len(train_sub) + ii + 1)/telapsed));
                    sys.stdout.flush();

        print('')
        return (all_scores, all_labels, all_trials);
    def evaluate_no_enroll_test(self, listfilename, print_interval=100, test_path='', num_eval=1, eval_frames=None, batch_size=128):        
        self.eval();
        stepsize = batch_size;
        counter = 0;
        index   = 0;
        loss    = 0;
        top1    = 0    # EER or accuracy
        top5 = 0
        lines       = []
        files       = []
        feats       = []
        table = []
        tables = []
        utt = []
        tstart = time.time()
        all_scores = [];
        all_labels = [];
        all_trials = [];
        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line):
                    break;
                data = line.split();
                files.append(data[1])
                utt.append(data[0])
                lines.append(line)
        # uttfiles = list(set(utt))
        # uttfiles = uttfiles.sort()
        utts = list(set([x.split()[0] for x in lines]))
        utts.sort()
        dictkeys = { key : ii for ii, key in enumerate(utts) }

        ## Save all features to file
        feat = []
        #print(dictkeys)
        print(len(utt))
        print(len(dictkeys))
        print(dictkeys)
        for idx, file in enumerate(files):

            inp1 = torch.FloatTensor(loadWAV(os.path.join(test_path,file), eval_frames, evalmode=True, num_eval=num_eval)).cuda()
            #[num_eval, E]
            #print(inp1.shape)
            ref_feat = self.__S__.forward(inp1).detach().cpu()
            feat.append(ref_feat)
            #print(utt[idx])
            #print(dictkeys[utt[idx]])
            table.append(dictkeys[utt[idx]])
            telapsed = time.time() - tstart

            # if idx % print_interval == 0:
            #     sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));
        print(len(feat))
        print(len(table))
        tem_feat, tem_table = [], []
        for id, embding in enumerate(feat):
            if id > 0 and id % stepsize == 0:
                feats.append(tem_feat)
                tables.append(tem_table)
                tem_feat, tem_table = [], []
                tem_feat.append(feat[id])
                tem_table.append(table[id])
            else:
                tem_feat.append(feat[id])
                tem_table.append(table[id])
            if id == len(feat) - 1:
                feats.append(tem_feat)
                tables.append(tem_table)
        all_scores, all_labels, all_trials = [], [], []
        for id, embding in enumerate(feats):
            # data = data.transpose(0,1)
            # self.zero_grad();
            # feat = []
            # for inp in embding:
            #     outp   = self.__S__.forward(inp.cuda())
            #     feat.append(outp)
            #feat = torch.stack(feat,dim=1).squeeze()
            #label   = torch.LongTensor(data_label).cuda()
            # print(feats[id])
            # print(tables[id])
            feats[id] =[t.numpy() for t in feats[id]]
            #tables[id] = [t.numpy() for t in tables[id]]
            # print(feats[id])
            # print(tables[id])
            # print(feats[id].shape)
            tem_table = torch.LongTensor(tables[id]).cuda()
            tem_feat = torch.tensor(feats[id], dtype=torch.float32).cuda()
            tem_feat = torch.squeeze(tem_feat, 1)
            # print(tem_feat)
            # print(tem_table)
            # print(tem_feat.shape)
            nloss, prec1, eer = self.__L__.forward(tem_feat, tem_table)
            #print(prec1)
            
            # all_scores.append(score);  
            # all_labels.append(int(lable));
            # all_trials.append(i+" "+data[1])

            loss    += nloss.detach().cpu();
            top1    += prec1
            top5 += eer
            counter += 1;
            if id < len(feats) - 1: 
                index  += stepsize;
            else:
                index += len(tables[id])
            #nloss.backward();
            #self.__optimizer__.step();

            telapsed = time.time() - tstart
            tstart = time.time()

            sys.stdout.write("\rProcessing (%d) "%(index));
            sys.stdout.write("Loss %f TEER/TAcc %2.3f%% - %.2f Hz "%(loss/counter, top1/counter, stepsize/telapsed));
            sys.stdout.flush()
        sys.stdout.write("\n");
        return (loss/counter, top1/counter, top5/counter);
    
    def evaluateFromList_s_norm(self, listfilename, sub_train_file_name, train_path='', print_interval=100, test_path='', num_eval=10, eval_frames=None):
        
        self.eval();
        #self.getTrainList(listfilename=train_file_name, print_interval=100, train_path=train_path, num_eval=1, eval_frames=eval_frames)
        print('Loading data')
        data = io.loadmat('save_ecapa_S4.mat')
        dev_id = data['utt_id']
        dev_vector = data['embedding']
        dev_vector = np.squeeze(dev_vector)
        dev_id = np.squeeze(dev_id)
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()

        enroll_txt = []
        test_txt = []
        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line):
                    break;

                data = line.split();

                ## Append random label if missing
                if len(data) == 2: data = [random.randint(0,1)] + data

                files.append(data[1])
                files.append(data[2])
                enroll_txt.append(data[1])
                test_txt.append(data[2])
                lines.append(line)
        train_sub = []
        with open(sub_train_file_name) as listfile:
            while True:
                line = listfile.readline();
                if (not line):
                    break;
                data = line.split();
                train_sub.append(data[1])
        test_du = list(set(test_txt))
        enroll_du = list(set(enroll_txt))
        train_du = list(set(train_sub))

        print(len(test_du))
        print(len(enroll_du))
        print(len(train_du))
        lines_test_train = []
        lines_enroll_train = []
        tem = []
        # produce enroll-train test-train
        for i in range(len(test_du)):
            for j in range(len(train_du)):
                tem.append(test_du[i])
                tem.append(train_du[j]) 
                lines_test_train.append(tem)
                tem=[]
        for i in range(len(enroll_du)):
            for j in range(len(train_du)):
                tem.append(enroll_du[i])
                tem.append(train_du[j]) 
                lines_enroll_train.append(tem)
                tem=[]
        print(len(lines_enroll_train))
        print(len(lines_test_train))
        #quit
        setfiles = list(set(files))
        setfiles.sort()
        nDataLoaderThread = 8
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, eval_frames=eval_frames)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )
        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp1                = data[0][0].cuda()
            ref_feat            = self.__S__.forward(inp1).detach().cpu()
            feats[data[1][0]]   = ref_feat
            telapsed            = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

        print('')
        train_dataset = test_dataset_loader(train_du, train_path, num_eval=num_eval, eval_frames=eval_frames)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )
        ## Extract features for every image
        for idx, data in enumerate(train_loader):
            inp1                = data[0][0].cuda()
            ref_feat            = self.__S__.forward(inp1).detach().cpu()
            feats[data[1][0]]   = ref_feat
            telapsed            = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(train_du),idx/telapsed,ref_feat.size()[1]));

        print('')
        all_scores = []
        all_labels = []
        all_trials = []
        tstart = time.time()
        all_scores_test_train = []
        all_scores_enroll_train = []
        all_trials_test_train = []
        all_trials_enroll_train  = []
        ## Read files and compute all scores
        for idx, line in enumerate(lines):
            data = line.split();
            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()
            if self.__L__.test_normalize:
                #print(ref_feat.shape)
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                #print(ref_feat.shape)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            # [I B, 1][I B ,1] --> [I B 1] [1 B I] = [I I]
            # [I I]
            #dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            #print(dist)
            #biao
            score =  numpy.mean(dist);
            #print(score)
            all_scores.append(score);  
            all_labels.append(int(data[0]));
            all_trials.append(data[1]+" "+data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        print('\n')
        for idx, line in enumerate(lines_enroll_train):
            ref_feat = feats[line[0]].cuda()
            com_feat = feats[line[1]].cuda()
            if self.__L__.test_normalize:
               # print(ref_feat.shape)
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                #print(ref_feat.shape)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            # [I B, 1][I B ,1] --> [I B 1] [1 B I] = [I I]
            # [I I]
            #dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            #print(dist)
            #biao
            score =  numpy.mean(dist);
           # print(score)
            all_scores_enroll_train.append(score);  
            all_trials_enroll_train.append(line[0]+" "+line[1])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();
        print('\n')
        for idx, line in enumerate(lines_test_train):
            ref_feat = feats[line[0]].cuda()
            com_feat = feats[line[1]].cuda()
            
            if self.__L__.test_normalize:
                #print(ref_feat.shape)
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                #print(ref_feat.shape)
                com_feat = F.normalize(com_feat, p=2, dim=1)
            dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            #print(dist)
            #biao
            score =  numpy.mean(dist);
           # print(score)
            all_scores_test_train.append(score);  
            all_trials_test_train.append(line[0]+" "+line[1])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();
        print('\n')

        return (all_scores, all_labels, all_trials, all_scores_enroll_train, all_trials_enroll_train, all_scores_test_train, all_trials_test_train);
    def evaluateFromList_s_norm2(self, listfilename, print_interval=100, test_path='', num_eval=1, eval_frames=None):
        
        self.eval();
        #self.getTrainList(listfilename=train_file_name, print_interval=100, train_path=train_path, num_eval=1, eval_frames=eval_frames)
        print('Loading data')
        data = io.loadmat('save_ecapa_S4.mat')
        dev_id = data['utt_id']
        dev_vector = data['embedding']
        dev_vector = np.squeeze(dev_vector)
        dev_id = np.squeeze(dev_id)
        # get 100 cohort avg embedding and sort 
        group_speaker = []
        tem_list = []
        train_sub = []
        speaker = []
        for i, evl in enumerate(dev_id):
            tem_list.append(dev_vector[i])
            if i+1 < len(dev_id) and dev_id[i+1] != dev_id[i]:
                train_sub.append(str('speaker_'+ str(dev_id[i])))
                group_speaker.append(tem_list)
                tem_list = []
            if i+1 == len(dev_id):
                train_sub.append(str('speaker_'+ str(dev_id[i])))
                group_speaker.append(tem_list)
            # if len(group_speaker) == 1000:
            #     break
        assert len(train_sub) == len(group_speaker)
        train_dic = {}
        for i, evl in enumerate(group_speaker):
            speaker.append(np.sum(group_speaker[i], axis=0) / len(group_speaker[i]))
            train_dic[train_sub[i]] = np.sum(group_speaker[i], axis=0) / len(group_speaker[i])
        assert len(train_sub) == len(speaker)
        #print(len(train_sub))
        #print(train_sub)
        #quit()
        lines       = []
        files       = []
        feats       = {}
        tstart      = time.time()
        enroll_txt = []
        test_txt = []
        ## Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline();
                if (not line):
                    break;
                data = line.split();
                files.append(data[1])
                files.append(data[2])
                enroll_txt.append(data[1])
                test_txt.append(data[2])
                lines.append(line)

        # with open(sub_train_file_name) as listfile:
        #     while True:
        #         line = listfile.readline();
        #         if (not line):
        #             break;
        #         data = line.split();
        #         train_sub.append(data[1])
        test_du = list(set(test_txt))
        enroll_du = list(set(enroll_txt))
        train_du = list(set(train_sub))

        print(len(test_du))
        print(len(enroll_du))
        print(len(train_du))
        lines_test_train = []
        lines_enroll_train = []
        tem = []
        # produce enroll-train test-train
        for i in range(len(test_du)):
            for j in range(len(train_du)):
                tem.append(test_du[i])
                tem.append(train_du[j]) 
                lines_test_train.append(tem)
                tem=[]
        for i in range(len(enroll_du)):
            for j in range(len(train_du)):
                tem.append(enroll_du[i])
                tem.append(train_du[j]) 
                lines_enroll_train.append(tem)
                tem=[]
        print(len(lines_enroll_train))
        print(len(lines_test_train))
        #quit
        setfiles = list(set(files))
        setfiles.sort()
        nDataLoaderThread = 8
        test_dataset = test_dataset_loader(setfiles, test_path, num_eval=num_eval, eval_frames=eval_frames)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=nDataLoaderThread,
            drop_last=False,
        )
        ## Extract features for every image
        for idx, data in enumerate(test_loader):
            inp1                = data[0][0].cuda()
            ref_feat            = self.__S__.forward(inp1).detach().cpu()
            feats[data[1][0]]   = ref_feat
            telapsed            = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(setfiles),idx/telapsed,ref_feat.size()[1]));

        print('')
        # train_dataset = test_dataset_loader(train_du, train_path, num_eval=num_eval, eval_frames=eval_frames)
        # train_loader = torch.utils.data.DataLoader(
        #     train_dataset,
        #     batch_size=1,
        #     shuffle=False,
        #     num_workers=nDataLoaderThread,
        #     drop_last=False,
        # )
        ## Extract features for every image
        # for idx, data in enumerate(train_loader):
        #     inp1                = data[0][0].cuda()
        #     ref_feat            = self.__S__.forward(inp1).detach().cpu()
        #     feats[data[1][0]]   = ref_feat
        #     telapsed            = time.time() - tstart

        #     if idx % print_interval == 0:
        #         sys.stdout.write("\rReading %d of %d: %.2f Hz, embedding size %d"%(idx,len(train_du),idx/telapsed,ref_feat.size()[1]));

        print('')
        all_scores = []
        all_labels = []
        all_trials = []
        tstart = time.time()
        all_scores_test_train = []
        all_scores_enroll_train = []
        all_trials_test_train = []
        all_trials_enroll_train  = []
        ## Read files and compute all scores
        for idx, line in enumerate(lines):
            data = line.split();
            ref_feat = feats[data[1]].cuda()
            com_feat = feats[data[2]].cuda()
            if self.__L__.test_normalize:
                #print(ref_feat.shape)
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                #print(ref_feat.shape)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            # [I B, 1][I B ,1] --> [I B 1] [1 B I] = [I I]
            # [I I]
            #dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            #print(dist)
            #biao
            score =  numpy.mean(dist);
            #print(score)
            all_scores.append(score);  
            all_labels.append(int(data[0]));
            all_trials.append(data[1]+" "+data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines),idx/telapsed));
                sys.stdout.flush();

        print('\n')
        for idx, line in enumerate(lines_enroll_train):
            ref_feat = feats[line[0]].cuda()
            #com_feat = feats[line[1]].cuda()
            com_feat = torch.tensor(train_dic[line[1]]).unsqueeze(0).cuda()
            #print(com_feat.shape)
            if self.__L__.test_normalize:
               # print(ref_feat.shape)
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                #print(ref_feat.shape)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            # [I B, 1][I B ,1] --> [I B 1] [1 B I] = [I I]
            # [I I]
            #dist = F.pairwise_distance(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            #print(dist)
            #biao
            score =  numpy.mean(dist);
           # print(score)
            all_scores_enroll_train.append(score);  
            all_trials_enroll_train.append(line[0]+" "+line[1])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines_enroll_train),idx/telapsed));
                sys.stdout.flush();
        print('\n')
        for idx, line in enumerate(lines_test_train):
            ref_feat = feats[line[0]].cuda()
            #com_feat = feats[line[1]].cuda()
            com_feat = torch.tensor(train_dic[line[1]]).unsqueeze(0).cuda()
            if self.__L__.test_normalize:
                #print(ref_feat.shape)
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                #print(ref_feat.shape)
                com_feat = F.normalize(com_feat, p=2, dim=1)
            dist = F.cosine_similarity(ref_feat.unsqueeze(-1), com_feat.unsqueeze(-1).transpose(0,2)).detach().cpu().numpy();
            #print(dist)
            #biao
            score =  numpy.mean(dist);
           # print(score)
            all_scores_test_train.append(score);  
            all_trials_test_train.append(line[0]+" "+line[1])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz"%(idx,len(lines_test_train),idx/telapsed));
                sys.stdout.flush();
        print('\n')

        return (all_scores, all_labels, all_trials, all_scores_enroll_train, all_trials_enroll_train, all_scores_test_train, all_trials_test_train);
    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Save parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def saveParameters(self, path):
        
        #torch.save(self.state_dict(), path);
        torch.save(self.state_dict(),path,_use_new_zipfile_serialization=False)

    ## ===== ===== ===== ===== ===== ===== ===== =====
    ## Load parameters
    ## ===== ===== ===== ===== ===== ===== ===== =====

    def loadParameters(self, path):

        self_state = self.state_dict();
        loaded_state = torch.load(path);
        for name, param in loaded_state.items():
            origname = name;
            if name not in self_state:
                name = name.replace("module.", "");

                if name not in self_state:
                    print("%s is not in the model."%origname);
                    continue;

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()));
                continue;

            self_state[name].copy_(param);

