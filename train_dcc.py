import time
import os
import sys, getopt
import ast
import datetime
import csv
import json
import numpy as np
import xgboost as xgb
import arff
import pandas as pd
import scipy.io.arff as sc
from scipy.sparse import csr_matrix
from sklearn import preprocessing
np.set_printoptions(threshold=sys.maxsize)

from mlcmeasures import *

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)          

def main():
    
    #np.set_printoptions(threshold=np.nan)
    
    dataset = ""
    num_labels = -1
    max_depth = 10
    num_rounds = 10
    chain_length = -1
    split = 1
    parameters = "{}"
    labels_to_transform = []
    models_to_train = []
    
    output_extra = True
    val_set_size = 0    

    output_large = dict()

    try:
        opts, args = getopt.getopt(sys.argv[1:], "f:l:d:r:c:s:p:t:oev:m:",["filename=","num_labels=","max_depth=","num_rounds=","chain_length=","split=","parameters=","labels_to_transform=","output_extra","validation=","models="])
    except getopt.GetoptError:
            tprint ('train.py -f <filename> -l <num_labels> -d <max_depth> -r <num_rounds> -c <chain_length> -s <split> -p <parameters> -t <labels_to_transform> -o [output_extra] -v <validation> -m <models>')
            sys.exit(2)
    for opt, arg in opts:
        if opt in("-h","--help"):
            tprint ('train.py -f <filename> -l <num_labels> -d <max_depth> -r <num_rounds> -c <chain_length> -s <split> -p <parameters> -t <labels_to_transform> -o [output_extra] -v <validation> -m <models>')
            sys.exit()
        elif opt in ("-f", "--filename"):
            dataset = arg
        elif opt in ("-l", "--num_labels"):
            num_labels = np.int32(arg)
        elif opt in ("-d", "--max_depth"):
            max_depth = np.int32(arg)
        elif opt in ("-r", "--num_rounds"):
            num_rounds = np.int32(arg)
        elif opt in ("-c", "--chain_length"):
            chain_length = np.int32(arg)
        elif opt in ("-s", "--split"):
            split = np.int32(arg)
        elif opt in ("-p", "--parameters"):
            parameters = arg
        elif opt in ("-t", "--features_to_transform"):
            labels_to_transform = arg.split(',')
        elif opt in ("-m", "--models"):
            models_to_train = arg.split(',')
        elif opt in ("-o", "--output_extra"):
            output_extra = True
        elif opt in ("-v", "--validation"):
            val_set_size = np.int32(arg)

    if(dataset == ""):
        tprint("No dataset specified - use parameter '-f'")
        sys.exit(2)
    if(num_labels == -1):
        tprint("No number of labels specified - use parameter '-l")
        sys.exit(2)
    if(chain_length == -1):
        chain_length = num_labels
                
    
    params = ast.literal_eval(parameters)
    params['objective'] = 'multilabel'
    params['num_label'] = num_labels
    params['max_depth'] = max_depth
    params['ml_split_method'] = split
    params['tree_method'] = 'multilabel'
    if not('eta' in params):
        params['eta'] = 0.3
    
    tprint("Init Model:\n")
    tprint("dataset: " + str(dataset))
    tprint("max_depth: " + str(max_depth))
    tprint("num_labels: " + str(num_labels))
    tprint("num_rounds: " + str(num_rounds))
    tprint("chain_length: " + str(chain_length))
    tprint("split: " + str(split))
    tprint("labels to encode: " + str(labels_to_transform))
    tprint("models to train: " + str(models_to_train))
    tprint("Parameters: " + str(parameters))
    tprint("Create extended JSON log: " + str(output_extra))
    tprint("Size Dev Set (Eval on Test if 0): " + str(val_set_size) + "\n")


    #======================
    #=== Log File Setup ===
    #======================
    evalSet = "Dev" if val_set_size > 0 else "Test" 
    if evalSet == "Test":
        out_file = '../logs/test_results_' + str(dataset) + ".csv"
    else:
        out_file = '../logs/eval_results_' + str(dataset) + ".csv"
    stamp = str(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    dump_file = '../logs/json_results_' + str(dataset) + "_" + '-'.join(models_to_train) + "_" + str(evalSet) + "_" + str(stamp) +".json"

    if not os.path.exists(out_file):
        headline = "Classifier;Info;Dataset;EvalSet;DevSize;xgbParams;numLabels;maxDepth;numRounds;chainLength;splitMethod;trainTime;predictTime;trainTimeCpp;predictTimeCpp;labelOrderTrain;labelOrderPredict;MacroPrecision(0);MacroPrecision(1);MacroPrecision(*);MacroPrecision#0;MacroRecall(0);MacroRecall(1);MacroRecall(*);MacroRecall#0;MacroF1(0);MacroF1(1);MacroF1(*);MacroF1#0;Example-basedPrecision(0);Example-basedPrecision(1);Example-basedPrecision(*);Example-basedPrecision#0;Example-basedRecall(0);Example-basedRecall(1);Example-basedRecall(*);Example-basedRecall#0;Example-based F1(0);Example-based F1(1);Example-based F1(0);Example-basedF1#0;SubsetAccuracy;PowersetPrecision(0);PowersetPrecision(1);PowersetPrecision(*);PowersetPrecision#0;PowersetRecall(0);PowersetRecall(1);PowersetRecall(*);PowersetRecall#0;Powerset F1(0);Powerset F1(1);Powerset F1(*);PowersetF1#0;MicroPrecision;MicroRecall;MicroF1;HammingAccuracy"
        with open(out_file, 'a') as f:
            f.write(headline+"\n")

    print()

    #=====================
    #===  Build Models ===
    #=====================
    if("dc" in models_to_train or "dcc" in models_to_train or "cc-dcc" in models_to_train or len(models_to_train) == 0):
        print()
        print("========================================================================================================================================")
        print()
        tprint("Dynamic Chain - Start Training\n")
        
        t0 = time.time()
        models, label_features, weights, train_order_pos, train_order_neg, train_time_total, train_time_model, intermediate_results = build_chain(filename=dataset,
                                                                        num_labels=num_labels,
                                                                        num_chain_nodes=chain_length,
                                                                        num_boosting_rounds=num_rounds, 
                                                                        parameters=params,
                                                                        val_set_size=val_set_size,
                                                                        labels_to_encode=labels_to_transform)    
        t1 = time.time()
        print()        

        order_train = []
        for tr in train_order_pos:
            tr[tr < 0] = 0
            order_train.append(np.argmax(tr))
        tprint("Train Order: ")
        print(str(order_train))
        tprint("Train Order - Positives:")
        print(train_order_pos)
        tprint("Train Order - Negatives:")
        print(train_order_neg)

        print()
        print()

        if val_set_size > 0:
            tprint("Start Predicting on Val\n")
        else:            
            tprint("Start Predicting on Test\n")
        
        preds_test, cum_preds_test, preds_raw_test, cum_preds_raw_test, y_test, prediction_order_pos_test, prediction_order_neg_test, confusion_matrix_test, predict_time_total_test, predict_time_model_test, confusion_matrix_cum_test, prediction_order_pos_cum_test, prediction_order_neg_cum_test = predict_chain(filename=dataset,
                                                                            num_labels=num_labels,
                                                                            labels_to_encode=labels_to_transform,
                                                                            eval_on_train=False,
                                                                            chain_models=models,
                                                                            val_set_size=val_set_size)
        t2 = time.time()

        print()
        tprint("Confusion Matrix - Predicting on Test / Val:")
        print(str(confusion_matrix_test))     
        print()
        tprint("Confusion Matrix - Predicting on Test / Val Cumulated:")
        print(str(confusion_matrix_cum_test))     
        print()

        order_predict_test = []
        for tr in prediction_order_pos_test:
            tr[tr < 0] = 0
            order_predict_test.append(np.argmax(tr))
        tprint("Prediction Order on Test / Val: ")
        print(str(order_predict_test))
        tprint("Prediction Order on Test / Val - Positives:")
        print(prediction_order_pos_test)
        tprint("Prediction Order on Test / Val - Negatives:")
        print(prediction_order_neg_test) 
        print()      
        tprint("Prediction Order on Test / Val - Cumulated Positives:")
        print(prediction_order_pos_cum_test)
        tprint("Prediction Order on Test / Val - Cumulated Negatives:")
        print(prediction_order_neg_cum_test) 
        
        print()
        print()       

        t3 = time.time()
        preds_train, cum_preds_train, preds_raw_train, cum_preds_raw_train, y_train, prediction_order_pos_train, prediction_order_neg_train, confusion_matrix_train, predict_time_total_train, predict_time_model_train, confusion_matrix_cum_train, prediction_order_pos_cum_train, prediction_order_neg_cum_train = predict_chain(filename=dataset,
                                                                            num_labels=num_labels,
                                                                            labels_to_encode=labels_to_transform,
                                                                            eval_on_train=True,
                                                                            chain_models=models,
                                                                            val_set_size=val_set_size)
        t4 = time.time()       

        print()
        tprint("Confusion Matrix - Predicting on Train:")
        print(str(confusion_matrix_train))
        print()
        tprint("Confusion Matrix - Predicting on Train Cumulated:")
        print(str(confusion_matrix_cum_train))
        print()

        order_predict_train = []
        for tr in prediction_order_pos_train:
            tr[tr < 0] = 0
            order_predict_train.append(np.argmax(tr))
        tprint("Prediction Order on Train: ")
        print(str(order_predict_train))
        tprint("Prediction Order on Train - Positives:")
        print(prediction_order_pos_train)
        tprint("Prediction Order on Train - Negatives:")
        print(prediction_order_neg_train)
        print()
        tprint("Prediction Order on Train - Cumulated Positives:")
        print(prediction_order_pos_cum_train)
        tprint("Prediction Order on Train - Cumulated Negatives:")
        print(prediction_order_neg_cum_train)

        print()
        print()

        
        tprint("Time to Train (with intermediate evaluation): " + str(round((t1-t0)*1000, 2)) + "ms ")
        tprint("Time to Train (pure train time in C++ and administration): " + str(round(train_time_total, 2)) + "ms ")
        tprint("Time to Train (pure train time C++): " + str(round(train_time_model, 2)) + "ms ")
        tprint("Time to Predict on Test/Val: " + str(round(predict_time_total_test, 2)) + "ms ")
        tprint("Time to Predict on Test/Val (only C++): " + str(round(predict_time_model_test, 2)) + "ms ")
        tprint("Time to Predict on Train: " + str(round(predict_time_total_train, 2)) + "ms ")
        tprint("Time to Predict on Train (only C++): " + str(round(predict_time_model_train, 2)) + "ms ")

        ## Write to CSV
        results = ["DC", "Standard", dataset, evalSet + "_pred_on_Test", val_set_size, parameters, num_labels, max_depth, num_rounds, chain_length, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(train_time_model, 2)), str(round(predict_time_model_test, 2)), str(order_train), str(order_predict_test)]
        result_pred_test = getBipartitionMeasures(np.nan_to_num(preds_test), np.nan_to_num(y_test))
        for key in result_pred_test:
            results.append(result_pred_test[key])
        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)
        results = ["DC", "Cumulated", dataset, evalSet + "_pred_on_Test", val_set_size, parameters, num_labels, max_depth, num_rounds, chain_length, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(train_time_model, 2)), str(round(predict_time_model_test, 2)), str(order_train), str(order_predict_test)]
        result_cum_pred_test = getBipartitionMeasures(np.nan_to_num(cum_preds_test), np.nan_to_num(y_test))
        for key in result_cum_pred_test:
            results.append(result_cum_pred_test[key])
        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)
        
        results = ["DC", "Standard", dataset, evalSet + "_pred_on_Train", val_set_size, parameters, num_labels, max_depth, num_rounds, chain_length, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(train_time_model, 2)), str(round(predict_time_model_train, 2)), str(order_train), str(order_predict_train)]
        result_pred_train = getBipartitionMeasures(np.nan_to_num(preds_train), np.nan_to_num(y_train))
        for key in result_pred_train:
            results.append(result_pred_train[key])
        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)
        results = ["DC", "Cumulated", dataset, evalSet + "_pred_on_Train", val_set_size, parameters, num_labels, max_depth, num_rounds, chain_length, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(train_time_model, 2)), str(round(predict_time_model_train, 2)), str(order_train), str(order_predict_train)]
        result_cum_pred_train = getBipartitionMeasures(np.nan_to_num(cum_preds_train), np.nan_to_num(y_train))
        for key in result_cum_pred_train:
            results.append(result_cum_pred_train[key])
        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)

        if(output_extra):
            dc_output = dict()
            setup_output = dict()
            setup_output['dataset'] = dataset
            setup_output['evalSet'] = evalSet
            setup_output['val_set_size'] = val_set_size
            setup_output['parameters'] = parameters
            setup_output['num_labels'] = num_labels
            setup_output['max_depth'] = max_depth
            setup_output['num_rounds'] = num_rounds
            setup_output['chain_length'] = chain_length
            setup_output['split'] = split
            dc_output['train_order_pos'] = train_order_pos
            dc_output['train_order_neg'] = train_order_neg
            dc_output['prediction_order_pos_test'] = prediction_order_pos_test
            dc_output['prediction_order_neg_test'] = prediction_order_neg_test
            dc_output['prediction_order_pos_train'] = prediction_order_pos_train
            dc_output['prediction_order_neg_train'] = prediction_order_neg_train
            dc_output['prediction_order_pos_cum_test'] = prediction_order_pos_cum_test
            dc_output['prediction_order_neg_cum_test'] = prediction_order_neg_cum_test
            dc_output['prediction_order_pos_cum_train'] = prediction_order_pos_cum_train
            dc_output['prediction_order_neg_cum_train'] = prediction_order_neg_cum_train
            dc_output['preds_def_test'] = preds_raw_test
            dc_output['preds_cum_test'] = cum_preds_raw_test
            dc_output['preds_def_train'] = preds_raw_train
            dc_output['preds_cum_train'] = cum_preds_raw_train
            dc_output['results_def_test'] = result_pred_test
            dc_output['results_cum_test'] = result_cum_pred_test
            dc_output['results_def_train'] = result_pred_train
            dc_output['results_cum_train'] = result_cum_pred_train
            dc_output['intermediate_stats'] = intermediate_results
            dc_output['time_train_with_intermed_preds'] = round((t1-t0)*1000, 2)
            dc_output['time_train'] = round(train_time_total, 2)
            dc_output['time_train_cpp_only'] = round(train_time_model, 2)
            dc_output['time_predict_test'] = round(predict_time_total_test, 2)
            dc_output['time_predict_test_cpp_only'] = round(predict_time_model_test, 2)
            dc_output['time_predict_train'] = round(predict_time_total_train, 2)
            dc_output['time_predict_train_cpp_only'] = round(predict_time_model_train, 2)
            dc_output['label_order_train'] = order_train
            dc_output['label_order_predict_test'] = order_predict_test
            dc_output['label_order_predict_train'] = order_predict_train
            dc_output['confusion_matrix_test'] = confusion_matrix_test 
            dc_output['confusion_matrix_train'] = confusion_matrix_train 
            dc_output['confusion_matrix_cum_test'] = confusion_matrix_cum_test 
            dc_output['confusion_matrix_cum_train'] = confusion_matrix_cum_train 
       
            output_large['setup'] = setup_output            
            output_large['dc'] = dc_output

    if("sxgb" in models_to_train or len(models_to_train) == 0):
        print()
        print("========================================================================================================================================")
        print()
        tprint(" SXGB - Start Training\n")
            
        t0 = time.time()
        models, label_features, _, _, _, train_time_total, train_time_model, intermediate_results = build_chain(filename=dataset,
                                                                        num_labels=num_labels,
                                                                        num_chain_nodes=1,
                                                                        num_boosting_rounds=num_rounds, 
                                                                        parameters=params,
                                                                        val_set_size=val_set_size,
                                                                        labels_to_encode=labels_to_transform)    
        t1 = time.time()

        if val_set_size > 0:
            tprint("Start Predicting on Val\n")
        else:            
            tprint("Start Predicting on Test\n")
        
        preds_test, cum_preds_test, preds_raw_test, cum_preds_raw_test, y_test, _, _, confusion_matrix_test, predict_time_total_test, predict_time_model_test, _, _, _ = predict_chain(filename=dataset,
                                                                            num_labels=num_labels,
                                                                            labels_to_encode=labels_to_transform,
                                                                            eval_on_train=False,
                                                                            chain_models=models,
                                                                            val_set_size=val_set_size)
        t2 = time.time()

        print()
        tprint("Confusion Matrix - Predicting on Test / Val:")
        print(str(confusion_matrix_test))     
        print()

        t3 = time.time()
        preds_train, cum_preds_train, preds_raw_train, cum_preds_raw_train, y_train, _, _, confusion_matrix_train, predict_time_total_train, predict_time_model_train, _, _, _ = predict_chain(filename=dataset,
                                                                            num_labels=num_labels,
                                                                            labels_to_encode=labels_to_transform,
                                                                            eval_on_train=True,
                                                                            chain_models=models,
                                                                            val_set_size=val_set_size)
        t4 = time.time()       

        print()
        tprint("Confusion Matrix - Predicting on Train:")
        print(str(confusion_matrix_train))
        print()
        
        tprint("Time to Train (with intermediate evaluation): " + str(round((t1-t0)*1000, 2)) + "ms ")
        tprint("Time to Train (pure train time in C++ and administration): " + str(round(train_time_total, 2)) + "ms ")
        tprint("Time to Train (pure train time C++): " + str(round(train_time_model, 2)) + "ms ")
        tprint("Time to Predict on Test/Val: " + str(round(predict_time_total_test, 2)) + "ms ")
        tprint("Time to Predict on Test/Val (only C++): " + str(round(predict_time_model_test, 2)) + "ms ")
        tprint("Time to Predict on Train: " + str(round(predict_time_total_train, 2)) + "ms ")
        tprint("Time to Predict on Train (only C++): " + str(round(predict_time_model_train, 2)) + "ms ")

        ## Write to CSV
        results = ["SXGB", "Standard", dataset, evalSet + "_pred_on_Test", val_set_size, parameters, num_labels, max_depth, num_rounds, chain_length, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(train_time_model, 2)), str(round(predict_time_model_test, 2)), "", ""]
        result_pred_test = getBipartitionMeasures(np.nan_to_num(preds_test), np.nan_to_num(y_test))
        for key in result_pred_test:
            results.append(result_pred_test[key])
        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)
        results = ["SXGB", "Cumulated", dataset, evalSet + "_pred_on_Test", val_set_size, parameters, num_labels, max_depth, num_rounds, chain_length, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(train_time_model, 2)), str(round(predict_time_model_test, 2)), "", ""]
        result_cum_pred_test = getBipartitionMeasures(np.nan_to_num(cum_preds_test), np.nan_to_num(y_test))
        for key in result_cum_pred_test:
            results.append(result_cum_pred_test[key])
        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)
        
        results = ["SXGB", "Standard", dataset, evalSet + "_pred_on_Train", val_set_size, parameters, num_labels, max_depth, num_rounds, chain_length, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(train_time_model, 2)), str(round(predict_time_model_train, 2)), "", ""]
        result_pred_train = getBipartitionMeasures(np.nan_to_num(preds_train), np.nan_to_num(y_train))
        for key in result_pred_train:
            results.append(result_pred_train[key])
        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)
        results = ["SXGB", "Cumulated", dataset, evalSet + "_pred_on_Train", val_set_size, parameters, num_labels, max_depth, num_rounds, chain_length, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(train_time_model, 2)), str(round(predict_time_model_train, 2)), "", ""]
        result_cum_pred_train = getBipartitionMeasures(np.nan_to_num(cum_preds_train), np.nan_to_num(y_train))
        for key in result_cum_pred_train:
            results.append(result_cum_pred_train[key])
        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)

        if(output_extra):
            dc_output = dict()
            dc_output['preds_def_test'] = preds_raw_test
            dc_output['preds_cum_test'] = cum_preds_raw_test
            dc_output['preds_def_train'] = preds_raw_train
            dc_output['preds_cum_train'] = cum_preds_raw_train
            dc_output['results_def_test'] = result_pred_test
            dc_output['results_cum_test'] = result_cum_pred_test
            dc_output['results_def_train'] = result_pred_train
            dc_output['results_cum_train'] = result_cum_pred_train
            dc_output['intermediate_stats'] = intermediate_results
            dc_output['time_train_with_intermed_preds'] = round((t1-t0)*1000, 2)
            dc_output['time_train'] = round(train_time_total, 2)
            dc_output['time_train_cpp_only'] = round(train_time_model, 2)
            dc_output['time_predict_test'] = round(predict_time_total_test, 2)
            dc_output['time_predict_test_cpp_only'] = round(predict_time_model_test, 2)
            dc_output['time_predict_train'] = round(predict_time_total_train, 2)
            dc_output['time_predict_train_cpp_only'] = round(predict_time_model_train, 2)
            dc_output['confusion_matrix_test'] = confusion_matrix_test 
            dc_output['confusion_matrix_train'] = confusion_matrix_train          
            output_large['sxgb'] = dc_output
    
    if("cc-dcc" in models_to_train or len(models_to_train) == 0):
        print()
        print("========================================================================================================================================")
        print()

        params = ast.literal_eval(parameters)
        params['objective'] = 'multilabel'
        params['tree_method'] = 'multilabel'
        params['num_label'] = 1
        params['max_depth'] = max_depth
        params['ml_split_method'] = split
        if not('eta' in params):
            params['eta'] = 0.3

        cc_pred, cc_pred_raw, y_test, cc_pred_test, cc_pred_raw_test, y_test_test, t0, t1, t2, t3, order, time_train_model, time_predict_model = classifier_chain(dataset, num_labels, labels_to_transform, train_order_pos, params, num_rounds, val_set_size, "dc")
        
        print()
        print()
        tprint("Evaluation of Classifier Chain Predictions with DC order\n")
        
        results = ["CC", "DC Order", dataset, evalSet, val_set_size, parameters, num_labels, max_depth, num_rounds, num_labels, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(time_train_model, 2)), str(round(time_predict_model, 2)), str(order), ""]
        result_pred = getBipartitionMeasures(np.nan_to_num(cc_pred), np.nan_to_num(y_test))
        for key in result_pred:
            tprint(key + ": " + str(result_pred[key]))
            results.append(result_pred[key])

        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)

        if(output_extra):
            cc_dc_output = dict()
            cc_dc_output['preds'] = cc_pred_raw
            cc_dc_output['results'] = result_pred
            cc_dc_output['time_train'] = round((t1-t0)*1000, 2)
            cc_dc_output['time_predict'] = round((t2-t1)*1000, 2)
            cc_dc_output['label_order'] = order   
            output_large['cc_dc'] = cc_dc_output
    
    if("cc-freq" in models_to_train or len(models_to_train) == 0):
        print()
        print("========================================================================================================================================")
        print()

        params = ast.literal_eval(parameters)
        params['objective'] = 'multilabel'
        params['tree_method'] = 'multilabel'
        params['num_label'] = 1
        params['max_depth'] = max_depth
        params['ml_split_method'] = split
        if not('eta' in params):
            params['eta'] = 0.3

        cc_pred, cc_pred_raw, y_test, cc_pred_test, cc_pred_raw_test, y_test_test, t0, t1, t2, t3, order, time_train_model, time_predict_model = classifier_chain(dataset, num_labels, labels_to_transform, 0, params, num_rounds, val_set_size, "freq")
        
        print()
        print()
        tprint("Evaluation of Classifier Chain Predictions with Freq order\n")
        
        results = ["CC", "Freq Order", dataset, evalSet, val_set_size, parameters, num_labels, max_depth, num_rounds, num_labels, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(time_train_model, 2)), str(round(time_predict_model, 2)), str(order), ""]
        result_pred = getBipartitionMeasures(np.nan_to_num(cc_pred), np.nan_to_num(y_test))
        for key in result_pred:
            tprint(key + ": " + str(result_pred[key]))
            results.append(result_pred[key])

        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)

        if(output_extra):
            cc_freq_output = dict()
            cc_freq_output['preds'] = cc_pred_raw
            cc_freq_output['results'] = result_pred
            cc_freq_output['time_train'] = round((t1-t0)*1000, 2)
            cc_freq_output['time_predict'] = round((t2-t1)*1000, 2)
            cc_freq_output['label_order'] = order   
            output_large['cc_freq'] = cc_freq_output
    
    if("cc-rare" in models_to_train or len(models_to_train) == 0):
        print()
        print("========================================================================================================================================")
        print()

        params = ast.literal_eval(parameters)
        params['objective'] = 'multilabel'
        params['tree_method'] = 'multilabel'
        params['num_label'] = 1
        params['max_depth'] = max_depth
        params['ml_split_method'] = split
        if not('eta' in params):
            params['eta'] = 0.3

        cc_pred, cc_pred_raw, y_test, cc_pred_test, cc_pred_raw_test, y_test_test, t0, t1, t2, t3, order, time_train_model, time_predict_model = classifier_chain(dataset, num_labels, labels_to_transform, 0, params, num_rounds, val_set_size, "rare")
        
        print()
        print()
        tprint("Evaluation of Classifier Chain Predictions with Rare order\n")
        
        results = ["CC", "Rare Order", dataset, evalSet, val_set_size, parameters, num_labels, max_depth, num_rounds, num_labels, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(time_train_model, 2)), str(round(time_predict_model, 2)), str(order), ""]
        result_pred = getBipartitionMeasures(np.nan_to_num(cc_pred), np.nan_to_num(y_test))
        for key in result_pred:
            tprint(key + ": " + str(result_pred[key]))
            results.append(result_pred[key])

        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)

        if(output_extra):
            cc_rare_output = dict()
            cc_rare_output['preds'] = cc_pred_raw
            cc_rare_output['results'] = result_pred
            cc_rare_output['time_train'] = round((t1-t0)*1000, 2)
            cc_rare_output['time_predict'] = round((t2-t1)*1000, 2)
            cc_rare_output['label_order'] = order   
            output_large['cc_rare'] = cc_rare_output
        
    if("cc-rand" in models_to_train or len(models_to_train) == 0):
        print()
        print("========================================================================================================================================")
        print()

        params = ast.literal_eval(parameters)
        params['objective'] = 'multilabel'
        params['tree_method'] = 'multilabel'
        params['num_label'] = 1
        params['max_depth'] = max_depth
        params['ml_split_method'] = split
        if not('eta' in params):
            params['eta'] = 0.3

        cc_pred, cc_pred_raw, y_test, cc_pred_test, cc_pred_raw_test, y_test_test, t0, t1, t2, t3, order, time_train_model, time_predict_model = classifier_chain(dataset, num_labels, labels_to_transform, 0, params, num_rounds, val_set_size, "rand")
        
        print()
        print()
        tprint("Evaluation of Classifier Chain Predictions with random order\n")
        
        results = ["CC", "Rand Order", dataset, evalSet, val_set_size, parameters, num_labels, max_depth, num_rounds, num_labels, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(time_train_model, 2)), str(round(time_predict_model, 2)), str(order), ""]
        result_pred = getBipartitionMeasures(np.nan_to_num(cc_pred), np.nan_to_num(y_test))
        for key in result_pred:
            tprint(key + ": " + str(result_pred[key]))
            results.append(result_pred[key])

        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)

        if(output_extra):
            cc_rand_output = dict()
            cc_rand_output['preds'] = cc_pred_raw
            cc_rand_output['results'] = result_pred
            cc_rand_output['time_train'] = round((t1-t0)*1000, 2)
            cc_rand_output['time_predict'] = round((t2-t1)*1000, 2)
            cc_rand_output['label_order'] = order   
            output_large['cc_rand'] = cc_rand_output

    if("br" in models_to_train or len(models_to_train) == 0):        
        print()
        print("========================================================================================================================================")
        print()

        params = ast.literal_eval(parameters)
        params['objective'] = 'multilabel'
        params['tree_method'] = 'multilabel'
        params['num_label'] = 1
        params['max_depth'] = max_depth
        params['ml_split_method'] = split
        if not('eta' in params):
            params['eta'] = 0.3
        
        br_pred, br_pred_raw, y_test, br_pred_test, br_pred_raw_test, y_test_test, t0, t1, t2, t3, time_train_model, time_predict_model = binary_relevance(dataset, num_labels, labels_to_transform, params, num_rounds, val_set_size)
        print()
        print()
        tprint("Evaluation of Binary Relevance Predictions\n")

        results = ["BR", "", dataset, evalSet, val_set_size, parameters, num_labels, max_depth, num_rounds, 0, split, str(round((t1-t0)*1000, 2)), str(round((t2-t1)*1000, 2)), str(round(time_train_model, 2)), str(round(time_predict_model, 2)), "", ""]
        result_pred = getBipartitionMeasures(np.nan_to_num(br_pred), np.nan_to_num(y_test))
        for key in result_pred:
            tprint(key + ": " + str(result_pred[key]))
            results.append(result_pred[key])

        with open(out_file, 'a', newline='') as f:
            writer = csv.writer(f, delimiter=';')
            writer.writerow(results)

        if(output_extra):
            br_output = dict()
            br_output['preds'] = br_pred_raw
            br_output['results'] = result_pred
            br_output['time_train'] = round((t1-t0)*1000, 2)
            br_output['time_predict'] = round((t2-t1)*1000, 2)
            output_large['br'] = br_output    

    if(output_extra):
        with open(dump_file, 'w') as file:
            file.write(json.dumps(output_large, cls=MyEncoder))



def prepare_dataset(filename, num_labels, features_to_encode=[], tprint_prefix=""):
    tprint("Prepare Dataset: " + str(filename), tprint_prefix)
    dataset = arff.load(open('./datasets/' + str(filename) + '.arff'), encode_nominal=True)
    column_names = [attribute[0] for attribute in dataset['attributes']]    
    df = pd.DataFrame(dataset['data'], columns=column_names)
    
    df_x = df.iloc[:,:(len(df.columns) - num_labels)]
    df_y = df.iloc[:,(len(df.columns) - num_labels):]

    for feature in features_to_encode:
        le = preprocessing.LabelEncoder()
        df_x.loc[:,str(feature)] = le.fit_transform(df_x[str(feature)])

    x = df_x.values
    x = x.astype(np.float)
    y = df_y.values
    y = y.astype(np.float)
    
    return x,y

def build_classifier(x_train, y_train, parameters, num_rounds, weights=[]):
    if len(weights) == len(y_train):
        dtrain = xgb.DMatrix(x_train, label=np.hstack(y_train), weight=weights)
    else:        
        dtrain = xgb.DMatrix(x_train, label=np.hstack(y_train))
    bst = xgb.train(parameters, dtrain, num_rounds)
    return bst

def predict(bst_model, x_test, round_preds):
    dtest = xgb.DMatrix(x_test)
    preds = bst_model.predict(dtest)
    if(round_preds):
        return np.array([[np.round(i) for i in inst] for inst in preds])
    return preds 

def build_chain(filename, num_labels, num_chain_nodes, num_boosting_rounds, parameters, val_set_size, labels_to_encode=[]):
    x_train, y_train = prepare_dataset(str(filename) + '-train', num_labels, labels_to_encode)
    train_time_total = 0
    train_time_model = 0
    intermediate_results = dict()

    if val_set_size > 0:
        size_train = int(np.round((100 - val_set_size)/100 * np.shape(x_train)[0]))

        x_dev = x_train[size_train:]
        y_dev = y_train[size_train:]
        x_train = x_train[:size_train]
        y_train = y_train[:size_train]

        tprint("Size of Dev Set: " + str(len(x_dev)) + " (=" + str(val_set_size) + "%)")
        print()

    models = []
    weights = np.zeros(np.shape(y_train)) + 1
    pred_matrix = np.zeros(np.shape(y_train))
    label_features = np.empty((len(x_train), num_labels))
    label_features[:] = np.nan
    prediction_order_pos = np.zeros((num_chain_nodes, num_labels))
    prediction_order_neg = np.zeros((num_chain_nodes, num_labels))
    
    for chain_round in range(num_chain_nodes):
        tprint("Train chain model: " +  str(chain_round))
        t_start = time.time()
        dtrain = xgb.DMatrix(np.hstack([x_train, label_features]), label=np.hstack(y_train), weight=np.hstack(weights))
        t_build_data = time.time()        
        bst = xgb.train(parameters, dtrain, num_boosting_rounds)
        t_train_model = time.time()     
        preds = predict(bst, np.hstack([x_train, label_features]), False)
        t_predict_model = time.time()  
        for p_count, pred in enumerate(preds):
            min_element = 0.5
            max_element = 0.5
            min_index = -1
            max_index = -1
            for i, p in enumerate(pred):
                if p >= max_element:# and pred_matrix[p_count, i] == 0:
                    max_element = p
                    max_index = i
                if p < min_element:# and pred_matrix[p_count, i] == 0:
                    min_element = p
                    min_index = i
            if(max_index != -1):
                if(np.isnan(label_features[p_count, max_index])):
                    label_features[p_count, max_index] = max_element
                    pred_matrix[p_count, max_index] = 1
                    weights[p_count, max_index] = 0
                elif(pred_matrix[p_count, max_index] == 1):
                    label_features[p_count, max_index] = max(max_element, label_features[p_count, max_index])
                prediction_order_pos[chain_round, max_index] += 1
            elif(min_index != -1):
                if(np.isnan(label_features[p_count, min_index])):
                    label_features[p_count, min_index] = min_element
                    pred_matrix[p_count, min_index] = -1
                    weights[p_count, min_index] = 0
                elif(pred_matrix[p_count, min_index] == -1):
                    label_features[p_count, min_index] = min(min_element, label_features[p_count, min_index])
                prediction_order_neg[chain_round, min_index] += 1
        t_stop = time.time()
        models.append(bst)

        times_detail_round = dict()
        time_total = round((t_stop-t_start) * 1000, 2)
        times_detail_round['total'] = time_total
        tprint("   - Train Time Total: " + str(round(time_total, 2)) + "ms ")
        
        time_load = round((t_build_data-t_start) * 1000, 2)
        times_detail_round['loadData'] = time_load
        tprint("   - Train Time Load Data: " + str(round(time_load, 2)) + "ms ")
        
        time_train = round((t_train_model-t_build_data) * 1000, 2)
        times_detail_round['trainCpp'] = time_train
        tprint("   - Train Time C++ Model: " + str(round(time_train, 2)) + "ms ")
        
        time_predict = round((t_predict_model-t_train_model) * 1000, 2)
        times_detail_round['predictCpp'] = time_predict
        tprint("   - Train Time Predict Model: " + str(round(time_predict, 2)) + "ms ")
        
        time_matrix = round((t_stop-t_predict_model) * 1000, 2)
        times_detail_round['buildPropMatrix'] = time_matrix
        tprint("   - Train Time Build Matrix: " + str(round(time_matrix, 2)) + "ms ")

        train_time_total += round((t_stop-t_start) * 1000, 2)
        train_time_model += round((t_train_model-t_build_data) * 1000, 2)

        temp_result_test, temp_result_train = getScores(filename, num_labels, labels_to_encode, models, val_set_size, chain_round, onTrain=True, onTestOrVal=True)

        round_results = dict()
        round_results['results_test'] = temp_result_test
        round_results['results_train'] = temp_result_train
        round_results['times'] = times_detail_round
        intermediate_results[chain_round] = round_results

        print()
        print()

    return models, label_features, weights, prediction_order_pos, prediction_order_neg, train_time_total, train_time_model, intermediate_results

def predict_chain(filename, num_labels, labels_to_encode, eval_on_train, chain_models, val_set_size, tprint_prefix=""):
    if eval_on_train:
        tprint("Eval on Train Set", tprint_prefix)
        x_test, y_test = prepare_dataset(str(filename) + '-train', num_labels, labels_to_encode, tprint_prefix)
    else:
        if val_set_size == 0:
            tprint("Eval on Test Set", tprint_prefix)
            x_test, y_test = prepare_dataset(str(filename) + '-test', num_labels, labels_to_encode, tprint_prefix)
        else:
            tprint("Eval on Dev Set", tprint_prefix)
            x_train, y_train = prepare_dataset(str(filename) + '-train', num_labels, labels_to_encode, tprint_prefix)
            size_train = int(np.round((100 - val_set_size)/100 * np.shape(x_train)[0]))

            x_test = x_train[size_train:]
            y_test = y_train[size_train:]
            
            tprint("Size of Dev Set: " + str(len(x_test)) + " (=" + str(val_set_size) + "%)", tprint_prefix)
            print()

    label_features = np.empty((len(x_test), num_labels))
    label_features[:] = np.nan
    pred_matrix = np.zeros(np.shape(y_test))
    prediction_order_pos = np.zeros((len(chain_models), num_labels))
    prediction_order_neg = np.zeros((len(chain_models), num_labels))
    prediction_order_pos_cum = np.zeros((len(chain_models), num_labels))
    prediction_order_neg_cum = np.zeros((len(chain_models), num_labels))
    confusion_matrix = np.zeros((len(chain_models), 4))
    confusion_matrix_cum = np.zeros((len(chain_models), 4))

    predict_time_total = 0
    predict_time_model = 0

    for model_num, bst_model in enumerate(chain_models):
        tprint("Predict Chain Node: " + str(model_num), tprint_prefix)
        t_start = time.time()
        preds = predict(bst_model, np.hstack([x_test, label_features]), False)
        t_model = time.time()
        if(model_num == 0):
            cum_preds = preds
        else:
            for i in range(len(cum_preds)):
                for k in range(num_labels):
                    if(pred_matrix[i, k] == 0):  
                        cum_preds[i,k] = max(cum_preds[i,k], preds[i,k])
                    if(np.round(cum_preds[i,k]) == 1):
                        prediction_order_pos_cum[model_num, k] +=  1
                    elif(np.round(cum_preds[i,k]) == 0):
                        prediction_order_neg_cum[model_num, k] +=  1
        for p_count, pred in enumerate(preds):
            min_element = 0.5
            max_element = 0.5
            min_index = -1
            max_index = -1
            for i, p in enumerate(pred):
                if p >= max_element:# and pred_matrix[p_count, i] == 0:
                    max_element = p
                    max_index = i
                if p < min_element:# and pred_matrix[p_count, i] == 0:
                    min_element = p
                    min_index = i
            if(max_index != -1):
                if(np.isnan(label_features[p_count, max_index])):
                    label_features[p_count, max_index] = max_element
                    pred_matrix[p_count, max_index] = 1
                elif(pred_matrix[p_count, max_index] == 1):                    
                    label_features[p_count, max_index] = max(max_element, label_features[p_count, max_index])
                prediction_order_pos[model_num, max_index] +=  1
            elif(min_index != -1):
                if(np.isnan(label_features[p_count, min_index])):                
                    label_features[p_count, min_index] = min_element
                    pred_matrix[p_count, min_index] = -1
                elif(pred_matrix[p_count, min_index] == -1):                    
                    label_features[p_count, min_index] = min(min_element, label_features[p_count, min_index])
                prediction_order_neg[model_num, min_index] +=  1
        t_stop = time.time()
        time_total = round((t_stop-t_start) * 1000, 2)
        predict_time_total += time_total
        tprint("   - Prediction Time Total: " + str(round(time_total, 2)) + "ms ", tprint_prefix)  
        time_model = round((t_model-t_start) * 1000, 2)
        predict_time_model += time_model    
        tprint("   - Prediction Time C++ model: " + str(round(time_model, 2)) + "ms ")
        confusion_matrix[model_num,:] = getConfusionMatrix([[np.round(i) for i in inst] for inst in label_features], np.nan_to_num(y_test))
        confusion_matrix_cum[model_num,:] = getConfusionMatrix([[np.round(i) for i in inst] for inst in cum_preds], np.nan_to_num(y_test))
    return [[np.round(i) for i in inst] for inst in label_features], [[np.round(i) for i in inst] for inst in cum_preds], label_features, cum_preds, y_test, prediction_order_pos, prediction_order_neg, confusion_matrix, predict_time_total, predict_time_model, confusion_matrix_cum, prediction_order_pos_cum, prediction_order_neg_cum

def getScores(filename, num_labels, labels_to_encode, models_to_train, val_set_size, extra_info, onTrain, onTestOrVal):
    tprint_prefix = "    "
    if onTrain:
        preds, cum_preds, _, _, y_train, _, _, _, _, _,_,_,_ = predict_chain(filename=filename,
                                                                num_labels=num_labels,
                                                                labels_to_encode=labels_to_encode,
                                                                eval_on_train=True,
                                                                chain_models=models_to_train,
                                                                val_set_size=val_set_size,
                                                                tprint_prefix=tprint_prefix)
        tprint("Results on Train:", tprint_prefix)
        result_pred_train = getBipartitionMeasures(np.nan_to_num(preds), np.nan_to_num(y_train))
        if ('dc' in models_to_train):
            tprint("Selected Evaluation: Standard", tprint_prefix)
        elif ('dcc' in models_to_train):
            tprint("Selected Evaluation: Cumulated", tprint_prefix)
        tprint("Evaluation of Standard Chain Predictions on Train:", tprint_prefix)
        for key in result_pred_train:
            tprint(key + ": " + str(result_pred_train[key]), tprint_prefix + "  - ")
        result_cum_pred = getBipartitionMeasures(np.nan_to_num(cum_preds), np.nan_to_num(y_train))
        tprint("Evaluation of Cumulated Chain Predictions on Train:", tprint_prefix)
        for key in result_cum_pred:
            tprint(key + ": " + str(result_cum_pred[key]), tprint_prefix + "  - ")
    tprint("")
    if onTestOrVal:
        preds, cum_preds, _, _, y_test, _, _, _, _, _,_,_,_ = predict_chain(filename=filename,
                                                                num_labels=num_labels,
                                                                labels_to_encode=labels_to_encode,
                                                                eval_on_train=False,
                                                                chain_models=models_to_train,
                                                                val_set_size=val_set_size,
                                                                tprint_prefix=tprint_prefix)
        if val_set_size > 0:
            tprint("Results on Val:", tprint_prefix)
        else:
            tprint("Results on Test:", tprint_prefix)
        result_pred_test = getBipartitionMeasures(np.nan_to_num(preds), np.nan_to_num(y_test))
        if ('dc' in models_to_train):
            tprint("Selected Evaluation: Standard", tprint_prefix)
        elif ('dcc' in models_to_train):
            tprint("Selected Evaluation: Cumulated", tprint_prefix)
        tprint("Evaluation of Standard Chain Predictions on Test:", tprint_prefix)
        for key in result_pred_test:
            tprint(key + ": " + str(result_pred_test[key]), tprint_prefix + "  - ")
        result_cum_pred = getBipartitionMeasures(np.nan_to_num(cum_preds), np.nan_to_num(y_test))
        tprint("Evaluation of Cumulated Chain Predictions on Test:", tprint_prefix)
        for key in result_cum_pred:
            tprint(key + ": " + str(result_cum_pred[key]), tprint_prefix + "  - ")

    return result_pred_test, result_pred_train

def tprint(text, prefix = ""):
    stamp = str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print(stamp + ' :: ' + prefix, text)


def classifier_chain(filename, num_labels, labels_to_encode, train_order, params, num_round, val_set_size, label_order):    
    tprint("-- Classifier Chains (" + str(label_order) + ") --\n")
    tprint("Parameters: " + str(params))    
    print()

    time_train_model = 0
    time_predict_model = 0
        
    if val_set_size == 0:
        tprint("Eval on Test Set")
        x_train, y_train = prepare_dataset(str(filename) + '-train', num_labels, labels_to_encode)
        x_test, y_test = prepare_dataset(str(filename) + '-test', num_labels, labels_to_encode)
    else:
        tprint("Eval on Dev Set")
        x_train, y_train = prepare_dataset(str(filename) + '-train', num_labels, labels_to_encode)
        size_train = int(np.round((100 - val_set_size)/100 * np.shape(x_train)[0]))

        x_test = x_train[size_train:]
        y_test = y_train[size_train:]
        x_train = x_train[:size_train]
        y_train = y_train[:size_train]

        x_test_real, y_test_real = prepare_dataset(str(filename) + '-test', num_labels, labels_to_encode)

        tprint("Size of Dev Set: " + str(len(x_test)))
        tprint("Size of Train Set: " + str(len(x_train)))
    print()
    
    order = []
    if label_order == "dc":
        for tr in train_order:
            order.append(np.argmax(tr))
        seen = set()
        unique_order = [x for x in order if not (x in seen or seen.add(x))]
        for i in range(num_labels):
            if i not in unique_order:
                unique_order.append(i)
    elif label_order == "freq":
        label_freq = np.sum(y_train, axis = 0)
        unique_order = [b[0] for b in sorted(enumerate(label_freq),key=lambda i:i[1])]
    elif label_order == "rare":
        label_freq = np.sum(y_train, axis = 0)
        unique_order = ([b[0] for b in sorted(enumerate(label_freq),key=lambda i:i[1])])[::-1]
    else:
        unique_order = np.array(range(num_labels))
        np.random.shuffle(unique_order)
        unique_order = list(unique_order)

    tprint("Prediction Order: " + str(unique_order) + "\n\n")
    
    models = []
    t0 = time.time()
    
    tprint("Start Training\n")
    for i, l in enumerate(unique_order):
        tprint("CC - build model " + str(i) + " for label " + str(l))
        t_model_start = time.time()
        bst = build_classifier(x_train, [y[l] for y in y_train], params, num_round)
        t_model_end = time.time()
        time_train_model += round((t_model_end-t_model_start)*1000, 2)
        dtrain = xgb.DMatrix(x_train)
        temp_pred = bst.predict(dtrain)
        temp = np.zeros((np.shape(x_train)[0], np.shape(x_train)[1]+1))
        temp[:,:-1] = x_train
        temp[:,-1] = temp_pred
        x_train = temp
        models.append(bst)
    t1 = time.time()

    print()
    print()
    tprint("Start Predicting\n")
    preds = np.zeros((len(x_test), num_labels))
    preds_raw = np.zeros((len(x_test), num_labels))
    for i, l in enumerate(unique_order):
        tprint("CC - predict model " + str(i) + " for label " + str(l))
        dtest = xgb.DMatrix(x_test)
        
        t_model_start = time.time()
        temp_pred = models[i].predict(dtest)
        t_model_end = time.time()
        time_predict_model += round((t_model_end-t_model_start)*1000, 2)
        temp = np.zeros((np.shape(x_test)[0], np.shape(x_test)[1]+1))
        temp[:,:-1] = x_test
        temp[:,-1] = temp_pred
        x_test = temp

        #pred = models[i].predict(dtest)
        preds_raw[:, l] = temp_pred
        preds[:, l] = np.round(temp_pred)
    t2 = time.time()
    print()

    if(val_set_size > 0):
        print()
        print()
        tprint("Start Predicting Test\n")
        preds_test = np.zeros((len(x_test_real), num_labels))
        preds_raw_test = np.zeros((len(x_test_real), num_labels))
        for i, l in enumerate(unique_order):
            tprint("CC - predict model " + str(i) + " for label " + str(l))
            dtest = xgb.DMatrix(x_test_real)
            
            temp_pred = models[i].predict(dtest)
            temp = np.zeros((np.shape(x_test_real)[0], np.shape(x_test_real)[1]+1))
            temp[:,:-1] = x_test_real
            temp[:,-1] = temp_pred
            x_test_real = temp

            #pred = models[i].predict(dtest)
            preds_raw_test[:, l] = temp_pred
            preds_test[:, l] = np.round(temp_pred)
        t3 = time.time()
        print()

    tprint("Time to Train: " + str(round((t1-t0)*1000, 2)) + "ms ")
    tprint("Time to Predict: " + str(round((t2-t1)*1000, 2)) + "ms ")

    if(val_set_size > 0):
        tprint("Time to Predict Test: " + str(round((t3-t2)*1000, 2)) + "ms ")

    if(val_set_size > 0):
        return preds, preds_raw, y_test, preds_test, preds_raw_test, y_test_real, t0, t1, t2, t3, unique_order, time_train_model, time_predict_model
    return preds, preds_raw, y_test, 0, 0, 0, t0, t1, t2, 0, unique_order, time_train_model, time_predict_model

def binary_relevance(filename, num_labels, labels_to_encode, params, num_round, val_set_size):
    tprint("-- Binary Relevance --\n")
    tprint("Parameters: " + str(params))
    print()

    time_train_model = 0
    time_predict_model = 0

    if val_set_size == 0:
        tprint("Eval on Test Set")
        x_train, y_train = prepare_dataset(str(filename) + '-train', num_labels, labels_to_encode)
        x_test, y_test = prepare_dataset(str(filename) + '-test', num_labels, labels_to_encode)
    else:
        tprint("Eval on Dev Set")
        x_train, y_train = prepare_dataset(str(filename) + '-train', num_labels, labels_to_encode)
        size_train = int(np.round((100 - val_set_size)/100 * np.shape(x_train)[0]))

        x_test = x_train[size_train:]
        y_test = y_train[size_train:]
        x_train = x_train[:size_train]
        y_train = y_train[:size_train]

        x_test_real, y_test_real = prepare_dataset(str(filename) + '-test', num_labels, labels_to_encode)

        tprint("Size of Dev Set: " + str(len(x_test)))
        tprint("Size of Train Set: " + str(len(x_train)))
    print()

    models = []

    t0 = time.time()
    tprint("Start Training\n")
    for i in range(num_labels):
        tprint("BR - build model " + str(i))
        t_model_start = time.time()
        bst = build_classifier(x_train, [y[i] for y in y_train], params, num_round)
        t_model_end = time.time()
        time_train_model += round((t_model_end-t_model_start)*1000, 2)
        models.append(bst)
    t1 = time.time()

    print()
    print()
    tprint("Start Predicting\n")
    preds = np.zeros((len(x_test), num_labels))
    preds_raw = np.zeros((len(x_test), num_labels))
    for i in range(num_labels):
        tprint("BR - predict model " + str(i))
        dtest = xgb.DMatrix(x_test)

        t_model_start = time.time()
        pred = models[i].predict(dtest)
        t_model_end = time.time()
        time_predict_model += round((t_model_end-t_model_start)*1000, 2)
        preds_raw[:, i] = pred
        preds[:, i] = np.round(pred)
    t2 = time.time()
    print()

    if val_set_size > 0:
        tprint("Start Predicting Test\n")
        preds_test = np.zeros((len(x_test_real), num_labels))
        preds_raw_test = np.zeros((len(x_test_real), num_labels))
        for i in range(num_labels):
            tprint("BR - predict model " + str(i))
            dtest = xgb.DMatrix(x_test_real)

            pred = models[i].predict(dtest)
            preds_raw_test[:, i] = pred
            preds_test[:, i] = np.round(pred)
        t3 = time.time()
        print()

    tprint("Time to Train: " + str(round((t1-t0)*1000, 2)) + "ms ")
    tprint("Time to Predict: " + str(round((t2-t1)*1000, 2)) + "ms ")

    if val_set_size > 0:
        tprint("Time to Predict Test: " + str(round((t3-t2)*1000, 2)) + "ms ")    

    if val_set_size > 0:
        return preds, preds_raw, y_test, preds_test, preds_raw_test, y_test_real, t0, t1, t2, t3, time_train_model, time_predict_model
    return preds, preds_raw, y_test, 0, 0, 0, t0, t1, t2, 0, time_train_model, time_predict_model

if __name__ == "__main__":
    main()
    
    
    
