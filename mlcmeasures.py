import sklearn
import sklearn.metrics
import numpy as np
from scipy.sparse import hstack, vstack, coo_matrix, csr_matrix, csc_matrix

conventionSuffixes=['(0)','(1)','(*)','#0'] #for /0 = 0; 1; ignore; count the occurrences of /0
def getConfusionMatrix(predictionMatrix,trueLabelsMatrix,suffixes=conventionSuffixes):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, row in enumerate(predictionMatrix):
        for k, p in enumerate(row):
            if not np.isnan(p):
                if p == trueLabelsMatrix[i,k] and p == 1:
                    tp += 1
                elif p == trueLabelsMatrix[i,k] and p == 0:
                    tn += 1
                elif p != trueLabelsMatrix[i,k] and p == 1:
                    fp += 1
                elif p != trueLabelsMatrix[i,k] and p == 0:
                    fn += 1
    return [tp, tn, fp, fn]
    
def getBipartitionMeasures(predictionMatrix,trueLabelsMatrix,suffixes=conventionSuffixes):
    output={} 
    #bring into right format. it actually should be sparse, but then "1-" operations do not work, so I use dense until another more efficient approach (without matrix operations but iterations)
    trueLabelsMatrix=csr_matrix(trueLabelsMatrix).A
    predictionMatrix=csr_matrix(predictionMatrix).A
   
    #do element-wise multiplications
    tp=trueLabelsMatrix*predictionMatrix
    tn=(1-trueLabelsMatrix)*(1-predictionMatrix)
    fp=(1-trueLabelsMatrix)*(predictionMatrix)
    fn=(trueLabelsMatrix)*(1-predictionMatrix)
    #aggregated confusion matrices
    tpcols=np.array([np.sum(col) for col in tp.T])
    fpcols=np.array([np.sum(col) for col in fp.T])
    fncols=np.array([np.sum(col) for col in fn.T])
    tprows=np.array([np.sum(row) for row in tp])
    fprows=np.array([np.sum(row) for row in fp])
    fnrows=np.array([np.sum(row) for row in fn])
    
    #compute macro measures
    output.update(computeFractions(tpcols,tpcols+fpcols,suffixes,prefix='Macro Precision'))
    output.update(computeFractions(tpcols,tpcols+fncols,suffixes,prefix='Macro Recall'))
    output.update(computeFractions(2*tpcols,2*tpcols+fncols+fpcols,suffixes,prefix='Macro F1'))

    #compute example based measures
    output.update(computeFractions(tprows,tprows+fprows,suffixes,prefix='Example-based Precision'))
    output.update(computeFractions(tprows,tprows+fnrows,suffixes,prefix='Example-based Recall'))
    output.update(computeFractions(2*tprows,2*tprows+fnrows+fprows,suffixes,prefix='Example-based F1'))
    
    subsetAccM=[1 for fprow,fnrow in zip(fprows,fnrows)  if fprow==0 and fnrow==0]
    output['Subset Accuracy']=len(subsetAccM)/float(len(fprows))

    output.update(computeFractions(tprows,(tprows+fprows)*(fprows+1),suffixes,prefix='Powerset Precision'))
    output.update(computeFractions(tprows,(tprows+fnrows)*(fnrows+1),suffixes,prefix='Powerset Recall'))
    output.update(computeFractions(2*tprows,tprows*(fprows+fnrows+2)+fprows*(fprows+1)+fnrows*(fnrows+1),suffixes,prefix='Powerset F1'))
    
    #compute micro measures
    #assume that there are no /0 anymore
    tptotal=float(np.sum(tpcols))
    fptotal=float(np.sum(fpcols))
    fntotal=float(np.sum(fncols))
    if tptotal+fptotal!=0:
        output['Micro Precision']=tptotal/(tptotal+fptotal)
    else:
        output['Micro Precision']=0
    if tptotal+fntotal!=0:
        output['Micro Recall']=tptotal/(tptotal+fntotal)
    else:
        output['Micro Recall']=0
    if tptotal+fptotal+fntotal!=0:
        output['Micro F1']=2*tptotal/(2*tptotal+fptotal+fntotal)
    else:
        output['Micro F1']=0
    output['Hamming Accuracy']=1.0-(fptotal+fntotal)/(tp.shape[0]*tp.shape[1])
    
    
    return output


def computeFractions(counter,denom,suffixes,prefix=''): #i.e. list of confusion matrizes
    res={}
    m1=0.0
    m0=0.0
    c0=0
    cNot0=0
    for i in range(len(counter)):
        if denom[i] == 0:
            #case that /0
            m1=m1+1
            c0=c0+1
            #TODO:case that 0/0 (it's assumed implicitly since all variables in counter appear in denom)
        else:
            frac=float(counter[i])/denom[i]
            m1=m1+frac
            m0=m0+frac
            cNot0=cNot0+1
    res[prefix+suffixes[0]]=m0/(cNot0+c0) #the average with /0 interpreted as 0
    res[prefix+suffixes[1]]=m1/(cNot0+c0) #the average with /0 interpreted as 1
    if(cNot0!=0):
        res[prefix+suffixes[2]]=m0/cNot0      #the average with /0 ignored
    else:
        res[prefix+suffixes[2]]=0      #this is very uncommon case and should only happen for chainLenght experiments when there are only few labels and these do not appear 
    res[prefix+suffixes[3]]=c0      #count of zero divisions, i.e. empty confusion matrices
    return res


    