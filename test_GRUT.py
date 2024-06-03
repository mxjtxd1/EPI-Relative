# --------------------------------------------------------------------
# To compare with pioneers, we quoted some of the test procedure from EPIVAN
# --------------------------------------------------------------------
import os
from models_GRUT import get_model
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score


models=['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK','fusion']
m=models[5]
model=None

max_len_en = 3000
max_len_pr = 2000
nwords = 4097
emb_dim = 100


names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK','fusion']

path_xlsx = './res_GRUT.xlsx'
model_list = []
cell_list = []
auc_list = []
aupr_list = []
epoch_list = []
data_dict = {}

#name = 'GM12878'
for model in models:
    model_name = model
    model = get_model(max_len_en, max_len_pr, nwords, emb_dim)

    for name in names:
        for epoch in range(0, 10):
            model.load_weights("./model/model_GRUT/%sModel%d.h5" % (model_name, epoch))
            Data_dir = './data/%s/' % name
            test = np.load(Data_dir + '%s_test.npz' % name)
            X_en_tes, X_pr_tes, y_tes = test['X_en_tes'], test['X_pr_tes'], test['y_tes']
            print("****************Testing %s cell line specific model on %s cell line****************" % (model_name, name))
            y_pred = model.predict([X_en_tes, X_pr_tes])
            auc = roc_auc_score(y_tes, y_pred)
            aupr = average_precision_score(y_tes, y_pred)
            print("AUC : ", auc)
            print("AUPR : ", aupr)
            model_list.append(model_name)
            cell_list.append(name)
            auc_list.append(auc)
            aupr_list.append(aupr)
            epoch_list.append(epoch)
data_dict['model'] = model_list
data_dict['cell'] = cell_list
data_dict['auc'] = auc_list
data_dict['aupr'] = aupr_list
data_dict['epoch'] = epoch_list
writer = pd.ExcelWriter(path_xlsx)
data = pd.DataFrame(data_dict)
data.to_excel(writer)
writer.save()
#for name in names:
#    for epoch in [1]:
#        model = get_model(max_len_en, max_len_pr, nwords, emb_dim)
#        model.load_weights("./model/%sModel%d.h5" % (name,epoch))
#        Data_dir='./data/%s/' % name
#        test=np.load(Data_dir+'%s_test.npz'%name)
#        X_en_tes,X_pr_tes,y_tes=test['X_en_tes'],test['X_pr_tes'],test['y_tes']
#        print("****************Testing %s cell line specific model on %s cell line****************"%(m,name))
#        y_pred = model.predict([X_en_tes,X_pr_tes])
#        auc=roc_auc_score(y_tes, y_pred)
#        aupr=average_precision_score(y_tes, y_pred)
#        print("AUC : ", auc)
#        print("AUPR : ", aupr)
#