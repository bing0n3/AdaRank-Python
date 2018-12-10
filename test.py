from __future__ import print_function, division
import numpy as np
from metric import NDCGScorer, NDCGScorer_qid, map_scorer
from adarank import AdaRank
from read_data import read_data


def cal():
	parser = read_data()
	# parser.read_mq2008('./mq2008')
	parser.read_ml()
	scores = []


	for k in range(5):
	    scores.append([])

	for i in range(5):
	    print("============fold{}==================".format(i+1))
	    train, vali, test = parser.get_fold(i)
	    X, y, qid = train

	    X_test, y_test, qid_test = test
	    X_vali, y_vali, qid_vali = vali

	    model = AdaRank(scorer=NDCGScorer_qid(K=5))
	    model.fit(X, y, qid, X_vali, y_vali, qid_vali)

	    pred = model.predict(X_test)
	    for k in range(5):
	        score = round(NDCGScorer_qid(K=k+1)(y_test,pred,qid_test).mean(),4)
	        scores[k].append(score)
	        print('nDCG@{}\t{}\n'.format(k+1,score))
	print("==============Mean NDCG==================")
	for f in range(5):
	    print("mean NDCG@{}\t{}\n".format(f+1,round(np.mean(scores[f]),4)))

cal()
