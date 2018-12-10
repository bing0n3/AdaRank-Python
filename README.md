# AdaRank

A python implementation of the AdaRank algorithm. It is based on weak ranker which use signle feature.

## How to Run

The following code will run Adarank for 500 iteration with optimzation function NDCG@5. If all features are selected 5 times, our algorithm will stop.

```py
model = Adaank(scorer=NDCGScorer_qid(K=5))
model.fit(X, y, qid, X_vali, y_vali, qid_vali)
pred = model.predict(X_test)
print(NDCGScorer_qid(K=5)(y_test,pred,qid_test).mean())
```



## References
Xu and Li. AdaRank: a boosting algorithm for information retrieval. In Proceedings of SIGIR '07, pages 391–398. ACM, 2007.

[rueycheng/AdaRank](https://github.com/rueycheng/AdaRank)

[The Lemur Project](https://sourceforge.net/p/lemur/wiki/RankLib/)