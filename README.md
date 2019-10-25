# Domain Adaptation on Amazon Reviews (four proudct) data
Resources of domain adaptation papers on sentiment analysis that have used Amazon reviews

## Dataset
The [Multi-Domain Sentiment Dataset](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/) was used in many domain adapation papers for sentiment analysis task.  It was first used in [Blitzer et al, (2007)](https://www.cs.jhu.edu/~mdredze/publications/sentiment_acl07.pdf). It contains more than 340, 000 reviews from 25 different types of products from Amazon.com (Chen et al.2012). Some domains (books and dvds) have hundreds of thousands of reviews. Others (musical instruments) have only a few hundred. Reviews contain star ratings (1 to 5 stars) that can be converted into binary labels if needed.  A subset of this dataset containing
four different product types: books, DVDs, electronics and kitchen appliances was used by [Blitzer et al. (2007)](https://www.cs.jhu.edu/~mdredze/publications/sentiment_acl07.pdf), which contains reviews of four types of products: books, DVDs, electronics, and kitchen appliances. And reviews with rating > 3 were labeled
positive, those with rating < 3 were labeled negative. Each domain (product type) consists of 2, 000 labeled inputs and approximately 4, 000 unlabeled ones (varying slightly between domains) and the two classes are exactly balanced.  Many works follow that convention, only experiment on this smaller set with its more manageable twelve domain adaptation tasks or multi-source domain adaptation where one domain is used as target all the others used as sources domains.  However,  the representations in different papers, such as how many features kept in bag-of-word representations, are different. Follow the early deep learning approach [Glorot et al.(2011) SDA](http://www.icml-2011.org/papers/342_icmlpaper.pdf) paper, the representations in [Chen et al. 2012 mSDA](http://www.cs.cornell.edu/~kilian/papers/msdadomain.pdf) paper was used in many works, where each reviews are preprocessed as a feature vector of unigram and bigram and you can choose to use the top 5000 most frequent features or use all the features.

Here is a brief list of the papers (to be continued) that have used this dataset along with results reported and also their implementation if there are any.

## Single source domain adaptation
- **SCL**: Bollywood, Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification. Association of Computational Linguistics [[ACL2007]](https://www.cs.jhu.edu/~mdredze/publications/sentiment_acl07.pdf)
![SCL paper reported accuracies](link-to-image)

- **SFA**:
- **MCT**:

- **SDA**: Domain Adaptation for Large-Scale Sentiment Classification:A Deep Learning Approach
[[ICML2011]](http://www.icml-2011.org/papers/342_icmlpaper.pdf)
- **mSDA**: Marginalized Denoising Autoencoders for Domain Adaptatio [[ICML2012]](http://www.cs.cornell.edu/~kilian/papers/msdadomain.pdf) 
[[Python]]()
[[Matlab]]()
[[Data]]()
- **TLDA**:
- **BTDNN**:
- **DANN**: Domain-Adversarial Training of Neural Networks. [[Journal of Machine Learning Research 2016]](http://jmlr.org/papers/v17/15-239.html)
[[Python]](https://github.com/GRAAL-Research/domain_adversarial_neural_network)


## Multi Source DA
- **MDAN**: Adversarial Multiple Source Domain Adaptation [[NIPS2018]](http://papers.nips.cc/paper/8075-adversarial-multiple-source-domain-adaptation) [[Pytorch]](https://github.com/KeiraZhao/MDAN)
[[Data]](https://github.com/KeiraZhao/MDAN)
- **MoE**: Multi-Source Domain Adaptation with Mixture of Experts 
[[EMNLP2018]](https://arxiv.org/abs/1809.02256)
[[Pytorch]](https://github.com/jiangfeng1124/transfer)
[[Data]](https://github.com/jiangfeng1124/transfer)
- **MAN**: Multinomial Adversarial Networks for Multi-Domain Text Classification [[NAACL2018]](https://www.aclweb.org/anthology/N18-1111/)
    [[Pytorch]](https://github.com/ccsasuke/man) [[tgzfromACL]](https://www.aclweb.org/anthology/attachments/N18-1111.Software.tgz)
- **DACL**: Dual Adversarial Co-Learning for Multi-Domain Text Classification [[arxiv2019]](https://arxiv.org/abs/1909.08203)
- **MDANet**: Learning Multi-Domain Adversarial Neural Networks for Text Classification [[IEEE Acess2019]](https://ieeexplore.ieee.org/document/8666710)

## Other Resources
- For a comprehensive list of papers and implementation(official or unofficial) for transfer learning/domain adapation methods on NLP/Computer Vision, please refer to [[Transfer Learning]](https://github.com/jindongwang/transferlearning).
- Another transfer learning resources [[Awesome-transfer-learning]](https://github.com/artix41/awesome-transfer-learning).
-  Longer list of papers on Domain adaptation for NLP/Computer Vision [[Awsome-domain-adaptation/]](https://github.com/zhaoxin94/awsome-domain-adaptation/blob/master/README.md).
- Another papers list on Unsupervised Domain adaptations, conference/journal/arxiv papers, [[Unsupervised Domain Adaptation]](https://github.com/barebell/DA).

