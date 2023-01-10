# NEEDED: Introducing Hierarchical Transformer to Eye Diseases Diagnosis
## Basic info:
This is the release code for :
[NEEDED: Introducing Hierarchical Transformer to Eye Diseases Diagnosis](https://arxiv.org/abs/2212.13408) 
which is accepted by SDM 2023!

Congratulations to [Ye, Xu](https://github.com/xuye2020), this work is his first research article during his master degree!

Recommended ref:
```
Xu Ye, Meng Xiao, Zhiyuan Ning, Weiwei Dai, Wenjuan Cui, Yi Du, Yuanchun Zhou. NEEDED: Introducing Hierarchical Transformer to Eye Diseases Diagnosis. SIAM International Conference on Data Mining 2023, 2023
```

Recommended Bib:
```
@inproceedings{ye2023needed,
  title={NEEDED: Introducing Hierarchical Transformer to Eye Diseases Diagnosis},
  author = {Ye, Xu and Xiao, Meng and Ning, Zhiyuan and Dai, Weiwei and Cui, Wenjuan and Du, Yi and Zhou, Yuanchun},
  journal={SIAM International Conference on Data Mining 2023},
  year={2023}
}
```
***
## Paper Abstract
With the development of natural language processing techniques(NLP), automatic diagnosis of eye diseases using ophthalmology electronic medical records (OEMR) has become possible. It aims to evaluate the condition of both eyes of a patient respectively, and we formulate it as a particular multi-label classification task in this paper. Although there are a few related studies in other diseases, automatic diagnosis of eye diseases exhibits unique characteristics. First, descriptions of both eyes are mixed up in OEMR documents, with both free text and templated asymptomatic descriptions, resulting in sparsity and clutter of information. Second, OEMR documents contain multiple parts of descriptions and have long document lengths. Third, it is critical to provide explainability to the disease diagnosis model. To overcome those challenges, we present an effective automatic eye disease diagnosis framework, NEEDED. In this framework, a preprocessing module is integrated to improve the density and quality of information. Then, we design a hierarchical transformer structure for learning the contextualized representations of each sentence in the OEMR document. For the diagnosis part, we propose an attention-based predictor that enables traceable diagnosis by obtaining disease-specific information. Experiments on the real dataset and comparison with several baseline models show the advantage and explainability of our framework.
***




## How to run:
### step 1: download the code and dataset:
```
git clone git@github.com:coco11563/NEEDED-Introducing-Hierarchical-Transformer-to-Eye-Diseases-Diagnosis.git
```
TBA
