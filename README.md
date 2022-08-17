# Regularized-Compelete-Adversarially-Learned-Anomaly-Detection
*********RCALAD official code*********

The code for the paper ["Regularized Compelete Adversarially Learned Anomaly Detection" (authors: Zahra Dehghanian, Saeed Saravani, Maryam Amirmazlaghani)](https://arxiv.org/abs/1812.02288) is now open source! 

Please reach us via emails or via github issues for any enquiries!

Please cite our work if you find it useful for your research and work.
```
@article{Zenati2018AdversariallyLA,
  title={Adversarially Learned Anomaly Detection},
  author={Houssam Zenati and Manon Romain and Chuan Sheng Foo and Bruno Lecouat and Vijay R. Chandrasekhar},
  journal={2018 IEEE International Conference on Data Mining (ICDM)},
  year={2018},
  pages={727-736}
}
```

## Prerequisites.
To run the code, follow those steps:

Download the project code:

```
https://github.com/zahraDehghanian97/RCALAD.git
```
Install requirements (in the cloned repository):

```
pip3 install -r requirements.txt
```


## Doing anomaly detection.

Running the code with different options

```
python main.py <model>  <dataset> --nb_epochs=<number_epochs> --label=<0, 1, 2, 3, 4, 5, 6, 7, 8, 9> --sn=<bool> --enable_dzz=<bool> --rd=<int> --d-<int> etc. 
```
Please refer to the argument parser in main.py for more details.
