# Regularized Complete Cycle Consistent GAN for Anomaly Detection
*********RCALAD official code*********

The code for the paper ["Regularized Complete Cycle Consistent GAN for Anomaly Detection" (authors: Zahra Dehghanian, Saeed Saravani, Maryam Amirmazlaghani and Mohamad Rahmati)](https://arxiv.org/abs/2304.07769) is now open source! 

Please reach us via emails or via github issues for any enquiries!

Please cite our work if you find it useful for your research and work.
```
@article{Dehghanian2023RegularizedCC,
  title={Regularized Complete Cycle Consistent GAN for Anomaly Detection},
  author={Zahra Dehghanian, Saeed Saravani, Maryam Amirmazlaghani and Mohamad Rahmati},
  year={2023}
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
