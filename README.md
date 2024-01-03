# Spot The Odd One Out: Regularized Complete Cycle Consistent Anomaly Detector GAN
*********RCALAD official code*********

The code for the paper ["Spot The Odd One Out: Regularized Complete Cycle Consistent Anomaly Detector GAN" (authors: Zahra Dehghanian, Saeed Saravani, Maryam Amirmazlaghani and Mohamad Rahmati)](https://arxiv.org/abs/2304.07769) is now open source! 

Please reach us via emails or via github issues for any enquiries!


## Prerequisites.
This code package was developed and tested with Python 3.7.6. Make sure all dependencies specified in the requirements.txt file are satisfied before running the model. This can be achieved by

```
pip3 install -r requirements.txt
```


## Usage.

Running the code with different options

```
python main.py <model>  <dataset> --nb_epochs=<number_epochs> --label=<0, 1, 2, 3, 4, 5, 6, 7, 8, 9> --sn=<bool> --enable_dzz=<bool> --rd=<int> --d-<int> etc. 
```
Please refer to the argument parser in main.py for more details.


## Cite.

Please cite our work if you find it useful for your research and work.
```
@article{Dehghanian2023Spot,
  title={Spot The Odd One Out: Regularized Complete Cycle Consistent Anomaly Detector GAN},
  author={Zahra Dehghanian, Saeed Saravani, Maryam Amirmazlaghani and Mohamad Rahmati},
  year={2023}
}
```
