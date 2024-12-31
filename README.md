# Anomaly Detection Using Complete Cycle Consistent Generative Adversarial Network
*********RCALAD official code*********

The code for the paper ["Anomaly Detection Using Complete Cycle Consistent Generative Adversarial Network" (authors: Zahra Dehghanian, Saeed Saravani, Maryam Amirmazlaghani and Mohamad Rahmati)](https://arxiv.org/abs/2304.07769) is now open source! 

Please reach us via emails or via github issues for any enquiries!


## Prerequisites.
This code package was developed and tested with Python 3.7.6. Make sure all dependencies specified in the requirements.txt file are satisfied before running the model. This can be achieved by

```
conda create --name tf1 python=3.7
conda activate tf1
pip3 install -r requirements.txt
```


## Usage.

Running the code with different options

```
python main.py <model>  <dataset> --nb_epochs=<number_epochs> --label=<0, 1, 2, 3, 4, 5, 6, 7, 8, 9> --sn=<bool> --enable_dzz=<bool> --rd=<int> --d-<int> etc. 
```
The default option will run the RCALAD model on Arrhythmia dataset with 1000 epoches.

Please refer to the argument parser in main.py for more details.


## Cite.

Please cite our work if you find it useful for your research and work.
```
@article{dehghanian2024anomaly,
  title={Anomaly Detection Using Complete Cycle Consistent Generative Adversarial Network},
  author={Dehghanian, Zahra and Saravani, Saeed and Amirmazlaghani, Maryam and Rahmati, Mohamad},
  journal={International journal of neural systems},
  pages={2550004},
  year={2024}
}
```
