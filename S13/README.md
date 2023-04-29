**Student of EVA7 Batch awaiting EVA Phase II submitting EVA8 Transformer Assignments** </br>
Repository github url : https://github.com/jai-mr/Session </br>
Assignment Repository : https://github.com/jai-mr/Session/blob/main/S13/README.md </br>
Submitted by : Jaideep R - No Partners</br>
Registered email id : jaideepmr@gmail.com</br>


**1. VAE - MNIST**</br>

**A. Design a variation of a VAE in Pytorch that takes in two inputs**</br>
i. an MNIST image, and</br>
ii. its label which is an one hot encoded vector sent through an embedding layer</br>
**B. Training as you would train a VAE**</br>
**C. Now randomly send an MNIST image, but with a wrong label. Do this 25 times, and share what the VAE makes i.e 25 images stacked in 1 image**</br>

- [VAE MNIST Implementation - Jupyter Notebook url](https://github.com/jai-mr/Session/blob/main/S13/1_s13_vae_mnist.ipynb)<br/>
- VAE-MNIST Output</br>
<img src="images/1_vae_mnist.png"></br>

**2. VAE - CIFAR10**</br>

**A. Design a variation of a VAE in Pytorch that takes in two inputs**</br>
1. a cifar10  image, and</br>
2. its label which is an one hot encoded vector sent through an embedding layer</br>
**B. Training as you would train a VAE**</br>
**C. Now randomly send an cifar10 image, but with a wrong label. Do this 25 times, and share what the VAE makes i.e 25 images stacked in 1 image**</br>

- [VAE Cifar10 Implementation - Jupyter Notebook url](https://github.com/jai-mr/Session/blob/main/S13/2_s13_vae_cifar10.ipynb)<br/>

- VAE - Cifar10</br>
<img src="images/1_vae_cifar10.png"></br>

**3. UNET - Oxfordiiit-pet-dataset**</br>

**A. train your own UNet from scratch**</br>
**B. using the dataset https://www.kaggle.com/tanlikesmath/the-oxfordiiit-pet-datase**t </br>
**C. strategy provided in this link https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406**</br>
**D. train it 4 times**</br>

i. MP+Tr+BCE</br>
 [Training Log](https://github.com/jai-mr/Session/blob/main/S13/logs/3_s13_Unet_MpTrBce.log)</br>
 [Jupyter notebook link](https://github.com/jai-mr/Session/blob/main/S13/3_s13_Unet_MpTrBce.ipynb)</br>
 
ii. MP+Tr+Dice Loss</br>
 [Training Log](https://github.com/jai-mr/Session/blob/main/S13/logs/4_s13_Unet_MpTrDice.log)</br>
 [Jupyter notebook link](https://github.com/jai-mr/Session/blob/main/S13/4_s13_Unet_MpTrDice.ipynb)</br>
 
iii. StrConv+Tr+BCE</br>
 [Training Log](https://github.com/jai-mr/Session/blob/main/S13/logs/5_s13_UNet_StrTrBce.log)</br>
 [Jupyter notebook link](https://github.com/jai-mr/Session/blob/main/S13/5_s13_UNet_StrTrBce.ipynb)</br>
 
iv. StrConv+Ups+Dice Loss</br>
 [Training Log](https://github.com/jai-mr/Session/blob/main/S13/logs/6_s13_UNet_StrUpsDice.log)</br>
 [Jupyter notebook link](https://github.com/jai-mr/Session/blob/main/S13/6_s13_UNet_StrUpsDice.ipynb)</br>
