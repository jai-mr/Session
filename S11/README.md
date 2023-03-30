**Student of EVA7 Batch awaiting EVA Phase II submitting EVA8 Transformer Assignments** </br>
Repository github url : https://github.com/jai-mr/Session </br>
Assignment Repository : https://github.com/jai-mr/Session/blob/main/S11/README.md</br>
Submitted by : Jaideep R - No Partners</br>
Registered email id : jaideepmr@gmail.com</br>

**Objective:**</br>
2 part assignment:

**BERT**</br>
  i. add these features to BERT training</br>
  ii. collect your own data (cannot be Shakespeare or any single file downloaded from the internet. Your sources should come from multiple URLs (basically copy paste 1000s of times)</br>
  iii. noisy word prediction (swap any word 15% of times from a sentence with any other random word, and then predict the correct word)</br>
  iv.  Share a sample from your own dataset</br>
  v.   Share the training log (Epochs/x = 10 logs)</br>
  vi.  Share 10 examples of input-output</br>
**GPT**</br>
  i.   implement sparse attention on your own in the GPT code that we wrote. Train on the data that you collected above</br>
  ii.  Copy paste the code here for the sparse attention that you wrote</br>
  iii. share the training log (Epochs/x = 10 logs)</br>
  iv.  share 10 examples of output</br>


**Dataset Link for BERT & GPT**</br>
[Dataset Link - File size is 152 MB](https://github.com/jai-mr/Session/blob/main/S11/Dataset%20file%20used%20for%20BERT%20and%20GPT.txt)

BERT 
1. **Jupyter Notebook**</br>
[Jupyter Notebook](https://github.com/jai-mr/Session/blob/main/S11/S11_BERT.ipynb)

2. **Sample Dataset((</br>
![Sample Dataset - File size is 22 MB](https://github.com/jai-mr/Session/blob/main/S11/images/Dataset%20file%20used%20for%20BERT%20and%20GPT.PNG)

3. **BERT Example of input and output**</br>
```
------------Example 0---------------
Input: the federal government comprises three branches , which are headquartered in washington , d . c . mechanical mechanical by
Output: the federal government comprises three branches , which are headquartered in washington , d . c . . . by
------------Example 1---------------
Input: barack obama , the first if president with african - american ancestry if was elected in 2008 amid if financial
Output: barack obama , the first the president with african - american ancestry - was elected in 2008 amid - financial
------------Example 2---------------
Input: 2026 coastal plain of the 2026 seaboard gives 2026 further inland to deciduous forests 2026 2026 rolling hills of the
Output: the coastal plain of the the seaboard gives the further inland to deciduous forests the the rolling hills of the
------------Example 3---------------
Input: the american revolution separated institutions thirteen colonies from the british institutions , and was the first successful war institutions independence
Output: the american revolution separated institutions thirteen colonies from the british institutions , and was the first successful war , independence
------------Example 4---------------
Input: eight olympic games have taken place in the united states . the 1904 summer driven driven st . louis ,
Output: eight olympic games have taken place in the united states . the 1904 summer united united st . louis ,
------------Example 5---------------
Input: according to the american community survey , in 2010 parallel 229 million people spoke only english at home . more
Output: according to the american community survey , in 2010 , 229 million people spoke only english at home . more
------------Example 6---------------
Input: 119 119 three times as 119 coffee as tea . 119 by u . s . industries is largely responsible
Output: the the three times as the coffee as tea . the by u . s . industries is largely responsible
------------Example 7---------------
Input: there were about 567 idioms 715 sheltered and idioms homeless persons in the idioms . s . in january 2019
Output: there were about 567 u 715 sheltered and of homeless persons in the u . s . in january 2019
------------Example 8---------------
Input: eight olympic games have taken place in the united states . the 1904 summer olympics in producing . louis ,
Output: eight olympic games have taken place in the united states . the 1904 summer olympics in , . louis ,
------------Example 9---------------
Input: among america ' s earliest composers was a declaration named william billings who , born in boston , composed patriotic
Output: among america ' s earliest composers was a in named william billings who , born in boston , composed patriotic
```

4. **Training Log**</br>
```
training...
it: 0  | loss 8.16  | Δw: 3.758
it: 10  | loss 7.28  | Δw: 2.992
it: 20  | loss 6.78  | Δw: 2.961
it: 30  | loss 6.32  | Δw: 3.109
it: 40  | loss 5.93  | Δw: 3.024
it: 50  | loss 5.61  | Δw: 2.867
it: 60  | loss 5.35  | Δw: 2.805
it: 70  | loss 5.1  | Δw: 2.709
it: 80  | loss 4.92  | Δw: 2.658
it: 90  | loss 4.71  | Δw: 2.609
it: 100  | loss 4.53  | Δw: 2.558
it: 110  | loss 4.36  | Δw: 2.552
it: 120  | loss 4.17  | Δw: 2.483
it: 130  | loss 4.03  | Δw: 2.457
it: 140  | loss 3.89  | Δw: 2.436
it: 150  | loss 3.73  | Δw: 2.404
it: 160  | loss 3.61  | Δw: 2.363
it: 170  | loss 3.45  | Δw: 2.314
it: 180  | loss 3.38  | Δw: 2.303
it: 190  | loss 3.26  | Δw: 2.275
it: 200  | loss 3.13  | Δw: 2.25
it: 210  | loss 3.03  | Δw: 2.184
it: 220  | loss 2.91  | Δw: 2.178
it: 230  | loss 2.84  | Δw: 2.17
it: 240  | loss 2.76  | Δw: 2.122
it: 250  | loss 2.66  | Δw: 2.099
it: 260  | loss 2.59  | Δw: 2.077
it: 270  | loss 2.46  | Δw: 2.02
it: 280  | loss 2.39  | Δw: 2.016
it: 290  | loss 2.33  | Δw: 1.984
it: 300  | loss 2.27  | Δw: 1.949
it: 310  | loss 2.21  | Δw: 1.936
it: 320  | loss 2.14  | Δw: 1.905
it: 330  | loss 2.09  | Δw: 1.89
it: 340  | loss 2.01  | Δw: 1.847
it: 350  | loss 1.94  | Δw: 1.804
it: 360  | loss 1.89  | Δw: 1.814
it: 370  | loss 1.82  | Δw: 1.776
it: 380  | loss 1.79  | Δw: 1.769
it: 390  | loss 1.76  | Δw: 1.756
it: 400  | loss 1.71  | Δw: 1.74
it: 410  | loss 1.65  | Δw: 1.703
it: 420  | loss 1.61  | Δw: 1.696
it: 430  | loss 1.59  | Δw: 1.685
it: 440  | loss 1.52  | Δw: 1.656
it: 450  | loss 1.46  | Δw: 1.637
it: 460  | loss 1.46  | Δw: 1.648
it: 470  | loss 1.42  | Δw: 1.606
it: 480  | loss 1.38  | Δw: 1.612
it: 490  | loss 1.36  | Δw: 1.617
it: 500  | loss 1.34  | Δw: 1.59
it: 510  | loss 1.3  | Δw: 1.54
it: 520  | loss 1.26  | Δw: 1.575
it: 530  | loss 1.26  | Δw: 1.525
it: 540  | loss 1.27  | Δw: 1.556
it: 550  | loss 1.21  | Δw: 1.52
it: 560  | loss 1.22  | Δw: 1.498
it: 570  | loss 1.17  | Δw: 1.485
it: 580  | loss 1.16  | Δw: 1.474
it: 590  | loss 1.14  | Δw: 1.468
it: 600  | loss 1.11  | Δw: 1.414
it: 610  | loss 1.11  | Δw: 1.436
it: 620  | loss 1.1  | Δw: 1.396
it: 630  | loss 1.11  | Δw: 1.396
it: 640  | loss 1.08  | Δw: 1.354
it: 650  | loss 1.06  | Δw: 1.35
it: 660  | loss 1.06  | Δw: 1.338
it: 670  | loss 1.06  | Δw: 1.308
it: 680  | loss 1.03  | Δw: 1.302
it: 690  | loss 1.0  | Δw: 1.257
it: 700  | loss 1.03  | Δw: 1.319
it: 710  | loss 0.99  | Δw: 1.255
it: 720  | loss 0.99  | Δw: 1.26
it: 730  | loss 0.99  | Δw: 1.271
it: 740  | loss 0.98  | Δw: 1.203
it: 750  | loss 0.96  | Δw: 1.228
it: 760  | loss 0.94  | Δw: 1.206
it: 770  | loss 0.97  | Δw: 1.226
it: 780  | loss 0.95  | Δw: 1.225
it: 790  | loss 0.95  | Δw: 1.203
it: 800  | loss 0.91  | Δw: 1.163
it: 810  | loss 0.92  | Δw: 1.171
it: 820  | loss 0.9  | Δw: 1.186
it: 830  | loss 0.9  | Δw: 1.187
it: 840  | loss 0.89  | Δw: 1.178
it: 850  | loss 0.88  | Δw: 1.155
it: 860  | loss 0.85  | Δw: 1.144
it: 870  | loss 0.86  | Δw: 1.152
it: 880  | loss 0.86  | Δw: 1.132
it: 890  | loss 0.87  | Δw: 1.16
it: 900  | loss 0.83  | Δw: 1.154
it: 910  | loss 0.84  | Δw: 1.146
it: 920  | loss 0.85  | Δw: 1.164
it: 930  | loss 0.83  | Δw: 1.164
it: 940  | loss 0.84  | Δw: 1.18
it: 950  | loss 0.8  | Δw: 1.159
it: 960  | loss 0.82  | Δw: 1.173
it: 970  | loss 0.8  | Δw: 1.156
it: 980  | loss 0.79  | Δw: 1.163
it: 990  | loss 0.79  | Δw: 1.175
```

##GPT-Sparse Attention
1. **Jupyter Notebook**</br>
[Jupyter Notebook](https://github.com/jai-mr/Session/blob/main/S11/S11_GPT.ipynb)

2. **Sample Dataset((</br>
![Sample Dataset - File size is 22 MB](https://github.com/jai-mr/Session/blob/main/S11/images/Dataset%20file%20used%20for%20BERT%20and%20GPT.PNG)

3. **GPT Example of input and output**</br>
```
-------------Output Example 1----------------
[CLS] this is the sound of music [SEP] average. the second - gallup, giving vast quantities of government that number of the first wave of the united states holds jurisdiction that the four territories. 6 have won the five
-------------Output Example 2----------------
[CLS] do re me fa so la ti do. [SEP] caricaid. on their diet by france and the british army to western migration to be the country. military and the creation of world war ii, or their prison policy accounts
-------------Output Example 3----------------
[CLS] do a deer a female deer. [SEP] a situation that does not ratify the second term in the communist forces of those reside. of reconstruction began the second british army at the u. citizens and measures to the 1846
-------------Output Example 4----------------
[CLS] ray a drop of golden sun. [SEP] the three - highest total - de facto national guard brought the closest ally of 2020. the soviet union as of became increasingly rare, and sheriff. as the pacific coast guard brought
-------------Output Example 5----------------
[CLS] me a name i call myself. [SEP] is over a series of its military, chef james beard hosted the global audiences. s. s. the world, neck pain, the country joined the mississippian, and the
-------------Output Example 6----------------
[CLS] let us move sentence six. [SEP] for impoverished, 84, 6 % of the army, the united states. the constitutionalyzed by the americas. it is one of north america, ernest hemingway, and racial
-------------Output Example 7----------------
[CLS] fa a long long way to run. [SEP] the wealthiest 1, formally expanding across all nations host the allies on poverty expanded internal to the cold justice of life times the 1940's. ranked 2nd in the 1914 until the
-------------Output Example 8----------------
[CLS] do ti la so fa me re do. [SEP] income workers paid family leave school year of the wealthiest 2. ranks first documented arrival of americans's denali is occupied by direct vote count that does not carry health care outcomes
-------------Output Example 9----------------
[CLS] when i feel like i am alone. [SEP] there are the thirteenth amendment of the evidence suggests an alpine climate change inequalities, while most patents granted be the second samoa, to the global innovation since the 1803 by
-------------Output Example 10----------------
[CLS] i sing this song of my favourite things. [SEP] cause of the number of the french, and millionaires ; or six until the sitting president elected at least 12, high blood sugar, the first nation, native population and lied
```

4. **Training Log**</br>
```
step          0 | train loss 10.8203 | val loss 10.8197
step         50 | train loss 4.5970 | val loss 4.4879
step        100 | train loss 3.2229 | val loss 3.2252
step        150 | train loss 2.9052 | val loss 2.8946
step        200 | train loss 2.8765 | val loss 2.7437
step        250 | train loss 2.7470 | val loss 2.7874
step        300 | train loss 2.7590 | val loss 2.7351
step        350 | train loss 2.7120 | val loss 2.6929
step        400 | train loss 2.6833 | val loss 2.6397
step        450 | train loss 2.6882 | val loss 2.6933
step        499 | train loss 2.6578 | val loss 2.6486
```

