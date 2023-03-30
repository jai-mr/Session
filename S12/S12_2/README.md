Student of EVA7 Batch awaiting EVA Phase II submitting EVA8 Transformer Assignments<br/>
Repository github url : https://github.com/jai-mr/Session<br/>
Assignment Repository : https://github.com/jai-mr/Session/tree/main/S12<br/>
Submitted by : Jaideep R - No Partners<br/>
Registered email id : jaideepmr@gmail.com<br/>

## Training Custom Dataset on Colab for YoloV3

1. [Refer to this Colab File](https://colab.research.google.com/drive/1LbKkQf4hbIuiUHunLlvY-cc0d_sNcAgS)
2. [Refer to this GitHub Repo](https://github.com/theschoolofai/YoloV3)
3. [Download this dataset](https://drive.google.com/file/d/1sVSAJgmOhZk6UG7EzmlRjXfkzPxmpmLy/view?usp=sharing)
   * This was annotated by EVA5 Students. Collect and add 25 images for the following 4 classes into the dataset shared:
   * class names are in custom.names file.
   * train the model
   * added your additional 100+ images & trained the model
4. Once done:
   * Downloaded a very small (~10-30sec) video from youtube which shows your classes.
   * Use ffmpeg to extract frames from the video.
   * Upload on your drive (alternatively you could be doing all of this on your drive to save upload time)
   * Infer on these images using detect.py file. Modify detect.py file if your file names do not match the ones mentioned on GitHub.
      python detect.py --conf-three 0.3 --output output_folder_name
   * Use ffmpeg to convert the files in your output folder to video
   * Upload the video to YouTube.
   * Also run the model on 16 images that you have collected (4 for each class)
5. Share the link to your GitHub project with the steps mentioned above - 1000 pts (only if all the steps were done, and it resulted in a trained model that you could run on video/images)
6. Share the link to your YouTube video
7. Share the link to the readme file where we can find the result of your model on YOUR 16 image

## Link to your GitHub project(for YoloV3 training on Colab). 
[Link to Github Colab Notebook](https://github.com/jai-mr/Session/blob/main/S12/S12_2/12_2.Assignment_Yolo3.ipynb)

## Images to show the result of your model for the 16+ images
<img src="https://github.com/jai-mr/Session/blob/main/S12/S12_2/images/fin.PNG" alt="Image for 16+ Images" title="Image for the resultant model of  16+ images" />

## Youtube Video
[Youtube Video Link](https://youtu.be/imZr6bXd5Qc)
