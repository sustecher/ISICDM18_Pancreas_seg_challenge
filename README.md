# ISICDM18_Pancreas_seg_challenge
The dataset can be found through Baidu Cloud.

Link: https://pan.baidu.com/s/1J7EtJH4NXVRjXuyvopSvdg 
Code: ****（To be confirmed by the data provider)

Our method can be divided into three steps.

Coarse Segmentation
--------------------------------------------------
It was used to localize pancreas in a 3D CT images.
This parts is based on our previour work. 
For more details, you can refer this link. 
https://github.com/KwuJohn/lddmmMASGD

Fine Segmentation
--------------------------------------------------
To learn the connection feature, a 3D patch-based convolutional neural network(CNN) and three 2D slice-based CNNs are jointly used to predict a fine segmentation based on a bounding box determined from the coarse segmentation.

Refine Segmentation
--------------------------------------------------
Finally, a level-set method is used, with the Fine segmentation being one of its constraints, to integrate information of the original image and the CNN-derived probability map to achieve a refine segmentation.

--------------------------------------------------
The detailed description of our method is published in [Medical Image Analysis](https://to be proof reading.). 
Please cite our work if you find the codeis or the ISICDM dataset is useful for your research.
