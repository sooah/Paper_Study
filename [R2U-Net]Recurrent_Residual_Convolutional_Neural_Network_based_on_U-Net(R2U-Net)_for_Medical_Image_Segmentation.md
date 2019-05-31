## Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation

### Introduction

----------------------

- Deep Learning based approach가 잘 되는 이유

  1. Activation function resolve training problem in DL approaches
  2. Dropout helps regularize the network
  3. Several efficient optimization techniques are avaliable

  → Semantic image segmentation task : small architecturally variant model 사용

  (ex. FCN, SegNet)



- CNN을 medical imaging에 적용

  - Goal : 많은 사람들에게 동시에 더 better treat 하는 것을 제공하여 더 빠르고, 더 나은 진단을 위해
  - 사람의 개입없이 자동으로 processing하게 되면, human error 줄일 수 있고, 시간과 비용을 줄일 수 있음

  → Segmentation에서 사람의 개입없이 빠르고, 정확학 segmentation을 위한 요구 많음 

  → but 한계 있음

   1. data 부족 : 많은 수의 데이터 존재 하지 않음 (expert가 직접 해줘야 하는데, 비싸고 시간과 노력이 더 많이 듦) 

      → data transformation / augmentation 적용

   2. class imbalance : patch based approach  사용 (pixel based approach로 확대)



- medical image processing의 경우 : global localization 과 context modulation이 localization task에 적용됨

  각 pixel : identification task에서 target lesion의 contour 과 관련된 desired boundary의 class label을 보임

  → target lesion의 boundary를 찾기 위해 관련된 pixel을 강조해야함



- 이 연구는 2개의 modified and improved segmentation model을 보여줌 
  1. Using recurrent convolution network
  2. Using recurrent residual convolutional network



- **contribution of this work**

  1. two new models RU-Net and R2U-Net are introduced for medical image segmentation

  2. The experiments are conducted on three different modalities of medical imaging 

     : retina blood vessel, skin cancer, lung

  3. Performance evaluation of the proposed models is conducted 

     - patch based method : retina blood vessel
     - end-to-end image-based approach : skin lesion, lung

  4. Comparison against recently proposed state of the art methods that shows superior performance against equivalent models with same number of network parameters

     

### Related Work

--------------------------

- training very deep model → difficult (∴ vanishing gradient problem)

  → resolve by ReLU/ELU

  → resolve by He, deep residual model (utilizing an identity mapping to facilitate the training process)



- CNN based segmentation
  - FCN : natural image 에서 good → RNN을 통해 improve ⇒ 매우 많은 data fine tuning 해서!
  - Random Architecture (image patch-based architecture) : 134.5M parameter compute해야함 → 너무 많은 수의 pixel이 overlay 되고, 같은 convolution이 많이 수행됨



### RU-Net and R2U-Net architecture

--------------------------

- Deep residual model, RCNN, U-Net에 영감받아 → RU-Net, R2U-Net 만듦 (3 model의 장점만을이용)
- RCNN : 다른 benchmark 사용하여 object recognition task에서 더 나은 performance를 보임

→ Recurrent Convolutional Layers (RCL)의 operation : RCNN에 따라 express된 각 discrete time step에 따라 수행



![image](https://user-images.githubusercontent.com/45067667/58673582-eb6a9d80-8386-11e9-9728-65419dbbb14a.png)



- $l_{th}$ layer의 residual RCNN(RRCNN) block의 input sample : $x_l$ 

  RCL에서 $k^{th}$ feature map 에서의 input sample이 $(i,j)$ pixel에 위치

  output of the network $O^l_{ijk}(t)$ at time step t (*t번째 time step에서 l번째 layer의 k번째 feature map 중 (i,j) pixel에 위치한 output*)
  $$
  O_{ijk}^l(t) = (W_k^f)^T*x_l^{f(i,j)}(t)+(W_k^r)^T*x_l^{r(i,j)}(t-1)+b_k
  $$

   - $x_l^{f(i,j)}(t)$ : standard convolution layer의 input

   - $x_l^{r(i,j)}(t-1)$ : $l_{th}$ RCL의 input

   - $W_k^f$ : standard convolution layer의 k번째 feature map의 weight

   - $W_k^r$ : RCL의 k번째 feature map의 weight

   - $b_k$ : bias

     

-  RCL의 output은 ReLU activation을 통과 : $F(x_l, w_l) = f(O_{ijk}^l(t)) = max(0,O_{ijk}^l(t))$ 

  - $F(x_l, w_l)$  : $l_{th}$ layer의 RCNN unit의 output
  - $F(x_l, w_l)$ 의 output : RU-Net에서 encoding과 decoding unit 내의 down-sampling과 up-sampling layer에서 사용 ~~(이 부분 잘 이해 안됨)~~

  

- R2U-Net : RCNN unit의 final output은 residual unit을 지남 (Fig 4. (d) 참고)



- RRCNN block 의 output : $x_{l+1}$ 
  $$
  x_{l+1} = x_l + F(x_l + w_l)
  $$

  - $x_l$ : RRCNN block의 input sample 
  - $x_{l+1}$  : sub-sampling /up-sampling layer의 input으로 들어감

→ RRCNN block 의 residual unit의 feature map의 갯수와 dimension과 동일함

![image](https://user-images.githubusercontent.com/45067667/58673598-03422180-8387-11e9-892b-f0ed6bdbe4e5.png)

- proposed method : 위 그림에서 (b)와 (d)를 stack하여 만듦



##### <4개의 다른 architecture 평가> 

1. U-Net with forward convolution layers and feature concatenation 

   : 기존 U-Net에서 볼 수 있는 crop and copy 대신 적용 (위 그림에서 (a)에 해당)

2. U-Net with forward convolutional layers with residual connectivity (residual U-Net/ResU-Net)  (위 그림에서 (c)에 해당)
3. U-Net with forward recurrent convolutional layer (RU-Net) (위 그림에서 (b)에 해당)
4. U-Net with recurrent convolutional layers



- Pictorial representation of time step에 대한 unfolded RCL layer

  ![image](https://user-images.githubusercontent.com/45067667/58673605-1228d400-8387-11e9-85ec-4f692f424107.png)

  : 1개의 convolution의 경우 2개의 연속적인 recurrent convolution이 뒤따라옴

  → 이를 concatenation (encoding과 decoding 사이)에 적용



##### proposed method와 U-Net의 차이

1. encoding과 decoding unit에서 regular forward convolutional layer 사용 대신 RCLs and RCLs with residual unit 사용

   → residual unit with RCLs : model이 더 효율적으로 깊어질 수 있도록 도와줌

2. 효율적인 feature accumulation method는 제안된 method의 RCL unit에  포함되어 있음 

   (feature accumulation : contract → expand 사이의 feature skip connection 의미하는 듯)

   - 기존 : U-Net model의 outside에서 element-wise feature summation 수행 

     → training process 동안만 더 나은 convergence 로의 benefit 보여줌

   - proposed model : model 내부에서 feature accumulation 

     → training과 testing phase 모두에서 benefit 보여줌

   ⇒ 다른 time-step에 대한 feature accumulation : 더 low level & 강력한 feature representation 보임

   ⇒ 매우 낮은 level의 feature를 추출하는 데도 도움 → medical imaging에서 다른 modality를 위한 segmentation에서 효과적임

3. 기존 U-Net의 crop and copy 사용 X → only concatenation operation 사용



##### U-Net에 비해 proposed architecture를 사용할 경우 advantage

​	network parameter 갯수의 효율성 : U-Net, ResU-Net과 같은 갯수의 parameter 사용 but segmentation task에서 더 나은 performance를 보임

​	⇒ recurrent and residual operation은 network parameter의 갯수를 증가시키지 않음



### Experimental setup and results

----------------------

#### B. Quantitative Analysis Approaches

- AC (accuracy) : $\dfrac{TP+TN}{TP+TN+FP+FN}$ 
- SE (sensitivity) : $\dfrac{TP}{TP+FN}$ 
- SP (specificity) : $\dfrac{TN}{TN+FP}$  
  - TP : True Positive
  - TN : True Negative
  - FP : False Positive
  - FN : False Negative
- DC (Dice Coefficient) : $2\dfrac{|GT∩SR|}{|GT|+|SR|}$ 
- JS (Jaccard similarity) : $\dfrac{|GT∩SR|}{GT∪SR}$ 
  - GT : ground truth
  - SR : segmentation result



