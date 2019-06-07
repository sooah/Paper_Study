# Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

[DeepLab V3+] : V3에 비해 약간의 변화 존재

- Encoder : ResNet with atrous convolution → Xception (Inception with separable convolution)
- ASPP → ASSPP (Atrous Separable Spatial Pyramid Pooling)
- Decoder : Bilinear up-sampling → Simplified U-Net style decoder



### Abstact

-----------

- DNN for semantic segmentation task : spatial pyramid pooling module / encoder-decoder structure 사용

  - spatial pyramid pooling module : filter / pooling operation 사용 → incoming feature 검색 ⇒ multi-scale contextual information encode!
  - encoder-decoder structure : spatial information을 점진적으로 recover → object의 더 sharp boundary capture 가능

  ⇒ in this method : 두 방법의 장점 combine

- DeepLab V3+ : simple yet effective decoder + DeepLab V3

  → object boundary 따라 segmentation refine

  - Xception model 사용
  - Atrous Spatial Pyramid Pooling & decoder module : depthwise separable convolution 적용

  ⇒ faster & stronger encoder-decoder network

  

### Introduction

----------------------

- Semantic segmentation 의 Goal : image의 모든 pixel에 semantic label 할당
- DCNN based on FCN : hand-crafted 작업에 비해 매우 개선

→ In this work : 2 종류의 neural network 고려

1. spatial pyramid pooling module 사용 (pooling feature로 다른 resolution의 rich contextual information capture)

2. encoder-decoder structure 사용 (sharp object boundary obtain 가능)



- capture contextual information at multi scale 위해

  - DeepLab V3 : 여러 parallel atrous convolution with different rates (Atrous Spatial Pyramid Pooling / ASPP) 적용
  - PSPNet : 다른 grid scale에서 pooling operation 수행

  → last feature map에 많은 semantic information 있음

  → but object boundary에 관한 detailed information missing (∵ network backbone의 striding operation 사용하는 pooling/convolution 존재해서)

  

- SOTA neural network design & limited GPU memory 고려

  → input resolution보다 8배 / 4배 작은 output feature map extract : 계산적으로 불가능

  - ResNet-101 : input resoluton보다 16배 작은 output feature 추출 위해 atrous convolution 적용 → 마지막 3개 residual block (9 layers)의 feature dilate 되어야함
  - output feature가 input 보다 8배 작기위해 → 26 residual blocks (78 layers) 영향 받음

  ⇒ denser output feature 얻기 위해 위 방법 사용 → computationally intensive



- encoder-decoder model : 

  - encoder path : faster computation (∵ no features are dilated)

  - decoder path : gradually recover sharp object boundary

  ⇒ 두 방법의 장점 합침! : multi-scale contextual information 통합 → encoder-decoder network의 encoder module 풍부하게!



- DeepLab V3+

  - DeepLab V3 extend : object boundary recover위해 simple yet effective decoder module adding

    ![image](https://user-images.githubusercontent.com/45067667/58694393-d8c78700-83cd-11e9-9b9d-16018da1cb3f.png)

  - encoder feature의 density : budget of computation resource에 따라 atrous convolution control하여 제어 가능 → output에 semantic information encoding
  - decoder module : detailed object boundary recovery 가능



- Depthwise separable의 성공 : Xception model 조정 & ASPP와 decoder module에 atrous separable convolution 적용 ⇒ 속도와 정확성 개선



- contribution

  - propose powerful encoder module & simple yet effective decoder module 

    → novel encoder-decoder structure

  - atrous convolution으로 resolution of extracted encoder feature 임의 제어 가능

    → trade-off precision and runtime (기존 encoder-decoder model은 불가능)

  - Xception model 사용 for segmentation task & ASPP module과 decoder module에 depthwise separable convolution 적용

    ⇒ faster & stronger encoder-decoder network



##### Note. 이해를 위해 정리 

( <https://m.blog.naver.com/PostView.nhn?blogId=laonple&logNo=221000648527&proxyReferer=https%3A%2F%2Fwww.google.com%2F> 참고 )

1. Atrous Convolution : 필터 내부에 zero-padding을 추가해 강제로 receptive field를 늘리는 방법 

![image](https://user-images.githubusercontent.com/45067667/58678112-91270800-8399-11e9-8f8c-d4ab175b246e.png)

- a) 1-dilated convolution  : 기존에 흔히 알고 있던 convolution과 동일
- b) 2-dilated convolution : 빨간점에 위치한 값들만 convolution 연산에 사용, 나머지는 0 → receptive field의 크기가 7x7 영역으로 커지는 것과 동일
- c) 4-dilated convolution : receptive field 크기가 15X15 로 커지는 것과 동일

cf) wavelet 이용 신호 분석(wavelet decomposition)에 사용되던 방식

-  장점
  1. receptive field는 커지지만 파라미터의 갯수는 늘어나지 않음 → 연산량 관점에서 탁월 (원래는 큰 receptive field를 취하기 위해선 파라미터 갯수 많아야함)
  2. dilation 계수 조정하면 다양한 scale에 대한 대응 가능 (다양한 scale에서의 정보를 꺼내기 위해 넓은 receptive field 봐야함 → dilated convolution 사용하면 어려움 없음)

![image](https://user-images.githubusercontent.com/45067667/58682172-888afd80-83aa-11e9-97b1-f6a18392606f.png)

- a) 기본적인 convolution : 인접 데이터를 이용해 kernel 크기가 3인 convolution
- b) 확장계수 k = 2 인 경우 : 인접한 데이터를 이용하는 것이 아닌 중간에 hole이 1개씩 들어오는 점의 차이 → kernel 3을 사용하더라도 대응하는 영역의 크기가 커졌음 확인 가능

→ kernel의 크기는 동일하게 유지되기 때문에 연산량은 동일하지만, receptive field의 크기가 커지는 효과! 



- 영상 데이터와 같은 2차원에서도 효과 좋음

![image](https://user-images.githubusercontent.com/45067667/58682076-2af6b100-83aa-11e9-8b8c-99c4671b4d99.png)



2. Atrous convolution 및 Bilinear interpolation

   VGG-16 & ResNet-101 을 DCNN 망으로 사용 → ResNet 구조 변형시킨 모델 통해 VGG-16 보다 성능 올림

   - DCNN에서 max-pooling layer 2개 제거 → 1/8 크기의 feature-map 얻음 & atrous convolution 통해 넓은 receptive field 볼 수 있도록 함

   - pooling 후 동일한 크기의 convolution 수행 → receptive field 넓어짐 

     = detail 위해 pooling layer 제거 → 이 부분을 atrous convolution이 더 넓은 receptive field를 볼 수 있도록 함 ⇒  pooling layer 사라지는 문제 해소

   - 이 후는 FCN / dilated convolution과 마찬가지로 bilinear interpolation 이용 → 원 영상 복원

     ![image](https://user-images.githubusercontent.com/45067667/58682532-e1a76100-83ab-11e9-98b9-14f016c696c8.png)

     

   

   - Atrous convolution  : receptive field 확대 → 특징 찾는 범위 넓게! ⇒ 전체 영상으로 찾는 범위 확대하면 제일 좋음 but 단계적 수행 필요 → 연산량 많이 필요

     ⇒ 적정 선에서 trade-off 나머지는 bilinear interpolation 선택 → but bilinear 만으로는 정확하게 비행기를 pixel 단위까지 정교한 segmentation 불가능  ⇒ 뒷부분은 CRF(Conditional Random Field) 이용 → post-processing 수행

     ![image](https://user-images.githubusercontent.com/45067667/58682896-42836900-83ad-11e9-8c4b-52b682d80189.png)

     ⇒ 전체적인 구조 : DCNN + CRF (DCNN 앞부분 : 일반적인 convolution / 뒷부분 : astrous convolution)



3. ASPP (Atrous Spatial Pyramid Pooling)

   multi-scale에 더 잘 대응 → 'fc6' layer 에서의 atrous convolution 확장 계수 = {6, 12, 18, 24} 적용 → 결과 취합 사용

   ![image](https://user-images.githubusercontent.com/45067667/58682993-bcb3ed80-83ad-11e9-9dc6-1574e4eb7fe5.png)

   - ResNet 설계자인 Kaiming He의 SPPNet 논문에 나오는 Spatial Pyramid Pooling 기법에 영감 받아 ASPP 로 이름지음
   - 확장 계수를 6부터 24까지 변화 시킴 → 다양한 receptive field 볼 수 있음
   - SPPNet 방식과 비슷 : 이전 단계 까지의 결과는 동일하게 사용 → fc6 layer 에서 atrous convolution 위한 확장 계수 r 값만 다르게 적용 → 결과 합침 ⇒ 연산 효율성 관점에서 큰 이득!

   ![image](https://user-images.githubusercontent.com/45067667/58683107-38159f00-83ae-11e9-8fbe-ea058d75f2b3.png)

   

   	- a) DeepLab V1 구조 : ASPP 지원하지 않는 경우 → fc6의 확장 계수 12로 고정	
   	- b) DeepLab V2 구조 : ASPP 수행 → fc6 계수 {6, 12, 18, 24} 설정

   ![image](https://user-images.githubusercontent.com/45067667/58683235-a65a6180-83ae-11e9-9bbe-c5b4ab1f34cc.png)

   	- LargeFOV : r = 12 고정
   	- ASPP-S : r = {2, 4, 8, 12} → 좁은 receptive field 대응
   	- ASPP-L : r = {6, 12, 18, 24} → 넓은 receptive field 대응
   	- 모두 VGG-16 사용

   ⇒ 단순하게 확장 계수 r = 12 로 고정시키는 것 보다 ASPP 지원하여 1.7% 성능 개선

   ⇒ scale 고정시키는 것 보다 multi-scale 사용하는 편이 좋음

   ⇒ 좁은 receptive field 보다는 넓은 receptive-field 보는 편이 좋음



4. Fully Connected CRF

   1/8크기의 해상도를 갖는 DCNN 결과 → bi-linear interpolation → 원 영상 크기로 확대

   ⇒ 해상도 떨어짐!

   ⇒ 문제 해결 위 해 CRF(Conditional Random Field) 사용 후처리과정!

![image](https://user-images.githubusercontent.com/45067667/58684095-e242f600-83b1-11e9-97e8-987481c9a194.png)

  - short-range CRF : segmentation 수행 뒤 생기는 segmentation 잡음 없애는 용도로 많이 사용

    → DCNN : conv+pooling 통해 크기 작아짐 → upsampling 통해 원 영상 크기로 확대 ⇒ 이미 충분히 smooth 

    → 여기에 short-range CRF 적용 ⇒ 결과 더 나빠짐

- 전체 픽셀 모두 연결 (fully connected) CFR 개발

  ( 공부 시 참고 : <http://swoh.web.engr.illinois.edu/courses/IE598/handout/fall2016_slide15.pdf>)



- shot-range CRF : local connection 정보만 사용 → detail 정보 얻을 수 없음

![image](https://user-images.githubusercontent.com/45067667/58684307-c855e300-83b2-11e9-96b5-a4e4b88b92e2.png)

- fully connected CRF : detail 살아있는 결과 얻을 수 있음

  ![image](https://user-images.githubusercontent.com/45067667/58684359-f509fa80-83b2-11e9-84b7-d432da9edb2b.png)

  - MCMC(Markov Chain Monte Carlo) 방식 사용 → 좋은 결과 but 시간 오래 걸림 

    ⇒ mean field approximation 적용 : message passing 사용 iteration 방법 적용 → 0.2초 수준으로 효과적으로 줄일 수 있음!

    ⇒ fully connected CRF 수행

  - mean field approximation : 복잡한 모델을 설명하기 위해 더 간단한 모델 선택하는 방식 (물리학/확률이론에서 많이 사용)

    : 수많은 변수들로 이루어진 복잡한 관계를 갖는 상황 

    → 특정 변수와 다른 변수들의 관계에 평균을 취함 → 평균으로부터 변화(fluctuation)해석 용이

    → 평균으로 근사된 모델 사용 → 전체 조망하기에 좋음

  - CRF 수식 

  ![image](https://user-images.githubusercontent.com/45067667/58684603-bcb6ec00-83b3-11e9-9af3-5a36a91bb16e.png)

  - unary term + pairwise term
  - $x$ : 각 pixel 위치에 해당하는 pixel label / $i, j$ : pixel 위치
  - Unary term : CNN 연산 통해서 얻을 수 있음 
  - pairwise term : pixel 간 예측에 중요 / pixel value 유사도와 위치적인 유사도 함께 고려 (like bi-lateral filter)
    - 2개의 가우시안 커널로 구성 → $\sigma_{\alpha}, \sigma_{\beta}, \sigma_{\gamma}$ 통해 scale 조절 가능 
    - 첫번째 가우시안 커널 : 비슷한 위치, 비슷한 컬러를 갖는 픽셀에 대해 비슷한 label이 붙을 수 있도록 해줌
    - 두번째 가우시안 커널 : 원래 픽셀의 근접도에 따라 smooth 수준 결정
    - $p_i, p_j$ : 픽셀의 position / $I_i, I_j$ : 픽셀의 intensity

  ⇒ 고속 처리 위해 mean field approximation 적용 →  feature space 에서 Gaussian convoluton으로 표현 가능 → 고속 연산 가능

  

5. Depth wise Separable Convolution (참고 : <https://blog.lunit.io/2018/07/02/deeplab-v3-encoder-decoder-with-atrous-separable-convolution-for-semantic-image-segmentation/>)

   - convolution 

     ![image](https://user-images.githubusercontent.com/45067667/58684950-0bb15100-83b5-11e9-9b42-06304122d508.png)

     입력 image : 8x8x3 (HxWxC) * convolution fiter : 3x3 (FxF)

     → filter 1개가 가지는 parameter 수 : 3x3x3 (FxFxC) = 27 

     → filter 4개 사용시 convolution의 총 parameter 수 : 3x3x3x4 (FxFxCxN)

     

   - Depthwise Convolution

   ![image](https://user-images.githubusercontent.com/45067667/58685123-92fec480-83b5-11e9-9005-efcd1709f04c.png)

   ​	convolution 연산에서 channel 축을 filter가 한번에 연산 대신 

   → input image의 channel 축 모두 분리 → channel 축 길이 항상 1로 가지는 여러개의 convolution filter로 대체

   

   - Depthwise separable convolution

     ![image](https://user-images.githubusercontent.com/45067667/58685173-c5a8bd00-83b5-11e9-9d97-a2e72f0a6edf.png)

     depthwise convolution으로 나온 결과 → 1x1xC 크기의 convolution filter 적용

     - 기존 convolution과 유사한 성능 & 사용되는 parameter 수와 연산량 획기적인 감소

     ex. input : 8x8x3 / 16개의 3x3 convolution filter

     → convolution : 3x3x3x16 = 432

     → depthwise separable convolution : 3x3x3 + 3x16 = 27 + 48 = 75

     - 기존 convolution filter : spatial dimension & channel dimension 동시 처리 

     → depthwise separable convolution : 따로 분리 시켜 각각 처리

     ⇒ 여러개의 필터 : spatial dimension 처리에 필요한 parameter 하나로 공유 → parameter 수 감소

     - 두 축을 분리하여 연산 수행 but 최종 결과 값은 두 축 모두 처리한 결과값 얻을 수 있음 → 기존 convolutional filter가 수행하던 역할 대체 가능

     ⇒ parameter 수 대폭 줄일 수 있음 

     ⇒ 깊은 구조로 확장하여 성능 향상 / 기존 대비 메모리 사용량 감소 & 속도 향상



### Related Work 

-------------

- FCNs based model : segmentation 결과 향상 

  → contextual information 추출 방법 다양

  	- employ multi-scale input
  	- probabilistic graphical model



- Spatial Pyramid pooling

  - PSPNet / DeepLab : 여러 grid scale에서 spatial pyramid pooling 수행 (image level pooling 포함) & 다른 비율로 여러 parallel atrous convolution 적용 (ASPP)

  → multi-scale information 추출



- Encoder-decoder : computer vision task에 성공적으로 적용(ex. human pose estimation / object detection / semantic segmentation)

  - encoder module : gradually reduce feature map & capture higher semantic information
  - decoder module : gradually recover spatial information

  ⇒ encoder로 DeepLab V3 사용 & sharper segmentation얻기 위해 simple yet effective decoder module 사용



- Depthwise Separable convolultion

  - computation cost & the number of parameters reduce
  - maintain similar performance

  → Xception model 사용 → accuracy & speed 향상



### Methods

------------------------

- atrous convolution, depthwise separable convolution 간단히 소개
- encoder module로 사용한 DeepLab V3 설명
- modified Xception model 소개 (faster computation & 성능 향상)

#### Encoder-Decoder with Atrous Convolution

##### Atrous Convolution

- DCNN 에서 계산된 resolution of feature 명시적으로 제어

- multi-scale information capture 위해 filter의 field of view 조정

- 2D image에 atrous convolution 적용 경우 

  each location $i$ , output feature map $y$ , convolution filter $w$, input feature map $x$
  $$
  y[i] = \sum_{k}x[i+r\cdot k]w[k]
  $$
  , atrous rate r : input signal sample할 stride

  ⇒ rate value r 바꿈 : filter의 field of view adaptively 수정



##### Depthwise Seprable Convolution

- standard convolution → depthwise convolution + pointwise convolution 

  ⇒ reduce computation complexity

- depthwise convolution : 각 input channel에 대해 따로 spatial convolution 수행

  → pointwise convolution ⇒ combine

- tensorflow 구현 시 : depthwise convolution 구현에서 atrous convolution 구현 가능

![image](https://user-images.githubusercontent.com/45067667/58696172-fb5b9f00-83d1-11e9-910c-1fe71e6a7518.png)

⇒ atrous separable convolution 수행 : computation complexity reduce, performance 유지



##### DeepLab V3 as encoder

- DeepLab V3 : atrous convolution 수행  → DCNN에서 임의의 resolution 의 feature 추출 위해

- output stride : final output resolution에 대한 input spatial resolution ratio (before global pooling / fully connect layer)

  - for image classification : output stride = 32

  - for semantic segmentation : output stride = 8 /16

    → denser feature extraction! (마지막 1개(2개) block 의 striding 제거 & 대응하는 atrous convolution 적용)

    ex. 마지막 2개 block에 각각 r = 2, r = 4, output = 8 적용

- DeepLab V3 : Atrous Spatial Pyramid Pooling module

  → atrous convolution에 모두 다른 rate 적용 ⇒ 다양한 scale에서 convolutional feature 검색

- DeepLab V3의 encoder output 바로 전 feature map 사용 → encoder output feature : 256 channel & rich semantic information 갖음

＋ computation budget에 따라 arbitrary resolution 에서 feature extract 가능 by applying atrous convolution



##### Proposed decoder 

- DeepLab V3의 encoder feature : output stride = 16 으로 계산
- factor 16으로 bilinearly upsampled : naive decoder module → object segmentation detail 성공적으로 회복 X 

![image](https://user-images.githubusercontent.com/45067667/58698415-36f86800-83d6-11e9-9bbb-13085aa01920.png)

⇒ propose simple yet effective decoder module

- encoder feature : up-sampling by 4 → 대응되는 low-level feature from network backbone (같은 spatial resolution)와 concatenate

- low level feature에 lxl conv 적용 (#channel 줄이기 위해) ∵ corresponding low-level feature는 주로 큰 수의 channel 갖음 (ex. 512, 256)

  → encoder feature 중요성 과하게 할 수 있음 & training 어렵게 함

- concatenate 후 → 3x3 conv 적용 (refine feature위해) → 다른 simple bilinear up-sampling 적용 (factor 4)

- encoder의 output stride = 16 사용 : best trade-off btw speed and accuracy

  output strdie = 8 사용 : cost of extra computation complexity 감수 → performance marginally 향상 

  (위 내용 Sec 4. 에서 확인 가능)



##### Modified Aligned Xception

- Xception : ImageNet 의 image classification 에서 좋은 결과 & fast computation

- Aligned Xception (modify Xception) : object detection task에서 성능 향상

  ⇒ semantic image segmentation task위해 Xception model 사용

  

- Aligned Xception 수정해서 사용

![image](https://user-images.githubusercontent.com/45067667/58744757-d161b580-8481-11e9-89cf-10e959b9be4a.png)

1. deeper Xception 동일 (fast computation 과 memory efficiency 위해 entry flow network 수정 X)

2. 모든 max pooling → depthwise separable convolution with striding 대체

   : atrous separable convolution 적용 가능 (arbitrary resolution 에서 feature map 추출 위해)

3. 각 3x3 depthwise convolution 후 → batch normalization & ReLU 추가 (MobileNet 과 비슷)



### Experiment

--------------------

ImageNet-1k pretrained ResNet-101 / modified aligned Xception 사용

→ extract dense feature map by atrous convolution 

##### 1. ResNet-101 as Network Backbone 

- ResNet-101 구조 encoder로 사용하였을 때

  ![image](https://user-images.githubusercontent.com/45067667/58692522-ba5f8c80-83c9-11e9-8764-2e8c907c9b1d.png)

  Decoder 부분 : bilinear upsampling → 단순 U-Net 구조 사용 : 기존 대비 mIOU 1.64% 향상



##### 2. Xception as Network Backbone

![image](https://user-images.githubusercontent.com/45067667/58692924-9badc580-83ca-11e9-97c9-74cbcb4b313f.png)

- ResNet-101 에 비해 Xception : 약 2% 정도 성능 향상
- ASPP & decoder 부분 convolution → separable convolution 대체 (SC 부분) : 성능 비슷하게 유지 but computation complexity 획기적인 감소 (Multiply-Adds) 