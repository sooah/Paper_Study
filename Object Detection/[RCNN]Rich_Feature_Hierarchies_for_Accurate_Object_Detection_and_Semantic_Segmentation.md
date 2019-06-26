# R-CNN : Rich feature hierarchies for accurate object detection and semantic segmentation

### Abstract 

-------------------

objection 분야에서 현재 까지 중 best 기술 : complex ensemble systems (multiple low level image feature + high level context)

→ simple & scalable detection algorithm 개발! (improve mean average precision(mAP))

⇒ approach : 2개의 insights 결합

1.  high-capacity CNN에 bottom-up region proposals 적용 → localize 와 segment object 위해

2.  labeled training data 부족한 경우 

   → auxiliary task 위한 supervised pre-training 

   → domain-specific fine-tuning 

   ⇒ performance 향상!

**CNN에 region proposal 접목! : R-CNN**

overfeat와 비교했을 때 성능 더 좋음



##### cf) Region Proposal

Naive approach : 물체가 존재할 수 있는 모든 크기의 영역(different scale & ratio)에 대해 sliding window로 image 모두 탐색 → classification 수행

- 이렇게 하면 탐색해야하는 영역 수 너무 많음
- classifier가 충분히 빠르지 않다면 연산 시간 오래 걸림

⇒ 비효율!

Region Proposal : sliding window의 비효율성을 개선하기 위해, input에서 '물체가 있을 법한' 영역을 빠른 속도로 찾아내는 알고리즘!

- 여러 종류가 있으나 그 중 **selective search & edge boxes** 가 좋음
- sliding window 방식에 비해 search space 확연하게 줄어듦 → 훨씬 빠른 속도로 object detection 수행 가능

⇒ image classification 수행하는 CNN + image에서 뭋레가 존재할 영역을 제안해주는 region proposal 연결

→ 높은 성능의 object detection 가능!



### Introduction

---------------------

visual recognition task : 주로 SIFT 와 HOG based 임

- SIFT : Scale Invariant Feature Transform

- HOG : Histogram of Gradient

  → blockwise & orientation histogram ( 대부분 영상 내에 존재하는 gradient 성분을 일정 block으로 나누고, 그 경향성을 이용하여 대상 검토)

→ V1(영장류의 visual pathwat에서 첫번째 cortical area)의 complex cell 과 대략적을 관련된 표현 가능

recognition : 여러 stage로 구성 

- visual recognition 위한 computing feature : 계층적, 다단계의 과정 있음
- 이를 "neocognition"에서 시도 but supervised training algorithm

⇒ this paper : CNN이 object detection에서 매우 높은 성능을 낼 수 있음을 처음으로 보여줌!

→ 이를 위해 2가지 문제에 집중 

1. deep network 로 localizing object
2. 적은 양의 annotated detection data 사용해서 high-capacity model training



Detection : localizing object 필요

이를 해결 위해 푸는 방법 다양 

1. regression problem 으로 보는 경우
2. sliding-window detector 사용 : object categories에 제한되어 사용되어옴 → high resolution quality 유지 위해 2개의 convolutional and pooling layer 갖음

but network 상층부 - 5 layer (receptive field : 195 X 195 / stride 32 X 32)

→ 정확한 localization 을 sliding window에서 어떻게?

⇒ **"recognition using regions"**사용해서 해결

1. input image에서 2000여개의 region proposal 생성
2. 각 proposal에서 CNN 사용해서 fixed-length vector 추출
3. category-specific linear SVM 사용 → 각 region 분류



region shape에 상관없이 각 region proposal에서 fixed-size의 CNN input 계산 위해 *affine image warping* 

 ![image](https://user-images.githubusercontent.com/45067667/60154073-37432200-9821-11e9-947d-a6df51f1a0e6.png)

: method overview & result highlight

⇒ region proposal을 CNN에 결합!

- overfeat(sliding window CNN) 보다 mAP 높음

  

- Detection에서 2번째 문제 발생

  : labeled data 매우 희박 & 사용 가능한 양이 큰 CNN 훈련 시키기에 불충분

  → conventional solution : unsupervised pre-training followed by supervised fine-tuniing

but 이 논문의 2번째 기여 : 큰 pre-data에 supervised pre-training → small data set에 domain-specific fine tuning!

→ data가 희박할 경우, high capacity CNN 학습의 효율적인 패러다임!!!

⇒ 매우 효율적임! class-specific computation : reasonably small matrix-vector product & greedy non-maximum suppression

→ 이런 연산 특성을 모든 카테고리에서 공유 & 이전에 사용된 region feature 보다 2 dim 낮은 형상에서 비롯


##### cf) non-maximum suppression 이란?

: image processing 통해 얻은 edge를 얇게 만들어 주는 것을 의미

gaussian mask / sobel mask 와 같은 것은 이용하여 찾은 edge → blurring 되어 있음(뭉게져 있음) ⇒ 더 선명한 선을 찾기 위해!

중심 pixel 기준 8방향 pixel value 비교 → 중심 pixel 가장 클 경우 놔두고, 아닐 경우 제거!

ex. 아래와 같은 pixel value 갖는 이미지 있다고 가정

![image](https://user-images.githubusercontent.com/45067667/60005681-eeb52880-96a9-11e9-8788-f9af89c3bca6.png)

1) 첫 번째 3 X 3 window 내의 값 비교

![image](https://user-images.githubusercontent.com/45067667/60005769-20c68a80-96aa-11e9-9c3e-b2ffbaebcd3f.png)

중심 pixel인 5 기준 8 방향 pixel value 비교 → 파란 테두리인 7이 가장 큰 것 확인 가능 

⇒ **중심 픽셀 값이 non-maximum 이므로 중심 픽셀 값은 0으로 제거!**

2) 다음 한 칸 옮겨서 진행

![image](https://user-images.githubusercontent.com/45067667/60005877-52d7ec80-96aa-11e9-8e7e-f652a4910731.png)

중심 pixel인 7 기준 8 방향 pixel value 비교 → 중심 pixel이 가장 큰 값을 갖음

⇒ **중심 픽셀 값이 maximum이므로 제거하지 않고 7 그대로 둠**

3) 전 영역에 대해 실시

![image](https://user-images.githubusercontent.com/45067667/60005963-81ee5e00-96aa-11e9-8055-820d7f5b972e.png)

non-maximum 값들에 0의 값을 주면 됨!

⇒ **뭉게진 직선을 보고 원래의 sharp한 직선을 찾는 과정**



### Object image warping

-------------------------

1. category-independent region proposal 생성 → detector가 사용 가능한 set of candidate detection 생성
2. large CNN : 각 region에서 fixed-length feature vector 추출
3. set of class-specific linear SVMs

#### 1. Module Design

##### Region Proposals

category-independent region proposal 생성 방법 다양

ex. objectness, selective search, category-independent object proposals, constrained parametric min-cuts(CPMC), multi-scale combinatorial grouping

→ R-CNN : particular region proposal method 가 아님(모든 region detect 할 필요 없음)

⇒ selective search 사용 : 이전 detection work 와 비교 위해!



##### Feature Extraction

각 region proposal에서 4096 dim의 feature vector 추출 by CNN

- feature : image 의 중심을 기준으로 227 X 227 잘라내고 (RGB image) → 5개의 convolutional layer + 2개의 fully connected layer

- region proposal에서 feature 계산

  - CNN에 input으로 넣어주기 위해 image data convert (이 architecture의 경우 fixed 227X227 pixel size 요구)

  ⇒ 다양한 transformation 존재 : 그 중 가장 간단한 것 사용!

  : candidate region의 size나 aspect ratio에 관계없이 모든 pixel을 요구된 size에 맞춰 만듦(우겨 넣는다 생각)

  - warping 이전에 tight bounding box 살짝 확대 → original box 주위로 p pixel을!



#### 2. Test-time detection

- Test 시 : test image에 selective search 적용 → 2000개의 region proposal 추출 (모든 실험에, selective search의 'fast model' 적용)

  -  각 proposal warp → CNN으로 feature 추출
  - 각 class에 대해 SVM  사용하여 각 class에 대한 extracted feature vector의 score 매김
  - image의 모든 점수가 매겨진 영역 고려 → greedy non-maximum suppression 적용 (각 class마다 따로)

  ⇒ 높은 scoring selected region 과 overlap 된 IOU가 학습된 threshold 보다 더 큰 값을 갖는 경우 reject!!!



##### Runtime analysis

2가지 특성이 detection을 효율적으로 만듦!

1. 모든 CNN parameter를 모든 카테고리들이 공유

2. CNN으로 계산된 feature vector는 low dimension을 갖음 

   (bag-of-visual-word encodings 사용한 spatial pyramid 와 같은 일반적인 방법에 비해!)

⇒ sharing : computing region proposal & feature에 걸리는 시간을 모든 class에 나눠서!

각 class 구체적인 계산 : SVM weight 와 feature 사이의 dot product , non-maximum suppression 만!



#### 3. Training

##### Supervised pre-training

large auxiliary dataset → pretrained CNN (using image-level annotation만)



##### Domain-specific fine-tunning

Detection 과 warped proposal window에 CNN 적용하기 위해

→ warped region proposal 만 사용하여, CNN parameter SGD training



##### Object category classifier

car detect 하는 binary classifier 라 가정 

- 차를 확실히 둘러싸는 것 : positive example / 차가 없는 것 : negative example
- but 차가 overlap 되어 있다면 label 어떻게?

→ IOU overlap threshold로 문제 해결! (어떤 region을 negative로 분류할지)

- overlap threshold : 0.3 ({0, 0.1, ... 0.5} 사이에서 valid set에 의해 선택) ⇒ 이 선택이 매우 중요

⇒ positive example : 각 class의 ground-truth bounding box로 정의!

- feature 추출 &  training label 적용 : 각 class 마다 linear SVM 적용

  : training data 가 memory 에 비해 너무 큼 → standard hard negative mining method 사용



### Visualiztion, ablation, and models of error

-----------

#### 1. Visualizing learned features

first-layer filter : oriented edge & opponent color capture



##### Reference 

--------------

- non-maximum suppression : <https://m.blog.naver.com/PostView.nhn?blogId=jinsoo91zz&logNo=220511441402&proxyReferer=https%3A%2F%2Fwww.google.com%2F>
- 

