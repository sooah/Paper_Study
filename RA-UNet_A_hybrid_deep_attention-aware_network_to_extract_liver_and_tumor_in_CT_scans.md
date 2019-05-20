## RA-U Net : 3D Hybrid Residual Attention-Aware Segmentation

### Introduction

----------------------

- High cost & High GPU memory 필요 → 3D FCN 구현 어려움 (depth가 2D에 비해 적음)

  이를 해결하기 위해 attention mechanism 과 residual network 사용

  End-to-end로 3D volumetric contextual feature 추출 가능

- U-Net 구조  + attention-residual learning mechanism → optimization & performance improvement 기대! (매우 깊은 network에 비해)



- contribution of this work

  1. Architecture에 residual block 쌓음

     → 더 깊은 architecture & gradient vanishing problem 다룰 수 있음

  2. Attention mechanism : img의 특정 part에 집중하는 능력이 있음

     → attention module 쌓음 → 다른 종류의 attention 가능 ⇒ attention-aware feature는 적응해서 변화 가능

  3. Basic architecture로 2D/3D U-Net 사용 →  다른 종류의 attention 가능 ⇒ end-to-end 가능



- Pretrained model 필요 없음 & post processing technique 필요 없음 (ex. 3D conditional random fields)



### Related Works

--------------------

1. Cascaded FCNs (CFCNs) : first FCN : segment liver / second FCN : segment its lesions

2. SurvivalNet (3D neural network) : lesion’s malignancy 예측



### Methodology

--------------------

##### A. Overview of our proposed architecture

1. 2D residual attention-aware U-Net (RA-Unet) : 전체적인 computational time 줄이기 위해 사용

   → U-Net의 connection 부분에 residual-attention mechanism 추가

   ⇒ coarse liver boundary box를 찾음

2. 3D RA-Unet : 정확한liver VOI를 얻기 위해 training

3. 2번째 3D RA-Unet : prior liver의 VOI 전달 → tumor region 뽑음

   ⇒ 다양하고 복잡한 조건에서 volume 다룰 수 있음 &  다른 liver/tumor data set 에서도 적용 가능



##### C. Data preprocessing

CT : HU 값 사용(-1000~1000) ⇒ liver region만 clean하게 남기기 위해 filter 사용 (주위 bone, air, 다른 tissue 제거) 

→ global windowing step 사용 : HU window 를 -100~200사이로 설정 ⇒  불필요한 organ/tissue 제거

→ zero-min normalize 적용



##### D. RA-Unet architecture

- residual block – network가 수백개의 layer 가질 수 있도록 해줌

- Attention mechanism – interest 부위만 구분하기 위해 관련된 지역만 focus

1. U-net as basic architecture : overall architecture 로 standard U-net 사용

   - Encoder : contextual information → complexity 갖는 hierarchical feature 추출

   - Decoder : diverse complexity 갖는 feature 받고, coarse to fine manner로 feature reconstruct

   → U-net : encoder와 decoder 잇는 connection 있음 → different hierarchical feature 전달 ⇒ network 더 precise, expansible 해짐

2. Residual learning mechanism : neural network deep → gradient vanishing 문제 발생 가능

   -  by He, residual of the identity map learn → residual learning framework! 

     ⇒ first layer, last layer 제외하고, residual block stock했음 (deeper하게 만들기 위해)

     

![1558333101148](C:\Users\soua\AppData\Roaming\Typora\typora-user-images\1558333101148.png)

- Stacked residual block : gradient vanishing problem 해결 
  -  neural network의 structure level에서 해결 → skip connection 으로 identity mapping 사용 → activation
- Residual block :  $ OR_{i,c}(x) = x+f_{i,c}(x) $ 
  - $ x $ : first input of residual block
  - $ OR $ : output of residual block
  - $ i $ : 모든 spatial position 에 대해
  - $ c ∈ {1,…, C} $ : index of channel ($ C $ : total number of channel)
  - $ f $ : residual mapping
- BN → activation(ReLU) → convolution

![1558334391028](C:\Users\soua\AppData\Roaming\Typora\typora-user-images\1558334391028.png)

3. Attention residual mechanism

- The attention module

  - trunk branch : original feature process 위해 사용 
  - soft mask branch : identity mapping construct 위해 사용

- Soft attention module output : $OA_{i,c}(x) = (1+S_{i,c}(x))F_{i,c}(x) $ , $S(x)∈[0,1]$ 

  → $S(x)$ 가 0에 가까우면 $OA(x)$ 는 original feature인 $F(x)$ 에 가까움

  → $S(A) $ : soft mask branch → trunk branch 에서 온 것 중, identical feature 선택, noise suppress

  ⇒ attention residual mechanism 에서 매우 중요한 역할

- Encoder in soft mask branch : max-pooling, residual block, long range residual block connected to the corresponding decoder

  → 다 하고 나서 sigmoid layer (output normalize하기 위해)

  ![1558334690544](C:\Users\soua\AppData\Roaming\Typora\typora-user-images\1558334690544.png)

⇒ trunk branch로부터 온 original feature information은 keep & soft mask branch 에서 liver tumor feature 에pay attention
