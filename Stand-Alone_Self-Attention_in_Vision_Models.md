# Stand-Alone Self-Attention in Vision Models

### Abstract

------------------

convolution : 현대 computer vision 에서 fundamental building block 으로써 자리잡고 있다. 

→ long-range dependency 의 경우 안 좋다는 단점이 있음

→ capture long-range dependency 위해 convolution 뛰어넘는 방법 연구!

⇒ content-based interaction 기반으로 convolution model 강화! (ex. self-attention, non-local means)

⇒ vision task 해결하고자 한다!

natural question : attention 이 단지 convolution 위에서 augmentation 되는 것 대신에, vision model에 독자적으로 사용가능한가?

- self-attention : 효과적인 stand-alone layer로 적용 가능!

- ResNet model에서, 모든 instance of spatial convolution 대신, self-attention 적용! ⇒ fully self-attention model 만듦

  → ImageNet Classification : 12% fewer FLOPS / 29% fewer parameters

  → COCO Object Detection : 39% fewer FLOPS / 34% fewer parameters

cf) FLOPS : 얼마나 연산량이 많은 지를 나타내는 지표

Detailed ablation studies : self-attention은 layer layer에 사용했을 경우 더 효과적!

(∵ stem에 해봤더니 별로였음, global한 건 self-attention이 더 잘 작동!)



### Introduction

---------

많은 양의 data set & computer source 등장 → CNN이 computer vision 응용의 backbone으로 자리잡게 되었음

CNN의 translation equivariance property : image에서 작동 위해 block 만들 수 있음!

but capturing long range interactions for convolution 문제 : 큰 receptive field에 대한 scaling property 별로여서..

cf) translation equivariance 

- equivariance : 함수의 입력이 바뀌면 출력 또한 바뀜

- translation equivariance : 입력의 위치가 변하면 출력도 동일하게 위치가 변화!

  

sequence modeling에서 : long range interaction 문제를 attention 을 사용해서 해결!

→ 최근, attention module : discriminative computer vision 문제에 적용 ⇒ traditional CNN의 성능 향상

- channel-based attention mechanism(Squeeze-Excite) : CNN channel의 scale 선택적으로 조절하기위해 적용
- spatially-aware attention mechanism : CNN architecture augment 위해 사용 → object detection & image classification 향상 위해 contextual information 제공!

→ 이런 것은 global attention layer를 이미 있는 convolutional model 위에 더한 것!

⇒ input의 모든 spatial location에 적용 → down sampling 되면서 작은 input에 영향을 줌



### Background

---------

#### 2.1 Convolutions

CNN : 적은 수의 neighborhoods 사용 (ex. kernel size 만큼) → 각 layer에서 local correlation structure 학습!

input : $x\in\mathbb{R}^{h\times w \times d_{in}}$ ($h$ : height, $w$ : width, $d_{in}$ : input channels)

- $x_{ij}$ pixel 주위의 local neighborhood : $\mathcal{N}_{k}$ 

→ spatial extent $k$를 사용하여 추출

⇒ region shape : $k \times k \times d_{in}$

![image](https://user-images.githubusercontent.com/45067667/61105008-6dd09c00-a4b3-11e9-9fbb-5887babd0915.png)

 weight matrix $W \in \mathbb{R}^{k \times k \times d_{out} \times d_{in}}$ → output  $y_{ij} \in \mathbb{R}^{d_{out}}$ : input 과 weight 의 depthwise matrix multiplication 의 sum!
$$
y_{ij} = \sum_{a,b\in \mathcal{N}_k(i,j)}W_{i-a,j-b}x_{ab} , \mathcal{N}_k(i,j) = \lbrace a,b||a-i|\le {k \over 2}, |b-j| \le {k \over 2} \rbrace
$$
![image](https://user-images.githubusercontent.com/45067667/61105303-6bbb0d00-a4b4-11e9-8360-4bb7fabfe9c9.png)



CNN : weight sharing 사용 → 모든 pixel position output 생성에 같은 weight resue

⇒ weight sharing : 학습된 표현의 translation equivariance 강화 & input size에서 convolution의 parameter count 분리



- ML application : convolution 사용 (ex. text-to-speech, generative sequence model)

→ convolution : predictive performance & model computational efficiency 향상

- depthwise-separable convolution : spatial-channel interaction 으로 low-rank factorization 얻음



#### 2.2 Self-Attention

Attention : neural sequence transduction model의 encoder-decoder에서 사용

→ source sentence의 length가 가변적! ⇒ context 기반 내용 요약 정보 제공 위해!

- ability : 중요한 region에 focus하여 context 학습 → 중요한 component 생성 → deep learning에 많이 적용 ⇒ recurrence with self attention으로 대체

self-attention : multiple context 대신, single context에 attention 적용 

(= 같은 context에서 모두 추출하여 query, key, value 만듦)

- ability : long-distance interaction 과 parallelizability → SOTA model 들에 많이 적용되어 있음



Vision : Self-attention은 non-local mean의 예시화라 볼 수 있음 → video classification  & object detection에서 성취!

⇒ **이 paper : convolution 제거 → network 전반적으로 local self-attention 적용**

다른 연구(H. Hu, Z. Zhang, Z. Xie, and S. Lin, “Local relation networks for image recognition,” arXiv
preprint arXiv:1904.11491, 2019) 

: model에 새로운 content-based layer 제안! → vision model에 self-attention의 form을 취한 이 paper 내용 보충해줄 수 있음



Stand-Alone self-attention layer : spatial convolution 대체, 완전히 attentional model 생성 → attention layer : simplicity에 집중!



pixel $x_{ij} \in \mathbb{R}^{d_{in}}$ 

- position $ab \in \mathcal{N}_k(i,j)$ : spatial extent $k$, $x_{ij}$ 주위에서 local region 추출 ⇒ 하나의 *memory block*

  → 모든 pixel에서 global(all to all) attention 하던 것과는 다름!

  ∵ global attention : input에 spatial down sampling 적용한 후에만 사용 가능

  (computationally expensive → fully attention model에서 모든 layer에 사용되는 것 방지!)

- output $y_{ij} \in \mathbb{R}^{d_{out}}$ : single-headed attention 사용

  ![image](https://user-images.githubusercontent.com/45067667/61105959-82626380-a4b6-11e9-9df0-2fd10b88631e.png)

$$
y_{ij} = \sum_{a,b \in \mathcal{N}_k(i,j)}\mathsf{softmax_{ab}}(q_{ij}^Tk_{ab})v_{ab}
$$

  - queries : $q_{ij} = W_Qx_{ij}$

  - keys : $k_{ab} = W_Kx_{ab}$ 

  - values : $v_{ab} = W_Vx_{ab}$

    : i,j 번째 pixel 과 그 neighborhoods 와의 linear transformation

- $\mathsf{softmax}_{ab}$ : $ij$의 neighborhood 로 계산된 모든 logit에 적용
- $W_Q,W_K,W_V \in \mathbb{R}^{d_{out} \times d_{in}}$ : 모두 학습된 transform



local self attention : neighborhood에 대해 spatial information 종합 (convolution과 비슷)

→ aggregation : content interaction으로 parameter화한 mixing weights($\mathsf{softmax}_{ab}$) 와 value vectors의 combination으로!

→ 모든 pixel $i,j$에 대해 반복

⇒ $x_{ij}$의 pixel feature를 N개의 group으로 partitioning! $x_{ij}^n \in \mathbb{R}^{d_{in} / N}$ 

- 각 group에 single-headed attention 따로 계산

- 각 head는 각각의 transform을 갖음 : $W_Q^n, W_K^n, W_V^n \in \mathbb{R}^{d_{out} / N \times d_{in} / N}$ 

  → output concatenate ⇒ final output : $y_{ij} \in \mathbb{R}^{d_{out}}$ 

대신, 이렇게 하는 경우 attention에 positional information이 없음 → permutation이 모두 똑같이 생성 가능 ⇒ vision task의 표현성 제한!

"Attention is all you need" 논문과 같이 image $i,j$를 absolute position of pixel로 사용하여 sinusoidal embedding 도 가능 

but *relative positional embedding* 이 더 좋은 결과를 냄



대신, 2D relative position embedding에 attention 적용!(relative attention)

- 각 position $ab \in \mathcal{N}_k(i,j)$ 에서 $i, j$ 에 대한 relative distance 정의로 시작

- relative distance : dimension 따라 쪼개짐 → 각 element $ab \in \mathcal{N}_k(i,j)$ : 2개의 distance 받음

  : row offset $a-i$  & column offset $b-j$ : 각각 dimension ${1 \over 2}d_{out}$과 관련된 embedding인 $r_{a-i}, r_{b-j}$ 갖음 

  → concatenate ⇒ form $r_{a-i,b-j}$

  ![image](https://user-images.githubusercontent.com/45067667/61106931-8d6ac300-a4b9-11e9-96f1-3c2b5e96dceb.png)

⇒ spatial-relative attention
$$
y_{ij} = \sum_{a,b \in \mathcal{R}(i,j)} \mathsf{softmax}_{ab}(q_{ij}^Tk_{ab}+q_{ij}^Tr_{a-i,b-j})v_{ab}
$$


- $\mathcal{N}_k(i,j)$의 element와 query 사이의 similarity 측정하는 logit : element의 content와 query로부터 element의 상대적인 거리 모두 조절

  → relative position information 사용 : self-attention도 translation equivariance 가능 (convolution과 비슷하겠지!)

- attention의 parameter 수 : spatial extent(kernel size 의미)의 크기와 무관 

  but convolultion의 parameter 수 : spatial extent에 quadratic하게 증가

- computational cost of attention : spatial extent로 느리게 증가

  convolution : $d_{in}, d_{out}$ typical value로 증가

ex. # of $d_{in}, d_{out} = 128, k = 3$인 convolution layer의 para 

= # of $k=19$인 attention layer의 para



### Fully Attentional Vision Model

---------

local attention layer를 primitive하게 여기는 경우, fully attentional architecture를 어떻게 구성하는가?

#### 3.1 Replacing Spatial Convolutions

spatial convolution : spatial extent k>1 사용한 convolution 의미

→ 1 × 1 convolution은 제외 (fully connected layer 만들 때 사용하는)

creating fully attentional vision model : 기존의 convolutional architecture 사용 → 모든 instance의 spatial convolution을 attention layer로 대체

2 × 2 average pooling with stride 2 : attention layer 뒤에 붙음 (spatial downsampling이 필요하든, 아니든)

⇒ ResNet family architecture에 적용

- core building block of ResNet : bottlenet block

  input → 1×1 down projection conv → 3×3 spatial conv → 1×1 up projection conv → output, input을 output으로 residual

  → ResNet 형성 위해 bottleneck block 여러번 반복 (한 bottleneck block의 output → 다음 bottleneck block의 input)

  ⇒ 3×3 spatial convolution을 self-attention layer로 대체
  $$
  y_{ij} = \sum_{a,b \in \mathcal{N}_k(i,j)} \mathsf{softmax}_{ab}(q_{ij}^Tk_{ab}+q_{ij}^Tr_{a-i,b-j})v_{ab}
  $$



#### 3.2 Replacing the Convolutional Stem

CNN의 initial layer(stem) : local feature(edge 같은) 학습

later layer : global object 확인

input image 크면 : stem - core block과 다름 → spatial downsampling으로 light weight operation에 초점

ex. ResNet : 7×7 conv (s=2) → 3×3 max pooling (s=2)

- stem layer에서 : content는 RGB pixel들로 구성 → 각각은 uninformative, 서로 매우 강하게 correlated 되어 있음

  → 이런 특성 : self-attention같은 content-based mechanism이 edge detector와 같은 의미있는 feature 학습하는 것을 어렵게 만듦

  ⇒ 그래서, stem에서 self-attention 사용한 것은 convolution 보다 성능 별로임

- distance based로 weight parametrization 하는 convolution : higher layer에서 필수적인 edge detector와 다른 local feature들을 쉽게 학습

  → convolution 과 self-attention 사이의 gap을 잇기 위해(computation 증가하지 않으면서) 

  : spatially-varying linear transformation 통한 pointwise 1×1 convolution으로 distance based information 주입!

  -  새로운 value transformation
    $$
    \tilde{v}_{ab} = (\sum_mp(a,b,m)W_V^m)x_{ab}
    $$
    multiple value matrices $W_V^m$ : 해당 position pixel의 neighborhood $p(a,b,m)$의 function인 factor들의 convex combination

  position dependent factor : convolution과 비슷 → pixel 위치의 neighborhood에 의존하여 scalar weight 학습

- stem : spatially aware value features를 갖는 attention layer로 구성 → 그 뒤에 max pooling

  → 간단하게 하기 위해, attention receptive field 는 max pooling window와 연관.

  자세한 사항은 appendix 참고 바람!