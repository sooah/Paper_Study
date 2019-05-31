# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

--------------------------

**[CycleGAN] : Image to Image translation (전체적인 형태는 유지하는 style transfer)**

-------------------------------

### Abstract

-------------------------

Paired example 없이 source domain X에서 target domain Y로 image translate하는 learning approach 제시

Goal : learn a mapping G : X → Y by adversarial loss (G(x)에서 나온 image의 distribution이 Y에서 나온 distribution과 구분되지 않음)

![image](https://user-images.githubusercontent.com/45067667/58673638-3e445500-8387-11e9-9507-ac87a1d62279.png)

### Introduction

---------------------------

- In this paper : 하나의 image collection에서 특별한 특징을 capture → 이 특징들이 다른 image collection에 적용되는 경우에 대한 learning method 제안 **이때 어느 paired training sample 존재 X **

- image translation 의 경우(ex. grayscale to color, image to semantic label, edge-map to photograph) → supervised setting에서는 image pair 존재해야만 가능!

  but, pair training example 얻는 것은 어렵고, 비쌈

⇒ **paired input-output example 없이 domain 사이의 translate 하는 것을 배우는 알고리즘 만들자!**

![image](https://user-images.githubusercontent.com/45067667/58673657-4dc39e00-8387-11e9-9c4d-ecc3dc64ca8c.png)



- domain 사이의 relationship을 가정 (같은 scene에 대해 2개의 다른 rendering을 갖고, 그 사이의 relationship인 scene을 찾자!)

  → paired example 형태의 supervision X / level of set에 대한 supervision 제시

  (ex. domain X에 대한 image set, 다른 domain Y에 대한 image set 제공)



- G : X → Y mapping 하는 것을 training 시킴 

  -  $\widehat{y} = G(x), x∈X$ 와 $y∈Y$ 구분하지 못하도록!
  - $y$ 와 $\widehat{y}$ 가 다른 것이라는 것 구분하도록!

  → $\widehat{y}$ 의 distribution이 empirical distribution인 $p_{data}(y)$ 와 match! (이때, $G$ 는 stochastic)

  ⇒ optimal $G$ : domain $X$ → domain $Y$ 로 translate (이때, $\widehat{Y}$은 $Y$와 동일한 분포를 갖음)



* but 문제점 있음

  1. 위의 translate : 개별적인 input x 와 output y가 의미있는 방법으로 pair라고 보장하지 X

     (∵ 같은 dist $\widehat{y}$ 를 만드는 수많은 G mapping 존재 가능)

  2. optimize adversarial objective in isolation 어려움 ~~(이미지 생성할때 제대로 converge 되지 못한 채로 쉽게 optimize 가능함)~~

     : 모든 input img → 같은 output img에 mapping ⇒ 더 이상 optimize X

  ⇒ 이 문제들을 해결하기 위해 translation의 **cycle consistent** 특성 이용 

  (ex. English to French → translate back French to English)

  ⇒ 수학적 해석 : $G:X→Y, F:Y→X$ (G와 F는 서로의 역함수이며, 각각은 일대일대응)



- 위의 cycle consistent를 assumption에 적용하기 위해
  1. mapping $G$ 와 $F$를 동시에 training
  2. cycle consistency loss 추가 ($F(G(x))\approx x, G(F(y))\approx y$ )



### Related Works

-------------------------

- Generative Adversarial Networks (GANs)

  - GAN의 성공 key : adversarial loss! → real photo와 generated image를 구분하지 못하도록!

  ⇒ adversarial loss : image generation task에서 매우 powerful

  - translated image가 target domain에서 나온 image와 구분되지 않기 위해 adversarial loss를 mapping learning에 사용



- Image to Image translation

  ~~추가하자~~~

 

- Unpaired Image-to-Image Translation

  - unpaired setting 다룬 선행 연구 몇개 있음 ($X$와 $Y$ 2개의 data domain 연결)

    1. Bayesian framework 제안 
       - source image로 부터 계산한 patch-based Markov Random Field를 prior로 사용
       - multiple style image로 얻는 likelihood term 사용
    2. CoGAN, cross-modal scene network : domains 사이에 공통적인 representation을 학습하기 위해 weight-sharing strategy 사용

    →  위의 framework expand : 기존 framework + variational autoencoder (VAE) / GAN

    

  - input과 output이 'style' 면에서 달라도, 구체적인 'content' feature 공유

    ▶ adversarial network 사용 (output이 input의 predefined metric space(ex. class label space, image pixel space, image feature space)에 가깝도록)

  ⇒ 우리는 이런 task-specific 에 의존 X (pre-defined similarity function btw input and output에 의존 하지 않음) & input과 output을 같은 low dimensional embedding space에 놓지 X



- Cycle consistency

  - supervise CNN training 에 transitivity 사용하기 위해 cycle consistency loss 사용 

    → $G$ 와 $F$가 서로 일치하도록 비슷한 loss 사용



- Neural Style transfer

  ~~추가하자~~~



### Formulation

---------------------------

**Goal : training sample ${x_i}^N_{i=1}, x_i∈X,  {y_j}^M_{j=1}, y_j∈ Y$ 주어졌을 때, domain $X$ 와  $Y$ 사이의 mapping function 학습** 

cf) 앞으로 $x$ 의 data distribution : $x \sim p_{data}(x)$ / $y$의 data distribution : $y \sim p_{data}(y)$ 라 쓰겠음



![image](https://user-images.githubusercontent.com/45067667/58673670-5fa54100-8387-11e9-974b-ffdcccb665b1.png)



- model : $G:X→Y, F:Y→X$ mapping 2개 사용

- adversarial discriminator : $D_x$ (image {$x$} 와 translated image {$F(y)$} 구분) , $D_y$ ({$y$}와 {$G(x)$} 구분)

  

**Objectives**

- adversarial losses : generated image의 distribution을 target domain의 data distribution으로 match
- cycle consistency losses : mapping $G$와 $F$가 서로 모순되는 것을 막기 위해



#### 1. Adversarial loss

For the mapping function $G : X → Y $ , discriminator $D_y$ 
$$
\mathcal{L}_{GAN}(G, D_Y, X, Y) = \mathbb{E}_{y\sim p_{data}(y)}[logD_Y(y)]+\mathbb{E}_{x\sim p_{data}(x)}[log(1-D_Y(G(x)))]
$$
( $D_Y$ : translated sample $G(x)$ 와 real sample $y$  구분 /  $G$ : domain $Y$의 image 와 비슷하게 image $G(x)$ 생성)

→ $G$ 는 위 식을 minimize 하려, adversary $D$ 는 이 식을 maxmize! 

⇒ $min_Gmax_{D_Y}\mathcal{L}_{GAN}(G, D_Y, X, Y)$



For mapping function $F : Y →X$ , discriminator $D_x$ 
$$
\mathcal{L}_{GAN}(F, D_X, Y, X) = \mathbb{E}_{x\sim p_{data}(x)}[logD_X(x)]+\mathbb{E}_{y\sim p_{data}(y)}[log(1-D_X(F(y)))]
$$
⇒ $min_Fmax_{D_X}\mathcal{L}_{GAN}(F, D_X, Y, X)$ 



#### 2. Cycle Consistency loss

- Adversarial training : identically distributed output을 내는 mapping $G$ 와 $F$를 각각 target domain $Y$와 $X$ 로 학습

  (strictly speaking, $G$와 $F$가 stochastic function 이어야 함) ~~∵ deterministic이면 1개의 분포만 나오니까~~ 

  

- but 충분히 큰 capacity가 있다면, network는 같은 set의 input image들을 target domain 의 random permutation image에 mapping 가능

→ 어떤 학습된 mapping이든 target distribution에 맞는 output distribution을 냄

⇒ adversarial loss 혼자서는 individual input $x_i$ 가 desired output $y_i$ 가 나오도록 mapping 하는 function을 학습시키도록 보장할 수 없음!



- mapping function의 가능한 space를 줄이기 위해, mapping function은 cycle-consistent 해야함

  → domain $X$에서 나온 각 image $x$ : image translation cycle은 $x$를 다시 original image로 되돌려놓아야함 ( ∴ $x → G(x) → F(G(x)) \approx x$ ) : forward cycle consistency

  → domain $Y$에서 나온 각 image $y$ : $y → F(y) → G(F(y)) \approx y $ : backward cycle consistency

⇒ cycle consistency loss를 통해 위 과정 수행
$$
\mathcal{L}_{cyc}(G, F) = \mathbb{E}_{x\sim p_{data}(x)}[||F(G(x)) - x||_1]+\mathbb{E}_{y\sim p_{data}(y)}[||G(F(y))-y||_1]
$$
: $F(G(x))$ 와 $x$ 사이의 L1 norm , $G(F(y))$ 와 $y$ 사이의 L1 norm



![image](https://user-images.githubusercontent.com/45067667/58673683-72b81100-8387-11e9-9cc2-8552b94e857d.png)  

 Fig 4 : cycle consistency loss의 역할 알 수 있으며, $F(G(x))$ 가 input image $x$와 거의 유사한 것 확인 가능



#### 3. Full Objective

$$
\mathcal{L}(G,F,D_X,D_Y) = \mathcal{L}_{GAN}(G, D_Y, X, Y) + \mathcal{L}_{GAN}(F, D_X, Y, X) + \lambda\mathcal{L}_{cyc}(G, F)
$$

$\lambda$ : two objective 사이의 상대적인 중요도 control

→ aim to solve : $G^*, F^* = argmin_{G,F}max_{D_X,D_Y}\mathcal{L}(G, F, D_X, D_Y)$



- model이 2개의 'autoencoder'를 training 하는 것 같이 보일 수 있음

  → 실제로는 1개의 autoencoder 학습 : $F\circ G : X → X$ jointly with another $G \circ F : Y → Y $ 

  

* 이 autoencoder 들은 특별한 internal structure 갖음

  * image를 다른 domain으로 변환하는 중간 매개를 통해 image를 자기자신에게 mapping

    → adversarial autoencoder의 특별한 경우! (임의의 target distribution에 matching 하기 위해 autoencoder의 bottleneck layer 학습을 adversarial loss를 사용함)

  → $X→X$ autoencoder의 target distribution은 domain $Y$ 





### Implementation

-------------------

##### Network Architecture

Generative network : ResNet 구조 사용 (residual connection 존재 → 정보 잃지 않아서 고해상도 처리에 좋음)

- 2개의 stride 2 convolution, residual block, 2개의 fractionally-strided convolution with stride $\frac{1}{2}$
- 6 block for 128 X 128, 9 block for 256 X 256 → instance normalization 사용



Discriminator network : 70X70 PatchGANs사용 (70X70의 overlapping image patch가 진짜인지, 가짜인지 분류하는 것이 목표)

→ patch level의 discriminator : full image 의 경우보다 parameter 적음 & fully convolutional fashion에서 임의의 사이즈에서 작동 가능



##### Training Details

- training 과정을 안정화 하기 위해 2가지 기술 사용

  1.  $\mathcal{L}_{GAN}$ : negative log likelihood를 least square loss로 대체! (∵ 기존 GAN loss 불안정)

     → training이 더 안정화 되고, 더 높은 품질의 결과를 만들어 냄

  ⇒ $\mathcal{L}_{GAN}(G, D, X, Y)$ 

  ​	train $G$ : minimize $\mathbb{E}_{x \sim p_{data}(x)}[(D(G(x))-1)^2]$

  ​	train $D$ : minimize $\mathbb{E}_{y \sim p_{data}(y)}[(D(y)-1)^2]+\mathbb{E}_{x \sim p_{data}(x)}[D(G(x))^2]$ 

> 더 좋은 이유 : 기존 GAN loss - cross entropy 형태 → vanishing gradient 발생 가능 ⇒ Generator까지 유의미한 gradient feedback 전달 X
>
> > 이를 labe 0과 1을 향한 MSE loss 같이 만들면 gradient feedback 잘 전달 (like LSGAN)



 2. model oscillation을 줄이기 위해, update discriminator using history of generated images

    → discriminator update로 최근 generator에서 생성된 image만이 아닌, 생성된 image의 history 사용 (50개이전 까지 저장해서 사용)

    ⇒ 생성된 이미지 저장을 위해 buffer 사용



### Result

-------------------

- unpaired image-to-image translation on paired dataset에 대한 최근 methods들을 사용하여 비교

- study the importance of both adversarial loss and cycle consistency loss



##### Analysis of the loss function

![image](https://user-images.githubusercontent.com/45067667/58673703-8a8f9500-8387-11e9-8d38-1325b54ac666.png)

각 loss를 어떻게 사용했느냐에 따른 결과

- Removing GAN loss or cycle-consistency loss : degrade result
- GAN + forward cycle loss = GAN loss + $\mathbb{E}_{x \sim p_{data}(x)}[||F(G(x))-x||_1]$ 
- GAN + backward cycle loss = GAN loss + $\mathbb{E}_{y \sim p_{data}(y)}[G(F(y))-y||_1]$ 

![image](https://user-images.githubusercontent.com/45067667/58673719-97ac8400-8387-11e9-90c2-570d526c8294.png)

