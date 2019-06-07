# StarGAN : Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation

------------------------------

**[StarGAN] : 여러개의 domain에서 Image to Image translation by one generator** 

(multi-domain image translation across different dataset)

----------

### Introduction

-----------------

* Image to image translation : 주어진 image의 particular aspect를 다른 aspect로 change (ex. 사람의 표정을 웃는 → 화난) → GAN을 통해 발전

* Training data는 2개의 다른 domain에서 나왔음 → model이 image를 하나의 domain에서 다른 domain으로 translate하는 것 학습
	* Denote
		1. Attribute : image에서 나온 의미있는 feature (ex. hair color, gender, age 등) 
		2. Attribute value : attribute의 특정 value (ex. 흑발/금발, 여성/남성 등)
		3. Domain : 같은 attribute value를 공유하는 image set
	
* Image dataset은 많은 labeled attribute 갖음 

  ex. CelebA) hair color, gender, age에 따라 40개의 label

  ​		RaFD) 표정에 다라 8개의 label

⇒ multi-domain image-to-image translation (여러 domain에서 나온 특성에 따라 image change)

![image](https://user-images.githubusercontent.com/45067667/58555020-6d59aa00-8253-11e9-97d4-ed09a1757f57.png)

i) CelebA img 전용 → 4개의 domain에 따라 translate (blond hair, gender, age, pale skin)

ii) CelebA와 RaFD 사용→ 다른 dataset에서 나온 multiple domain으로 training ⇒ RaFD에서 배운 feature를 사용하여 CelebA의 facial expression change



- 현존하는 model들은 multi-domain img translation하기에 inefficient & ineffective

  ![image](https://user-images.githubusercontent.com/45067667/58555732-35ebfd00-8255-11e9-904b-74eae5b5366a.png)

  Fig2 : 4개의 다른 domain 사이에서 image translate 하기 위해 12개의 다른 generator network training 하는 것 설명

  - inefficiency : K domain 사이에서 learning 할 경우 k(k-1) generator가 training되어야 함
  - ineffective : 모든 domain의 image에 learning 할 수 있는 global feature 존재 but 각 generator가 전체 training data 사용하지 않고, k개 중의 2개의 domain에서 나온 training data만 사용해서 learning 

  ⇒ training data의 전체적인 사용 X → generate image의 quality 측면에서 한계 발생



- 다른 dataset에서 나온 domain을 jointly training할 능력 X (∵ 각 dataset이 partially labeled)



⇒ 위 문제들 해결 위해 아래 제시

1. propose 'StarGAN' : multiple domain 사이에서의 mapping 학습 방법으로 novel하고 scalable한 방법! (Fig2.(b) : **multiple domain의 training data 사용 & 단 1개의 generator 사용하여 모든 가능한 domain 사이의 mapping 학습**)

   - fixed translation (ex. black to blond hair) 대신, input으로 img와 domain information 사용 → img를 대응하는 domain으로 flexible하게 translate 하는 것 learning

   - domain information 나타내기 위해 label 사용 (ex. binary / one-hot vector)

   - training 중 : target domain level random하게 생성 → model이 input img를 target domain으로 flexible하게 translate하게 train

     ⇒ domain label control & testing phase에서 원하는 domain으로 image translate 가능

2. Simple but effective approach : 다른 dataset의 domain 사이에서 jointly training 가능 by **"adding a mask vector to the domain label"**

   → model이 unknown label 무시, 특정 dataset에서 나온 label에 집중

   ⇒ RaFD에서 나온 feature 사용해서 CelebA image의 facial expression 합성에서 잘 작동!

⇒ **multi-domain image translation across different dataset 처음으로 성공**



< contribution>

- propose StarGAN : 오직 1개의 generator와 discriminator 사용 → multiple domain 사이의 mapping 학습(image의 모든 domain을 effective하게 training)
- multiple dataset 사이의 multi-domain image translation 학습 성공적! by mask vector method → StarGAN이 모든 domain label control 가능
- facial attribute transfer & facial expression 생성에 qualitative & quantitative 한 결과 생성



### Related Works

-------------------

##### Generative Adversarial Networks

- discriminator 와 generator로 구성
- discriminator : real 과 fake sample 구분하도록 학습
- generator : real sample 과 구분되지 않는 fake sample 학습

⇒ generated image가 가능한 realistic 하기 위해 adversarial loss 극대화



##### Conditional GANs

- conditional image generation 기반 GAN

- discriminator 와 generator 에게 conditioned on the class sample generate 하기 위해 class information 제공 

  ex. 주어진 text description 과 관련된 image generate

⇒ conditioning domain information 제공 → 다양한 target domain으로 image translation 할 수 있는 scalable GAN 제안 



##### Image-to-Image Translation

- image-to-image translation에 대해 많은 연구 있음 → but 선행 연구들은 한 번에 2개의 다른 domain 사이에서의 관계만 학습 가능함

  → multiple domain을 다루기에는 확장성의 측면에서 제한적임 (∵ 각 domain의 쌍씩 training해야)

⇒ 우리의 framework는 1개의 model을 사용해서 multiple domain 사이의 relation 학습 가능



### Star Generative Adversarial Networks

---------------

먼저 framework 설명 → 다른 label set을 갖는 multiple dataset 통합하여 image translation flexible하게 수행 방법

#### Multi-Domain Image-to-Image Translation

**Goal : multiple domain 사에에서 mapping 학습을 하는 1개의 generator G training!**

→ training $G$  : target domain label c로 conditioned 하여 input img $x$ → output img $y$ translate 

⇒ $G(x,c) → y$ 

-  target domain label c : randomly generate 

  → input image flexible하게 translate 하도록 $G$  training 가능

- auxiliary classifier 사용 : single discriminator가 multiple domain control 가능하게 함⇒

⇒ discriminator : source 와 domain label에 대한 probability distribution 생성 $D: x → {D_{src}(x), D_{cls}(x)}$  

![image](https://user-images.githubusercontent.com/45067667/58558360-8d8d6700-825b-11e9-9d4f-e65ad6800975.png)



##### Adversarial loss

- adversarial loss : $\mathcal{L}_{adv} = \mathbb{E}_x[logD_{src}(x)] + \mathbb{E}_{x,c}[log(1-D_{src}(G(x,c)))]$ 

  - $G$ : input image $x$ 와 target domain label $c $ 에 의 해 conditioned 된 image $G(x,c)$ 생성
  - $D$ : real 과 fake 구분하려 함

  cf) 이 논문에서, $D_{src}(x)$ : $D$에 의해 제공된 source에 대한 probability distribution

  ​	→ generator $G$ :  $\mathcal{L}_{adv}$ minimize 하려함

  ​	→ discriminator $D$ : $\mathcal{L}_{adv}$ maximize 하려함



##### Domain Classification loss

**Goal : 주어진 input image $x$와 target domain label $c$ 에 대해, target domain $c$를 제대로 구분하여 $x$를 output image $y$로 translate**

- $D$의 top에 auxiliary classifier 추가 & $D$와 $G$를 optimize하는 domain classification loss 부과 

  → objective를 2개의 관점으로 나눔

  1. real image 에 대한 domain classification loss → optimize $D$
     $$
     \mathcal{L}_{cls}^r = \mathbb{E}_{x,c'}[-logD_{cls}(c'|x)]
     $$

     -  $D_{cls}(c'|x) $ : $D$ 에 의해 계산된 domain label에 대한 probability distribution)
     - minimize $\mathcal{L}_{cls}^r$ : $D$가 real image $x$를 original domain $c'$ 로 분류하도록 학습 (training data로 input image와 domain label pair $(x,c')$ 주어짐)

  2. fake image 에 대한 domain classification loss → optimize $G$ 
     $$
     \mathcal{L}_{cls}^f = \mathbb{E}_{x,c}[-logD_{cls}(c|G(x,c))]
     $$

     - minimize $\mathcal{L}_{cls}^f$ : target domain $c$로 분류되는 generate image를 $G$가 생성하도록 학습



##### Reconstruction loss

- adversarial & classification loss minimize

  → $G$ : image가 realistic & 정확한 target domain으로 분류되도록 training

- $\mathcal{L}_{adv}$ & $\mathcal{L}_{cls}^f$ minimize → translated image가 input의 관련된 부분의 domain만 바꾸면서 input image의 내용 보존한다 보장 X

⇒ 문제 완화 위해 **cycle-consistency loss를 generator에 적용** 
$$
\mathcal{L}_{rec} = \mathbb{E}_{x,c,c'}[\parallel x-G(G(x,c),c')\parallel]
$$

	- $G$ 가 translated image $G(x,c)$와 original domain label $c'$ 을 input으로 받음 → original image $x$ 다시 reconstruct 하도록!	
	- reconstruction loss로 $L_1 norm$ 사용



**NOTE : 1개의 generator를 2번 사용** 

1. original image를 target domain의 image로 translate
2. translated image를 original image로 reconstruct



##### Full objective

- optimize $G,D$ 

  $\mathcal{L}_D = -\mathcal{L}_{adv} + \lambda \mathcal{L}_{cls}^r$ 

  $\mathcal{L}_G = \mathcal{L}_{adv} + \lambda_{cls} \mathcal{L}^f_{cls} + \lambda_{rec} \mathcal{L}_{rec}$  

  - $\lambda_{cls}$ : domain classification loss의 adversarial loss에 대한 상대적인 중요도를 control 하는 hyperparameter

  - $\lambda_{rec}$ : reconstruction loss의 adversarial loss에 대한 상대적인 중요도를 control 하는 hyperparameter

    (이 논문의 모든 연구는 $\lambda_{cls} = 1$,  $\lambda_{rec} = 10$ 사용)



#### Training with multiple datasets

- StarGAN의 중요한 장점 : 다른 종류의 label을 갖는 multiple dataset을 동시에 통합가능

  → test phase에서 모든 label control 가능

→ but, multiple dataset learning의 문제 : label information은 각 dataset에 대해 partially known함

​	ex. CelebA : hair color, gender 등의 특징에 대한 label O / facial expression에 대한 label X

​			RaFD : vice versa

⇒ label vector $c'$의 complete information : translated image $G(x,c)$로 부터 input image $x$를 reconstruct할 떄 필요

→ 문제..



##### Mask vector

- 문제 완화 위해 **mask vector m** 제안!

  : unspecified label 무시 & 특정 dataset에서 제공된 명시적으로 알려진 label에 집중

  - n-dimensional one-hot vector 사용 (n : # dataset)

  - label의 unified version 을 vector로 정의 

    $\tilde{c} = [c_1, c_2, …, c_n, m]$ : concatenate, $c_i$ : $ith$ dataset의 label

  -  known label $c_i$의 vector : binary attribute에 대한 binary vector / categorical attribute에 대한 one-hot vector
  - 1개의 domain에 대해 학습할 경우, 나머지 n-1개의 domain의 label에 대해서는 모두 0을 주어 학습 진행



##### Training Strategy

- multiple dataset training → generator에 domain label $\tilde{c}$ 를 input으로 사용

  ⇒ generator가 unspecified label (zero vector) 무시 & explicitly give label에 집중

- generator의 structure
  - single dataset training하는 것과 같음
  - input으로 label $\tilde{c}$ 사용하는 차이만
- discriminator에 auxiliary classifier 추가
  - 모든 dataset의 label에 대한 probability distribution 생성 위해

⇒ multi-task learning setting에서 model training (discriminator : known label에 대한 classification error만 minimize 하도록)

​	ex. CelebA의 image에 대해 training 

   - discriminator : CelebA의 attribute의 label에 대한 classification error만 minimize 

     ​							RaFD에 대한 facial expression 에 대해서는 하지 않음

     → 이를 위해 

     discriminator : CelebA와 RaFD 번갈아가며 학습 ⇒ 두 dataset에 대한 discriminative feature 모두 학습

     generator : 두 dataset의 모든 label 다루도록 학습

     

### Implementation

----------------

#####  Improved GAN training

- training process 안정화 & higher quality image 생성 필요

  → $\mathcal{L}_{adv}$ 를 Wasserstein GAN objective with gradient penalty로 대체
  $$
  \mathcal{L}_{adv} = \mathbb{E}_x[D_{src}(x)]-\mathbb{E}_{x,c}[D_{src}(G(x,c))]-\lambda_{gp}\mathbb{E}_x[(\parallel\bigtriangledown_{\hat{x}}D_{src}(\hat{x})\parallel_2-1)^2]
  $$

   - $\hat{x}$ : real과 generated image 쌍 사이의 직선을 따라 균일하게 sampling 

     (이 논문에선 $\lambda_{gp} = 10$ 사용)



##### Network Architecture

- Generator network 

![image](https://user-images.githubusercontent.com/45067667/58621195-cc76f780-8303-11e9-878c-c17ac35dacb3.png)

​	- down sampling : 2개의 convolutional layer with stride size 2	

​	- up sampling : 2개의 transposed convolutional layer with stride size 2

​	- instance normalization for generator



- Discriminator network

![image](https://user-images.githubusercontent.com/45067667/58621467-6b9bef00-8304-11e9-83e6-83b8e7f2df44.png)

​	- Patch GANs 사용 (local image patch가 real인지 fake 인지 구분)	

​	- Normalization 사용 X



### Experiment

-------------

StarGAN과 최근 방법들 비교

facial expression synthesis에서 classification 실험 수행

##### Baseline Models

- DIAT, CycleGAN 채택 (2개의 다른 domain 사이에서 image-to-image translation 수행)

  → 비교 위해 2개의 다른 domain에 대한 모든 쌍을 multiple time training

- IcGAN 채택 (cGAN을 사용하여 attribute transfer 수행)

  

**DIAT**

- adversarial loss 사용 : $x∈X → y∈Y$ mapping learning

  $x, y$ : 다른 domain $X$와 $Y$에서 나온 face image

- mapping에 regularization term 갖음 : $\parallel x-F(G(X))\parallel_1$ 

  → source image의 identity feature preserve 위해!

  $F$ : face recognition task에서 미리 trained 된 feature extractor

  

##### CycleGAN

- adversarial loss 사용 : 2개의 다른 domain X 와 Y 사이의 mapping learning

- cycle consistency loss로 mapping regularize : $\parallel x-(G_{YX}(G_{XY}(x)))\parallel_1$ and $\parallel y-(G_{XY}(G_{YX}(y)))\parallel_1$ 

  → 2개의 다른 domain의 각 쌍에 대해 2개의 generator와 discriminator 요구



##### IcGAN

- cGAN model에 encoder 통합

- cGAN : $G{z,c} → x$ mapping 학습

  → latent vector $z$ 와 conditional vector $c$에 conditioned 된 image $x$ 생성

- encoder 소개 : cGAN의 inverse mapping 학습

  → $E_z : x → z$ and $E_c : x → c $ 

  ⇒ conditional vector $c$ 만 바꾸고, latent vector $z$ 보존한 image 생성 가능! 

