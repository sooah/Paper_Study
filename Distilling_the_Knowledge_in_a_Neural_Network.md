# Distilling the Knowledge in a Neural Network

### Abstract

---------

ML에서 성능 향상 시키는 가장 간단한 방법 : 많은 다른 모델들을 같은 데이터에 대해 학습 → 그들의 prediction average

⇒ 모델들의 전체 앙상블을 사용하여 prediction : 너무 크고 무거움 & 많은 사람들이 사용할 수 있도록 배치하는 데 너무 많은 비용이 들어감 (특히 각 모델들이 너무 큰 뉴럴넷인 경우)

이전 연구"Model compression " : ensemble의 것을 하나의 모델로 compress하는 것이 가능함을 보여줌! → 이를 더 발전시켜 다른 compression 기술을! 

knowledge in ensemble of model을 single model로 distill : MNIST에서 놀라운 결과를 이끌어 냄 & commercial system에 주로 사용되는 acoustic model을 급향상시킴 

하나 혹은 그 이상의 모델로 구성된 앙상블의 새로운 종류를 소개 & 모델들이 혼란스러워하는 fine-grained class 구분하는 새로운 모델 학습

다른 mixture of experts 와는 달리, 이 특화된 모델들은빠르고 평행하게 학습 가능함



### Introduction

--------

large scale ML : 매우 비슷한 모델을을 training 과 deployment stage에서 사용 (but 각각은 매우 다른 요구사항이 있음)  

ex. speech and object recognition : 매우 크고 많은 데이터셋의 structure를 뽑아내기 위해 학습 but 실시간으로 동작할 필요 없음 & 매우 많은 양의 계산 필요

but 다수의 사용자들을 위해 적용될 때 : latency 와 computation resource를 위한 훨씬 더 중요한 요구사항들이 있음 → 데이터로 부터 structure를 뽑아내는 것을 쉽게 하기 위해 매우 큰 모델을 학습!

매우 큰 모델 : 개별적으로 학습된 모델들의 ensemble / dropout 과 같이 강력한 정규화를 통해 학습된 매우 큰 단일 모델

→ 이런 큰 모델이 학습되면, 이를 사용하여 다른 종류의 학습이 가능! : **distillation** 

⇒ 큰 모델의 knowledge를 작은 모델에 전달하기 위해! (deployment에 더 적합함) 

이미 이전 논문이 있음 : 큰 모델들의 앙상블이 작은 단일 모델로 전달되는데 필요한 knowledge를 믿을 수 있게 보여줌! 



이런 매우 유망한 방법에 대해 조사하는 것을 막는 conceptual block :  학습할 모델의 파라미터 값에 대한 지식을 확인하려는 경향 → 모델의 형태를 바꾸면서 같은 knowledge를 갖는 것을 확인하는 것을 어렵게 함!

→ knowledge에 대한 큰 관점 갖는 것(특정한 예시화를 들고자 하는 것에서 좀 벗어나야?) : input vector를 output vector로 mapping 하는 학습하는 것 의미! 

매우 많은 수의 class를 구분하기 위해 학습하는 매우 큰 모델을 위한 일반적인 학습 목적

: correct answer의 log probability의 평균을 최대화! but 이런 학습의 side effect : 모든 부정확한 answer에도 probability 할당 → 확률이 매우 작을 지라도 일부는 다른 것들 보다는 훨씬 더 클 수 있음! 

⇒ *incorrect answer의 상대적인 확률 : cumbersome model이 얼마나 일반화 할 수 있는 경향이 있는 지를 의미!*

ex. BMW 사진이 있을 때 이것을 쓰레기차로 헷갈릴 확률이 매우 적지만 있을 수는 있음! but 당근으로 헷갈릴 확률보다는 높겠지? 



training 시 목적함수 : 실제 사용자의 목적을 최대한 가능한 한 반영하려 함! but 모델은 학습 데이터에서 성능을 최적화 하고, 새로운 데이터에서는 일반화하는 경향이 있음! 

모델이 일반화도 잘 하도록 학습하면 좋겠지만, 일반화를 제대로 하는 방법에 대한 정보가 필요하고, 이 정보는 일반적으로 사용할 수 없는 정보임! (사실상 얻을 수 없는 정보라는 것임)

→ 큰 모델에서 작은 모델로 knowledge를 distilling 할 때 : 큰 모델이 일반화 하는 것과 같은 방식으로 작은 모델을 학습시킴! 

ex. cumbersome model이 일반화를 잘 한다면, 이것이 다른 모델들의 ensemble의 평균이었을 경우, 작은 모델이 같은 방식으로 일반화 한다면 같은 training set을 갖고 일반적인 방식으로 학습시킨 작은 모델모다 test data에서 더 좋은 결과를 일반적으로 낼 것이다! 

> ensemble model의 경우 일단, 여러 모델들의 평균을 내어 test에서 좋은 결과를 낼 수 있게 한 모델임(일반화를 좀 더 잘하는 모델임) → 이 모델에 대해 knowledge distillation ⇒ 불필요하게 많은 parameter가 ensemble에 있지만, 이런 불필요한 parameter들을 분리해내어 generalization 성능을 기존 대비 향상시킬 수 있는 knowledge만 뽑아내겠다!!



cumbersome model의 일반화 정도를 작은 model로 transfer 하는 분명한 방법 : cumbersome model에서 만들어진 class probability들을 **soft target**으로 사용! 

→ 이 transfer stage : 같은 학습 데이터 사용 / transfer set 분리

cumbersome model이 간단한 모델들의 큰 ensemble 이라면 : 각각의 predictive distribution의 arithmetic 혹은 geometric mean을 soft target으로 사용

soft target이 높은 엔트로피를 갖고 있다면 : hard target 보다 각각의 학습데이터에 대해 더 많은 정보 제공 & 학습 데이터 사이 gradient에 더 작은 분산 제공! ⇒ 작은 모델이 원래 cumbersome model 보다 더 적은 학습 데이터 사용해서 학습할 수 있고, 더 높은 learning rate를 사용할 수 있게 해줌!



MNIST의 경우 cumbersome model에서 매우 믿을만한 정확한 정답을 항상 제공 : 학습된 함수에 대한 많은 정보들이 soft target에 매우 작은 확률의 비율로 있음! ex. '2'의 한 사진 : '3'의 경우에 대한 확률이 $10^{-6}$ , '7'의 경우에 대한 확률이 $10^{-9}$ / 다른 '2'의 사진 경우도 아마 비슷할 것! 

⇒ 데이터 전체에 대해 비슷한 구조를 정의할 수 있는 매우 귀중한 정보! ('2'가 각각 '3'이랑 '7'과 얼마나 비슷한지 알려줄 수 있음) but transfer stage의 cross-entropy cost function 에 매우 적은 영향을 줌(거의 0에 가깝기 때문)

→ 이전 논문 : 작은 모델을 학습하기 위해 logit을 사용하는 것으로 해결(final softmax에 들어가는 input들)  ← softmax에 의해 생성된 probability를 target으로 사용하지 않고! 

& cumbersome model에 의해 생성된 logit과 small model에 의해 생성된 logit사이의 squared difference를 최소화 하도록 함! 

⇒ 이 논문의 더 일반적인 solution : **distillation** : cumbersome model이 적합한 target의 soft set을 만들 때 까지 final softmax의 temporature 올림! 

→ small model을 학습시킬 때 이 soft target들을 matching하기 위해 같은 high temperature를 사용한다. 

(나중에 cumber some model의 logit을 matching하는 것이 실제로 distillation의 특별한 경우임을 보여줌)



small model을 학습하기 위해 사용된 transfer set은 전체가 unlabeled data로 구성되어 있거나, original training set을 사용함

→ original training set을 사용하는 것이 더 잘 동작 : small model 이 true target을 cumbersome model에 의해 제공된 soft target matching 만큼 잘 예측하기 위해 objective function에 small term을 더함! 

small model이 soft target 과 correct answer의 방향을 잘못 matching하고 있는 것이 도움이 되는 것으로 밝혀짐

> 이 부분들 이해 잘 안되는 데 아래 블로그 참고!
>
> <https://blog.lunit.io/2018/03/22/distilling-the-knowledge-in-a-neural-network-nips-2014-workshop/>
>
> image recognition task를 풀기 위해 NN 학습 방법 : input data에 대한 softmax output이 label과 최대한 비슷해 지도록 softmax cross-entropy loss를 최소화 하는 방법 사용! → 이렇게 얻은 NN로 부터 얻을 수 있는 softmax output에는 생각보다 많은 knowledge 들 담겨 있음!
>
> 입력 데이터들이 바로 knowledge를 전달하기 위한 매개체 역할을 함! → 원래 모델의 학습에 사용되었던 original training data를 사용하거나 학습에 전혀 사용되지 않았던 external unlabeled data가 될 수도 있음! ⇒ 둘의 차이점 : 전자 - 데이터에 대한 정답이 이미 주어져 있음 / 후자 - 없음!
>
> 논문에서는 전자가 더 잘 나옴! but 그냥 효과가 더 좋았던 것은 아니고, 기존 학습에 사용한 original training data는 정답이 주어져 있기 때문에 학습시에 기존의 정답 (hard label)과 모델이 유추해낸 softmax output(soft label)을 둘 다 사용할 수 있었기 때문! 



### Distillation

------------

NN : softmax output layer 이용하여 class probability 생성! ← logit $z_i$ 를 다른 logit들의 $z_j$와 비교하여 각 class에 해당하는 probability인 $q_i$를 생성! 
$$
q_i = {exp(z_i/T) \over \sum_jexp(z_j/T)}
$$

- $T$ : temperature (normally set to 1)

  → $T$에 더 큰 값을 사용하면 전체적인 class에 대해 더 부드러운 확률 분포를 생성함!



Distillation의 간단한 버전 : transfer set에서 distilled model을 학습 & softmax에서 높은 temperature를 사용한 cumbersome model을 사용하여 생성된 transfer set의 각 case의 soft target distribution 사용 

distilled model 학습 시 같은 높은 temperature 사용 → training 후 (test 말하는 듯?)에는 temperature로 1 사용



transfer set의 일부 혹은 전체에 대해 정확한 label을 알고 있다면, distilled model이 더 정확한 label을 형성할 수 있도록 많은 도움을 줄 수 있음 → 방법 : soft target 수정 시 correct label을 사용! 

→ 더 좋은 방법 있음 : 두 개의 다른 objective function에 weighted average 사용! 

- 첫 번째 objective function : soft target에 대한 cross entropy → cumbersome model에서 soft target 생성할 때 사용한 것 만큼 높은 temperature를 distilled model의 softmax에 사용! 

- 두 번째 objective function : correct label을 사용한 cross entropy → temperature이 1이어도 distilled model의 softmax에서 정확히 같은 logit 생성! 

- 두 번째 objective function에 더 낮은 weight를 주었을 때 더 좋은 결과 얻음

- soft target에 의해 생성된 gradient의 크기를 $1/T^2$만큼 scaling 해주었음 → hard와 soft target 모두를 사용하였을 때 $T^2$를 곱해주어야함! 

  ⇒ hard 와 soft target 사이의 상대적인 기여가 상대적으로 덜 변화 되는 것을 보장해줌!  (만약 실험하는 동안 metaparameter로서 distillation의 temperature이 변화한다해도!)

  

  

##### 2.1 Matching logits is a special case of distillation

 transfer set의 각 경우 : distilled model의 각 logit $z_i$에 관하여 corss-entropy gradient$dC/dz_i$에 기여! 

cumbersome model의 logit $v_i$ : soft target probability $p_i$ 생성 & temperature $T$ 사용하여 transfer training 

→ 그 때의 gradient 
$$
{\partial{C} \over \partial{z_i}} = {1 \over T}(q_i - p_i) = {1 \over T}({e^{z_i/T} \over 
\sum_je^{z_j/T}} - {e^{v_i/T} \ \over \sum_je^{v_j/T}})
$$
temperature이 logit의 크기에 비해 높다면 aprroximate가능
$$
{\partial{C} \over \partial{z_i}} \approx {1 \over T}({1 + z_i/T \over N + \sum_jz_j/T} - {1+v_i/T \over N+\sum_jv_j/T})
$$
각transfer case가 zero-meaned 라 가정 → $\sum_jz_j = \sum_jv_j = 0$ 

위 식 아래와 같이 더 간단하게!
$$
{\partial{C} \over \partial{z_i}} \approx {1\over NT^2}(z_i - v_i)
$$

- high temperature limit : distillation은 $1/2(z_i - v_i)^2$를 minimize하는 것과 같음! (대신, logits이 transfer case에 대해 zero-meaned여야함)

- low temperature : distillation이 logit matching 하는 데 매우 낮은 집중 → 평균보다 더 낮게 나옴! 

  → 더 나을 수도 있음 : 이런 logit들은 cumbersome model 학습시키는데 사용한 cost function에 의해 완전히 제한되어 있지 않음! ⇒ logit들이 매우 nosy 함! 

  ⇒ 이런 negative logit들이 cumbersom model에 의해 얻어지는 knowledge에 대해 사용할만한 정보를 전달 할 수 있음!

- 이런 영향들을 지배하는 것이 무엇이냐는 것은 매우 경험적으로 얻어진 질문!

  →  distilled model이 cumbersome model의 knowledge를 모두 담기에 매우 작은 경우 : intermediate temperature들이 매우 잘 작동함 가장 큰 negative logit을 무시한 경우가 제일 helful하다고! 

> softmax output으로 부터 모델 학습 시 더 많은 knowledge를 전달하기 위해 temperature라는 파라미터 $T$ 추가! 
>
> $T$ 높을 수록 기존보다 더 soft 한 probability distribution 얻을 수 있음!
>
> 만약 일부 클래스에 대한 probability가 0에 가깝 → 학습시에 정보 잘 전달되지 않을 수 있음! ⇒ 더 soft하게 만들어 학습에 잘 반영될 수 있도록! 
>
> $T$가 2-4 사이일 때 distillation이 가장 효과적으로 적용! $T$가 1이 되면 기존 softmax function과 동일!



### Preliminary experiments on MNIST

-----------

distillation 이 얼마나 잘 작동하는지 보기 위해 : 1개의 큰 NN(1200개의 ReLU hidden unit으로 이루어진 2개의 hidden layer로 구성)을 60000개의 training set으로 학습!

dropout 사용해서 정규화함 → dropout : model들의 매ㅐㅐ우 큰 ensemble이 weight를 학습하는 것 같이 보이게 함! 

input : 모든 방향으로 2 pix 까지 jitter됨 (augment 시켰다는 뜻인듯)

⇒ 67개의 test error 발생

smaller net(800개의 ReLU hidden unit으로 이루어진 2개의 hidden layer로 구성 & 정규화 하지 않음) : 146개의 error 발생

→ large net에서 temperature 20으로 생성한 soft target 의 matching에 대한 추가적인 task를 추가하여 정규화 할 경우 : 74개의 error 발생

⇒ **soft target이 distilled model에 매우 많은 양의 knowledge 전달 가능!** (transfer set이 어떤 translation을 포함하지 않더라도, translated training data에서 학습된 generalization 방법에 대한 knowledge 포함!)



distilled net이 2개의 hidden layer 각각에 300개 혹은 그 이상의 unit을 가지는 경우 : 8을 넘는 모든 temperature - 꽤 비슷한 결과 생성! but layer 마다 30개 unit 정도로 낮추는 경우 - 더 낮거나 더 높은 temperature 보다 2.5-4 사이에서 가장 잘 작동!



transfer set에서 digit '3'인 경우를 모두 제외하고 실험 : distilled model의 관점에서 '3'은 한 번도 보지 못한 전설적인 존재! but distilled model은 test set 1010개 중 '3'이 133개 있는 경우에 대해서도 206개의 error 발생

→ 대부분의 error : '3'에 대해 학습된 bias가 매우 낮아 생성

만약 이 bias를 3.5까지 올리면(test set에서 성능에 대해 최적화) : distilled model - 109개의 error 중 14개만 '3'에서 error! 

→ 제대로 된 bias에 의해 : distilled model - '3'을 training 중 한 번도 보지 못했지만, '3'에 대해 98.6%의 정확도를 보여줌! 

만약 training set에 '7'과 '8'만 남겨두고 transfer set 구성 : distilled model은 47.3%의 test error 발생 but '7'과 '8'에 대한 bias를 7.6까지 낮춘다면 - 13.2%까지 test error 감소!



###### Reference

- <https://jamiekang.github.io/2017/05/21/distilling-the-knowledge-in-a-neural-network/>
- <https://blog.lunit.io/2018/03/22/distilling-the-knowledge-in-a-neural-network-nips-2014-workshop/>

