# ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)

- 주요 배경(아이디어) :  Masked Language Modeling(MLM) 사전 훈련 방법은 일부 토큰을 [MASK]로 대체하여 입력을 손상시킨 다음 모델을 훈련하여 원래 토큰을 재구성하는 방식이다. 다운스트림 NLP 작업으로 전송될 때 좋은 결과를 얻지만 일반적으로 효과적이려면 많은 양의 컴퓨팅이 필요하다는 단점이 존재한다. <br>
이러한 단점을 보완하기 위해 replaced token detection라는 보다 sample-efficient인 pre-training 방법을 제안한다. 이 방법은 입력을 마스킹하는 대신 일부 토큰을 소형 발전기 네트워크에서 샘플링된 그럴듯한 대안으로 교체하여 입력을 손상시킨다. 그런 다음 손상된 토큰의 원래 정체를 예측하는 모델을 훈련하는 대신, 우리는 예측되는 discriminative model을 훈련하는 방식이다.
<br><br>
- Introduction : language에 대한 SotA representation learning은 denoising Autoencoder(노이즈 제거 자동인코더)를 학습하는 것으로 볼 수 있다. 입력 시퀀스의 토큰 중 약 15% 정도를 마스킹하고 이를 복원하는 masked language modeling (MLM) 태스크를 통해 학습을 진행 하는데 기존의 autoregressive language modeling 학습에 비해 양방향 정보를 고려한다는 점에서 효과적인 학습을 할 수 있지만 토큰의 15%에서만 학습하기 때문에 상당한 계산 비용이 발생한다는 단점이 존재한다. <br>
대안으로 모델이 실제 입력 토큰을 그럴듯하지만 합성적으로 생성된 대체 토큰과 구별하는 방법을 배우는 사전 교육 작업인 replaced token detection task를 제안한다. masking 대신 일부 토큰을 small masked language model의 output을 통해 샘플로 대체하여 입력을 손상시킨다. 이 손상 절차는 네트워크가 pre-training 중에 인공 [MASK] 토큰을 보지만 다운스트림 작업에서 fine-tuning 될 때는 볼 수 없는 BERT의 mismatch문제를 해결한다. 그런 다음 모든 토큰에 대해 original, replacement를 예측하는 discriminator(판별기)로 pre-training을 한다.<br>
접근방식이 GAN과 유사해보이지만 text에 GAN을 적용하기 어렵기 때문에 maximum-likelihood로 훈련한다는 점에서 제안하는 방식은 “adversarial”이 아니다.
    - 타 모델과의 GLUE score 비교 <br>.
    ![compare_model](https://user-images.githubusercontent.com/86700191/183028796-ce1e2116-6bf9-4a0b-97ac-76b6a0ab0379.PNG) <br>
    ELECTRA-Small의 경우 single gpu에서 4일만에 학습된다. (BERT-Large의 1/20 파라미터 수, 1/135 계산량을 가지고 있다) 또한 BERT-Small과 비교하여 GLUE 성능이 5point 더 높고, 훨씬 더 큰 모델인 GPT 보다도 더 높다. ELECTRA-Large는 RoBERTa나 XLNet보다 더 적은 파라미터와 계산량의 1/4만 사용하였으나 비슷한 성능을 가졌다.
<br><br>
- Method
  - replaced token detection의 개요 <br>
  ![replaced_token_detection](https://user-images.githubusercontent.com/86700191/183081658-c8518414-55c2-43cd-a1b7-bb809d2cfd9b.PNG) <br>
  RTD(replaced token detection) 태스크를 통해 학습하기 위해서 generator G 와 discriminator D , 두 개의 네트워크로 구성되어 있으며 공통적으로 Transformer 인코더 구조를 따른다. <br>
  position t에 대해 generator는 softmax layer로 token x_t를 generation할 확률을 출력한다. <br>
  ![generator](https://user-images.githubusercontent.com/86700191/183089546-81cf4960-c51d-4c7c-b572-c904eecaeefc.PNG) <br><br>
  e는 token embedding. 주어진 position t에 대해 discriminator는 token x_t가 “fake”인지, 즉 data distribution이 아닌 generator에서 나온것인지 sigmoid output layer로 예측한다. <br>
  ![discriminator](https://user-images.githubusercontent.com/86700191/183089954-62e2f25a-cf96-4ddf-99ad-fc7e75d836fd.PNG) <br><br>
  generator는 Masked Language Modeling(MLM)을 수행하도록 훈련한다. 입력 x = [x1, x2, ..., xn]이 주어지면 MLM은 먼저 m = [m1, ..., mk]를 마스크할 set of positions(1과 n 사이의 위치)를 무작위로 선택하여 [MASK] 토큰으로 대체한다. 그런 다음 generator는 mask된 토큰의 original ID를 예측하는 방법을 학습한다. discriminator는 데이터의 토큰을 generator 샘플로 대체된 토큰과 구별하도록 학습한다.<br>
    - 모델의 입력 구조 <br>
    ![input](https://user-images.githubusercontent.com/86700191/183092177-11719d91-24bd-4d38-89b3-3a3b32af0595.PNG) <br><br>
    - 모델의 Loss function <br>
    ![Loss](https://user-images.githubusercontent.com/86700191/183092185-47a40440-cab1-4e69-84ec-31f76c1311c9.PNG) 
  <br><br>
  - GAN(Generative Adversarial Networks)과의 차이점
    - generator가 원래 토큰과 동일한 토큰을 생성했을 때, "fake" 대신 "real"로 간주한다.
    - generator가 discriminator를 속이기 위해 adversarial(적대적)하게 학습하는 게 아니고 maximum likelihood(최대가능도)로 학습한다. (샘플링하는 과정에서 역전파가 불가능)
    - generator의 입력으로 노이즈 벡터를 넣어주지 않는다. 
<br><br>
- Experiments
  - Experimental Setup
    - General Langauage Understanding Evaluation (GLUE) 벤치마크와 Stanford Question Answering (SQuAD) 데이터셋을 사용
    - BERT와 동일하게 Wikipedia와 BooksCorpus를 사용해서 pre-training (Large 모델의 경우에는 XLNet에서 사용한 ClueWeb, CommonCrawl, Gigaword를 사용)
    - 모델의 구조와 대부분의 하이퍼 파라미터를 BERT와 동일하게 설정
    - 10번의 fine-tuning 결과의 중간값(median)을 최종 성능으로 사용
  <br><br>
  - Model Extensions
    - Weight sharing(가중치 공유) : pre-training의 효율성을 증가시키기 위해 generator, discriminator 사이의 가중치를 공유한다. generator, discriminator의 크기가 같으면 Transformer의 가중치가 동일할 수 있다.<br>
    generator, discriminator의 크기가 같을 때 weight tying strategies의 GLUE 점수를 비교하면 no weight tying < tying token embeddings < tying all weights이 된다. 하지만 tying all weights는 generator, discriminator의 크기가 같아야 한다는 점이 상당한 단점이 생긴다. <br>
    Discriminator는 입력으로 들어온 토큰만 학습하는 반면, generator는 출력 레이어에서 softmax를 통해 사전에 있는 모든 토큰에 대해서 밀도 있게 학습하여 MLM이 이러한 표현을 잘 학습하기 떄문에 ELECTRA는 tied token embeddings에서 이득을 얻는다는 가설을 세우고 이후 실험에 대해 진행한다.
    <br><br>
    - Smaller Generators : generator, discriminator의 크기가 동일할 경우 ELECTRA는 MLM보다 훈련할 때 단계당 약 두 배의 compute가 필요해진다. 이것을 줄이기 위해 더 적은 generator를 사용한다. 특히, 하이퍼파라미터를 유지하면서 Layer 크기를 줄여 모델의 크기를 작게 만든다. 
      - generator 와 discriminator 크기에 대한 GLUE 점수<br>
      ![Glue_G_D](https://user-images.githubusercontent.com/86700191/183343142-a7763c05-a0e5-4c73-a845-2ba92232d664.PNG) <br>
      모두 동일한 스텝(500K)만큼을 학습했기 때문에 작은 모델은 똑같은 계산량, 시간만큼 학습하면 더 많은 step을 돌 것이고, 결과적으로 작은 모델 입장에서는 계산량 대비 성능을 손해를 보게 된다. 그럼에도 불구하고 discriminator의 크기 대비 1/4 - 1/2 크기의 generator를 사용했을 때 가장 좋은 성능을 가진다는 것을 실험을 통해 알수 있었다.<br>
      이러한 이유로 generator가 너무 강력하면 discriminator의 태스크가 너무 어려워져 효과적인 학습을 방해한다 추측할 수 있다. 특히, discriminator의 파라미터를 실제 데이터 분포가 아닌 generator를 모델링하는 데 사용할 수도 있다.
    <br><br>
    - Training Algorithms : generator와 discriminator를 jointly(공동으로) 학습시키는 방식으로 채택헸다.
      - Two-stage training : generator만 학습시킨 다음 discriminator를 generator의 학습된 가중치로 초기화, generator의 가중치는 고정하고 학습시키는 방식이다. 단, generator, discriminator의 크기가 같아야 하며, 이를 통해 점점 discriminator를 위한 학습과정이 훈련 내내 나아지게 된다.
      <br><br>
      - Adversarial training : GAN처럼 generator를 reinforcement learning(강화학습)을 통해 adversarial training하는 방식이다.
      <br><br>
      - Training Algorithm 비교<br>
      ![compare_twostage_adversarial](https://user-images.githubusercontent.com/86700191/183397388-3f2e02ae-f4f7-4eb1-83a3-5b4e1644b882.PNG) <br>
      generator와 discriminator를 jointly 학습시키는 방식이 가장 좋다는 것을 알 수 있다. Two-stage 학습에서 discriminative objective로 바꿨을 때, 성능이 쭉 오른 것을 볼 수있으며, Adversarial 학습이 maximum likelihood 기반의 학습보다 성능이 낮다는 것을 볼 수 있다.<br>
      이러한 이유로 Adversarial training에 대한 MLM 성능이 안 좋다는 점이다. MLM 정확도가 58% 밖에 안되는데 이것은 텍스트를 생성하는 큰 작업 공간에서 작업할 때 강화 학습의 샘플 효율성이 떨어졌기 때문이다.
      또 다른 이유로는 학습된 generator가 만드는 분포의 엔트로피가 낮다는 점이다. Softmax 분포는 하나의 토큰에 확률이 쏠려있는데 이렇게 되면 샘플링할 때 다양성이 많이 떨어지게 된다.
  <br><br>
  - Small Models
  <br><br>
  - Large Models
  <br><br>
  - Efficiency Analysis
<br><br>
- Conclusion
<br><br>