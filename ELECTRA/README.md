# ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately) : Pre-training Text Encoders as Discriminators Rather Than Generators

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
  - Small Models : 연구의 목표는 pre-training의 효율성을 향상시키는 것이므로 단일 GPU에서 빠르게 훈련할 수 있는 작은 모델을 개발한다. BERT-Base의 하이퍼파라미터 기준으로 sequence length는 512->128, batch size는 256 -> 128, hidden size는 768 -> 256, token embedding은 768 -> 128으로 줄였다. <br>
  또한 공정한 비교를 위해 동일한 하이퍼파라미터를 사용하여 BERT-Small 모델도 교육한다. BERT-Small을 1.5M step으로 훈련하기 때문에 1M step으로 훈련한 ELECTRA-Small과 동일한 훈련 FLOP를 사용한다.
    - GLUE dev set에 대한 small model 간 비교<br>
    ![model_GLUE](https://user-images.githubusercontent.com/86700191/183630464-b2180a44-5d17-48d5-b90b-e436f54e79eb.PNG) <br>
    ELECTRA-Small은 BERT-Small보다 무려 5 포인트나 높은 성능을 보였고, 심지어는 훨씬 큰 모델인 GPT보다도 좋은 성능을 보였다. 또한, 수렴 속도가 매우 빠른것을 볼 수 있는데 하나의 GPU로 6시간 만에 꽤 괜찮은 성능을 보여준다. Base 크기의 경우에도 ELECTRA-Base는 BERT-Base를 능가할 뿐 아니라 심지어 BERT-Large보다도 더 좋은 성능을 기록했다.
  <br><br>
  - Large Models : replaced token detection task의 효과를 측정하기 위해 큰 ELECTRA 모델을 훈련시킨다. ELECTRA-Large는 BERT-Large와 크기가 같지만 훨씬 더 오래 훈련된다. 특히, 400k step(ELECTRA-400K, RoBERTa의 약 1/4 pre-training compute)와 1.75M step(ELECTRA-1.75M, RoBERTa와 유사한 compute)에 대한 모델을 훈련한다. 
  2048의 batch size과 XLNet pre-training 데이터를 사용했으며, XLNet 데이터가 RoBERTa를 교육하는 데 사용된 데이터와 유사하지만, 비교가 완전히 직접적인 것은 아니라는 점에 주목한다. 비교에 사용할 BERT-Large 모델은 ELECTRA-400K과 동일한 하이퍼파라미터와 훈련시간을 사용했다. 
    - GLUE dev set에 대한 large model 간 비교<br>
    ![GLUE_dev_large](https://user-images.githubusercontent.com/86700191/183820655-38dc1b2d-2d29-467f-9e1b-1a62fb5fa0b9.PNG) <br>
    ELECTRA-400k는 RoBERTa 및 XLNet과 동등하게 작동하나 RoBERTa와 XLNet을 교육하는 데 비해 ELECTRA-400K를 교육하는데 드는 compute는 1/4(FLOPs) 미만이므로 ELECTRA의 샘플 효율성 향상이 대규모로 유지된다는 것을 입증할 수 있다. 더 오래 훈련한 ELECTRA-1.75M의 경우 대부분의 GLUE task에서 가장 높은 점수를 얻었고, pre-training compute를 적게 요구하는 모델이 되었다.
    <br><br>
    - SQuAD에 대한 non-ensemble 모델 간 비교<br>
    ![squad_large](https://user-images.githubusercontent.com/86700191/183827391-b04b38fa-2946-4cb1-a298-59e5585d6530.PNG) <br>
    GLUE 결과와 일관되게, 동일한 계산 리소스가 주어진 경우 ELECTRA는 MLM 기반 방법보다 더 나은 점수를 받는다. ELECTRA-400k를 기준으로 비슷한 compute를 가지는 RoBERTa-100k와 BERT보다 성능이 좋으며, 약 4배의 compute를 가지는 RoBERTa-500k와 비슷한 성능을 가진다. <br>
    ELECTRA는 일반적으로 SQuAD 1.1보다 2.0에서 더 좋은 성능을 가지는데 이는 replaced token detection이 SQuAD의 answerability classification으로 이전될 수 있기 때문이다.
  <br><br>
    - Efficiency Analysis : small subset of tokens에 대한 훈련 목표를 제시하는 것이 MLM을 비효율적으로 만든다고 제안했다. 하지만 적은 수의 masked token만 예측하지만 여전히 많은 수의 input token을 받기 떄문에 명백하지 않다고 판단하여 BERT와 ELECTRA 사이의 “stepping stones(디딤돌)”이 되도록 설계된 일련의 다른 pre-training 목표를 비교한다.
      - ELECTRA 15% : ELECTRA의 구조를 유지하되, discriminator loss를 입력 토큰의 15%만으로 만들도록 설정한다.
      - Replace MLM : Discriminator를 MLM 학습을 하되 [MASK]로 치환하는 게 아니고 generator가 만든 토큰으로 치환힌디.
      - All-Tokens MLM : Replace MLM처럼 하되, 일부(15%) 토큰만 치환하는 게 아니고 모든 토큰을 generator가 생성한 토큰으로 치환한다.
      <br><br>
      - Compute-efficiency 성능 비교<br>
      ![compare_effcoency](https://user-images.githubusercontent.com/86700191/184106469-fedf90b2-73f8-4789-9707-a4caef191933.PNG) <br>
      ELECTRA-15%가 ELECTRA보다 성능이 낮은 것을 보며 ELECTRA가 부분 집합이 아닌 모든 입력 토큰에 대해 정의된 loss를 통해 큰 이익을 얻고 있다는 것을 발견했다. 또한, Replace MLM이 BERT보다 성능이 좋은 것을 보고, [MASK] 토큰의 pre-training과 fine-tuning 간의 불일치 문제로 인해 BERT의 성능이 약간 손상되고 있음을 발견했다.
      하지만 BERT는 이를 위한 트릭이 포함되어 있다는 점을 주목하여 이 휴리스틱한 방법이 문제를 완전히 해결하지 못한다는 것을 알 수 있다. 마지막으로 All-Tokens MLM이 BERT와 ELCTRA 사이의 격차를 좁힌다는 것을 발견했다. 전체적으로 ELECTRA의 향상의 많은 양은 모든 토큰에서 학습한 결과이며, 적은 양은 pre-training과 fine-tuning 간의 불일치를 해결한 점에서 온다고 시사할 수 있다.
      <br><br>
      - 다양한 모델 크기에 대한 BERT와 ELCTRA의 비교 <br>
      ![compare_BERT_ELECTRA](https://user-images.githubusercontent.com/86700191/184106474-0372f18d-d7a6-4bba-af60-8e21a7f0716f.PNG) <br>
      ELECTRA의 이득이 모델이 작아질수록 더 커진다는 것을 발견했다. 또한 소형 모델은 완전히 훈련되면 수렴하게되며, ELECTRA가 BERT보다 높은 다운스트림 정확도를 보여준다. All-Tokens MLM에 비해 ELECTRA의 향상은 단지 ELECTRA의 이득이 더 빠른 훈련에서 나온다는 것에서 비롯된다고 시사할 수 있다.
  <br><br>
- Conclusion : language representation learning을 위한 새로운 self-supervision 태스크인 Replaced Token Detection을 제안했다. 이 제안의 주요 아이디어는 작은 generator가 만들어 낸 질 좋은 negative sample과 input token을 구별하도록 텍스트 인코더를 학습시키는 것이다. Masked language modeling에 비해 pre-training objective는 훨씬 효율적(compute-efficient)이고 downstream tasks에서 더 좋은 성능을 나타내었다. 또한, 상대적으로 적은 compute를 사용하는 경우에도 잘 작동한다는 것을 알 수 있었다.
<br><br>
- 예시 코드에 쓰인 모델 구조<br><br>
![ELECTRA](https://user-images.githubusercontent.com/86700191/185046068-842401b1-e101-48cc-89c9-c46441758089.png)