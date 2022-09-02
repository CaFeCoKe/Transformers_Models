# XLNet (Generalized Autoregressive Pretraining for Language Understanding)

- Introduction : Unsupervised representation learning은 NLP 분야에서 좋은 성과를 내고 있다. 이 방법은 먼저 방대한 양의 unlabeled된 text corpora로 신경망을 pre-training한 후 downstream task에 맞게 모델이나 representation을 fine-tune하는 것이다. Pre-training 단계에서도 여러 objective들이 이용되어 왔는데, 그 중 가장 대표적인 두 가지를 소개한다.
  - Autoregressive (AR) : 일반적인 Language Model (LM)의 학습 방법으로 이전 token들을 보고 다음 token을 예측하는 방식이다. text sequence가 주어지면 AR 모델은 likelihood를 forward/backward(순방향/역방향)로 각 conditional distribution(조건부 분포)를 모델링하도록 훈련된다.<br>
  이렇게 AR 모델은 uni directional context(단방향 컨텍스트)를 인코딩하도록 훈련되기 때문에 deep bidirectional contexts(심층 양방향 컨텍스트)를 모델링하는 데 효과적이지 않다. 하지만 downstream language understanding task에는 bidirectional context 정보가 필요한 경우가 많기 떄문에 AR 언어 모델링과 효과적인 pre-training 사이의 격차가 발생한다.
  <br><br>
  - Auto Encoding (AE) : AE를 기반으로한 pre-training은 분포를 추정하지 않고 손상된 데이터를 원래 데이터로 reconstruct(재구성)하는 것을 목표로 삼는다. 가장 대표적인 예시는 BERT이다. 입력 토큰 시퀀스가 주어지면 특정한 비율의 토큰을 특정 기호인 [MASK]로 대체하고, 모델은 손상된 버전에서 원래의 토큰을 복구하도록 학습된다. density estimation(밀도 추정)은 목적의 일부분이 아니므로 BERT는 재구성을 위해 bidirectional contexts를 활용할 수 있다.<br>
  하지만 BERT에서 pre-training 시 사용되는 [MASK]와 같은 인공기호은 실제 데이터를 fine-tuning 할 때에는 존재하지 않으므로 'pretrain-finetune discrepancy' 문제가 발생한다. 또한 예측된 토큰들은 입력에서 마스킹이 되기 때문에 BERT에서는 AR 언어모델과 같이 product rule을 사용하여 joint probability(결합확률)을 모델링할수 없다. 
  즉, BERT에서는 unmasked 토큰을 고려할 때 예측된 토큰이 서로 독립적이라 가정하는데 이것은 자연어에서 일반적인 high-order(고차), long-range dependency(장거리 의존성) 특성을 지나치게 단순화하게 된다.
  <br><br>
  - 기존 language pretraining objectives의 장단점에 따라 AR, AR의 장점을 모두 활용하는 일반화된 AR 언어 모델인 XLNet을 제안함
    - 기존 AR 모델에서처럼 고정된 순방향 또는 역방향 인수분해 순서를 사용하는 대신 가능한 모든 순열의 예상 로그 가능성을 최대화한다.
    - 일반화된 AR 언어 모델로서 XLNet은 데이터 손상에 의존하지 않아 'pretrain-finetune discrepancy'를 겪지 않는다. 또한, product rule을 사용하여 joint probability(결합확률)을 모델링하기 위해 BERT에서 만든 independence assumption(독립성 가정)을 없앤다.
    - Transformer-XL의 segment recurrence mechanism(세그먼트 반복 메커니즘)과 relative encoding scheme(상대 인코딩 체계)을 pre-training에 통합시킨다.
    - 인수분해 순서가 임의적이고 대상이 모호하기 때문에 Transformer(-XL) architecture를 바로 적용하지 못한다. 모호성을 제거하기 위해 Transformer(-XL) network를 다시 reparameterize(매개변수화)시킨다.
<br><br>
- Proposed Method
  - Background : AR language modeling과 BERT의 language model pre-training을 비교한다. AR 모델은 likelihood를 maximizing하는 방향으로 pre-traing을 진행한다. BERT는 Denosing AE가 기반인 모델이다. 두 가지 pre-train의 장단점을 비교한다. 
    - Independence Assumption : BERT는 마스킹된 token이 독립되게 재구성된다는 가정에 기초하여 joint conditional probability를 인수분해 한다. 반면에 AR language modeling은 product rule을 사용하여 인수분해 한다.
    <br><br>
    - Input noise : BERT의 input에는 downstream task에서는 사용하지 않는 [MASK]와 같은 기호가 사용되기 때문에 'pretrain-finetune discrepancy'가 발생한다. 그에 비해 AR 언어 모델링은 input corruption(입력 손상)에만 의존하지 않으며 이 문제로 어려움을 겪지 않는다.
    <br><br>
    - Context dependency : AR representation은 특정 위치의 토큰까지 단방향으로 계산되지만 BERT representation은 bidirectional contextual information에 접근할 수 있기 때문에 BERT가 bidirectional context를 더 잘 capture할 수 있도록 pre-train된다.
  <br><br>
  - Objective: Permutation Language Modeling 
  <br><br>
  - Architecture: Two-Stream Self-Attention for Target-Aware Representations
  <br><br>
  - Incorporating Ideas from Transformer-XL
  <br><br>
  - Modeling Multiple Segments
  <br><br>
  - Discussion
<br><br>
- Experiments
<br><br>
- Conclusions
