# XLNet (Generalized Autoregressive Pretraining for Language Understanding)

- Introduction : Unsupervised representation learning은 NLP 분야에서 좋은 성과를 내고 있다. 이 방법은 먼저 방대한 양의 unlabeled된 text corpora로 신경망을 pre-training한 후 downstream task에 맞게 모델이나 representation을 fine-tune하는 것이다. Pre-training 단계에서도 여러 objective들이 이용되어 왔는데, 그 중 가장 대표적인 두 가지를 소개한다.
  - Autoregressive (AR) : 일반적인 Language Model (LM)의 학습 방법으로 이전 token들을 보고 다음 token을 예측하는 방식이다. text sequence가 주어지면 AR 모델은 likelihood를 forward/backward(순방향/역방향)로 각 conditional distribution(조건부 분포)를 모델링하도록 훈련된다.<br>
  이렇게 AR 모델은 uni directional context(단방향 컨텍스트)를 인코딩하도록 훈련되기 때문에 deep bidirectional contexts(심층 양방향 컨텍스트)를 모델링하는 데 효과적이지 않다. 하지만 downstream language understanding task에는 bidirectional context 정보가 필요한 경우가 많기 떄문에 AR 언어 모델링과 효과적인 pre-training 사이의 격차가 발생한다.
  <br><br>
  - Auto Encoding (AE) : AE를 기반으로한 pre-training은 분포를 추정하지 않고 손상된 데이터를 원래 데이터로 reconstruct(재구성)하는 것을 목표로 삼는다. 가장 대표적인 예시는 BERT이다. 입력 토큰 시퀀스가 주어지면 특정한 비율의 토큰을 특정 기호인 [MASK]로 대체하고, 모델은 손상된 버전에서 원래의 토큰을 복구하도록 학습된다. density estimation(밀도 추정)은 목적의 일부분이 아니므로 BERT는 재구성을 위해 bidirectional contexts를 활용할 수 있다.<br>
  하지만 BERT에서 pre-training 시 사용되는 [MASK]와 같은 인공기호은 실제 데이터를 fine-tuning 할 때에는 존재하지 않으므로 'pretrain-finetune discrepancy' 문제가 발생한다. 또한 예측된 토큰들은 입력에서 마스킹이 되기 때문에 BERT에서는 AR 언어모델과 같이 product rule을 사용하여 joint probability(결합확률)을 모델링할수 없다. 
  즉, BERT에서는 unmasked 토큰을 고려할 때 예측된 토큰이 서로 독립적이라 가정하는데 이것은 자연어에서 일반적인 high-order(고차), long-range dependency(장거리 의존성) 특성을 지나치게 단순화하게 된다.
  <br><br>
  