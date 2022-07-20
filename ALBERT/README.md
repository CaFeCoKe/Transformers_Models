# ALBERT (A Lite BERT for Self-supervised Learning of Language)

- 주요 배경(아이디어) : pre-training → fine-tuning 구조를 가진 이전의 언어 모델들은 모델의 사이즈 증가로 성능을 향상시켜왔다. 이를 적용하기 위해 대형 모델을 사전 학습 후 소형 모델로 distill을 하는 과정을 실행한다.<br>
여기서 사이즈가 큰 모델은 수억, 수십억개의 파라미터를 가진다는 것을 고려하면 하드웨어의 memory limitation에 부딪히게 되고, 통신 오버헤드도 파라미터의 수와 정비례하기 떄문에 분산 훈련에서도 Training speed가 크게 저해 된다는 문제점이 있다.
ALBERT는 이 두 문제점에 대해 접근하여 해결방법을 제시하기 위해 나온 언어 모델이다.
<br><br>
- ALBERT가 제안하는 방법
  - Two parameter reduction : 성능을 크게 손상시키지 않으면서 BERT에 대한 매개 변수 수를 크게 줄여 매개 변수 효율성을 향상시킨다. BERT-large와 유사한 ALBERT 구성은 매개 변수가 18배 더 적고 약 1.7배 더 빠르게 훈련할 수 있다.
    - factorized embedding parameterization (인수분해 임베딩 매개변수화) : 큰 어휘 임베딩 행렬을 두 개의 작은 행렬로 분해하여 은닉층의 크기와 어휘 임베딩의 크기를 분리하게 되면 어휘 임베딩의 파라미터 크기를 크게 늘리지 않고도 hidden size를 쉽게 늘릴 수 있다.
    - cross-layer parameter sharing (교차 계층 파라미터 공유) : 파라미터가 네트워크의 깊이에 따라 증가하는 것을 방지한다.
  <br><br>
  - sentence-order prediction (SOP) : 문장 순서 예측으로 inter-sentence coherence(문장 간의 일관성)에 중점을 두고 있으며. BERT에서 제안된 next sentence prediction (NSP)의 비효과적인 부분을 해결 할 수 있게 설계가 되어있다. 
  SOP 도입으로 BERT-large보다 더 적은 매개 변수를 여전히 가지고 있지만 훨씬 더 나은 성능을 달성하는 훨씬 더 큰 구성으로 확장할 수 있게 되었다.
<br><br>
- Related Work