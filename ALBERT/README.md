# ALBERT (A Lite BERT for Self-supervised Learning of Language)

- 주요 배경(아이디어) : pre-training → fine-tuning 구조를 가진 이전의 언어 모델들은 모델의 사이즈 증가로 성능을 향상시켜왔다. 이를 적용하기 위해 대형 모델을 사전 학습 후 소형 모델로 distill을 하는 과정을 실행한다.<br>
여기서 사이즈가 큰 모델은 수억, 수십억개의 파라미터를 가진다는 것을 고려하면 하드웨어의 memory limitation에 부딪히게 되고, 통신 오버헤드도 파라미터의 수와 정비례하기 떄문에 분산 훈련에서도 Training speed가 크게 저해 된다는 문제점이 있다.
ALBERT는 이 두 문제점에 대해 접근하여 해결방법을 제시하기 위해 나온 언어 모델이다.
<br><br>
- ALBERT가 제안하는 방법
  - Two parameter reduction : 성능을 크게 손상시키지 않으면서 BERT에 대한 파라미터 수를 크게 줄여 파라미터의 효율성을 향상시킨다. BERT-large와 유사한 ALBERT 구성은 매개 변수가 18배 더 적고 약 1.7배 더 빠르게 훈련할 수 있다.
    - factorized embedding parameterization (인수분해 임베딩 매개변수화) : 큰 어휘 임베딩 행렬을 두 개의 작은 행렬로 분해하여 은닉층의 크기와 어휘 임베딩의 크기를 분리하게 되면 어휘 임베딩의 파라미터 크기를 크게 늘리지 않고도 hidden size를 쉽게 늘릴 수 있다.
    - cross-layer parameter sharing (교차 계층 파라미터 공유) : 파라미터가 네트워크의 깊이에 따라 증가하는 것을 방지한다.
  <br><br>
  - sentence-order prediction (SOP) : 문장 순서 예측으로 inter-sentence coherence(문장 간의 일관성)에 중점을 두고 있으며. BERT에서 제안된 next sentence prediction (NSP)의 비효과적인 부분을 해결 할 수 있게 설계가 되어있다. 
  SOP 도입으로 BERT-large보다 더 적은 매개 변수를 여전히 가지고 있지만 훨씬 더 나은 성능을 달성하는 훨씬 더 큰 구성으로 확장할 수 있게 되었다.
<br><br>
- Related Work
  - Scaling up Representation Learning for Natural Language (자연어를 위한 표현 학습 스케일 업) : 자연어의 학습 표현은 광범위한 NLP 작업에 유용한 것으로 나타났다. 논문 기준 2년동안 가장 큰 변화는 사전 훈련 단어 임베딩에서 전체 네트워크 사전 훈련으로 전환한 후 작업별 미세조정을 한다는 것이다.
  이 작업에서 모델 크기(hidden size, hidden layers, attention heads)가 클수록 성능이 향상된다는 것을 보여준다. 그러나 모델 크기와 계산 비용 문제로 인해 1024의 hidden size에서 멈추게 되었다. <br>
  최신 모델들이 수억, 수십억개의 파라미터를 가진다는 것을 고려하면 GPU/TPU의 메모리 한계로 대형 모델로 실험하기 어렵다는 것을 알 수 있다. 이를 위한 과거의 해결책으로는 gradient checkpointing(메모리 요구 사항을 sublinear로 줄임)이라는 방법과 각 layer의 활성화를 재구성하여 intermediate activation을 저장할 필요가 없도록 하는 방법이 있었다.
  그러나 두 방법 모두 메모리 소비를 줄이는 대신 속도를 희생시킨다. 따라서, parameter-reduction techniques을 통해 메모리 소비를 줄이면서 훈련속도를 높이게 되었다.
  <br><br>
  - Cross-Layer Parameter Sharing (교차 계층 파라미터 공유) : 이전 트랜스포머 구조에서도 연구 되었지만, 사전 교육/미세 조정 설정보다는 표준 인코더/디코더 작업에 대한 학습에 중점을 두었다. Cross-Layer Parameter Sharing이 있는 네트워크는 트랜스포머보다 language modeling 및 subject-verb agreement 에서 더 나은 성능을 보였다.<br>
  트랜스포머 내트워크를 위한 DQE(Deep Equilibrium model)을 제안한 팀에서는 특정 layer의 input embedding과 output embedding이 동일하게 유지되는 equilibrium point(평형점) 도달한다는 것을 보여주었고, ALBERT 논문 저자들의 임베딩은 수렴하기보다 진동하고 있음을 보여주었다. <br>
  Parameter-Sharing 트랜스포머를 표준 트랜스포머와 결합하여 실험한 팀에서는 두 트랜스포머를 결합한 것이 표준 트랜스포머보다 더 높은 성능을 달성한다는 것을 보여주었다.
  <br><br>
  - Sentence Ordering Objectives : 두 개의 text segment의 순서를 예측하여 pre-train loss로 사용한다. 기존 BERT는 쌍의 두 번째 세그먼트가 다른 문서의 세그먼트와 스왑되었는지 여부를 예측(NSP)하는 데 기초하여 손실을 사용한다. 그러나 sentence ordering이 더 어려운 사전 훈련 작업이며 특정 다운스트림 작업에 더 유용하다는 것을 알아내었다.
  또한, 다른 팀에서도 텍스트의 연속된 두 세그먼트의 순서를 예측하려고 했지만, 그들은 둘을 경험적으로 비교하기보다는 세 가지 분류 작업에서 원래의 다음 문장 예측과 결합한다는 점에서 다르다.
<br><br>
- The Elements of ALBERT
  - Model Architecture Choices : ALBERT의 backbone은 GELU 비선형성을 가진 트랜스포머 인코더를 사용한다는 점에서 BERT와 유사하다. BERT 표기법에 따라 어휘 임베딩 크기를 E로, 인코더 레이어의 수를 L로, 은닉층의 크기를 H로 표기한다.  feed-forward/filter의 크기를 4H, attention heads의 수를 H/64로 설정하고 다음 세가지 아이디어를 반영한다.
    - Factorized embedding parameterization
    <br><br>
    - Cross-layer parameter sharing
    <br><br>
    - Inter-sentence coherence loss
  <br><br>
  - Model setup :
<br><br>