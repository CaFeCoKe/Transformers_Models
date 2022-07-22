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
    - Factorized embedding parameterization : BERT와 이후 후속 연구들에서 WordPiece Embedding의 차원(E)과 Contextual Embedding의 차원(H)이 같은 크기를 가졌다. 모델링 관점에서 WordPiece 임베딩은 context-independent representations(문맥 독립적인 표현)을 학습하는 반면, 은닉층 임베딩은 context-dependent representations(문맥 의존적인 표현)을 학습한다. <br>
    실용적인 관점에서 일반적으로 vocabulary size V가 클수록 좋다. E ≡ H일때, H의 크기를 증가시키면 크기가 V × E인 임베딩 행렬의 크기가 증가하기 때문에 수십억개의 parameter가 있는 모델이 쉽게 생성될 수 있으며, 대부분은 training에서 드물게 업데이트 된다. <br>
    ALBERT는 embedding parameter를 factorization(인수 분해)하여 두 개의 작은 matrices로 분해한다. 이를 통해 O(V × H)였던 것을 O(V × E + E × H)로 바꾸면서 임베딩 파라미터를 줄였다. 이것은 H > E일 때 중요하며,  모든 단어 조각에 대해 동일한 E를 사용한다. 이 방법은 각 단어에 대한 각각 다른 임베딩 크기를 갖는 whole-word embedding에 비해 문서 전체에 훨씬 균등하게 분포되기 때문이다.
    <br><br>
    - Cross-layer parameter sharing : 파라미터 공유 방법은 계층 간 피드 포워드 네트워크(FFN)의 파라미터만 공유하거나 attention의 파라미터만 공유 하는 방법 등이 있는데, ALBERT는 계층 간의 모든 파라미터를 공유하는 것을 사용한다.<br>
      - BERT-large 및 ALBERT-large에 대한 각 레이어의 입력 및 출력 임베딩의 L2 거리와 코사인 유사성(정도의 관점에서)
      ![L2_cosine](https://user-images.githubusercontent.com/86700191/180207183-7e7c9e85-4ba0-4005-a333-589dbf8a7714.PNG) <br>
      BERT는 진동, ALBERT는 수렴된다는 점을 볼 때 계층에서 계층으로의 전환이 BERT보다 ALERT의 경우 훨씬 더 매끄럽다는 것을 볼 수 있다. 이러한 결과는 weight-sharing가 네트워크 파라미터를 안정화하는 데 영향을 미친다는 것을 보여준다.
    <br><br>
    - Inter-sentence coherence loss (문장 간 일관성 loss) : BERT는 두 가지 loss(Masked LM + Next Sentence Prediction)를 이용한다. NSP 목표는 문장 쌍 간의 관계에 대한 추론이 필요한 자연어 추론(NLI)과 같은 다운스트림 작업에서 성능을 향상시키기 위해 설계되었으나 후속 연구에서 NSP의 영향을 신뢰할 수 없다고 판단하고 제거되었다.
    ALBERT 저자들은 NSP의 비효과적인 이유가 주제 예측(topic prediction)이 일관성 예측(coherence prediction)에 비해 쉽고, MLM loss와 겹치기 떄문이라 보았다. <br>
    문장 간 모델링이 언어 이해의 중요한 측면이지만 주제 예측을 피하고 대신 inter-sentence coherence(문장 간 일관성)를 모델링하는 데 중점을 두는 SOP loss(문장 순서 예측(SOP) 손실)을 사용한다. SOP loss는 동일한 문서에서 두 개의 연속 segment를 positive sample로 사용(BERT와 동일), 두 개의 segment의 순서가 바뀐것은 negative sample로 사용한다.
    이를 통해 모델은 담화 수준의 일관성 속성에 대해 더 세분화된 차이를 학습해야 하며 NSP가 풀어내지 못하는 과제와 NSP가 풀어내는 과제를 모두 SOP가 풀어내는 결과를 내면서 multi-sentence encoding tasks(다중 문장 인코딩 작업)에 대한 다운스트림 작업 성능을 지속적으로 향상시켰다.
  <br><br>
  - Model setup :
<br><br>