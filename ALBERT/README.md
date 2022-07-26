# ALBERT (A Lite BERT for Self-supervised Learning of Language)

- 주요 배경(아이디어) : pre-training → fine-tuning 구조를 가진 이전의 언어 모델들은 모델의 사이즈 증가로 성능을 향상시켜왔다. 이를 적용하기 위해 대형 모델을 사전 학습 후 소형 모델로 distill을 하는 과정을 실행한다.<br>
여기서 사이즈가 큰 모델은 수억, 수십억개의 파라미터를 가진다는 것을 고려하면 하드웨어의 memory limitation에 부딪히게 되고, 통신 오버헤드도 파라미터의 수와 정비례하기 떄문에 분산 훈련에서도 Training speed가 크게 저해 된다는 문제점이 있다.
ALBERT는 이 두 문제점에 대해 접근하여 해결방법을 제시하기 위해 나온 언어 모델이다.
<br><br>
- ALBERT가 제안하는 방법
  - Two parameter reduction : 성능을 크게 손상시키지 않으면서 BERT에 대한 파라미터 수를 크게 줄여 파라미터의 효율성을 향상시킨다. BERT-large와 유사한 ALBERT 구성은 파라미터가 18배 더 적고 약 1.7배 더 빠르게 훈련할 수 있다.
    - factorized embedding parameterization (인수분해 임베딩 매개변수화) : 큰 어휘 임베딩 행렬을 두 개의 작은 행렬로 분해하여 은닉층의 크기와 어휘 임베딩의 크기를 분리하게 되면 어휘 임베딩의 파라미터 크기를 크게 늘리지 않고도 hidden size를 쉽게 늘릴 수 있다.
    - cross-layer parameter sharing (교차 계층 파라미터 공유) : 파라미터가 네트워크의 깊이에 따라 증가하는 것을 방지한다.
  <br><br>
  - sentence-order prediction (SOP) : 문장 순서 예측으로 inter-sentence coherence(문장 간의 일관성)에 중점을 두고 있으며. BERT에서 제안된 next sentence prediction (NSP)의 비효과적인 부분을 해결 할 수 있게 설계가 되어있다. 
  SOP 도입으로 BERT-large보다 더 적은 파라미터를 여전히 가지고 있지만 훨씬 더 나은 성능을 달성하는 훨씬 더 큰 구성으로 확장할 수 있게 되었다.
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
  - Model setup : 위와 같은 설계 선택으로 ALBERT의 파라미터는 BERT에 비해 훨씬 적다.<br>
  ![compare](https://user-images.githubusercontent.com/86700191/180598149-4e495615-8bd8-4e9a-8fc2-c203436bf9de.PNG) <br>
  ALBERT-large와 BERT-large를 비교하자면 각각 파라미터의 수가 18M 대 334M로 약 18배 차이가 난다. 그리고 ALBERT-xxlarge의 경우 12-Layer와 24-Layer의 성능이 크게 차이나지 않고, 오히려 Layer가 늘어남에 따른 계산비용이 늘어나는 단점으로 인해 12-Layer의 결과를 채택했다.
<br><br>
- Experimental Results
  - Experimental Setup
    - Pre-train corpora : BookCorpus, Wikipedia (약 16GB)
    - BERT와 동일한 input format : [CLS] x [SEP] y [SEP] 
    - Maximum input length : 512 (10% 확률로 512보다 짧은 input sequence를 random하게 생성)
    - Vocab size : 30,000 (SentencePiece 사용) 
    - N-gram masking을 사용하며 길이를 random 선택 (최대 길이 3)
    - batch size : 4096 
    - step : 125000 (optimizer: Lamb / learning rate: 0.00176)
    - Cloud TPU v3에서 진행하였으며 사용된 TPU의 수는 model size에 따라 64~512
    <br><br>
  - Evaluation Benchmarks
    - Intrinsic Evaluation (내재적 평가) : 훈련 진행 상황을 모니터링하기 위해 SQuAD와 RACE의 development sets를 기반으로 development sets를 만들어 MLM과 SOP 성능을 확인한다. 단, 다운스트림 평가에 영향은 주지 않는다.<br><br>
    - Downstream Evaluation (다운스트림 평가) : General Language Understanding Evaluation(GLUE), Stanford Question Answering Dataset(SQuAD), ReAding Comprehension from Examinations(RACE)를 사용하여 모델을 평가한다. 단, GLUE의 경우 개발세트의 변동이 크기 때문에 5 run 이상의 중앙값을 쓴다. 
  <br><br>
  - Overall Comparison between BERT and ALBERT
    - BERT와 ALBERT의 성능 비교표<br>
    ![compare_model](https://user-images.githubusercontent.com/86700191/180699819-727991ab-45c5-48ec-a89d-6445332cc084.PNG) <br>
    ALBERT-xxLarge는 BERT-Large의 약 70%의 파라미터 수로 몇 가지 대표적인 다운스트림 작업에 대해 BERT-Large 보다 더 좋은 성능을 보여준다.<br>
  또 다른 흥미로운 점은 동일한 training configuration(동일한 수의 TPU)에서 training time의 데이터 처리 속도이다. 통신과 계산이 적기 때문에 ALBERT는 BERT에 비해 데이터 처리량이 더 높다. 같은 Large 모델로 비교했을 때 1.7배 빠른 반면, ALBERT-xxlarge는 더 큰 구조로 인해 3배가량 느리다는 것을 앟 수 있다.
  <br><br>
  - Factorized Embedding Parameterization
    - embedding size 변화에 따른 성능 비교표<br>
    ![embedding_size](https://user-images.githubusercontent.com/86700191/180701493-47b4b6e0-1e94-430c-9d0a-480919f27827.PNG) <br>
    non-shared condition(BERT-style)에서는 embedding size가 클수록 성능이 향상되지만 그다지 크진 않다. all-shared condition(ALBERT-style)에서는 128의 크기가 가장 좋다. 따라서, E = 128로 고정한다.
  <br><br>
  - Cross-layer parameter sharing
    - cross-layer parameter-sharing의 전략 비교<br>
    ![compare_cross_layer](https://user-images.githubusercontent.com/86700191/180703898-9d887fb4-a21e-42e0-bc68-d100521254ab.PNG) <br>
    두가지 embedding size(768, 128)에 대해서 all-shared stategy(ALBERT-style), non-shared strategy(BERT-style) 및 only the attention parameter share, FFN parameter share 4가지 전략에 대해 비교한다. <br>
    all-shared strategy는 두 조건 모두에서 성능을 저하시킨다. 그러나 E=768(Avg에서 -2.5)에 비해 E=128(Avg에서 -1.5)에 대해서는 덜 감소한다. 또한 대부분의 성능 저하는 FFN layer parameter sharing에서 비롯된 것으로 나타났지만 attention parameter를 sharing하면 E=128(Avg에서 +0.1)일 때 하락이 발생하지 않으며 E=768(Avg에서 -0.7)에서 약간의 하락이 발생한다.<br>
    또 다른 전략으로는 L 레이어를 크기 M의 N개의 그룹으로 나누고 각 크기 M 그룹이 파라미터를 공유하는 전략이 있다. 이때, M이 작을수록 성능이 좋아지지만 전체 파라미터의 수가 증가하게 된다. 따라서, all-shared strategy를 채택한다.
  <br><br>
  - Sentence order prediction (SOP)
    - NSP와 SOP가 intrinsic and downstream tasks에 미치는 영향<br>
    ![sop](https://user-images.githubusercontent.com/86700191/180920576-856687f9-e38a-4805-a9b3-bebf72f1971d.PNG) <br>
    ALBERT 기반 구성으로 None(XLNet and RoBERTa), NSP(BERT), SOP(ALBERT)의 세가지 실험을 head-to-head로 비교한다. intrinsic tasks의 결과는 NSP loss가 SOP task에 비해 이점이 없음을 보여준다.(none과 비슷한 52.0%의 정확도) 이를 통해 NSP는 topic shift만 모델링 한다는 결론을 낼 수 있다.<br>
    대조적으로 SOP loss는 NSP task를 잘 해결하고(78.9% 정확도), SOP task도 잘 수행한다.(86.5% 정확도) 더 중요한 점은 SOP loss가 multi-sentence encoding task에 대한 작업성능을 지속적으로 향상시킨다는 점이다.(+1% for SQuAD1.1, +2% for SQuAD2.0, +1.7% for RACE)
  <br><br>
  - What if we train for the same amount of time?
    - 학습 시간 제어의 효과<br>
    ![same_time](https://user-images.githubusercontent.com/86700191/180920580-94387d78-294a-473a-a9b7-0c79ecaa4935.PNG) <br>
    training time이 길수록 성능이 향상되므로 training step을 제어하는 대신 실제 training time을 제어하는 비교를 수행한다. BERT-large(400k step, 34h)과 ALBERT-xxlarge(125k step, 32h)의 training time은 거의 같았고 이를 학습 후 성능을 확인하면 ALBERT-xxlarge가 BERT-large에 비해 average 1.5% 향상, RACE에서는 5.2% 향상되었다.
  <br><br>
  - Additional training data and dropout effects
    - 데이터 추가 및 dropout 제거 효과<br>
    ![add_data](https://user-images.githubusercontent.com/86700191/180929920-875c95c2-ea8a-442e-b054-cd3bd8ca64d7.PNG)
    ![dropout](https://user-images.githubusercontent.com/86700191/180929931-0c68794c-5011-4d19-b25b-674e00ec908f.PNG)
    ![add_data_dropout](https://user-images.githubusercontent.com/86700191/180929929-0013edf4-31c5-4eea-8de8-c16b0c07c9ca.PNG) <br>
    
  <br><br>
  - Current State-of-the-aft on NLU Tasks
