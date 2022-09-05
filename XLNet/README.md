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
  - Objective: Permutation Language Modeling : AR 모델의 장점을 유지하고 모델이 bi-directional context를 capture할 수 있도록 permutation language modeling objective를 제안한다.
    - permutation language modeling의 수식<br>
    ![permute](https://user-images.githubusercontent.com/86700191/188071252-7d7e9fae-9f35-4b7a-b768-3ac04d4d8761.PNG) <br>
    x : 텍스트 시퀀스, z :  factorization order , Z(T) : 길이가 t인 시퀀스의 모든 가능한 순열조합들, Θ : 모델 파라미터(공유됨)  <br>
    길이가 T인 시퀀스 x에는 T!개의 서로다른 순서가 autoregressive factorization을 위해 존재한다. 직관적으로 만약 모델 파라미터들이 모든 factorization orders에 대하여 공유된다면, 모델은 양방향의 모든 위치에서 정보를 얻을 수 있도록 학습된다.
    <br><br>
    - Remark on Permutation : 제안하는 방식은 sequence 순서가 아닌 인수분해 순서만 바꾼다. 즉 원래의 sequence 순서를 유지하고 원본 sequence에 해당하는 positional encoding을 사용하여 인수분해 순서 permutation에 해당하는 attention mask를 얻는다. <br>
    ![permutation_language_modeling](https://user-images.githubusercontent.com/86700191/188073268-09536c2d-78b9-4372-85ea-965a2f07b519.PNG)
  <br><br>
  - Architecture: Two-Stream Self-Attention for Target-Aware Representations : permutation language modeling은 standard Transformer parameterization에서 naive하게 작동되지 않는다. 이 문제를 확인하기 위해 표준 소프트맥스 공식을 사용하여 차수 분포를 parameterize해본다. <br>
  ![softmax](https://user-images.githubusercontent.com/86700191/188366768-a4aae51a-577f-4061-867c-03cb40babc72.png) <br>
  hΘ(X z<t)은 마스킹 후 공유된 Transformer 네트워크를 통해 생성된  X z<t의 hidden representation으로, 예측할 단어의 위치 zt에 의존하지 않는다. 결과적으로, target 위치에 상관없이 같은 분포가 예측되기 때문에 유용한 representation을 학습할 수 없다. 이런 문제를 피하기 위해서 target 위치를 인식하기 위해 아래와 같이 다음 토큰의 분포를 re-parameterize 하는 것을 제안한다. <br>
  ![reparameterize](https://user-images.githubusercontent.com/86700191/188367043-5ddf154a-4f4b-496f-8995-f9c8d5d64086.PNG) <br>
  gΘ(X z<t, zt) 는 target position인 zt를 추가적으로 입력값으로 받는 새로운 유형의 representation이다.
    - Two-Stream Self-Attention : target-aware representaiton 아이디어는 target 예측에 있어서 모호함을 없애주는 반면, gθ(X z<t, zt) 를 어떻게 계산할 것인가에 대한 문제가 남는다. 목표 위치 zt에 "stand"하고 Attention를 통해 컨텍스트 X z<t에서 정보를 수집하기 위해 위치 zt에 의존할 것을 제안한다. 이 parameterization을 위해  standard Transformer 아키텍처와 모순되는 2가지 요구 조건이 있다. <br>
      - 토큰 x (zt) 와 gΘ(X z<t, zt)을 예측하기 위해서는 zt의 위치만 사용하고, x(zt)의 내용은 사용해서는 안된다.
      - 다른 토큰 x(zj)를 예측할때(j>t), 전체 문맥 정보를 제공하기 위해서는  gθ(X z<t, zt)가 반드시 x (zt)를 인코딩해야한다. <br>

    이러한 문제를 해결하기 위해서 1개가 아닌 2개의 hidden representation을 제안한다.
      - content representation hθ(xz≤t)은 Standard Transformer의 hidden state과 같은 역할로, 문맥과 x(zt) 자신을 인코딩한다.
      - query representation gθ(xz<t , zt)은 X(z<t)와 위치 zt에 대한 문맥 정보에 접근할수 있으며 내용 x(zt)에는 접근 할 수 없다. <br>
    
    각 self attention layer마다 2개의 representation이 공유된 파라미터 셋을 가지고 업데이트된다. <br>
    ![representation](https://user-images.githubusercontent.com/86700191/188372347-b0625644-c017-4962-a4af-8cd7ecf208e4.PNG) <br><br>
    ![two_stream_selfattention](https://user-images.githubusercontent.com/86700191/188372248-72205696-b04b-4fa6-89a0-40d780d2cd5b.png) <br>
    - A detailed illustration of the content stream <br><br>
    ![content](https://user-images.githubusercontent.com/86700191/188372847-80c94e06-ed74-4405-8181-55b59e537b3b.png) 
    <br><br>
    - A detailed illustration of the query stream <br><br>
    ![query](https://user-images.githubusercontent.com/86700191/188372852-98ce7e91-6ff5-4b40-998a-e7678eab09b6.png)
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
