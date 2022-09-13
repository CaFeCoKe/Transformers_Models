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
    x : 텍스트 시퀀스, z :  factorization order , Z(T) : 길이가 t인 시퀀스의 모든 가능한 순열조합들, θ : 모델 파라미터(공유됨)  <br>
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
      - 토큰 x (zt) 와 gθ(X z<t, zt)을 예측하기 위해서는 zt의 위치만 사용하고, x(zt)의 내용은 사용해서는 안된다.
      - 다른 토큰 x(zj)를 예측할때(j>t), 전체 문맥 정보를 제공하기 위해서는  gθ(X z<t, zt)가 반드시 x (zt)를 인코딩해야한다. <br><br>

      이러한 문제를 해결하기 위해서 1개가 아닌 2개의 hidden representation을 제안한다.
      - content representation hθ(xz≤t)은 Standard Transformer의 hidden state과 같은 역할로, 문맥과 x(zt) 자신을 인코딩한다.
      - query representation gθ(xz<t , zt)은 X(z<t)와 위치 zt에 대한 문맥 정보에 접근할수 있으며 내용 x(zt)에는 접근 할 수 없다. <br><br>
    
      각 self attention layer마다 2개의 representation이 공유된 파라미터 셋을 가지고 업데이트된다. <br>
      ![representation](https://user-images.githubusercontent.com/86700191/188372347-b0625644-c017-4962-a4af-8cd7ecf208e4.PNG) <br><br>
      ![two_stream_selfattention](https://user-images.githubusercontent.com/86700191/188372248-72205696-b04b-4fa6-89a0-40d780d2cd5b.png) <br><br>
    - Partial Prediction : permutation language modeling은 몇 가지 이점이 있지만 순열로 인해 최적화가 어렵고 수렴이 오래걸린다. 이를 위해 인수분해 순서에서 마지막 몇 개의 토큰의 예측만 이용하는 방법을 사용한다. Z를 non-target subsequence Z≤c and a target subsequence Z>c로 분할한다. c는 cutting point이다.
    objective는 non-target subsequence에서 target subsequence conditioned의 log-likelihood를 maximize하는 것이다. Z>c는 인수분해 순서 Z가 주어진 sequence에서 가장 긴 context를 가지므로 target으로 선택된다. hyperparameter K는 prediction단계에서 1/K개의 token을 선택하기 위해 사용된다. unselected token들은 query representation이 계산되지 않기 때문에 메모리를 아끼고 속도를 향상시킬 수 있다.<br>
    ![Partial Prediction](https://user-images.githubusercontent.com/86700191/188594117-ec5f3326-0bb9-4e0b-a67a-e1dd7a8ab2b5.PNG)
  <br><br>
  - Incorporating Ideas from Transformer-XL : Transformer-XL의 segment recurrence mechanism(세그먼트 반복 메커니즘)과 relative positional encoding scheme(상대 인코딩 체계)을 pre-training에 통합시킨다. 
  제안된 permutation setting에 recurrence mechanism을 통합하고 previous segment의 hidden state를 다시 사용할 수 있는 방법을 설명한다. 긴 sequence에서 두개의 segment를 입력으로 받는다고 가정한다.<br>
  ![1](https://user-images.githubusercontent.com/86700191/188584205-c01ffe9c-7020-4f9e-8d92-6236e788b9fd.PNG) <br>
  z˜ 와 z를 [1 · · · T] 와 [T + 1 · · · 2T]의 permutation이라 가정 후 permutation z˜ 를 통해 첫 번째 segment를 계산하고 각 layer m에 대해 얻어진 content representation h˜(m)을 cache한다. 다음 segment x에서 memory를 포함하는 attention update는 다음과 같다.<br>
  ![2](https://user-images.githubusercontent.com/86700191/188584212-0b6e3577-4b3c-440e-a411-25e02fa6c838.PNG) <br> ([.,.]은 sequence dimension사이에 concatenation 의미) <br>
   positional encoding은 original sequence의 실제 position에만 의존하기 때문에 h˜(m)이 계산되면 z˜ 와는 독립적이게 된다. 이를 이용하여 이전 segment의 인수분해 순서에 대한 정보 없이 memory를 caching하여 다시 사용할 수 있다. 모델은 마지막 segmentt의 모든 인수분해 순서에 대해 메모리를 활용하는 방법을 train한다고 추측된다.(아래 그림 참조)
  그리고 query stream도 동일한 방법으로 계산할 수 있다.
  <br><br>
    - A detailed illustration of the content stream <br><br>
    ![content](https://user-images.githubusercontent.com/86700191/188372847-80c94e06-ed74-4405-8181-55b59e537b3b.png) 
    <br><br>
    - A detailed illustration of the query stream <br><br>
    ![query](https://user-images.githubusercontent.com/86700191/188372852-98ce7e91-6ff5-4b40-998a-e7678eab09b6.png)
    <br><br>
  - Modeling Multiple Segments : 많은 downstream task들은 multiple input segment를 포함하고 있다. 예를 들면 QA같은 경우 question과 context paragraph로 구성되어있다. BERT와 마찬가지로 랜덤하게 선택된 두개의 segment를 concatenation(연결)하여 하나의 sequence로 permutation language modeling을 진행한다. 
  모델에 대한 입력은 BERT와 동일하게 [CLS, A, SEP, B, SEP]로 되어있다. two-segment data format을 따르지만 ablation study에서 일관적인 향상을 하지 못했기 때문에 next sentence prediction을 수행하지 않는다.
    - Relative Segment Encodings : BERT의 경우 word embedding에 각 position에 따라 absolute segment embedding을 적용하였지만  XLNet은 transformer-XL에서 제안한 relative segment encoding을 확장하여 사용한다.
    sequence에서 i와 j인 position pair이 주어지고 동일한 segment에서 온 경우 Sij = S+ 또는 Sij = S-를 인코딩하는 segment를 사용한다. S+와 S-는 각 attention head에 대해 learnable(학습 가능한) 모델 파라미터이다. position pair가 어떤 특정 segment로부터 왔는지 반대로 두 위치가 동일한 segment 내에 있는지 여부만 고려한다. 즉, 위치 간의 관계를 모델링하는 것이다.<br>
    relative segment encoding의 이점은 두가지가 있다. 첫번째는 relative encoding의 inductive bias(귀납적 편향)는 일반화를 향상시킨다. 두번째는 absolute segment encoding을 사용하여 불가능한 두 개 이상의 segment가 있는 task에 대한 fine-tune 가능성을 열어준다는 것이다.
  <br><br>
  - Discussion : BERT와 XLNet은 sequence token의 subset만을 예측한다는 것을 알 수 있었다. 모든 토큰이 mask처리되면, BERT는 의미있는 예측을 할 수 없기 때문에 필수적인 선택이다. 또한 BERT와 XLNet에서 partial prediction은 충분한 context가 있는 token만 예측하여 최적화가 어려워진다. 
    - BERT와 XLNet의 차이점
    ![BERTvsXLNet](https://user-images.githubusercontent.com/86700191/188823106-7c71febc-ed1f-426d-b0b0-05859e207fa9.PNG) <br>
    [New,York,is,a,city]에 대해 비교한다. 예시에서 [New,York] 두 개의 token을 예측하고 log p(New York | is a city)를 maximize한다고 가정한다. 또한, XLNet의 인수분해 순서는 [is,a,city,New,York]이라고 가정한다. <br>
    XLNet은 쌍(New, York) 간의 dependency를 capture 할 수 있으며, BERT는 이를 생략하게 된다. BERT는 (New, city) 및 (York, city)와 같은 일부 dependency pair를 학습하지만, XLNet은 항상 동일한 대상이 주어지면 더 많은 dependency pair을 학습하고 "denser(더 밀집)"한 효과적인 훈련을 한다는 것은 분명하다.
<br><br>
- Experiments
  - Pre-training and Implementation : Pre-training을 위해서 XLNet도 BERT를 따라서 합이 16GB 정도 되는 BooksCorpus와 English Wikipedia를 사용하며, 추가적으로 15GB의 Giga5, 너무 짧거나 질이 떨어지는 문장을 휴리스틱하게 필터링하는 전처리 작업을 한 19GB의 Clue Web2012-B와 78GB의 Common Crawl dataset도 사용했다. 
  Google의 SentencePiece tokenizer를 사용하였고 위의 5개의 dataset 각각 순서대로 2.78B, 1.09B, 4.75B, 4.30B, 19.97B 개의 token을 얻을 수 있었고, 따라서 총 32.89B의 token으로 pre-training을 진행했다. <br>
  XLNet-Large는 512 TPU v3 환경에서 5.5일 동안 약 500K step으로 학습되었다. Batch size는 8192이었으며, Linear learning rate decay를 적용한 Adam optimizer를 사용했다. recurrence mechanism이 도입되었기 때문에 순방향과 역방향 각각이 배치 크기의 절반을 차지하는 bidirectional 데이터 입력 pipeline을 사용한다.
  <br><br>
  - Fair Comparison with BERT
  ![compare_BERT](https://user-images.githubusercontent.com/86700191/189803843-4f245b44-44b3-43c2-9132-0ca3f68901ad.PNG) <br><br>
  가장 좋은 성능을 보인 3가지의 BERT 변형과 동일한 데이터 및 하이퍼파라미터로 훈련된 XLNet을 비교한다. XLNet은 모든 데이터 세트에서 BERT의 성능을 능가한다는 것을 볼 수 있다.
  <br><br>
  - Comparison with RoBERTa: Scaling Up
    - RACE
    ![compare_RoBERTa_RACE](https://user-images.githubusercontent.com/86700191/189803846-46a7816e-b478-49ea-9c86-cb6f7cf79389.PNG) <br><br>
    - SQuAD
    ![compare_RoBERTa_SQuAD](https://user-images.githubusercontent.com/86700191/189803848-12e6641c-7964-405a-90eb-20eb8a5b9d03.PNG) <br><br>
    - GLUE
    ![compare_GLUE](https://user-images.githubusercontent.com/86700191/189803850-f24f0c2b-746c-474c-b492-98c3fd6fab20.PNG) <br><br>
    - Error rate (text classification tasks)
    ![compare_textclassification](https://user-images.githubusercontent.com/86700191/189805379-f48ef277-a5a3-498f-b498-61f0273548ed.PNG) <br><br>
  
    RoBERTa와 비교적 공정한 비교를 위해 전체 데이터를 기반으로 하며 RoBERTa의 하이퍼파라미터를 재사용한다. 더 긴 context를 포함하는 SQuAD 및 RACE와 같은 explicit reasoning task(명시적 추론 작업)의 경우, XLNet의 성능 이득은 일반적으로 더 크다. 이 우월성은 XLNet의 Transformer-XL backcone에서 비롯된다 볼 수 있다.
    또한, MNLI(>390K), Yelp(>560K), Amazon(>3M)과 같이 이미 풍부한 감독 예제를 가지고 있는 분류 작업의 경우 XLNet은 여전히 상당한 이득으로 이어진다.
  <br><br>
  - Ablation Study
  ![ablation](https://user-images.githubusercontent.com/86700191/189807920-70c952d6-ad95-447d-9000-be782e8e6fc6.PNG) <br><br>
  설계 선택의 중요성을 이해하기 위해 크게 다음의 3가지 측면에서 ablation study를 진행한다. 
    - permutation language modeling objective의 효과
    - Transformer-XL backbone과 segment-level recurrence (i.e. using memory)의 중요성
    - span-based prediction과 bidirectional input pipeline, 그리고 next sentence prediction의 필요성
    <br><br>
    
    XLNet과 비교하는 모델로는 Original BERT-Base (row 1)과 BERT에서 쓰는 Denoising auto-encoding (DAE) objective로 학습하고 bidirectional input pipeline이 적용된 Transformer-XL (row 2)을 선정했다. 모든 모델은 BERT-Base의 hyperparameter와 동일하게 맞춘 12 layer의 구조를 갖고 있으며, BooksCorpus와 Wikipedia로 pre-training하였다. 위의 모든 결과는 5번의 결과의 중간값(median)이다.<br>
    rows 1~4에서 Transformer-XL와 permutation language modeling objective의 우수성을 XLNet-Base 모델이 BERT보다 좋은 성능을 보인 것으로 증명된다. 또한, row 5에서 memory caching mechanism을 제거하게 되면 가장 긴 context를 처리하는 RACE의 경우 성능이 확실히 떨어지는 것으로 memory caching을 사용해야 성능이 좋다는 것을 알 수있다. 
    rows 6~7에서 span-based prediction과 bidirectional input pipeline이 XLNet에서 중요한 역할을 한다는 것을 보여준다. 마지막으로, row 8에서는 원래의 BERT에서 제안된 next sentence prediction이 성능의 향상으로 이어지는 것은 아니라는 것을 발견하여 XLNet은 next sentence prediction을 사용하지 않는다.
<br><br>
- Conclusions : XLNet은 AR과 AE 방법의 장점을 결합하기 위해 permutation language modeling(순열 언어 모델링) objective를 사용하는 일반화된 AR pretraining 방법이다. XLNet의 신경 아키텍처는 Transformer-XL와 two-stream attention mechanism의 통합을 포함하여 AR objective와 함께 원활하게 작동하도록 개발되었다. XLNet은 다양한 task에 대한 이전의 pretraining objective에 비해 상당한 개선을 보여준다.
