# BART (Bidirectional Auto-Regressive Transformer)

- Abstract & Introduction : Self-supervised methods은 다양한 NLP tasks에서 놀라운 성공을 거두었다. 가장 성공적인 접근 방식은 단어의 무작위 subset(하위 집합)이 마스킹된 텍스트를 재구성하도록 훈련된 denoising autoencoder인 MLM(masked language model)의 변형들이다. 그러나 이러한 방법은 일반적으로 특정 유형의 end tasks에 초점을 맞춰 활용성이 떨어진다는 단점이 있다.<br>
BART는 Bidirectional Transformers와 Auto-Regressive Transformers를 결합한 모델을 pretrain하며, 매우 광범위한 end tasks에 적용할 수 있는 sequence-to-sequence 모델로 구축된 denoising autoencoder이다. arbitrary noising function으로 텍스트를 손상(corrupting)시키고, 원본 텍스트를 재구성하기 위해 sequence-to-sequence 모델을 학습하며, 표준 트랜스포머 기반 신경 기계 번역 아키텍처를 사용함으로써 BERT(bidirectional encoder), GPT(left-to-right decoder), 최신 pretraining schemes를 일반화하는 것으로 볼 수 있다.(아래 그림 참조) <br>
이러한 설정의 장점으로 noising flexibility(노이즈 유연성)이 있다. noising flexibility는 길이 변경을 포함한 arbitrary transformations(임의 변환)를 원본 텍스트에 적용하는 것을 말한다. 원본 문장의 순서를 무작위로 섞고 텍스트의 임의의 범위(길이 0도 포함)가 단일 마스크 토큰으로 대체되는 새로운 in-filling scheme(채우기 방식)을 사용하여 최적의 성능을 찾는 여러 노이즈 접근 방식을 평가한다.  이 접근법은 모델이 전체 문장 길이에 대해 더 많이 추론하고 입력에 대해 더 긴 범위 변환을 하도록 하여 BERT에서의 original word masking과 next sentence prediction 목표를 일반화한다.<br>
BART는 text generation(텍스트 생성)을 위해 fine tune될 때 특히 효과적이지만 comprehension tasks(이해 작업)에도 잘 작동한다. RoBERTa와 유사한 학습 환경에서 abstractive dialogue(추상적 대화), question answering(질문&답변), summarization(요약) Task들에서 SOTA 성능을 보인다.<br>
BART는 몇 개의 추가 transformer layers 위에 쌓이는 machine translation의 새로운 scheme을 제시하여 fine tune에 대한 새로운 사고 방식을 개방한다. 이 계층은 BART를 통한 전파를 통해 기본적으로 외국어를 노이즈가 있는 영어로 번역되게 훈련하도록 target-side의 언어로 된 prt-trained BART를 사용한다.<br><br>
  - BART, BERT, GPT의 도식적(schematic) 비교<br><br>
 ![schematic_comp](https://user-images.githubusercontent.com/86700191/185347742-2d717009-4f34-4752-85f3-0e902db491c7.png)
<br><br>
- Model : BART는 손상된 문서를 원본 문서에 매핑하는 denoising autoencoder이다. 손상된 텍스트에 대한 bidirectional encoder와 left-to-right autoregressive decoder를 가진 sequence-to-sequence 모델로 구현된다. pre-training을 위해 원본 문서의 negative log likelihood를 최적화한다.
  - Architecture : ReLU 활성화 함수를 GeLU로 수정하고, 파라미터를 초기화하는 sequence-to-sequence Transformer 구조를 사용한다. Base 모델의 경우 인코더와 디코더에 6개의 레이어를 사용하고, Large 모델의 경우 각각 12개의 레이어를 사용한다.<br>
  BERT와의 차이점으로는 디코더의 각 레이어가 인코더의 마지막 히든 레이어에 추가적으로 cross-attention을 수행한다는 점과 word prediction 전에 추가적인 feed-forward network을 사용하지 않는다는 점이다. 또한, 모델이 동일한 크기라면 BART는 BERT보다 10% 정도 많은 parameters를 가진다.
  <br><br>
  - Pre-training BART : BART는 손상된 문장들을 복구하는 방식으로 학습하는데 디코더의 출력과 원본 문서의 cross-entropy loss를 최적화하여 훈련된다. 기존 denosing atoencoder는 특정 nosing scheme으로 한정되어 있었으나 BART는 어떤 타입의 document corruption이라도 사용할 수 있다.<br><br>
    - 5가지의 noise 기법<br>
    ![noising](https://user-images.githubusercontent.com/86700191/185781649-2d0780dc-86b1-463f-82fa-23714507d439.PNG) <br>
      - Token Masking : BERT처럼 랜덤한 위치의 Token을 [MASK] 토큰으로 대체한다.
      - Token Deletion : 입력에서 랜덤한 token들을 삭제한다. Token Masking와의 차이점은 모델은 삭제된 토큰의 위치가 어디인지 알아내야 한다는 점이다.
      - Text Infilling : 여러 개의 text span을 선택하고, 이를 하나의 [MASK] 토큰으로 대체한다. 이때 span의 길이는 Poisson distribution (λ = 3)를 통해 구하게 된다. (SpanBERT의 아이디어)
      - Sentence Permutation : document를 문장 단위로 나누고, 랜덤한 순서로 섞는다.
      - Document Rotation : 랜덤으로 토큰을 하나 선택 후 문서가 해당 토큰부터 시작하도록 문장의 순서를 회전시킨다. 모델은 document의 시작점을 예측해야한다.
<br><br>
- Fine-tuning BART
  - Sequence Classification Tasks (시퀀스 분류 작업) : 동일한 입력이 인코더 및 디코더에 공급되고 최종 디코더 토큰의 final hidden state가 새로운 multi-class linear classifier(다중 클래스 선형 분류기)에 공급된다. BERT의 CLS 토큰과 유사하지만, 디코더에서 토큰에 대한 representation이 전체 입력에 대한 디코더 상태에 주의할 수 있도록 끝에 추가 토큰을 추가한다.
  ![classification](https://user-images.githubusercontent.com/86700191/185827684-f4663ea7-a8a5-4235-9b69-e3763bfd6eea.PNG)
  <br><br>
  - Token Classification Tasks (토큰 분류 작업) : 전체 document를 인코더와 디코더에 입력한다. 디코더의 top hidden state를 각 단어에 대한 representation으로 사용한다.
  <br><br>
  - Sequence Generation Tasks (시퀀스 생성 작업) : BART는 autoregressive 디코더를 갖고 있으므로 abstractive question answering(추상적 질문 답변), summarization(요약)와 같은 Sequence Generation Tasks (시퀀스 생성 작업)에 대해 직접 fine-tuning이 가능하다. 인코더의 input은 input sequence이고, 디코더는 output을 autoregressive하게 생성한다.
  <br><br>
  - Machine Translation (기계 번역) : 이전에 있었던 연구에서 pre-trained 인코더를 통합하여 모델을 개선할 수 있음을 보여주었지만, 디코더에서 pre-trained 언어 모델을 사용함으로써 얻는 이득은 제한적이었다. 이에 BART 모델은 모델 전체를 기계번역을 위한 단일 디코더로 사용하고, 여기에 bitext로 학습된 새로운 Encoder 파라미터를 추가하여 해결하였다.<br>
  BART 인코더의 임베딩 레이어를 새로운 랜덤하게 초기화된 인코더로 교체한다. 이 모델은 end-to-end로 훈련되어 BART가 영어로 denoising 할 수 있는 입력에 외래어를 매핑하도록 새로운 인코더를 훈련시킨다. 이때의 새 인코더는 기존 BART 모델과 다른 vocabulary를 사용할 수 있다. <br>
  소스 인코더를 두 단계로 훈련하는데, 두 경우 모두 BART 모델의 출력에서 cross-entropy loss(교차 엔트로피 손실)을 backpropagating(역전파)한다. 첫 단계에서는 대부분의 BART parameter를 동결하고 랜덤하게 초기화된 source encoder, BART positional embedding, BART 인코더의 첫번째 레이어의 self-attention input projection matrix만 업데이트한다. 두 번째 단계에서는 적은 수의 iterations에 대해 모든 모델의 parameter를 훈련한다.<br>
  ![machine_translation](https://user-images.githubusercontent.com/86700191/185827691-50877819-ff1f-46ed-881d-e47f0b87759a.PNG)
<br><br>
- Comparing Pre-training Objectives
  - Comparison Objectives
  - Tasks
  - Results
<br><br>
- Large-scale Pre-training Experiments
  - Experimental Setup
  - Discriminative Tasks
  - Generation Tasks
  - Translation
<br><br>
- Qualitative Analysis