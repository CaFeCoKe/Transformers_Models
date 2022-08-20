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
  - Pre-training BART
<br><br>
- Fine-tuning BART
  - Sequence Classification Tasks
  - Token Classification Tasks
  - Sequence Generation Tasks
  - Machine Translation
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