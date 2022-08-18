# BART (Bidirectional Auto-Regressive Transformer)

- Abstract : BART는 seq2seq 모델의 pretraining을 위한 denoising autoencoder이며, arbitrary noising function으로 텍스트를 손상(corrupting)시키고, 원본 텍스트를 재구성하는 모델을 학습함으로써 학습된다. 표준 트랜스포머 기반 신경 기계 번역 아키텍처를 사용함으로써 BERT(bidirectional encoder), GPT(left-to-right decoder), 최신 pretraining schemes를 일반화하는 것으로 볼 수 있다. <br>
본 논문에서는 원본 문장의 순서를 무작위로 섞고 텍스트 범위가 단일 마스크 토큰으로 대체되는 새로운 in-filling scheme(채우기 방식)을 사용하여 최적의 성능을 찾는 여러 노이즈 접근 방식을 평가한다. BART는 text generation(텍스트 생성)을 위해 fine tune될 때 특히 효과적이지만 comprehension tasks(이해 작업)에도 잘 작동한다. 
또한, RoBERTa와 유사한 학습 환경에서 abstractive dialogue(추상적 대화), question answering(질문&답변), summarization(요약)Task들에서 SOTA 성능을 보인다.
<br><br>
- Introduction
<br><br>
- Model
  - Architecture
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