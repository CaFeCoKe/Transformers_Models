# XLM (Cross-lingual Language Model Pretraining)

- Introduction : large unsupervised text corpus이 학습되고 자연어 이해(NLU) 작업에 대해 미세 조정된 Transformer 언어 모델이 나온 이후 general-purpose sentence representations(범용 문장 표현)에 대한 관심이 급증했지만 대부분 하나의 언어(특히 영어)에만 초점이 맞춰지고 있다.
많은 언어에서 cross-lingual sentence representations(교차 언어 문장 표현)을 학습하고 평가하는 최근의 발전은 영어 중심 bias(편향)를 완화하는 것을 목표로 하며, 임의의 문장을 shared embedding space(공유 임베딩 공간)으로 인코딩할 수 있는 universal cross-lingual encoder(범용 교차 언어 인코더)를 구축하는 것이 가능하다는 것을 시사한다. <br>
본 논문에서 기여한 점
  - cross-lingual language modeling(교차 언어 모델링)을 사용하여 cross-lingual representation(교차 언어 표현)을 학습하기 위한 새로운 unsupervised method(비지도 방법)을 소개하고 두 가지 monolingual pretraining objective(단일 언어 사전 훈련 목표)를 조사한다.
  - 병렬 데이터를 사용할 수 있을 때 cross-lingual pretraining(교차 언어 사전 훈련)을 개선하는 새로운 supervised learning objective(지도 학습 목표)를 소개한다.
  - cross-lingual classification(언어 간 분류), unsupervised machine translation(비지도 기계 번역) 및 supervised machine translation(지도 기계 번역)에서 SOTA를 달성한다.
  - cross-lingual language model(교차 언어 모델)이 low-resource languages(저자원 언어)의 복잡성에 대한 상당한 개선을 제공할 수 있음을 보여준다.
  - 본 논문의 코드와 사전 훈련된 모델을 사용할 수 있도록 공개한다.
<br><br>
- Cross-lingual language models : 세 가지의 language modeling objective를 제시한다. (단일언어 기반 비지도 학습 2개 + 병렬 말뭉치 기반 지도 학습 1개)
  - Shared sub-word vocabulary : Byte Pair Encoding (BPE)를 통해 생성된 동일한 shared vocabulary를 가진 모든 언어를 처리한다. 이것은 동일한 알파벳 또는 숫자 또는 고유 명사와 같은 anchor token을 공유하는 언어 간 임베딩 공간의 정렬을 크게 향상시킨다.
  monolingual corpora에서 무작위로 샘플링된 문장의 연결에 대한 BPE 분할을 배운다.(α = 0.5)<br>
  ![math](https://user-images.githubusercontent.com/86700191/191441460-af3d690b-1ea3-4062-b17d-2d0ef89d9a4c.PNG) <br>
  이 분포를 사용하여 샘플링하면 low-resource language와 관련된 토큰 수가 증가하고 high-resource language(고자원 언어)에 대한 bias가 완화된다. 특히, 이것은 low-resource language의 단어가 문자 수준에서 분할되는 것을 방지한다.
  <br><br>
  - Causal Language Modeling (CLM) : 문장 P(wt | w1,..., wt-1, θ)에서 주어진 단어의 확률을 모델링하도록 훈련된 Transformer language model으로 구성된다. Transformers는 이전 hidden state를 현재 batch로 전달하여 batch의 첫 번째 단어에 context를 제공할 수 있다.
  그러나 이 기술은 cross-lingual setting(교차 언어 설정)으로 확장되지 않으므로 단순성을 위해 각 배치의 첫 번째 단어를 context 없이 그대로 둔다.
  <br><br>
  - Masked Language Modeling (MLM) : text stream에서 BPE 토큰의 15%를 무작위로 샘플링하고, 80%는 [MASK] 토큰으로 대체하며, 10%는 무작위 토큰으로 변경하고, 10%는 시간을 변경하지 않고 유지한다.
   기존 MLM와의 차이점에는 문장 쌍 대신 임의의 문장 수(256개의 토큰으로 자름)의 text stream이 사용된다. 희귀 토큰과 빈번한 토큰 사이의 불균형에 대응하기 위해 text stream의 토큰은 가중치가 invert frequencies(반전 주파수)의 제곱근에 비례하는 다항 분포에 따라 샘플링한다. <br><br>
  ![MLM](https://user-images.githubusercontent.com/86700191/191736531-e4634d76-dda2-434c-96dd-456af8740fb6.PNG)
  <br><br>
  - Translation Language Modeling (TLM) : CLM 및 MLM objective는 unsupervised하며 monolingual data(단일 언어)만을 필요로 한다. 하지만 사용 가능한 병렬 데이터를 활용하질 못한다. cross-lingual pretraining을 개선하기 위해 새로운 translation language modeling (TLM)을 만들었다.
  TLM objective는 monolingual text stream를 고려하는 대신 병렬 문장을 연결하는 MLM의 확장이다. source 및 target 문장 모두에서 단어를 무작위로 마스킹한다. <br>
  예를 들어 영어 문장에서 마스킹된 단어를 예측하기 위해 모델은 주변 영어 단어 또는 프랑스어 번역에 주의를 기울일 수 있으며, 모델이 영어와 프랑스어 표현을 정렬하도록 장려한다. 특히 영어 문맥이 마스킹된 영어 단어를 추론하기에 충분하지 않을 경우 모델은 프랑스어 문맥을 활용할 수 있다. 정렬을 용이하게 하기 위해 target 문장의 위치도 재설정한다. <br><br>
  ![tlm](https://user-images.githubusercontent.com/86700191/191895601-5cae9263-e94d-434c-adf8-d47240627bb4.PNG)
  <br><br>
  - Cross-lingual Language Models : 3개의 방법으로 pretraining을 진행했다; CLM, MLM, MLM used in combination with TLM. CLM 및 MLM은 256개의 토큰으로 구성된 64개의 연속 문장 스트림으로 구성된 배치로 모델을 훈련한다. 각 반복에서 배치(batch)는 동일한 언어에서 온 문장으로 구성된다. TLM과 함께 사용되는 MLM은 두 가지 목표를 번갈아 가며, 유사한 접근 방식으로 language pairs(언어 쌍)을 샘플링한다.
<br><br>
- Cross-lingual language model pretraining