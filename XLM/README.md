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
  - Causal Language Modeling (CLM)
  <br><br>
  - Masked Language Modeling (MLM)
  <br><br>
  - Translation Language Modeling (TLM)
  <br><br>
  - Cross-lingual Language Models