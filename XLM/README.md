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
  ![TLM_](https://user-images.githubusercontent.com/86700191/191912543-980ba0a0-2081-4ff6-8d6b-8e4723eba11c.PNG)
  <br><br>
  - Cross-lingual Language Models : 3개의 방법으로 pretraining을 진행했다; CLM, MLM, MLM used in combination with TLM. CLM 및 MLM은 256개의 토큰으로 구성된 64개의 연속 문장 스트림으로 구성된 배치로 모델을 훈련한다. 각 반복에서 배치(batch)는 동일한 언어에서 온 문장으로 구성된다. TLM과 함께 사용되는 MLM은 두 가지 목표를 번갈아 가며, 유사한 접근 방식으로 language pairs(언어 쌍)을 샘플링한다.
<br><br>
- Cross-lingual language model pretraining : cross-lingual language model을 사용하여 다음을 얻을 수 있는 방법을 설명한다.
  1. zero-shot cross-lingual classification를 위한 문장 인코더의 더 나은 초기화
  2. supervised, unsupervised neural machine translation(지도, 비지도 신경망 기계 번역) 시스템의 더 나은 초기화
  3. low-resource languages(저자원 언어)에 대한 언어 모델
  4. unsupervised cross-lingual word embedding(비지도 교차 언어 단어 임베딩)
  <br><br>
  - Cross-lingual classification : pretrained XLM 모델은 general-purpose cross-lingual text representation(범용 교차 언어 텍스트 표현)을 제공한다. 영어 분류 작업에 대한 monolingual language model을 fine-tuning하는 것과 유사하게 cross-lingual classification benchmark(교차 언어 분류 벤치마크)에서 XLM을 fine-tuning한다. 모델 성능 평가에는 cross-lingual natural language inference (XNLI) dataset를 사용한다.
  pretrained Transformer의 첫 번째 hidden state위에 linear classifier(선형 분류기)를 추가하고 영어 NLI 학습 데이터 세트의 모든 파라미터를 fine-tuning한다. 그런 다음 15개의 XNLI 언어로 정확한 NLI 예측을 할 수 있는 모델의 용량을 평가한다. <br>
  ![tabel1](https://user-images.githubusercontent.com/86700191/191938867-8dd280b5-9f1b-493e-9227-4f6f85457875.PNG)
  <br><br>
  - Unsupervised Machine Translation : Pretraining은 unsupervised neural machine translation (UNMT)(비지도 신경 기계 번역)의 핵심 요소이다. lookup table을 초기화하는 데 사용되는 pretrained crosslingual word embeddings(사전 훈련된 교차 언어 단어 임베딩)의 품질은 UNMT의 성능에 상당한 영향을 미친다.
  여기에 UNMT의 반복 프로세스를 bootstrap하기 위해 cross-lingual language model을 사용하여 전체 인코더 및 디코더를 pretraining함으로써 한 단계 더 발전시킬 것을 제안한다. "WMT 14 En-Fr", "WMT 16 En-Dr", "WMT 16 En - Ro" 에 대해 평가한다. <br>
  ![UNMT](https://user-images.githubusercontent.com/86700191/192256839-9962af9f-a3b2-4e8b-bffe-694b8047a113.PNG)
  <br><br>
  - Supervised Machine Translation : supervised machine translation(지도 기계 번역)을 위한 cross-lingual language modeling pretraining의 영향을 조사하고, 접근 방식을 다국어 NMT로 확장한다. CLM과 MLM pretraining이 "WMT 16 En-Ro"에 미치는 영향을 평가한다. <br>
  ![SMT](https://user-images.githubusercontent.com/86700191/192256845-7725f70a-6d8e-40d4-a08c-c9262e799e32.PNG)
  <br><br>
  - Low-resource language modeling : vocabulary의 상당 부분을 공유할 때 higher-resource languages의 데이터를 활용하는 것이 종종 도움이 된다. Wikipedia에는 100k 문장 정도의 네팔어가 있고 힌두어는 약 6배정도 더 많다. 또한, 두 개의 언어는 약 80% 정도의 100k subword units의 BPE vocab을 공유한다. 네팔어 언어 모델, 힌두어를 조합한 언어 모델, 힌두어와 영어를 조합한 언어 모델을 가지고 perplexity를 비교한다. <br>
  ![Nepali_LM](https://user-images.githubusercontent.com/86700191/192460120-c94f5527-a71f-499f-834e-d34961ee6259.PNG)
  <br><br>
  - Unsupervised cross-lingual word embeddings : 이전의 연구들에서 monolingual word embedding spaces(단일 언어 단어 임베딩 공간)을 adversarial training(적대적 훈련)과 함게 정렬하여 unsupervised word translation(비지도 단어 번역)을 수행하는 방법을 보여주었으며, 또한 두 언어 간의 공유 어휘를 사용한 다음 monolingual corpora(단일 언어 말뭉치)연결에 fastText를 적용하는 것도 공통 알파벳을 공유하는 언어에 대한 high-quality cross-lingual word embedding(고품질 교차 언어 단어 임베딩) (Concat)을 직접 제공한다는 것도 보여주었다.
  이 논문에서는 공유 어휘를 사용하지만 단어 임베딩은 cross-lingual language model (XLM)의 lookup table을 통해 얻는다. 이 접근방식들을 cosine similarity, L2 distance, cross-lingual word similarity의 세 가지 metrics에 대해 비교한다<br>
  ![cross-ligual_word_embedding](https://user-images.githubusercontent.com/86700191/192460192-cdde25ed-f9a2-4688-a8bc-3d81eb33e1db.PNG)
<br><br>
- Experiments and results : cross-lingual language model pretraining이 여러 벤치마크에 미치는 강한 영향을 empirically하게(실증적으로) 보여주고, 현 논문의 접근 방식을 현재 SOTA와 비교한다.
  - Training details : 1024 hidden units, 8 heads, GELU activation, 0.1 rate의 dropout, learned positional embeddings가 있는 Transformer architecture를 사용한다. 또한, Adam optimizer, linear warmup, 1e-4에서 5e-4까지의 learning rate를 사용하여 모델을 훈련한다. <br>
  CLM 및 MLM 목표를 위해 256개 토큰의 스트림과 64 size의 mini batch를 사용한다. 이때의 mini batch는 두 개 이상의 연속된 문장을 포함할 수 있다. TLM 목표를 위해 유사한 길이의 문장으로 구성된 4000개의 토큰의 mini batch를 샘플링한다. 이 논문에서는 언어에 대한 평균적인 perplexity를 훈련의 중지 기준으로 사용한다. 기계 번역은 6개의 layer를 사용하고, 2000개의 토큰의 mini batch를 사용한다. <br>
  XNLI에서 fine-tuning할 때, 8 또는 16 크기의 mini batch를 사용하고, 문장 길이를 256단어로 잘라낸다. 80k BPE 분할과 95k의 어휘를 사용하고 XNLI 언어의 Wikipedia에서 12-layer model을 훈련한다. 5e-4에서 2e-4 까지의 값으로 Adam optimizer의 learning rate를 샘플링하고 20000 random sample의 evaluation epochs를 갖는다.
  임의로 초기화된 final linear classifier(최종 선형 분류기)에 대한 입력으로 transformer의 마지막 layer의 첫 번째 hidden state를 사용하고 모든 파라미터를 fine-tuning한다. 현 실험에서 마지막 layer에 대해 max-pooling 또는 mean-pooling을 사용하는 것이 첫 번째 hidden state를 사용하는 것보다 더 잘 작동하지 않았다. 마지막으로 훈련 속도를 높이고 모델의 메모리 사용량을 줄이기 위해 float16 연산을 사용한다.
  <br><br>
  - Data preprocessing : WikiExtractor2를 사용하여 Wikipedia dump에서 원시 문장을 추출하고 CLM 및 MLM 목표에 대한 monolingual data(단일 언어 데이터)로 사용한다. TLM 목표를 위해 영어를 포함하는 병렬 데이터만 사용한다. 정확하게 프랑스어, 스페인어, 러시아어, 아랍어 및 중국어에는 MultiUN을 사용하고, 힌두어에는 IIT Bombay corpus를 사용한다.
  OPUS 3 website Tiedemann에서 다음과 같은 corpora를 추출한다: EUbookshop corpus에서는 독일어, 그리스어 및 불가리아어, OpenSubtitles 2018에서는 터키어, 베트남어 및 태국어, Tanzil에서는 우르두어와 스와힐리어, GlobalVoices에서는 스와힐리어를 추출하였다. 중국어, 일본어는 Kytea4 tokenizer를 태국어는 PyThaiNLP5 tokenizer를 사용하고 다른 모든 언어의 경우 Moses가 제공한 tokenizer와 필요할 때 default English tokenizer를 사용한다.
  단어를 subword unit으로 분할하고 BPE codes를 학습 하기 위해 fastBPE6를 사용한다. BPE codes는 모든 언어에서 샘플링된 문장의 연결에서 학습된다.
  <br><br>
  - Results and analysis
    - Cross-lingual classification <br>
    ![cross-lingual_result](https://user-images.githubusercontent.com/86700191/192981569-a23386e2-c687-4c5a-86c2-10fdab9943b1.PNG) <br><br>
    두 가지 유형의 pretrained cross-lingual encoders를 평가한다: monolingual corpora만에 대해서만 MLM 목표를 사용하는 unsupervised cross-lingual language model 그리고 추가 병렬 데이터를 사용하여 MLM과 TLM 손실을 모두 결합하는 supervised cross-lingual language model. 또한, 두 가지 기계 번역 기준선을 포함한다: 영어 MultiNLI 훈련 세트가 각 XNLI 언어로 기계 번역되는 TRANSLATETRAIN 그리고 XNLI의 모든 개발 및 테스트 세트가 영어로 번역되는 TRANSLATER-TEST. <br>
    fully unsupervised MLM 방법은 zero-shot cross-lingual classification에 대한 2억 2,300만 개의 병렬 문장을 사용하는 Artetxe와 Schwenk의 supervised 접근방식을 크게 능가한다. 각 XNLI 언어(TRANSLATE-TRAIN)의 훈련 세트에서 fine-tune된 경우, 논문의 supervised model은 zero shot 접근방식을 능가하여 XLM이 강력한 성능과 함께 모든 언어에서 fine-tune 될 수 있음을 보여준다.
    <br><br>
    - Unsupervised machine translation
    <br><br>
    - Supervised machine translation
    <br><br>
    - Low-resource language model
    <br><br>
    - Unsupervised cross-lingual word embeddings
<br><br>
- Conclusion