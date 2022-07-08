# RoBERTa (A Robustly Optimized BERT Approach)

- 주요 배경(아이디어) : BERT 모델이 Undertrain 되어 있는 것을 발견하여, 사전학습(pre-training) 과정에서의 Hyper-parameter들을 튜닝하며 학습을 진행하여 기존 BERT에 비해 더 좋은 성능을 얻었다. (GLUE, RACE, SQuAD에서 SOTA 달성)
<br><br>
- Hyper-parameter 튜닝
  - 더 큰 배치로 더 오래 모델을 학습한다.
  - Next Sentence Prediction (NSP) Task를 제거한다.
  - Sequence 길이(크기)를 더 늘려 학습을 진행한다.
  - Masking 패턴을 동적으로 바꾼다.
  - 대용량의 새로운 데이터셋을 모은다.
<br><br>
- Static vs Dynamic Masking (정적 vs 동적)
  - Static Masking : 기존 BERT에서 사용하는 Masking 방법이다. Mask를 한번만 하고 고정되어 동일한 Mask를 갖는 학습 데이터를 매 학습 단계(epoch)에서 반복해서 보게된다. <br>
  이를 해결하기 위해 학습 데이터를 10번 복사해서 각각의 시퀀스가 40번의 epoch를 진행하는 동안 각기 다른 10개의 Mask를 적용하였다. 이는 곧 시퀀스가 동일한 Mask를 4번만 보게 된다는 뜻이다. <br>
  하지만 이는 데이터의 용량이 커질수록 메모리의 관점에서 보게 되면 점점 효율이 낮아진다. 
  - Dynamic Masking : 