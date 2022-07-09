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
  <br><br>
  - Dynamic Masking : RoBERTa에서 사용한 Masking 방법이다. Static Masking과는 달리 매 epoch마다 Mask를 새로 적용하는 방식으로 학습 시간은 늘어나지만 메모리를 아낄 수 있다는 장점이 있다.
  따라서, 더 많은 step, 더 큰 데이터셋으로 사전학습을 할 경우에 Dynamic Masking은 중요하다. <br>
  동일한 환경에서 진행된 실험에서 Dynamic Masking의 성능은 Static Masking의 성능과 비슷하거나 조금 향상된 성능을 보여준다.
<br><br>
- Input format & Next Sentence Prediction (NSP)
  - NSP의 필요성에 대한 의문 : 기존 BERT에서는 두 개의 Segment(문장)를 이어 붙여 입력으로 사용하고 두 Segment가 문맥상 이어지는지에 대해 판단하는 Pair 단위의 태스크를 위해 NSP를 추가하여 학습을 진행하였다. 이로 인해 NSP가 있는 모델이 더 뛰어난 성능을 보여주게 되었다.<br>
  하지만 해당 실험에선 단순히 NSP의 유무에 대해서만 판단하였고, 이는 입력 형태는 그대로 사용하였기 때문에 나온 결과였다. NSP가 없다면 입력 형태가 굳이 두 개의 Segment를 이어 붙인 형태를 사용 할 필요가 없었고 이를 재검증하기 위해 RoBERTa에서는 4가지의 입력 형태를 준비하였다.
  <br><br>
  - Input format setting
    - Segment-Pair + NSP : 기존 BERT와 동일한 설정이다. 결합된 토큰의 길이는 512개 미만이여야한다.
    - Sentence-Pair + NSP : 각 Segment가 하나의 문장으로만 구성된다. 각 segment가 매우 짧기 때문에 토큰의 길이가 512개보다 훨씬 적다. 따라서 Segment-Pair + NSP의 토큰 수와 비슷하게 유지되도록 batch size를 늘린다.
    - Full-Sentences (removed NSP) : 각 Input은 하나 이상의 문서에서 전체 길이가 최대 512 토큰이 되도록 연속적으로 샘플링된 전체 문장으로 구성된다. 문서의 경계를 넘을 수 있으며, 이 경우 separator token(구분자 토큰)을 추가한다. 
    - Doc-Sentences (removed NSP) : Full-Sentences와 유사하게 구성되나 하나의 문서가 끝나면 다음 문서를 이용하지 않는다. 이로 인해 512개 토큰보다 짧을 수 있어 batch size를 동적으로 늘려 Full-Sentences와 유사한 총 토큰 수를 가지게 된다.
  <br><br>
  - NSP의 유무, 4가지의 Input format의 결과<br><br>
  ![input_format_exp](https://user-images.githubusercontent.com/86700191/177953169-5456cf55-1b21-45f9-b8d1-a0289b0a2ccb.PNG) <br><br>
  Pair 단위의 입력과 NSP가 존재하는 모델에서는 두 개의 Segment(문장)를 이어 붙인 Segment-Pair + NSP의 성능이 높게 나왔다. Sentence-Pair + NSP에서는 long-range dependencies를 학습할 수 없었기 떄문이다. <br>
  NSP의 유무로 따진 성능에서는 NSP를 제거한 모델이 약간 더 향상된 성능을 보여준다. 이중에 Doc-Sentences가 더 높은 성능을 나타냈지만 batch size가 동적으로 변하기 떄문에 RoBERTa 논문에서는 이 부분에 있어 일관성 있는 설정을 위해 Full-Sentences의 입력 형태를 사용한다.
<br><br>
- Training with Large Batches