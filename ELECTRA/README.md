# ELECTRA (Efficiently Learning an Encoder that Classifies Token Replacements Accurately)

- 주요 배경(아이디어) :  Masked Language Modeling(MLM) 사전 훈련 방법은 일부 토큰을 [MASK]로 대체하여 입력을 손상시킨 다음 모델을 훈련하여 원래 토큰을 재구성하는 방식이다. 다운스트림 NLP 작업으로 전송될 때 좋은 결과를 얻지만 일반적으로 효과적이려면 많은 양의 컴퓨팅이 필요하다는 단점이 존재한다. <br>
이러한 단점을 보완하기 위해 replaced token detection라는 보다 sample-efficient인 pre-training 방법을 제안한다. 이 방법은 입력을 마스킹하는 대신 일부 토큰을 소형 발전기 네트워크에서 샘플링된 그럴듯한 대안으로 교체하여 입력을 손상시킨다. 그런 다음 손상된 토큰의 원래 정체를 예측하는 모델을 훈련하는 대신, 우리는 예측되는 discriminative model을 훈련하는 방식이다.
<br><br>
- Introduction : language에 대한 SotA representation learning은 denoising Autoencoder(노이즈 제거 자동인코더)를 학습하는 것으로 볼 수 있다. 입력 시퀀스의 토큰 중 약 15% 정도를 마스킹하고 이를 복원하는 masked language modeling (MLM) 태스크를 통해 학습을 진행 하는데 기존의 autoregressive language modeling 학습에 비해 양방향 정보를 고려한다는 점에서 효과적인 학습을 할 수 있지만 토큰의 15%에서만 학습하기 때문에 상당한 계산 비용이 발생한다는 단점이 존재한다. <br>
대안으로 모델이 실제 입력 토큰을 그럴듯하지만 합성적으로 생성된 대체 토큰과 구별하는 방법을 배우는 사전 교육 작업인 replaced token detection task를 제안한다. masking 대신 일부 토큰을 small masked language model의 output을 통해 샘플로 대체하여 입력을 손상시킨다. 이 손상 절차는 네트워크가 pre-training 중에 인공 [MASK] 토큰을 보지만 다운스트림 작업에서 fine-tuning 될 때는 볼 수 없는 BERT의 mismatch문제를 해결한다. 그런 다음 모든 토큰에 대해 original, replacement를 예측하는 discriminator(판별기)로 pre-training을 한다.<br>
접근방식이 GAN과 유사해보이지만 text에 GAN을 적용하기 어렵기 때문에 maximum-likelihood로 훈련한다는 점에서 제안하는 방식은 “adversarial”이 아니다.
    - 타 모델과의 GLUE score 비교 <br>.
    ![compare_model](https://user-images.githubusercontent.com/86700191/183028796-ce1e2116-6bf9-4a0b-97ac-76b6a0ab0379.PNG) <br>
    ELECTRA-Small의 경우 single gpu에서 4일만에 학습된다. (BERT-Large의 1/20 파라미터 수, 1/135 계산량을 가지고 있다) 또한 BERT-Small과 비교하여 GLUE 성능이 5point 더 높고, 훨씬 더 큰 모델인 GPT 보다도 더 높다. ELECTRA-Large는 RoBERTa나 XLNet보다 더 적은 파라미터와 계산량의 1/4만 사용하였으나 비슷한 성능을 가졌다.
<br><br>
- Method
<br><br>
- Experiments
<br><br>
- Conclusion
<br><br>