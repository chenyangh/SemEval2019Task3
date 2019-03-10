# ANA at SemEval-2019 Task 3
This repo contains the code for our paper,
 
 *ANA at SemEval-2019 Task 3: Contextual Emotion detection in Conversations through hierarchical LSTMs and BERT*


An overview of the proposed *HRLCE* Model:

![HRLCE](img/hred.jpg )


The results are shown in the following table:

|        |    F1    |   Happy  |   Angry  |    Sad   | Harm. Mean  |
| ------ | :------: | :------: | :------: | :------: | :---------: |
| *SL*   |   Dev  <br/>  Test  |  0.6430  <br/>  0.6400  |  0.7530 <br/>  0.7190 |  0.7180  <br/> 0.7300  |  0.7016  <br/> 0.6939    |
| *SLD*   |   Dev  <br/>  Test  |  0.6470  <br/>  0.6350  |  0.7610 <br/>  0.7180 |  0.7360  <br/> 0.7360  |  0.7112  <br/> 0.6934    |
| *HRLCE*   |   Dev  <br/>  Test  |  0.7460  <br/>  0.7220  |  0.7590 <br/>  0.7660 |  0.8100  <br/> 0.8180  |  **0.7706**  <br/> **0.7666**    |
| *BERT*   |   Dev  <br/>  Test  |  0.7138  <br/>  0.7151  |  0.7736 <br/>  0.7654 |  0.8106  <br/> 0.8157  |  0.7638  <br/> 0.7631    |

*CODE CLEANING IS STILL IN PROGRESS*

# Acknowledgement
This code is relying on the work of the following projects, 
please do not forget to get them credits if you find the work of ours is helpful:

[AllenNlp](https://github.com/allenai/allennlp)

[torchMoji](https://github.com/huggingface/torchMoji)

[pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)

[ekphrasis](https://github.com/cbaziotis/ekphrasis)