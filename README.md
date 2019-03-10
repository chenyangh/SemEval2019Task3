# ANA at SemEval-2019 Task 3 

*NOTE: CODE CLEANING IS STILL IN PROGRESS*


This repo contains the code for our paper,
 
 *ANA at SemEval-2019 Task 3: Contextual Emotion detection in Conversations through hierarchical LSTMs and BERT*


An overview of the proposed *HRLCE* Model:

![HRLCE](img/hred.jpg )


HRLCE is a single model that can achieve a score of 0.7666 on the final test set while only using the training dataset. 

We also finetune the *BERT-LARGE* model on this task. The results of BERT and HRLCE are combined to get the 0.7709 on the leaderboard of SemEval 2019 Task3.
 
## Instructions
PyTorch1.0 with Python 3.6 serve as the backbones of this project.

The code is using one GPU by default, you have to modify the code to make it running on CPU or multiple GPUs.

The code might include more features than what has been described in the paper. For example, we experienced with multi-task learning and focal loss, but we found no significant difference.

To run the code, you have to specify the path to a 

Another note, to get your submissions measured the same way as the CodaLab, you need to look at the harmonic mean of the three macro F1 scores of the three emotion categories. 
It is slightly different than using the micro F scores of the three emotion categories directly. 

## Performance
The results are shown in the following table:

|        |    F1    |   Happy  |   Angry  |    Sad   | Harm. Mean  |
| ------ | :------: | :------: | :------: | :------: | :---------: |
| *SL*   |   Dev  <br/>  Test  |  0.6430  <br/>  0.6400  |  0.7530 <br/>  0.7190 |  0.7180  <br/> 0.7300  |  0.7016  <br/> 0.6939    |
| *SLD*   |   Dev  <br/>  Test  |  0.6470  <br/>  0.6350  |  0.7610 <br/>  0.7180 |  0.7360  <br/> 0.7360  |  0.7112  <br/> 0.6934    |
| *HRLCE*   |   Dev  <br/>  Test  |  0.7460  <br/>  0.7220  |  0.7590 <br/>  0.7660 |  0.8100  <br/> 0.8180  |  **0.7706**  <br/> **0.7666**    |
| *BERT*   |   Dev  <br/>  Test  |  0.7138  <br/>  0.7151  |  0.7736 <br/>  0.7654 |  0.8106  <br/> 0.8157  |  0.7638  <br/> 0.7631    |

From the table you will notice that BERT performs better on *Angry* but worse on *Happy*, therefore it makes sense to combine the results of these two. 


## Acknowledgement
This code is relying on the work of the following projects:

* [AllenNlp](https://github.com/allenai/allennlp)

* [torchMoji](https://github.com/huggingface/torchMoji)

* [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)

* [ekphrasis](https://github.com/cbaziotis/ekphrasis)

Many thanks to my supervisor [Osmar R. Zaïane](http://webdocs.cs.ualberta.ca/~zaiane/) for supporting me working on this shared task. 