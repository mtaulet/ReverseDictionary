# ReverseDictionary
A reverse dictionary to retrieve words given a vague definition input

Consists of 4 primary python notebooks that implement DistilBert+LSTM, Bert+LSTM, BART and Pegasus to solve the objective. Various iterations of the datasets used during development can be found in the datasets directory.

While running the notebooks, make sure that the paths point to the appropriate file locations. Additionally BARTv3_Text_summarization_Multiple_Outputs and Pegasus_v1 were developed on Google colab and hence contain the cell to connect to google drive.

Src directory contains files used to create the datasets.

Final accuracy scores on the testset:

|  | Top 1    | Top 10    | Top 100 |
| :---:   | :---: | :---: | :---: |
| DistilBert+LSTM | 0%   | 0%   | 6% |
| Bert+LSTM | 3%   | 6%   | 20% |
| Bart | 23%   | 39%   | 48% |
| Pegasus | 33%   | 57%   | 73% |

Developed as a capstone project for CSCI 544 at USC.

Team details:
Group 17
- Devanshi Krishna Shah 
- Likhita Arun Navali 
- Marta Taulet Sanchez
- Shashank Ramesh
- Sindhura Bagodu Ramachandra

