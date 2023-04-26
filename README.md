# ReverseDictionary
A reverse dictionary to retrieve words given a vague definition input

Consists of 4 primary python notebooks that implement DistilBert+LSTM, Bert+LSTM, BART and Pegasus to solve the objective. Various iterations of the datasets used during development can be found in the datasets directory.

Final accuracy scores on the testset:

|  | Top 1    | Top 10    | Top 100 |
| :---:   | :---: | :---: | :---: |
| DistilBert+LSTM | 0%   | 0%   | 6% |
| Bert+LSTM | 3%   | 6%   | 20% |
| Bart | 15%   | 38%   | 46% |
| Pegasus | 33%   | 57%   | 73% |

Developed as a capstone project for CSCI 544 at USC.

Team details:
Group 17
- Devanshi Krishna Shah 
- Likhita Arun Navali 
- Marta Taulet Sanchez
- Shashank Ramesh
- Sindhura Bagodu Ramachandra

