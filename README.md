# Normal_Covid_Non_covid_Classification

Chen Yan (1003620) 
Goh Yi Lin,Tiffany (1003674)               
Xavier Tan De Jun (1003376)


To train the Normal-Infected Model
1) Uncomment the Model 1 training block
2) Comment out all other blocks (Covid-Non-Covid block and Graph Plotting block)
3) Run: python main.py "./dataset" --save_dir_norm_inf_model "./checkpoints/modelTestModel1" --epochs 15 --gpu

To train the Covid-Non-Covid Model
1) Uncomment the Model 2 training block
2) Comment out all other blocks (Normal-Infected block and Graph Plotting block)
3) Run: python main.py "./dataset" --save_dir_covid_non_model "./checkpoints/modelTestModel2" --epochs 15 --gpu

To run the final validation set testing and plotting
1) Uncomment the Graph Plotting block
2) Comment out all other blocks (Normal-Infected block and Covid-Non-Covid block)
3)Change variable path1 = './checkpoints/modelTestModel1/epoch-14.pt'
4)Change variable path2 = './checkpoints/modelTestModel2/epoch-14.pt'
5) Run: python main.py "./dataset"

0: Normal
1: Infected, Non-Covid
2: Infected, Covid
