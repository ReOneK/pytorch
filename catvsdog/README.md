pytorch_application
=

* dataset: catvsdog  [download](https://www.kaggle.com/c/dogs-vs-cats/data)

* env: 
       
       python3.6
       
       cpu/gpu
       
       reqiurement.txt
       
* train command:
        
        python main.py train --train-data-root=./data/train --use-gpu --env=classifier
        
   
* test command:
        
        python main.py test --data-root=./data/test  --batch-size=256 --load-path='checkpoints/squeezenet.pth'
        
        
   
if you want use visdom to visualizatioon,you have to run this command first:

        python -m visdom.server
        
        （and i hate visdom,it is urgly...）       
     