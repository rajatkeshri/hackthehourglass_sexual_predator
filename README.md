# hackthehourglass_sexual_predator

The dataset requires author persmission. Please register for permission in this link. I have added the dataset temporarily in the drive link, and will remove it from the github repo after the hack.
https://zenodo.org/record/3713280#.XyVATSgzbIU
<br>
<br>
Drive link : https://drive.google.com/drive/folders/1hh3aWjBUlIyGH9Z_dbUyodAdTIE2HM3n?usp=sharing
<br>
<br>

<h1> Usage </h1>
After getting the dataset, extract the xml file and put it in a folder named test. Run the jupyter notebook "Data Preprocessing.ipynb" to genereate a csv file
<br>
<br>
The first model trained on the csv egenerated above is trained by running the file "Training with sentences.ipynb". The model is saved in the Models folder. Here we train Logistic regression, SVM and neural network using scikitlearn library.
The second model is trained using the goodwords.txt which contains all normally used words and bagofwords_predator.txt which contains words commonly used to identify predators. The 2nd model is trained by running "Training with only words.ipynb".
<br>
<br>
Once both the models are trained, run the "Predator Bot.py" file. This will extract the chats and and save it in chats.json file. This will also do the predictions on the chats.json file
