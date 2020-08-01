from time import sleep
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

#######################
name_msg_dic = {}
msgs = []
model_sentence_LR = "Models/Lr_model_1.pkl"
model_words_LR = "Models/Lr_model_words.pkl"
model_words_SVM = "Models/svm_model_words.pkl"


name_prediction_dic = {}

f = open("bagofwords_predator.txt", "r")
predator_words = f.readlines()
f = open("goodwords.txt", "r")
goodwords = f.readlines()

p_temp = []
g_temp = []
for i in range(0,len(predator_words)):
    predator_words[i] = predator_words[i].split("\n")[0]
    goodwords[i] = goodwords[i].split("\n")[0]
    p_temp.append(1)
    g_temp.append(0)

X_train = predator_words + goodwords
y_train = p_temp + g_temp
#######################


def extract_msg():
    chat_number = 0
    #dm_set = browser.find_elements_by_xpath("//div[@class='        DPiy6            Igw0E     IwRSH      eGOV_         _4EzTm                                                                                                              ']")
    i = 0
    #for i in range(0,len(dm_set)):
    print("GETTING CHAT DATA")
    while True:

        msgs = []
        dm_set = browser.find_elements_by_xpath("//div[@class='        DPiy6            Igw0E     IwRSH      eGOV_         _4EzTm                                                                                                              ']")
        x = dm_set[i]
        link = x.find_element_by_xpath(".//a[@class='-qQT3 rOtsg']")
        elements_within_link = x.find_elements_by_xpath(".//a[@class='-qQT3 rOtsg']")
        link.click()
        sleep(2)

        try:

            name = browser.find_element_by_xpath(".//div[@class='_7UhW9    vy6Bb      qyrsm KV-D4              fDxYl     ']")
            print(name.text)
            i+=1
            if i == 22:
                break

            c = 0
        except:
            pass
        #for j in range(0,5):
        try:
            bottom_up_msg  = browser.find_elements_by_xpath(".//div[@class='_7UhW9   xLCgt      MMzan  KV-D4             p1tLr      hjZTB']")
            #print(len(bottom_up_msg))
            #bottom_up_msg = bottom_up_msg[(len(bottom_up_msg) - 1) - c]
            #bottom_up_msg.click()
            for k in bottom_up_msg:
                #print(k.text)
                #print("\n")
                msgs.append(k.text)

            name_msg_dic[name.text] = msgs

        except:
            pass

            #el=browser.find_element_by_xpath(".//div[@class='QBdPU ']")

            #action = webdriver.common.action_chains.ActionChains(browser)
            #action.move_by_offset(1284-30,190)
            #action.click()
            #action.perform()
            #c+=1
        chat_number+=1
        if chat_number == 11:
            break

    with open('chats.json', 'w') as fp:
        json.dump(name_msg_dic, fp,indent = 4)

    return name_msg_dic


def prediction(name_msg_dic):

    is_predator_number = 0
    ################################
    #Temporary
    spi = pd.read_csv('train_nourl.csv')
    spi_sub = spi
    # splitting test train data
    spi_sub.loc[(pd.isna(spi_sub["predator"])),['predator']] = 0
    spi_sub.loc[(pd.isna(spi_sub["msg"])),['msg']] = "whatsup"


    # We need to make sure the classes have the same proportion in both sets
    X_train, X_test, y_train, y_test = train_test_split(np.array(spi_sub['msg'].apply(lambda x: np.str_(x))),
                                                        np.array(spi_sub['predator']), stratify = np.array(spi_sub['predator']),
                                                        test_size = 0.20, random_state = 100)

    tv = TfidfVectorizer(min_df = 0., max_df = 1., norm = 'l2', use_idf = True, smooth_idf = True)

    train_tfidf = tv.fit_transform(X_train)
    test_tfidf = tv.transform(X_test)

    p_temp = []
    g_temp = []
    for i in range(0,len(predator_words)):
        predator_words[i] = predator_words[i].split("\n")[0]
        goodwords[i] = goodwords[i].split("\n")[0]
        p_temp.append(1)
        g_temp.append(0)

    X_train1 = predator_words + goodwords
    y_train1 = p_temp + g_temp
    X_train1, X_test1, y_train1, y_test1 = train_test_split(np.array(X_train1),
                                            np.array(y_train1), stratify = np.array(y_train1),
                                            test_size = 0.10, random_state = 100)
    tv1 = TfidfVectorizer(min_df = 0., max_df = 1., norm = 'l2', use_idf = True, smooth_idf = True)
    train_tfidf1 = tv1.fit_transform(X_train1)
    test_tfidf1 = tv1.transform(X_test1)

    ####################################

    loaded_model_LR = pickle.load(open(model_sentence_LR, 'rb'))
    loaded_model_wordLR = pickle.load(open(model_words_LR, 'rb'))
    loaded_model_wordSVM = pickle.load(open(model_words_SVM, 'rb'))

    for name in name_msg_dic:
        msg = name_msg_dic[name]
        list_of_predator_messages = []
        for mm in msg:
            print(mm)
            is_predator_number = 0

            messages = mm.split()
            m = np.array(messages)
            m = tv.transform(m)
            predictions_sentence_LR = loaded_model_LR.predict(m)

            m = np.array(messages)
            m = tv1.transform(m)
            predictions_word_LR = loaded_model_wordLR.predict(m)
            predictions_word_SVM = loaded_model_wordSVM.predict(m)

            # For sentence model prediction
            count_of_words = 0
            total_words = len(predictions_sentence_LR)
            indices_predicted_1 = [index for index, element in enumerate(predictions_sentence_LR) if element == 1]
            for i in indices_predicted_1:
                if messages[int(i)] in predator_words:
                    count_of_words+=1
            #print(count_of_words)
            if ((count_of_words/total_words)*100) >= 50:
                print("------------------------Predator ",mm,predictions_word_LR )
                is_predator_number +=1

            # For word model prediction
            indices_predicted_wordlr = [index for index, element in enumerate(predictions_word_LR) if element == 1]
            if len(indices_predicted_wordlr) > (len(predictions_word_LR)-len(indices_predicted_wordlr)):
                print("------------------------Predator ",mm,predictions_word_LR)
                is_predator_number+=1

            indices_predicted_wordsvm = [index for index, element in enumerate(predictions_word_SVM) if element == 1]
            if len(indices_predicted_wordsvm) > (len(predictions_word_SVM)-len(indices_predicted_wordsvm)):
                print("------------------------Predator ",mm,predictions_word_SVM)
                is_predator_number+=1

            if is_predator_number >= 2:
                print("------------------------Message is Predator classified ", mm,is_predator_number)
                list_of_predator_messages.append(mm)


        name_prediction_dic[name] = list_of_predator_messages
    return(name_prediction_dic)





if __name__=="__main__":


    browser = webdriver.Firefox()
    browser.implicitly_wait(5)

    browser.get('https://www.instagram.com/')

    #login_link = browser.find_element_by_xpath("//a[text()='Log In']")
    #login_link.click()
    sleep(2)

    username_input = browser.find_element_by_css_selector("input[name='username']")
    password_input = browser.find_element_by_css_selector("input[name='password']")

    username_input.send_keys("silver_halide_crystals")
    password_input.send_keys("mynameisrajat98")

    login_button = browser.find_element_by_xpath("//button[@type='submit']")
    login_button.click()
    sleep(5)
    print("LOGIN SUCCESSFUL")

    try:
        notnow_button = browser.find_element_by_xpath("//button[@class='sqdOP yWX7d    y3zKF     ']")
        notnow_button.click()
        sleep(3)

        notnow_button = browser.find_element_by_xpath("//button[@class='aOOlW   HoLwm ']")
        notnow_button.click()
        sleep(3)

    except:
        pass

    dm = browser.find_element_by_xpath("//a[@class='xWeGp']")
    dm.click()
    sleep(5)


    #######################################
    while True:
        extract_msg()
        browser.get(browser.current_url);

        with open('chats_demo.json') as f:
            name_msg_dic = json.load(f)

        name_prediction_dic= prediction(name_msg_dic)
        #print(name_prediction_dic)
        for i in name_prediction_dic:
            if len(name_prediction_dic[i])>0:
                print("Possible Predator :" + str(i) + ", messages: " + str(name_prediction_dic[i]) )
        sleep(10)
