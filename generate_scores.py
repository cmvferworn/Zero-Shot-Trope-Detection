import csv
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import drive

base_dir = "/TropeID/"
folder = "full_run/"
result_file = base_dir + folder + "model_results.csv"
test_file = base_dir + "test.json"
TiMoQ2T = base_dir + "TiMoQ2T.csv"

def main():
    films = get_films(result_file)
    tropes, category, subcategory = get_tropes(TiMoQ2T)

    #Dealing with annoying backslash duplication
    #by removing double quotes from trope names
    temp_tropes = []
    for trope in tropes:
        trope = trope.replace("\\", "")
        trope = trope.replace('"', "")
        temp_tropes.append(trope)
    tropes = temp_tropes

    score_collector = []
    
    #Questions with No Entailment (BoolQ)
    Q_noEnt_binary = find_tropes_Q_noEnt(result_file, films)
    binary_to_trope_file(Q_noEnt_binary, tropes, films, 'Q_noEnt_results')

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Q_noEnt_results', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["QBool", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Q_noEnt_Scores', category, subcategory)

    #Templates - SQUAD 1
    t_ent, t_con_neu, t_ent_neu = find_tropes(result_file, films, 'T1_Score')
    binary_to_trope_file(t_ent, tropes, films, 'Q2T_SQUAD1_ent_results')
    binary_to_trope_file(t_con_neu, tropes, films, 'Q2T_SQUAD1_con_neu_results')
    binary_to_trope_file(t_ent_neu, tropes, films, 'Q2T_SQUAD1_ent_neu_results')

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Q2T_SQUAD1_ent_results', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["Q2TS1Ent", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Q2T_SQUAD1_ent_Scores', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Q2T_SQUAD1_con_neu_results', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["Q2TS1ConNeu", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Q2T_SQUAD1_con_neu_Scores', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Q2T_SQUAD1_ent_neu_results', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["Q2TS1EntNeu", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Q2T_SQUAD1_ent_neu_Scores', category, subcategory)

    #Templates - SQUAD 2
    t_ent, t_con_neu, t_ent_neu = find_tropes(result_file, films, 'T1_S2_Score')
    binary_to_trope_file(t_ent, tropes, films, 'Q2T_SQUAD2_ent_results')
    binary_to_trope_file(t_con_neu, tropes, films, 'Q2T_SQUAD2_con_neu_results')
    binary_to_trope_file(t_ent_neu, tropes, films, 'Q2T_SQUAD2_ent_neu_results')

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Q2T_SQUAD2_ent_results', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["Q2TS2Ent", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Q2T_SQUAD2_ent_Scores', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Q2T_SQUAD2_con_neu_results', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["Q2TS2ConNeu", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Q2T_SQUAD2_con_neu_Scores', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Q2T_SQUAD2_ent_neu_results', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["Q2TS2EntNeu", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Q2T_SQUAD2_ent_neu_Scores', category, subcategory)

    #Templates - no QAing
    t_ent, t_con_neu, t_ent_neu = find_tropes(result_file, films, 'T1_noQ_Score')
    binary_to_trope_file(t_ent, tropes, films, 'E_noQ_ent_results')
    binary_to_trope_file(t_con_neu, tropes, films, 'E_noQ_con_neu_results')
    binary_to_trope_file(t_ent_neu, tropes, films, 'E_noQ_ent_neu_results')

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'E_noQ_ent_results', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["TEnt", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'E_noQ_ent_Scores', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'E_noQ_con_neu_results', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["TConNeu", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'E_noQ_con_neu_Scores', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'E_noQ_ent_neu_results', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["TEntNeu", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'E_noQ_ent_neu_Scores', category, subcategory)

    #QA thresholding
    trope_bin, trope_bin_2 = find_tropes_2(result_file, films, 0.5)
    binary_to_trope_file(trope_bin, tropes, films, 'Thresholding_S1_results_0.5')
    binary_to_trope_file(trope_bin_2, tropes, films, 'Thresholding_S2_results_0.5')

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S1_results_0.5', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS1.5", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S1_Scores_0.5', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S2_results_0.5', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS2.5", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S2_Scores_0.5', category, subcategory)

    #QA thresholding
    trope_bin, trope_bin_2 = find_tropes_2(result_file, films, 0.3)
    binary_to_trope_file(trope_bin, tropes, films, 'Thresholding_S1_results_0.3')
    binary_to_trope_file(trope_bin_2, tropes, films, 'Thresholding_S2_results_0.3')

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S1_results_0.3', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS1.3", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S1_Scores_0.3', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S2_results_0.3', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS2.3", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S2_Scores_0.3', category, subcategory)

    #QA thresholding
    trope_bin, trope_bin_2 = find_tropes_2(result_file, films, 0.4)
    binary_to_trope_file(trope_bin, tropes, films, 'Thresholding_S1_results_0.4')
    binary_to_trope_file(trope_bin_2, tropes, films, 'Thresholding_S2_results_0.4')

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S1_results_0.4', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS1.4", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S1_Scores_0.4', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S2_results_0.4', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS2.4", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S2_Scores_0.4', category, subcategory)

    #QA thresholding
    trope_bin, trope_bin_2 = find_tropes_2(result_file, films, 0.2)
    binary_to_trope_file(trope_bin, tropes, films, 'Thresholding_S1_results_0.2')
    binary_to_trope_file(trope_bin_2, tropes, films, 'Thresholding_S2_results_0.2')

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S1_results_0.2', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS1.2", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S1_Scores_0.2', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S2_results_0.2', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS2.2", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S2_Scores_0.2', category, subcategory)

    #QA thresholding
    trope_bin, trope_bin_2 = find_tropes_2(result_file, films, 0.6)
    binary_to_trope_file(trope_bin, tropes, films, 'Thresholding_S1_results_0.6')
    binary_to_trope_file(trope_bin_2, tropes, films, 'Thresholding_S2_results_0.6')

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S1_results_0.6', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS1.6", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S1_Scores_0.6', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S2_results_0.6', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS2.6", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S2_Scores_0.6', category, subcategory)

    #QA thresholding
    trope_bin, trope_bin_2 = find_tropes_2(result_file, films, 0.1)
    binary_to_trope_file(trope_bin, tropes, films, 'Thresholding_S1_results_0.1')
    binary_to_trope_file(trope_bin_2, tropes, films, 'Thresholding_S2_results_0.1')

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S1_results_0.1', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS1.1", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S1_Scores_0.1', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S2_results_0.1', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS2.1", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S2_Scores_0.1', category, subcategory)

    #QA thresholding
    trope_bin, trope_bin_2 = find_tropes_2(result_file, films, 0.05)
    binary_to_trope_file(trope_bin, tropes, films, 'Thresholding_S1_results_0.05')
    binary_to_trope_file(trope_bin_2, tropes, films, 'Thresholding_S2_results_0.05')

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S1_results_0.05', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS1.05", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S1_Scores_0.05', category, subcategory)

    tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring = compare_results(test_file, 'Thresholding_S2_results_0.05', tropes, films)
    f1_score, accuracy, f1_score_tropes, accuracy_tropes = calculate_scores(tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring, tropes)
    score = ["ThreshS2.05", f1_score, accuracy]
    score_collector.append(score)
    generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, 'Thresholding_S2_Scores_0.05', category, subcategory)


    file_csv = base_dir + folder + 'TotalF1Accuracy.csv'
    field_names = ['pipeline', 'f1_score', 'accuracy']
    with open(file_csv,'w') as csvfile:
        writer = csv.writer(csvfile)
        header = ['trope', 'f1_score', 'accuracy']
        writer.writerow(header)
        for scr in score_collector:
            writer.writerow(scr)


    #plot_results(f1_collector, acc_collector)

#headings = ['Q1_Score', 'Q1_Answer', 'Q1_S2_Score', 'Q1_S2_Answer',
#'Q1_noEnt_Score_Yes', 'Q1_noEnt_Score_No', 
#'T1_Score_Ent', 'T1_Score_Neu', 'T1_Score_Con', 
#'T1_S2_Score_Ent', 'T1_S2_Score_Neu', 'T1_S2_Score_Con', 
#'T1_noQ_Score_Ent', 'T1_noQ_Score_Neu', 'T1_noQ_Score_Con']

def plot_results(f1_collector, acc_collector):
    f1_names = list(f1_collector.keys())
    f1_values = list(f1_collector.values())
    
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(f1_collector)), f1_values, tick_label=f1_names)
    plt.show()

    names = list(acc_collector.keys())
    values = list(acc_collector.values())
    
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(acc_collector)), values, tick_label=names)
    plt.show()

    plt.figure(figsize=(15, 15))
    plt.scatter(values, f1_values)
    i = 0
    for name in names:
        plt.annotate(name, (values[i], f1_values[i]), textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center')
        i = i + 1
    plt.show()





#def plot_tropes():
#    print()
def find_tropes_2(result_file, films, threshold):
    S1_score = 'Q1_Score' 
    S2_score = 'Q1_S2_Score'
    trope_bin = []
    trope_bin_2 = []
    with open(result_file, 'r') as f:
        read_result = csv.DictReader(f)
        
        #Iterate over movie results
        for row in read_result:
            trope_bin_movie = []
            trope_bin_movie_2 = []
            i = 1
            #Iterate over trope results
            while i < 96:
                S1 = S1_score.replace("1", str(i))
                S2 = S2_score.replace("1", str(i))
                if float(row[S1]) > threshold:
                    trope_bin_movie.append(1)
                else:
                    trope_bin_movie.append(0)
                if float(row[S2]) > threshold:
                    trope_bin_movie_2.append(1)
                else:
                    trope_bin_movie_2.append(0)
                i = i+1
            trope_bin.append(trope_bin_movie)
            trope_bin_2.append(trope_bin_movie_2)
    
    return trope_bin, trope_bin_2


def find_tropes(result_file, films, prefix):
    ent_score = prefix + '_Ent' 
    neu_score = prefix + '_Neu'
    con_score = prefix + '_Con'

    trope_bin_ent = []
    trope_bin_con_neu = []
    trope_bin_ent_neu = []

    with open(result_file, 'r') as f:
        read_result = csv.DictReader(f)
  
        #Iterate over movie results
        for row in read_result:
            trope_bin_movie_ent = []
            trope_bin_movie_con_neu = []
            trope_bin_movie_ent_neu = []
            i = 1
            #Iterate over trope results
            while i < 96:
                ent = ent_score.replace("1", str(i))
                neu = neu_score.replace("1", str(i))
                con = con_score.replace("1", str(i))
                #Entailment larger than contradiction. Ignore neutral
                if float(row[ent]) > float(row[con]):
                    trope_bin_movie_ent.append(1)
                else:
                    trope_bin_movie_ent.append(0)
                #Entailment larger than contradiction and neutral. (Contradiction = Contradiction + Neutral)
                if float(row[ent]) > float(row[con]) and float(row[ent]) > float(row[neu]):
                    trope_bin_movie_con_neu.append(1)
                else: 
                    trope_bin_movie_con_neu.append(0)
                #Contradiction smaller than neutral and entailment (Entailment = Neutral + Entailment)
                if float(row[con]) < float(row[ent]) and float(row[con]) < float(row[neu]):
                    trope_bin_movie_ent_neu.append(1)
                else:
                    trope_bin_movie_ent_neu.append(0)
                i = i+1
            trope_bin_ent.append(trope_bin_movie_ent)
            trope_bin_con_neu.append(trope_bin_movie_con_neu)
            trope_bin_ent_neu.append(trope_bin_movie_ent_neu)
    #print("Trope bin: ")

    return trope_bin_ent, trope_bin_con_neu, trope_bin_ent_neu

def generate_score_file(f1_score, accuracy, f1_score_tropes, accuracy_tropes, tropes, f_name, category, subcategory):
    file_loc = base_dir + folder + f_name + '.txt'
    file_csv = base_dir + folder + f_name + '.csv'
    with open(file_loc, 'w') as f, open(file_csv,'w') as csv_f:
        csv_writer = csv.writer(csv_f)
        header = ['category', 'sub-category', 'trope', 'f1_score', 'accuracy']
        csv_writer.writerow(header)

        f.write('Total F1 Score: ' + str(f1_score) + '\n')
        f.write('Total Accuracy: ' + str(accuracy) + '\n \n')

        i = 0
        for trope in tropes:
            f.write(trope + ' F1 Score: ' + str(f1_score_tropes[trope]) + '\n')
            f.write(trope + ' Accuracy Score: ' + str(accuracy_tropes[trope]) + '\n \n')
            row = [category[i], subcategory[i], trope, f1_score_tropes[trope], accuracy_tropes[trope]]
            i = i+1
            csv_writer.writerow(row)


#headings = ['Q1_Score', 'Q1_Answer', 'Q1_S2_Score', 'Q1_S2_Answer','Q1_noEnt_Score_Yes', 'Q1_noEnt_Score_No', 'T1_Score_Ent', 'T1_Score_Neu', 'T1_Score_Con', 'T1_S2_Score_Ent', 'T1_S2_Score_Neu', 'T1_S2_Score_Con', 'T1_noQ_Score_Ent', 'T1_noQ_Score_Neu', 'T1_noQ_Score_Con'
def compare_results(test_file, result_file, tropes, films):
    #Over all and per trope
    tru_pos = 0
    fal_pos = 0
    tru_neg = 0
    fal_neg = 0
    
    res_file = base_dir + folder + result_file + ".json"

    true_file = open(test_file)
    guess_file = open(res_file)
    true_data = json.load(true_file)
    guess_data =json.load(guess_file)

    rows, cols = (len(tropes), 4)
    trope_scoring = [[0 for i in range(cols)] for j in range(rows)]

    for film in films:
        true_tropes = true_data.get(film)
        guess_tropes = guess_data.get(film)
        counter = 0
        for trope in tropes:
            if trope in true_tropes and trope in guess_tropes:
                tru_pos = tru_pos + 1
                trope_scoring[counter][0] = trope_scoring[counter][0]+1
            elif trope in true_tropes and trope not in guess_tropes:
                fal_neg = fal_neg + 1
                trope_scoring[counter][1] = trope_scoring[counter][1]+1
            elif trope not in true_tropes and trope not in guess_tropes:
                tru_neg = tru_neg + 1
                trope_scoring[counter][2] = trope_scoring[counter][2]+1
            elif trope not in true_tropes and trope in guess_tropes:
                fal_pos = fal_pos + 1
                trope_scoring[counter][3] = trope_scoring[counter][3]+1
            else:
                print('f1 scoring error.')
            counter = counter + 1
      
    return tru_pos, fal_pos, tru_neg, fal_neg, trope_scoring

def binary_to_trope_file(trope_bin, tropes, films, f_name):
    file_loc = base_dir + folder + f_name + '.json'
    trope_results = {}
    j = 0
    for row in trope_bin:
        found_tropes = []
        i = 0
        for elem in row:
            if elem == 1: 
                found_tropes.append(tropes[i])
            i = i + 1
        film = films[j]
        trope_results[film] = found_tropes
        j = j + 1
    dict_to_json(trope_results, file_loc)

def find_tropes_Q_noEnt(result_file, films):
    yes_score = 'Q1_noEnt_Score_Yes' 
    no_score = 'Q1_noEnt_Score_No'
    trope_bin = []
    with open(result_file, 'r') as f:
        read_result = csv.DictReader(f)
        
        #Iterate over movie results
        for row in read_result:
            trope_bin_movie = []
            i = 1
            #Iterate over trope results
            while i < 96:
                yes = yes_score.replace("1", str(i))
                no = no_score.replace("1", str(i))
                if float(row[yes]) > float(row[no]):
                    trope_bin_movie.append(1)
                else:
                    trope_bin_movie.append(0)
                i = i+1
            trope_bin.append(trope_bin_movie)
    return trope_bin

def get_films(data):
    data_file = pd.read_csv(data)
    films = data_file['film'].tolist()
    #f = open(data)
    #film_data = json.load(f)
    #films = list(film_data.keys())
    return films

def get_tropes(f_name):
    tropes = []
    category = []
    subcategory = []
    with open(f_name, newline='') as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            tropes.append(row['trope'])
            category.append(row['category'])
            subcategory.append(row['sub-category'])
    return tropes, category, subcategory

def calculate_scores(true_positive, false_positive, true_negative, false_negative, trope_scoring, tropes):
    #Calculate F1 Scores
    if (true_positive+false_positive) > 0:
        precision = true_positive/(true_positive + false_positive)
    else:
        precision = 0
    if (true_positive+false_negative) > 0:
        recall = true_positive/(true_positive + false_negative)
    else: 
        recall = 0
    if (precision+recall) > 0:
        f1_score = 2 * (precision * recall)/(precision + recall)
    else:
        f1_score = 0

    #Calculate Accuracy
    accuracy = (true_positive + true_negative)/(true_positive + true_negative + false_positive + false_negative)

    f1_score_tropes = {}
    accuracy_tropes = {}
    counter = 0

    for trope in tropes:
        tru_pos = trope_scoring[counter][0]
        fal_neg = trope_scoring[counter][1]
        tru_neg = trope_scoring[counter][2]
        fal_pos = trope_scoring[counter][3]
        #print("Tru Pos, Fal Neg, Tru Neg, Fal Pos: " + str(true_positive) + ", " + str(false_negative) + ", " + str(true_negative) + ", " + str(false_positive) )
        if (tru_pos+fal_pos) > 0:
          precision = tru_pos/(tru_pos + fal_pos)
        else:
          precision = 0
        if (tru_pos+fal_neg) > 0:
          recall = tru_pos/(tru_pos + fal_neg)
        else: 
          recall = 0
        if (precision+recall) > 0:
          trope_f1 = 2 * (precision * recall)/(precision + recall)
          f1_score_tropes[trope] = trope_f1
        else:
          f1_score_tropes[trope] = 0
        accuracy_tropes[trope] = (tru_pos + tru_neg)/(tru_pos + tru_neg + fal_pos + fal_neg)
        counter = counter + 1

    return f1_score, accuracy, f1_score_tropes, accuracy_tropes

def dict_to_json(data, f_name):
    json_object = json.dumps(data, indent=4)
    with open(f_name, "w") as outfile:
        outfile.write(json_object)

def result_csv_headings():
    field_names = ['film']
    headings = ['Q1_Score', 'Q1_Answer', 'Q1_S2_Score', 'Q1_S2_Answer','Q1_noEnt_Score_Yes', 'Q1_noEnt_Score_No', 'T1_Score_Ent', 'T1_Score_Neu', 'T1_Score_Con', 'T1_S2_Score_Ent', 'T1_S2_Score_Neu', 'T1_S2_Score_Con', 'T1_noQ_Score_Ent', 'T1_noQ_Score_Neu', 'T1_noQ_Score_Con']
    for names in headings:
        field_names.append(names)
    i = 2
    while i < 96:
        for name in headings:
            new_name = name.replace("1", str(i))
            field_names.append(new_name)
        i = i+1
    return field_names

main()
