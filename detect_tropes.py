import json
import csv
import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

base_dir = "./"
TiMoQ2T = "../input/timoq2t/TiMoQ2T.csv"
test_file = "../input/timoq2t/test.json"


def detect_tropes():
    
    #Name folder for attempt
    folder = ""

    #Initiate results CSV file
    field_names = result_csv_headings()
    model_results = base_dir + folder + "/model_results.csv"
    if os.path.isfile(model_results) == False:
        create_result_csv(folder, field_names)

    #Get data
    questions, tropes, templates, questions_noEnt = grab_questions(TiMoQ2T)
    film_trope_data = test_file
    films = grab_films(film_trope_data)

    #For quick testing:
    #questions = ["Who is the cause of all the bad happenings?", "Who is completely obnoxious?"]
    #tropes = ["Big Bad", "Jerkass"]
    #templates = ["[blank] is the cause of all the bad happenings.", "[blank] is completely obnoxious."]
    #questions_noEnt = ["Is someone the cause of all the bad happenings?", "Is someone completely obnoxious?"]


    #Add "In the text " to the beginning of all templates.
    templates = add_text(templates)

    #Change to templates for no question answering
    templates_noQ = remove_blanks(templates, questions)

    #QA Distilbert SQuAD Model Pipeline
    qa_model = pipeline("question-answering")
    #QA Roberta SQuAD2 Model Pipeline
    roberta_qa_model = pipeline("question-answering", model = "deepset/tinyroberta-squad2", tokenizer = "deepset/tinyroberta-squad2")
    #Entail Model
    entail_tokenizer = AutoTokenizer.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
    entail_model = AutoModelForSequenceClassification.from_pretrained("ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
    #Roberta BoolQ Model
    boolq_model = AutoModelForSequenceClassification.from_pretrained("shahrukhx01/roberta-base-boolq")
    #boolq_model.to(device) 
    boolq_tokenizer = AutoTokenizer.from_pretrained("shahrukhx01/roberta-base-boolq")

    #Cut Number of films for testing purposes
    #films = films[560:620]
    
    #films.reverse()

    film_counter = 1
    cached = load_cache(folder)
 

    for film in films:
        if film in cached:
            print(film + " found in cache.")
            film_counter = film_counter + 1
            continue
    
        context = grab_context(film)

        question_counter = 0
        template_counter = 0

        result_dict = {}
        result_dict["film"] = film

        print("Processing the film: " + film)
        print("Film count: " + str(film_counter) + " out of " + str(len(films)))
        
        for question in questions:
            question_counter = question_counter + 1
            #print("Question " + str(question_counter) + " out of " + str(len(questions)))
            #Check if There is a Question to Answer
            if question != '[ENTAIL]':
                #DistilBert SQuAD QA Pipeline
                score, answer = QuestionAnswer(qa_model, question, context)
                result_dict["Q" + str(question_counter) + "_Score"] = score
                result_dict["Q" + str(question_counter) + "_Answer"] = answer
                #RoBERTa SQuAD 2 QA Pipeline
                score, answer = QuestionAnswer(roberta_qa_model, question, context)
                result_dict["Q" + str(question_counter) + "_S2_Score"] = score
                result_dict["Q" + str(question_counter) + "_S2_Answer"] = answer

            else:
                result_dict["Q" + str(question_counter) + "_Score"] = 0
                result_dict["Q" + str(question_counter) + "_Answer"] = "no_answer"
                result_dict["Q" + str(question_counter) + "_S2_Score"] = 0
                result_dict["Q" + str(question_counter) + "_S2_Answer"] = "no_answer"
        
        question_counter = 0

        #Answer Questions without Entailment with BoolQ Model
        for question in questions_noEnt:
            question_counter = question_counter + 1
            #print("Question (no Ent) " + str(question_counter) + " out of " + str(len(questions)))
            yes, no = BoolQ(boolq_model, boolq_tokenizer, question, context)
            result_dict["Q" + str(question_counter) + "_noEnt_Score_Yes"] = yes
            result_dict["Q" + str(question_counter) + "_noEnt_Score_No"] = no
            
        
        for template in templates:
            trope = tropes[template_counter]
            template_counter = template_counter + 1
            #print("Template " + str(template_counter) + " out of " + str(len(templates)))

            #Fill Templates Using DistilBert SQuAD v1
            try:
                fill = result_dict["Q" + str(template_counter) + "_Answer"]
            except KeyError:
                result_dict['T'+ str(template_counter) +'_Score_Ent'] = 0
                result_dict['T'+ str(template_counter) +'_Score_Neu'] = 0
                result_dict['T'+ str(template_counter) +'_Score_Con'] = 0
            else:
                result = Entailment(entail_model, entail_tokenizer, template, fill, context)
                result_dict['T'+ str(template_counter) +'_Score_Ent'] = result[0]
                result_dict['T'+ str(template_counter) +'_Score_Neu'] = result[1]
                result_dict['T'+ str(template_counter) +'_Score_Con'] = result[2]
            
            #Fill templates using tinyRoBERTa SQuAD v2 
            try:
                fill = result_dict["Q" + str(template_counter) + "_Answer"]
            except KeyError:
                result_dict['T'+ str(template_counter) +'_S2_Score_Ent'] = 0
                result_dict['T'+ str(template_counter) +'_S2_Score_Neu'] = 0
                result_dict['T'+ str(template_counter) +'_S2_Score_Con'] = 0
            else:
                result = Entailment(entail_model, entail_tokenizer, template, fill, context)
                result_dict['T'+ str(template_counter) +'_S2_Score_Ent'] = result[0]
                result_dict['T'+ str(template_counter) +'_S2_Score_Neu'] = result[1]
                result_dict['T'+ str(template_counter) +'_S2_Score_Con'] = result[2]



        template_counter = 0

        for template in templates_noQ:
            #print("Template_noQ " + str(template_counter) + " out of " + str(len(templates)))
            trope = tropes[template_counter]
            template_counter = template_counter + 1

            fill = "none_fill"
            #result = Entailment(entail_model, entail_tokenizer, template, fill, context)
            result = Entailment(entail_model, entail_tokenizer, template, fill, context)
            #result_dict[film][template] = result

            result_dict['T'+ str(template_counter) +'_noQ_Score_Ent'] = result[0]
            result_dict['T'+ str(template_counter) +'_noQ_Score_Neu'] = result[1]
            result_dict['T'+ str(template_counter) +'_noQ_Score_Con'] = result[2]

            
        append_result(model_results, result_dict, field_names)
        cache_film(film, folder)
        film_counter = film_counter + 1

def load_cache(folder):
    cached_films = []
    cache_file = "/cached_films.txt"
    cache_loc = base_dir + folder + cache_file
    try:
        with open(cache_loc, "r") as filehandle:
            for line in filehandle:
                #remove line break
                curr_place = line[:-1]
                #append film to list
                cached_films.append(curr_place)
    except FileNotFoundError:
        print("No cached films.")
        
    return cached_films

def cache_film(film, folder):
    cache_file = "/cached_films.txt"
    cache_loc = base_dir + folder + cache_file
    with open(cache_loc, 'a') as filehandle:
        filehandle.write(film)
        filehandle.write("\n")
        filehandle.close()

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

def create_result_csv(folder, field_names):
    model_results = base_dir + folder + "/model_results.csv"
    with open(model_results, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(field_names)

def append_result(model_results, result_dict, field_names):
    with open(model_results, 'a') as f:
        writer = csv.DictWriter(f, fieldnames = field_names)
        writer.writerow(result_dict)
        f.close()

def dict_to_json(data, f_name):
    json_object = json.dumps(data, indent=4)
    with open(f_name, "w") as outfile:
        outfile.write(json_object)

def grab_context(film):
    context_dir = '../input/timoq2t/synopses/synopses/'
    f_ext = '.json'
    file_name = context_dir + film + f_ext
    f = open(file_name)
    context_data = json.load(f)
    context = context_data.get("plot")
    return context

def add_text(templates):
    temps = []
    for template in templates:
        low_temp = template.lower()
        temp = "In the text " + low_temp
        temps.append(temp)
    return temps

def remove_blanks(templates, questions):
    count = 0
    templates_noQ = []
    for template in templates:
        templates_noQ.append(template)
    for question in questions:
        all_words = question.split()
        first_word= all_words[0]
        temp = templates_noQ[count]
        if first_word == "Who":
            if "[blank]" in temp:
                templates_noQ[count] = temp.replace("[blank]", "someone")
        if first_word == "What":
            if "[blank]" in temp:
                templates_noQ[count] = temp.replace("[blank]", "there")
        count = count + 1
    return templates_noQ

def grab_questions(f_name):
    questions = []
    tropes = []
    templates = []
    questions_noEnt = []
    with open(f_name, newline='') as in_file:
        reader = csv.DictReader(in_file)
        for row in reader:
            questions.append(row['question'])
            tropes.append(row['trope'])
            templates.append(row['template'])
            questions_noEnt.append(row['question_noEnt'])
    return questions, tropes, templates, questions_noEnt

def grab_films(data):
    f = open(data)
    film_data = json.load(f)
    films = list(film_data.keys())
    return films

def QuestionAnswer(qa_model, question, context):
    result = qa_model(context = context, question = question)
    
    score = result['score']
    answer = result['answer']

    return score, answer

def QARoberta(model, tokenizer, question, context):
    result = model(context = context, question = question)

    score = result['score']
    answer = result['answer']
    
    return score, answer

def Entailment(model, tokenizer, template, fill, context):
    premise = context
    if "[blank]" in template:
        hypothesis = template.replace("[blank]", fill)
    else:
        hypothesis = template
  
    tokenized_input_seq_pair = tokenizer.encode_plus(premise, hypothesis , return_token_type_ids=True, truncation=True)
    input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
  
    token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
    attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=None)
    result = torch.softmax(outputs[0], dim=1)[0].tolist()
    

    return result

def BoolQ(model, tokenizer, question, context):
    sequence = tokenizer.encode_plus(question, context, return_tensors="pt", truncation=True)['input_ids'] #.to(device)
    logits = model(sequence)[0]
    probabilities = torch.softmax(logits, dim=1).tolist()[0]   #.detach().cpu().tolist()[0]
    
    yes = probabilities[1]
    no = probabilities[0]

    return yes, no
    #result = {}
    #result["yes"] = probabilities[1]
    #result["no"] = probabilities[0]
    #return result

detect_tropes()
