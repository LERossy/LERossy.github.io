from pathlib import Path
import numpy as np

# Function to parse a file into a list of words for each document and corresponding list of class labels
def read_documents(doc_file):
    docs = []
    labels = []
    with open(Path(__file__).parent / doc_file, encoding='utf-8') as f:
        for line in f:
            words = line.strip().split()
            docs.append(words[3:])
            labels.append(words[1])
    return docs, labels

# Derive Prior and Likelhood probabilities for all the classes and words - P(cj) and P(wk|cj)
def train_nb(documents, labels):
    # Set up data structures containing probabilities
    P_c = {"neg" : 0.0, "pos" : 0.0}    # Prior
    P_w_c = {"neg" : {}, "pos" : {}}    # Likelihood
    N_c = {"neg" : 0, "pos" : 0}        # Total number of words in each class
    vocabulary = 0                      # Vocabulary count
    
    # Count through all the occurances of classes, and words in each class for each document
    for i in range (len(documents)):
        label = labels[i]
        P_c[label] += 1
        for j in range(len(documents[i])):
            word = documents[i][j]
            # For each new word, increment the vocabulary and enter it in each class dictionary
            if not (word in P_w_c["neg"] or word in P_w_c["pos"]):
                vocabulary += 1
                # Adding the word to both positive and negative ensures scoring doesn't crash 
                # later on if one of the words never occurs for one of the classes
                P_w_c["neg"][word] = 0
                P_w_c["pos"][word] = 0
                
            # Increment the word count for the class
            P_w_c[label][word] += 1
            N_c[label] += 1
                
    # Likelihood: Calculate Smoothed log probabilities for each word given that the class is negative
    for P_wk_c in P_w_c["neg"]:
        P_w_c["neg"][P_wk_c] = np.log( (P_w_c["neg"][P_wk_c] + 1) / float(N_c["neg"] + vocabulary) )
            
    # Likelihood: Calculate Smoothed log probabilities for each word given that the class is positive
    for P_wk_c in P_w_c["pos"]:
        P_w_c["pos"][P_wk_c] = np.log( (P_w_c["pos"][P_wk_c] + 1) / float(N_c["pos"] + vocabulary) )
            
    # Prior: Calculate log probability of each of the possible classes occurring
    for P_cj in P_c:
        P_c[P_cj] = np.log(P_c[P_cj]/float(len(documents)))
                
    return P_c, P_w_c, vocabulary

# Calculate the probability that the given document
# is classified under the given class label with the given prior and likelihood
def score_doc_label(document, label, P_c, P_w_c):
    sum_P_w_c = 0
    # Add all the log probabilities from likelhood of each word in the document given the class label
    for word in document:
        if word in P_w_c[label]:
            sum_P_w_c += P_w_c[label][word]
        
    # Calculate the posterior by adding the prior to the likelihood of the given class label
    P_c_w = P_c[label] + sum_P_w_c
    
    # Returns the log probability (score)
    return P_c_w
    
# Classifies a document based on the given likelhoods and priors
def classify_nb(document, P_c, P_w_c):
    P_neg = score_doc_label(document, "neg", P_c, P_w_c)        # Negative classification score
    P_pos = score_doc_label(document, "pos", P_c, P_w_c)        # Positive classification score
    
    # Return the class with the higher score - probability
    return "neg" if P_neg > P_pos else "pos"
    
# Classify all the documents in a given test list
def classify_documents(docs, P_c, P_w_c):
    classifications = []
    for doc in docs:
        classifications.append(classify_nb(doc, P_c, P_w_c))
        
    return classifications
  
# Calculate the accuracy of a Naive Bayes model classification   
def accuracy(true_labels, guessed_labels):
    correct = 0
    wrong = []
    # Count all the correct classifications
    for i in range(len(true_labels)):
        if true_labels[i] == guessed_labels[i]:
            correct += 1
        else:
            wrong.append(i) # Keep track of misclassifications for error analysis
    
    # Accuracy is the number of correct classifications divided by the total number of classifications
    accuracy = float(correct) / len(true_labels)
    return accuracy, wrong

# Read the documents from file and split them into training and test sets
all_docs, all_labels = read_documents('all_sentiment_shuffled.txt')
split_point = int(0.80*len(all_docs))
train_docs = all_docs[:split_point]
train_labels = all_labels[:split_point]
eval_docs = all_docs[split_point:]
eval_labels = all_labels[split_point:]

# print(train_docs)

# Train the Naive-Bayes Model on the training set
P_c, P_w_c, vocabulary = train_nb(train_docs, train_labels)

# Print the vocabulary as a check
print(vocabulary)
print()

# Sanity check on the scoring - testing and comparing scores for small documents to see if they're reasonable
print( np.exp(score_doc_label(["fun"], "pos", P_c, P_w_c)))
print( np.exp(score_doc_label(["great"], "neg", P_c, P_w_c)))
print( np.exp(score_doc_label(["great"], "pos", P_c, P_w_c)))
print( np.exp(score_doc_label(["bad"], "neg", P_c, P_w_c)))
print( np.exp(score_doc_label(["bad"], "pos", P_c, P_w_c)))
print( np.exp(score_doc_label(["a", "top-quality", "performance"], "pos", P_c, P_w_c)))
print( np.exp(score_doc_label(["a", "top-quality", "performance"], "neg", P_c, P_w_c)))

print()

# Sanity check on classifications - see if the classifications of short documents are reasonable
print(classify_nb(["great"], P_c, P_w_c))
print(classify_nb(["fantastic"], P_c, P_w_c))
print(classify_nb(["bad"], P_c, P_w_c))
print(classify_nb(["absolute"], P_c, P_w_c))
print(classify_nb(["boring"], P_c, P_w_c))

print()

# Classify all the test documents and calculate the accuracy of the model
classifications = classify_documents(eval_docs, P_c, P_w_c)
acc, wrong = accuracy(eval_labels, classifications)
print(acc)
print()

# Look at misclassified documents
# This is a correct classification printed for sanity and comparison
print(eval_docs[0])
print( score_doc_label(eval_docs[0], "neg", P_c, P_w_c)) 
print( score_doc_label(eval_docs[0], "pos", P_c, P_w_c)) 
print( classify_nb(eval_docs[0], P_c, P_w_c))

# Misclassifications
print(eval_docs[wrong[0]])
print( score_doc_label(eval_docs[wrong[0]], "neg", P_c, P_w_c)) 
print( score_doc_label(eval_docs[wrong[0]], "pos", P_c, P_w_c))
print( classify_nb(eval_docs[wrong[0]], P_c, P_w_c))

print(eval_docs[wrong[10]])
print( score_doc_label(eval_docs[wrong[10]], "neg", P_c, P_w_c)) 
print( score_doc_label(eval_docs[wrong[10]], "pos", P_c, P_w_c))
print( classify_nb(eval_docs[wrong[10]], P_c, P_w_c))

print(eval_docs[wrong[100]])
print( score_doc_label(eval_docs[wrong[100]], "neg", P_c, P_w_c)) 
print( score_doc_label(eval_docs[wrong[100]], "pos", P_c, P_w_c))
print( classify_nb(eval_docs[wrong[100]], P_c, P_w_c))