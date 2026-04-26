 Project Title

Word2Vec Implementation in C

 Overview

Word2Vec is a popular word embedding technique in Natural Language Processing (NLP) that represents words as vectors in a continuous vector space. Words that share similar meanings or appear in similar contexts are positioned close to each other in this space.

This project implements the Word2Vec algorithm (introduced by Google in 2013) from scratch using the C programming language, without relying on any external machine learning libraries. The goal is to understand the internal working of word embeddings at a low level.

 Objectives

The main objective of this project is to:

Learn and implement word embedding techniques from scratch
Represent vocabulary words as dense vectors
Capture semantic and contextual relationships between words
Build a lightweight NLP tool using pure C
⚙️ Features
1.  Word Similarity

Computes similarity between two words using cosine similarity.

Examples:

similarity(road, bus) ≈ 0.36
similarity(good, great) ≈ 0.89
2. Vector Arithmetic (Word Analogy)

Performs mathematical operations on word vectors to capture relationships.

Examples:

king − man + woman ≈ queen
running − run + walk ≈ walking
3.  Similar Words Retrieval

Finds the most contextually similar words to a given input word.

Examples:

good → (great, best, excellent)
king → (monarch, prince, ruler)
4.  Sentence Similarity

Measures similarity between sentences by averaging word vectors.

Examples:

"dog is a loyal animal"
"cat is a domestic animal"
→ similarity ≈ 0.7
5. Add One Out Detection

Identifies the word that does not belong in a group.

Example:

(apple, banana, mango, car) → car
6. Word Clustering:

Groups words into K clusters based on vector similarity (e.g., K-means).

Examples:

K = 2 → {bus}, {apple, mango}
K = 3 → {grape}, {bus, car}, {lion}

7. 🔎 Semantic Search

Searches the most contextually relevant sentence from the .txt file based on the user input.

Example:

File contains:

"The first mechanical computer was invented by Charles Babbage."
"Fiber optic cables transmit global internet data across the ocean floor using pulses of light."
"Artificial intelligence algorithms are trained by finding hidden patterns in massive datasets."
"Modern smartphones contain microchips that are exponentially more powerful than early supercomputers."
"Cybersecurity experts use encryption to protect sensitive digital information from hackers."

Input:
Computer is a very useful device

Output:
The first mechanical computer was invented by Charles Babbage.


8. Concept Connector:

Connects multiple related input words and predicts a common concept or category.

Example:
Input:castle crown throne
Output:royalty / king / monarch


PROJECT MATERIAL:
Language: C

Topics: Natural Language Processing, Word2Vec, File Handling, Dynamic Memory Management

Technology: GitHub

How to run:
git clone https://github.com/your-username/word2vec.c.git
cd word2vec.c
ls
word2vec.c
vectors.txt
gcc word2vec.c -o word2vec -lm
./word2vec


Dataset source:
https://www.kaggle.com/datasets/sugataghosh/google-word2vec
