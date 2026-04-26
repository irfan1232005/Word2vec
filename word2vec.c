//word2vec algorithm of word embeddding in Natural language processing 
//developing it from scratch 


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//declaring global variable
#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define TABLE_SIZE 1e8

long long layer1_size = 300;
long long min_count = 50;
//maximum window size
int window = 5;
//negative sample number
int negative = 5;
float alpha = 0.025;
float sample = 1e-3;

struct vocab_word 
{
    long long cn;
    char *word;
};

struct vocab_word *vocab;
int *vocab_hash;
int *table;
float *syn0;
float *syn1neg;
float *expTable;

const int vocab_hash_size = 30000000;
long long vocab_max_size = 1000;
long long vocab_size = 0;
long long train_words = 0;
long long word_count_actual = 0;
long long min_reduce = 1;

void AllocateExpTable()
{
    unsigned long size;
    size = (EXP_TABLE_SIZE + 1) * sizeof(float);
    expTable = (float *)malloc(size);
    
    if (expTable == NULL) 
    {
        printf("Memory allocation failed for ExpTable");
        printf("/n");
        exit(1);
    }
}

void FillExpTable()
{
    int i;
    float f;
    float numerator;
    float denominator;
    float exponent_val;
    float range_val;
    float normalized_i;

    i = 0;
    while (i < EXP_TABLE_SIZE) 
    {
        normalized_i = (float)i / (float)EXP_TABLE_SIZE;
        range_val = normalized_i * 2 - 1;
        exponent_val = range_val * MAX_EXP;
        
        f = exp(exponent_val);
        
        numerator = f;
        denominator = f + 1;
        
        expTable[i] = numerator / denominator;
        i++;
    }
}

void InitExpTable() 
{
    AllocateExpTable();
    FillExpTable();
}
//sigmoid function implementation

static inline float GetSigmoid(float f) 
{
    float result;
    int index;
    float formula_val;
    float mid_calc;

    if (f > MAX_EXP) 
    {
        return 1.0f;
    }
    
    if (f < -MAX_EXP) 
    {
        return 0.0f;
    }
    
    mid_calc = f + MAX_EXP;
    formula_val = mid_calc * (EXP_TABLE_SIZE / MAX_EXP / 2);
    index = (int)formula_val;
    
    if (index < 0)
    {
        index = 0;
    }
    
    if (index >= EXP_TABLE_SIZE)
    {
        index = EXP_TABLE_SIZE - 1;
    }

    result = expTable[index];
    
    return result;
}


// This function calculates how much error needs to be sent back to the Hidden Layer.
static inline void Backpropagation(long long l2, float g, float *neu1_error) 
{
    int c;
    float product;
    
    // Loop through neurons
    c = 0;
    while (c < layer1_size) 
    {
        // Calculate error contribution from Output Layer weights
        product = g * syn1neg[l2 + c];
        
        // Accumulate error for the Hidden Layer (to be updated later)
        neu1_error[c] = neu1_error[c] + product;
        c++;
    }
}


// This function updates the weights of the Output Layer using the explicit Hidden Layer.
static inline void GradientDescentOptimization(long long l2, float g, float *hidden_layer) 
{
    int c;
    float product;
    
    c = 0;
    while (c < layer1_size) 
    {
        // Calculate the adjustment value: Gradient * Hidden Layer Activation
        product = g * hidden_layer[c];
        
        // Update Rule: NewWeight = OldWeight + Adjustment
        syn1neg[l2 + c] = syn1neg[l2 + c] + product;
        c++;
    }
}


//  (Artificial Neural Network)

static inline void ann(long long input_idx, long long target_idx, int label, float alpha, float *neu1_error) 
{
    long long l1; // Index for Input Layer weights
    long long l2; // Index for Output Layer weights
    float f;      // Raw Output Score
    float g;      // Gradient
    float pred;   // Predicted Probability
    float diff;   // Error
    int c;
    
    l1 = input_idx * layer1_size; 
    l2 = target_idx * layer1_size; 
    f = 0;
    
    //the hidden layer 
    float hidden_layer[3000]; 
    
    
    // Copy the activated input word's weights directly into our explicit Hidden Layer
    c = 0;
    while (c < layer1_size) 
    {
        hidden_layer[c] = syn0[l1 + c];
        c++;
    }
    

    // Perform Dot Product: Hidden Layer * Output Weights (syn1neg)
    c = 0;
    while (c < layer1_size) 
    {
        f = f + (hidden_layer[c] * syn1neg[l2 + c]);
        c++;
    }

    // Apply Activation Function  to get probability (0 to 1)
    pred = GetSigmoid(f);
    
    // Calculate Prediction Error using error function
    //binary entropy
    diff = label - pred;
    
    // Calculate Gradient (Error * Learning Rate
    g = diff * alpha;

    
    Backpropagation(l2, g, neu1_error);

    // optimization
    GradientDescentOptimization(l2, g, hidden_layer);
}

void ReadCharFromFile(FILE *fin, int *ch)
{
    *ch = fgetc(fin);
}
//word reading from the corpus
void ReadWord(char *word, FILE *fin) 
{
    int a;
    int ch;
    
    a = 0;
    while (!feof(fin)) 
    {
        ReadCharFromFile(fin, &ch);
        
        if (ch == 13) 
        {
            continue;
        }

        if (ch == ' ') 
        {
            if (a > 0) 
            {
                break;
            }
            continue;
        }
        
        if (ch == '\t') 
        {
            if (a > 0) 
            {
                break;
            }
            continue;
        }
        
        if (ch == '\n') 
        {
            if (a > 0) 
            {
                ungetc(ch, fin);
                break;
            }
            strcpy(word, "</s>");
            return;
        }

        word[a] = ch;
        a++;
        
        if (a >= MAX_STRING - 1) 
        {
            a--;
        }
    }
    word[a] = 0;
}
//calculate hash of the words
unsigned long long CalculateHash(char *word, int len)
{
    unsigned long long hash;
    int a;
    
    hash = 0;
    a = 0;
    while (a < len) 
    {
        hash = hash * 257 + word[a];
        a++;
    }
    return hash;
}
//gets the hash of every required words
int GetWordHash(char *word) 
{
    unsigned long long hash;
    int len;
    
    len = strlen(word);
    hash = CalculateHash(word, len);
    hash = hash % vocab_hash_size;
    
    return (int)hash;
}

int SearchVocab(char *word) 
{
    unsigned int hash;
    int index;
    
    hash = GetWordHash(word);
    
    while (1) 
    {
        index = vocab_hash[hash];
        
        if (index == -1) 
        {
            return -1;
        }
        
        if (!strcmp(word, vocab[index].word)) 
        {
            return index;
        }
        
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}

void EnsureVocabCapacity()
{
    long long new_size;
    
    if (vocab_size + 2 >= vocab_max_size) 
    {
        vocab_max_size = vocab_max_size + 1000;
        new_size = vocab_max_size * sizeof(struct vocab_word);
        vocab = (struct vocab_word *)realloc(vocab, new_size);
        
        if (vocab == NULL) 
        {
            printf("Memory allocation failed for vocab resize\n");
            exit(1);
        }
    }
}

int AddWordToVocab(char *word) 
{
    unsigned int hash;
    int length;
    
    hash = GetWordHash(word);
    length = strlen(word) + 1;
    
    if (length > MAX_STRING) 
    {
        length = MAX_STRING;
    }
    
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    
    if (vocab[vocab_size].word == NULL) 
    {
        printf("Memory allocation failed for word string\n");
        exit(1);
    }
    
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    
    EnsureVocabCapacity();
    
    while (vocab_hash[hash] != -1) 
    {
        hash = (hash + 1) % vocab_hash_size;
    }
    
    vocab_hash[hash] = vocab_size - 1;
    
    return vocab_size - 1;
}

int VocabCompare(const void *a, const void *b) 
{
    long long count_a;
    long long count_b;
    struct vocab_word *word_a;
    struct vocab_word *word_b;
    
    word_a = (struct vocab_word *)a;
    word_b = (struct vocab_word *)b;
    
    count_a = word_a->cn;
    count_b = word_b->cn;
    
    return count_b - count_a;
}

void RebuildHashTable()
{
    int i;
    unsigned int hash;
    int a;
    
    i = 0;
    while (i < vocab_hash_size) 
    {
        vocab_hash[i] = -1;
        i++;
    }
    
    a = 0;
    while (a < vocab_size) 
    {
        hash = GetWordHash(vocab[a].word);
        
        while (vocab_hash[hash] != -1) 
        {
            hash = (hash + 1) % vocab_hash_size;
        }
        
        vocab_hash[hash] = a;
        a++;
    }
}

void SortVocab() 
{
    long long remaining_words;
    int a;
    int b;
    int i;
    unsigned int hash;
    long long size;
    
    remaining_words = vocab_size - 1;
    
    if (remaining_words > 0) 
    {
        qsort(&vocab[1], remaining_words, sizeof(struct vocab_word), VocabCompare);
    }
    
    i = 0;
    while (i < vocab_hash_size) 
    {
        vocab_hash[i] = -1;
        i++;
    }
    
    size = vocab_size;
    train_words = 0;
    
    a = 0;
    while (a < size) 
    {
        if ((vocab[a].cn < min_count) && (a != 0)) 
        {
            vocab_size--;
            free(vocab[a].word);
        } 
        else 
        {
            hash = GetWordHash(vocab[a].word);
            
            while (vocab_hash[hash] != -1) 
            {
                hash = (hash + 1) % vocab_hash_size;
            }
            
            vocab_hash[hash] = a;
            train_words = train_words + vocab[a].cn;
        }
        a++;
    }
    
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    if (vocab == NULL) 
    {
        printf("Memory re-allocation failed after sorting\n");
        exit(1);
    }
}
//removing rare words from our vocab
void ReduceVocab() 
{
    int a;
    int b;
    
    b = 0;
    a = 0;
    
    while (a < vocab_size) 
    {
        if (vocab[a].cn > min_reduce) 
        {
            vocab[b].cn = vocab[a].cn;
            vocab[b].word = vocab[a].word;
            b++;
        } 
        else 
        {
            free(vocab[a].word);
        }
        a++;
    }
    
    vocab_size = b;
    
    RebuildHashTable();
    
    min_reduce++;
}

void ProcessWordDuringPass1(char *word)
{
    int index;
    int new_index;
    long long threshold;
    
    index = SearchVocab(word);
    
    if (index == -1) 
    {
        new_index = AddWordToVocab(word);
        vocab[new_index].cn = 1;
    } 
    else 
    {
        vocab[index].cn++;
    }
    
    threshold = vocab_hash_size * 0.7;
    
    if (vocab_size > threshold) 
    {
        ReduceVocab();
    }
}

void LearnVocabFromTrainFile() 
{
    char word[MAX_STRING];
    FILE *fin;
    long long a;
    
    a = 0;
    while (a < vocab_hash_size) 
    {
        vocab_hash[a] = -1;
        a++;
    }
    
    fin = fopen("input.txt", "rb");
    if (fin == NULL) 
    {
        printf("ERROR: input.txt not found!\n");
        exit(1);
    }
    
    vocab_size = 0;
    AddWordToVocab("</s>");
    
    printf("Pass 1: Reading file to build Vocabulary...\n");
    
    while (1) 
    {
        ReadWord(word, fin);
        
        if (feof(fin)) 
        {
            break;
        }
        
        train_words++;
        
        if ((train_words % 100000) == 0) 
        {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        
        ProcessWordDuringPass1(word);
    }
    
    SortVocab();
    
    printf("\nVocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
    
    fclose(fin);
}

void AllocateNet()
{
    long long syn0_size;
    long long syn1_size;
    
    syn0_size = vocab_size * layer1_size * sizeof(float);
    syn0 = (float *)malloc(syn0_size);
    
    if (syn0 == NULL) 
    {
        printf("Memory allocation failed for syn0\n");
        exit(1);
    }
    
    syn1_size = vocab_size * layer1_size * sizeof(float);
    syn1neg = (float *)malloc(syn1_size);
    
    if (syn1neg == NULL) 
    {
        printf("Memory allocation failed for syn1neg\n");
        exit(1);
    }
}
//randomization to apply the weight for ann and backpropagation
//randomization for other important parts
void RandomizeWeights()
{
    long long a;
    long long b;
    unsigned long long next_random;
    float random_val;
    float normalized_val;
    
    next_random = 1;
    a = 0;
    while (a < vocab_size) 
    {
        b = 0;
        while (b < layer1_size) 
        {
            next_random = next_random * 25214903917 + 11;
            random_val = (next_random & 0xFFFF) / (float)65536;
            normalized_val = (random_val - 0.5) / layer1_size;
            syn0[a * layer1_size + b] = normalized_val;
            b++;
        }
        a++;
    }
}

void ZeroOutputWeights()
{
    long long a;
    long long b;
    
    a = 0;
    while (a < vocab_size) 
    {
        b = 0;
        while (b < layer1_size) 
        {
            syn1neg[a * layer1_size + b] = 0;
            b++;
        }
        a++;
    }
}

void InitNet() 
{
    AllocateNet();
    RandomizeWeights();
    ZeroOutputWeights();
}

void AllocateUnigramTable()
{
    table = (int *)malloc(TABLE_SIZE * sizeof(int));
    if (table == NULL) 
    {
        printf("Memory allocation failed for Unigram Table\n");
        exit(1);
    }
}

void InitUnigramTable() 
{
    long long a;
    long long i;
    double train_words_pow;
    double d1;
    double power;
    double word_pow;
    double ratio;
    
    AllocateUnigramTable();
    
    power = 0.75;
    train_words_pow = 0;
    
    a = 0;
    while (a < vocab_size) 
    {
        word_pow = pow(vocab[a].cn, power);
        train_words_pow = train_words_pow + word_pow;
        a++;
    }
    
    i = 0;
    d1 = pow(vocab[i].cn, power) / train_words_pow;
    
    a = 0;
    while (a < TABLE_SIZE) 
    {
        table[a] = i;
        
        ratio = a / (double)TABLE_SIZE;
        
        if (ratio > d1) 
        {
            i++;
            word_pow = pow(vocab[i].cn, power);
            d1 = d1 + (word_pow / train_words_pow);
        }
        
        if (i >= vocab_size) 
        {
            i = vocab_size - 1;
        }
        
        a++;
    }
}

long long ReadWordIndex(FILE *fin) 
{
    char word[MAX_STRING];
    int index;
    
    ReadWord(word, fin);
    
    if (feof(fin)) 
    {
        return -1;
    }
    
    index = SearchVocab(word);
    return index;
}

void UpdateProgressMonitor(long long word_count, long long *last_word_count)
{
    long long diff_count;
    float progress;
    float ratio;
    
    diff_count = word_count - *last_word_count;
    
    if (diff_count > 10000) 
    {
        word_count_actual = word_count_actual + diff_count;
        *last_word_count = word_count;
        
        ratio = word_count_actual / (float)(train_words + 1);
        progress = ratio * 100;
        
        printf("Alpha: %f  Progress: %.2f%% \r", alpha, progress);
        fflush(stdout);
        
        alpha = 0.025 * (1 - ratio);
        
        if (alpha < 0.0001) 
        {
            alpha = 0.0001;
        }
    }
}
//removes a,the,is these kinds of words
int SubsampleWord(long long word, unsigned long long *next_random)
{
    float prob_ratio;
    float ran;
    float term1;
    float term2;
    
    if (sample > 0) 
    {
        prob_ratio = sample * train_words;
        term1 = sqrt(vocab[word].cn / prob_ratio) + 1;
        term2 = prob_ratio / vocab[word].cn;
        ran = term1 * term2;
        
        *next_random = *next_random * 25214903917 + 11;
        
        if (ran < (*next_random & 0xFFFF) / 65536.f) 
        {
            return 1; // Discard this word
        }
    }
    return 0; // Keep this word
}
//training function starter
void PerformTrainingForPair(long long input_word, long long target_word, int label, float *neu1)
{
    ann(input_word, target_word, label, alpha, neu1);
}
//negative sampling
long long GetNegativeSample(unsigned long long *next_random)
{
    long long target;
    int table_index;
    
    *next_random = *next_random * 25214903917 + 11;
    table_index = (*next_random >> 16) % (int)TABLE_SIZE;
    target = table[table_index];
    
    if (target == 0) 
    {
        target = *next_random % (vocab_size - 1) + 1;
    }
    
    return target;
}
//decideing the context window in skipgram
void TrainContextWindow(long long sentence_pos, long long sent_len, long long *sen, unsigned long long *next_random, float *neu1)
{
    long long word;
    long long last_word;
    long long target;
    long long a;
    long long b;
    long long c;
    long long d;
    long long l1;
    long long start_window;
    long long end_window;
    long long input_offset;
    float new_syn0;
    int label;
    
    word = sen[sentence_pos];
    if (word == -1) return;
    
    c = 0;
    while (c < layer1_size) 
    {
        neu1[c] = 0;
        c++;
    }
    
    *next_random = *next_random * 25214903917 + 11;
    b = *next_random % window;
    
    start_window = b;
    end_window = window * 2 + 1 - b;
    
    a = start_window;
    while (a < end_window) 
    {
        if (a == window) 
        {
            a++;
            continue;
        }
        
        c = sentence_pos - window + a;
        
        if (c < 0) 
        {
            a++;
            continue;
        }
        
        if (c >= sent_len) 
        {
            a++;
            continue;
        }
        
        last_word = sen[c];
        
        // Negative Sampling Loop
        d = 0;
        while (d < negative + 1) 
        {
            if (d == 0) 
            {
                target = word;
                label = 1;
            } 
            else 
            {
                target = GetNegativeSample(next_random);
                
                if (target == word) 
                {
                    d++;
                    continue;
                }
                label = 0;
            }
            
            PerformTrainingForPair(last_word, target, label, neu1);
            d++;
        }
        
        l1 = last_word * layer1_size;
        c = 0;
        while (c < layer1_size) 
        {
            input_offset = l1 + c;
            new_syn0 = syn0[input_offset] + neu1[c];
            syn0[input_offset] = new_syn0;
            c++;
        }
        a++;
    }
}

void SaveVectorFile()
{
    FILE *fo;
    long long a;
    long long b;
    
    printf("\n");
    printf("Training complete. Saving vectors to vectors.txt...");
    printf("\n");
    
    fo = fopen("vectors.txt", "wb");
    if (fo == NULL) 
    {
        printf("Error creating vectors.txt");
    printf("\n");
        exit(1);
    }
    
    fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
    
    a = 0;
    while (a < vocab_size) 
    {
        fprintf(fo, "%s ", vocab[a].word);
        
        b = 0;
        while (b < layer1_size) 
        {
            fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
            b++;
        }
        fprintf(fo, "\n");
        a++;
    }
    
    fclose(fo);
}

void TrainModel() 
{
    long long word;
    long long sentence_length;
    long long sentence_position;
    long long word_count;
    long long last_word_count;
    long long sen[MAX_SENTENCE_LENGTH + 1];
    
    unsigned long long next_random;
    float *neu1;
    FILE *fi;
    
    next_random = 1;
    sentence_length = 0;
    sentence_position = 0;
    word_count = 0;
    last_word_count = 0;
    
    fi = fopen("input.txt", "rb");
    if (fi == NULL) 
    {
        printf("Error opening input.txt for training");
        
    printf("\n");
        exit(1);
    }
    
    neu1 = (float *)calloc(layer1_size, sizeof(float));
    if (neu1 == NULL) 
    {
        printf("Memory allocation failed for neu1\n");
        exit(1);
    }

    printf("Pass 2: Training Neural Network...\n");
    
    while (1) 
    {
        UpdateProgressMonitor(word_count, &last_word_count);
        
        if (sentence_length == 0) 
        {
            while (1) 
            {
                word = ReadWordIndex(fi);
                
                if (feof(fi)) 
                {
                    break;
                }
                
                if (word == -1) 
                {
                    continue;
                }
                
                if (word == 0) 
                {
                    break;
                }
                
                if (SubsampleWord(word, &next_random))
                {
                    continue;
                }
                
                sen[sentence_length] = word;
                sentence_length++;
                
                if (sentence_length >= MAX_SENTENCE_LENGTH) 
                {
                    break;
                }
            }
            sentence_position = 0;
        }
        
        if (feof(fi)) 
        {
            if (sentence_length == 0) 
            {
                break;
            }
        }
        //For one center word, it finds nearby context words and trains
        TrainContextWindow(sentence_position, sentence_length, sen, &next_random, neu1);
        
        sentence_position++;
        word_count++;
        
        if (sentence_position >= sentence_length) 
        {
            sentence_length = 0;
            continue;
        }
    }
    
    fclose(fi);
    free(neu1);
    
    SaveVectorFile();
}

void LoadWordFromFile(FILE *f, char *word, int *index_ptr, long long b)
{
    unsigned int hash;
    char buffer[MAX_STRING];
    int len;
    
    fscanf(f, "%s", buffer);
    len = strlen(buffer) + 1;
    
    vocab[b].word = (char *)calloc(len, sizeof(char));
    strcpy(vocab[b].word, buffer);
    
    hash = GetWordHash(vocab[b].word);
    while (vocab_hash[hash] != -1) 
    {
        hash = (hash + 1) % vocab_hash_size;
    }
    vocab_hash[hash] = b;
}

void LoadVectorFromFile(FILE *f, long long b, long long size)
{
    float len;
    float val;
    float sq_sum;
    float normalized;
    long long a;
    
    sq_sum = 0;
    a = 0;
    while (a < size) 
    {
        fscanf(f, "%f", &val);
        syn0[b * size + a] = val;
        sq_sum = sq_sum + (val * val);
        a++;
    }
    
    len = sqrt(sq_sum);
    
    a = 0;
    while (a < size) 
    {
        normalized = syn0[b * size + a] / len;
        syn0[b * size + a] = normalized;
        a++;
    }
}

void LoadModel() 
{
    long long words;
    long long size;
    long long a;
    long long b;
    long long vocab_bytes;
    long long syn0_bytes;
    long long vocab_hash_bytes;
    char word[MAX_STRING];
    int dummy_idx;
    
    FILE *f;
    f = fopen("vectors.txt", "rb");
    
    if (f == NULL) 
    {
        printf("Error: vectors.txt not found. Train first!\n");
        exit(1);
    }
    
    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);
    
    vocab_size = words;
    layer1_size = size;
    
    printf("Loading Model: %lld words, %lld dimensions\n", words, size);
    
    vocab_bytes = words * sizeof(struct vocab_word);
    vocab = (struct vocab_word *)malloc(vocab_bytes);
    
    vocab_hash_bytes = vocab_hash_size * sizeof(int);
    vocab_hash = (int *)malloc(vocab_hash_bytes);
    
    a = 0;
    while (a < vocab_hash_size) 
    {
        vocab_hash[a] = -1;
        a++;
    }
    
    syn0_bytes = words * size * sizeof(float);
    syn0 = (float *)malloc(syn0_bytes);
    
    b = 0;
    while (b < words) 
    {
        LoadWordFromFile(f, word, &dummy_idx, b);
        LoadVectorFromFile(f, b, size);
        b++;
    }
    fclose(f);
}
//calculates the cosine similarity
float CalculateCosineSimilarity(long long word1_idx, long long word2_idx)
{
    float dist;
    float dot_prod;
    long long a;
    
    dist = 0;
    a = 0;
    while (a < layer1_size) 
    {
        dot_prod = syn0[a + word1_idx * layer1_size] * syn0[a + word2_idx * layer1_size];
        dist = dist + dot_prod;
        a++;
    }
    return dist;
}
//performs similarity function
void PerformSimilarityCheck()
{
    char st1[MAX_STRING];
    long long b;
    long long c;
    long long a;
    long long d;
    float dist;
    float bestd[40];
    long long bi[40];
    long long current_index;
    float current_dist;
      printf("\n");
    printf("Enter word: ");
    printf("\n");
    scanf("%s", st1);
    
    b = SearchVocab(st1);
    
    if (b == -1) 
    {
        printf("Out of dictionary word!\n");
        return;
    }
    
    a = 0;
    while (a < 20) 
    {
        bestd[a] = 0;
        bi[a] = 0;
        a++;
    }
    
    printf("\n");
    printf("                                              Word       Cosine Distance\n");
    printf("------------------------------------------------------------------------\n");
    
    c = 0;
    while (c < vocab_size) 
    {
        if (c == b) 
        {
            c++;
            continue;
        }
        
        dist = CalculateCosineSimilarity(b, c);
        
        a = 0;
        while (a < 20) 
        {
            if (dist > bestd[a]) 
            {
                d = 19;
                while (d > a) 
                {
                    bestd[d] = bestd[d - 1];
                    bi[d] = bi[d - 1];
                    d--;
                }
                bestd[a] = dist;
                bi[a] = c;
                break;
            }
            a++;
        }
        c++;
    }
    
    a = 0;
    while (a < 15) 
    {
        current_index = bi[a];
        current_dist = bestd[a];
        printf("%50s\t\t%f\n", vocab[current_index].word, current_dist);
        a++;
    }
}
//word arithmatic(king - man +woman =queen)
void PerformFlexibleArithmetic()
{
    char input_line[MAX_STRING * 10]; 
    char *token;
    char words[10][MAX_STRING]; 
    char ops[10];               
    int word_count = 0;
    int op_count = 0;
    long long indices[10];
    float vec[3000];
    float len;
    float dist;
    float bestd[40];
    long long bi[40];
    long long a, c, d;
    int i;
      printf("\n");
    printf("Enter Equation (e.g., king - man + woman): ");
    
    int ch;
    while ((ch = getchar()) != '\n' && ch != EOF); 
    
    if (fgets(input_line, sizeof(input_line), stdin) == NULL) return;
    
    input_line[strcspn(input_line, "\n")] = 0;

    token = strtok(input_line, " ");
    while (token != NULL) 
    {
        if (strcmp(token, "+") == 0 || strcmp(token, "-") == 0) 
        {
            ops[op_count++] = token[0];
        } 
        else 
        {
            strcpy(words[word_count++], token);
        }
        token = strtok(NULL, " ");
    }

    if (word_count < 2) 
    {
        printf("Error: Need at least 2 words.\n");
        return;
    }

    for (i = 0; i < word_count; i++) 
    {
        indices[i] = SearchVocab(words[i]);
        if (indices[i] == -1) 
        {
            printf("Error: Word '%s' not in dictionary.\n", words[i]);
            return;
        }
    }

    for (a = 0; a < layer1_size; a++) 
    {
        vec[a] = syn0[a + indices[0] * layer1_size];
    }

    for (i = 1; i < word_count; i++) 
    {
        char op = (i-1 < op_count) ? ops[i-1] : '+'; 
        
        for (a = 0; a < layer1_size; a++) 
        {
            if (op == '+') 
            {
                vec[a] += syn0[a + indices[i] * layer1_size];
            } 
            else 
            {
                vec[a] -= syn0[a + indices[i] * layer1_size];
            }
        }
    }

    len = 0;
    for (a = 0; a < layer1_size; a++) len += vec[a] * vec[a];
    len = sqrt(len);
    for (a = 0; a < layer1_size; a++) vec[a] /= len;

    for (a = 0; a < 20; a++) { bestd[a] = 0; bi[a] = 0; }

    printf("\n                                              Word       Cosine Distance\n");
    printf("------------------------------------------------------------------------\n");

    for (c = 0; c < vocab_size; c++) 
    {
        int skip = 0;
        for(i=0; i<word_count; i++) if (c == indices[i]) skip = 1;
        if (skip) continue;

        dist = 0;
        for (a = 0; a < layer1_size; a++) dist += vec[a] * syn0[a + c * layer1_size];
        
        for (a = 0; a < 20; a++) 
        {
            if (dist > bestd[a]) 
            {
                for (d = 19; d > a; d--) 
                {
                    bestd[d] = bestd[d - 1];
                    bi[d] = bi[d - 1];
                }
                bestd[a] = dist;
                bi[a] = c;
                break;
            }
        }
    }

    for (a = 0; a < 15; a++) 
    {
        printf("%50s\t\t%f\n", vocab[bi[a]].word, bestd[a]);
    }
}
//two words and calculate their similarity
void PerformPairSimilarity()
{
    char st1[MAX_STRING];
    char st2[MAX_STRING];
    long long w1;
    long long w2;
    float dist;
      printf("\n");
    printf("Enter two words to check similarity (e.g. mango banana)\n");
    printf("Input: ");
    scanf("%s %s", st1, st2);
    
    w1 = SearchVocab(st1);
    w2 = SearchVocab(st2);
    
    if (w1 == -1) 
    {
        printf("Error: Word '%s' not in vocabulary.\n", st1);
        return;
    }
    if (w2 == -1) 
    {
        printf("Error: Word '%s' not in vocabulary.\n", st2);
        return;
    }
    
    dist = CalculateCosineSimilarity(w1, w2);
    
    printf("\nCosine Similarity (%s, %s) = %f\n", st1, st2, dist);
}
//remove sthe odd word from the input words
void PerformOddOneOut()
{
    char input_line[MAX_STRING * 10];
    char *token;
    char words[10][MAX_STRING];
    long long indices[10];
    float avg_vec[3000];
    int word_count = 0;
    int i;
    long long a;
    float len;
    float min_dist = 100.0; 
    int odd_one_index = -1;
    float dist;
      printf("\n");
    printf("Enter list of words (e.g., apple banana car): ");
    
    int ch;
    while ((ch = getchar()) != '\n' && ch != EOF); 
    
    if (fgets(input_line, sizeof(input_line), stdin) == NULL) return;
    input_line[strcspn(input_line, "\n")] = 0;

    token = strtok(input_line, " ");
    while (token != NULL) 
    {
        strcpy(words[word_count++], token);
        token = strtok(NULL, " ");
    }

    if (word_count < 3) 
    {
        printf("Error: Need at least 3 words.\n");
        return;
    }

    for (i = 0; i < word_count; i++) 
    {
        indices[i] = SearchVocab(words[i]);
        if (indices[i] == -1) 
        {
            printf("Error: Word '%s' not in dictionary.\n", words[i]);
            return;
        }
    }

    for (a = 0; a < layer1_size; a++) 
    {
        avg_vec[a] = 0;
    }

    for (i = 0; i < word_count; i++) 
    {
        for (a = 0; a < layer1_size; a++) 
        {
            avg_vec[a] += syn0[a + indices[i] * layer1_size];
        }
    }
    
    len = 0;
    for (a = 0; a < layer1_size; a++) 
    {
        len += avg_vec[a] * avg_vec[a];
    }
    len = sqrt(len);
    
    for (a = 0; a < layer1_size; a++) 
    {
        avg_vec[a] /= len;
    }
    
    for (i = 0; i < word_count; i++) 
    {
        dist = 0;
        for (a = 0; a < layer1_size; a++) 
        {
            dist += avg_vec[a] * syn0[a + indices[i] * layer1_size];
        }
        
        if (dist < min_dist) 
        {
            min_dist = dist;
            odd_one_index = i;
        }
    }
    
    if (odd_one_index != -1) 
    {
        printf("\nOdd one out: %s\n", words[odd_one_index]);
    }
}
//gets the vector of every token of the sentence
void GetSentenceVector(char *sentence, float *vector)
{
    char temp[MAX_STRING * 20];
    char *token;
    int word_count = 0;
    long long idx;
    long long a;
    float len;

    strcpy(temp, sentence);
    
    for (a = 0; a < layer1_size; a++) vector[a] = 0;

    token = strtok(temp, " \n");
    while (token != NULL) 
    {
        idx = SearchVocab(token);
        if (idx != -1) 
        {
            for (a = 0; a < layer1_size; a++) 
            {
                vector[a] += syn0[a + idx * layer1_size];
            }
            word_count++;
        }
        token = strtok(NULL, " \n");
    }

    if (word_count > 0) 
    {
        len = 0;
        for (a = 0; a < layer1_size; a++) 
        {
            vector[a] /= word_count; 
            len += vector[a] * vector[a];
        }
        len = sqrt(len);
        for (a = 0; a < layer1_size; a++) vector[a] /= len; 
    }
}

void PerformSentenceSimilarity()
{
    char sent1[MAX_STRING * 20];
    char sent2[MAX_STRING * 20];
    float vec1[3000]; 
    float vec2[3000];
    float dist;
    long long a;

    int ch;
    while ((ch = getchar()) != '\n' && ch != EOF);

    printf("Enter Sentence A: ");
    if (fgets(sent1, sizeof(sent1), stdin) == NULL) return;
    sent1[strcspn(sent1, "\n")] = 0;
      printf("\n");
      printf("\n");

    printf("Enter Sentence B: ");
    if (fgets(sent2, sizeof(sent2), stdin) == NULL) return;
    sent2[strcspn(sent2, "\n")] = 0;

    GetSentenceVector(sent1, vec1);
    GetSentenceVector(sent2, vec2);

    dist = 0;
    for (a = 0; a < layer1_size; a++) 
    {
        dist += vec1[a] * vec2[a];
    }

    printf("\nSemantic Similarity Score: %f\n", dist);
}

// clustering into k groups 
void PerformWordClustering()
{
    char input_line[MAX_STRING * 20];
    char words[50][MAX_STRING];
    long long indices[50];
    int cluster_assignment[50];
    float centroids[10][300]; // Max 10 clusters, 300 dimensions
    int word_count = 0, k_clusters = 0;
    int i, j, c, iter;

    printf("\n--- K-Means Word Clustering ---");
    printf("maximum 10 clustering is possible");
      printf("\n");
    printf("How many groups (K) do you want to create? ");
    scanf("%d", &k_clusters);
    
    if (k_clusters < 2 || k_clusters > 10) {
        printf("Please choose K between 2 and 10.\n");
        return;
    }

    // Clear buffer and get words
    int ch; while ((ch = getchar()) != '\n' && ch != EOF);
    printf("Enter words (e.g., car banana tiger mango lion bus): ");
    if (fgets(input_line, sizeof(input_line), stdin) == NULL) 
        return;
    input_line[strcspn(input_line, "\n")] = 0;

    // Tokenize words and find indices
    char *token = strtok(input_line, " ");
    while (token != NULL && word_count < 50) {
        long long idx = SearchVocab(token);
        if (idx != -1) 
        {
            strcpy(words[word_count], token);
            indices[word_count] = idx;
            word_count++;
        } 
        else 
        {
            printf("Skipping '%s' (not in vocab).\n", token);
        }
        token = strtok(NULL, " ");
    }

    if (word_count < k_clusters) {
        printf("Error: Not enough valid words for %d clusters.\n", k_clusters);
        return;
    }

    // initialization
    int c_idx[10];
    c_idx[0] = 0;
    for (c = 0; c < layer1_size; c++) 
    {
        centroids[0][c] = syn0[c + indices[0] * layer1_size];
    }

    for (j = 1; j < k_clusters; j++) 
    {
        float min_max_sim = 2.0; 
        int best_candidate = -1;

        for (i = 0; i < word_count; i++) 
        {
            int already_chosen = 0;
            for (int prev = 0; prev < j; prev++) 
            {
                if (c_idx[prev] == i) already_chosen = 1;
            }
            if (already_chosen)
                continue;

            float max_sim_to_centroids = -2.0;
            for (int prev = 0; prev < j; prev++) 
            {
                float sim = 0;
                for (c = 0; c < layer1_size; c++)
                    {
                    sim += syn0[c + indices[i] * layer1_size] * centroids[prev][c];
                }
                if (sim > max_sim_to_centroids) 
                {
                    max_sim_to_centroids = sim;
                }
            }

            if (max_sim_to_centroids < min_max_sim) 
            {
                min_max_sim = max_sim_to_centroids;
                best_candidate = i;
            }
        }
        c_idx[j] = best_candidate;
        for (c = 0; c < layer1_size; c++) 
        {
            centroids[j][c] = syn0[c + indices[best_candidate] * layer1_size];
        }
    }

    // k means iteration
    for (iter = 0; iter < 10; iter++)
        {
        // assignment step
        for (i = 0; i < word_count; i++) 
        {
            float max_sim = -2.0;
            int best_cluster = 0;
            for (j = 0; j < k_clusters; j++) 
            {
                float sim = 0;
                // Cosine similarity between word and centroid
                for (c = 0; c < layer1_size; c++) 
                {
                    sim += syn0[c + indices[i] * layer1_size] * centroids[j][c];
                }
                if (sim > max_sim) 
                {
                    max_sim = sim;
                    best_cluster = j;
                }
            }
            cluster_assignment[i] = best_cluster;
        }

        //update steps
        for (j = 0; j < k_clusters; j++) 
        {
            float new_vec[300] = {0};
            int count = 0;
            for (i = 0; i < word_count; i++) 
            {
                if (cluster_assignment[i] == j) 
                {
                    for (c = 0; c < layer1_size; c++) new_vec[c] += syn0[c + indices[i] * layer1_size];
                    count++;
                }
            }
            if (count > 0) 
            {
                float len = 0;
                for (c = 0; c < layer1_size; c++) 
                {
                    new_vec[c] /= count;
                    len += new_vec[c] * new_vec[c];
                }
                len = sqrt(len);
                for (c = 0; c < layer1_size; c++) centroids[j][c] = new_vec[c] / len;
            }
        }
    }

    //display result
      printf("\n");
    printf("-- Clustering Results ---");
  printf("\n");
    for (j = 0; j < k_clusters; j++)
        {
        printf("Group %d: ", j + 1);
        int found = 0;
        for (i = 0; i < word_count; i++) 
        {
            if (cluster_assignment[i] == j) 
            {
                printf("%s ", words[i]);
                found = 1;
            }
        }
        if (!found)
            printf("[Empty]");
        printf("\n");
    }
}
//finds the semantic sentence from the document 
void PerformSemanticSearch() 
{
    char filename[MAX_STRING];
    char query[MAX_STRING * 20];
    char line[MAX_STRING * 100]; 
    char best_sentence[MAX_STRING * 100];
    float query_vec[3000];
    float sentence_vec[3000];
    float best_score = -2.0;
    float current_score;
    long long a;
    FILE *f;

     printf("\n");  
    printf("--- Semantic Search Engine ---\n");
    
    
    int ch;
    while ((ch = getchar()) != '\n' && ch != EOF);
    
    
    printf("Enter the name of your text file (e.g., document.txt): ");
    if (fgets(filename, sizeof(filename), stdin) == NULL)
     return;
    filename[strcspn(filename, "\n")] = 0; 

    
    f = fopen(filename, "r");
    if (f == NULL) 
    {
        printf("Error: Could not find '%s' in the current folder.\n", filename);
        return;
    }

    
    printf("Enter your search query: ");
    if (fgets(query, sizeof(query), stdin) == NULL)
     {
         fclose(f);
          return;
         }
    query[strcspn(query, "\n")] = 0; 

    
    GetSentenceVector(query, query_vec);

    
    best_sentence[0] = '\0'; 
    while (fgets(line, sizeof(line), f) != NULL) 
    {
        line[strcspn(line, "\n")] = 0;
        if (strlen(line) < 5) 
        continue;

        // Get the vector for the current sentence
        GetSentenceVector(line, sentence_vec);
        
        current_score = 0;
        for (a = 0; a < layer1_size; a++) 
        {
            current_score += query_vec[a] * sentence_vec[a];
        }

        
        if (current_score > best_score) 
        {
            best_score = current_score;
            strcpy(best_sentence, line);
        }
    }
    fclose(f);

    // Output the results
    if (best_score == -2.0) 
    {
        printf("\n");
        printf(" No valid sentences found in the document to compare.\n");
    } 
    else 
    {
        printf("\n=> Best Match (Similarity Score: %.4f):\n", best_score);
        printf("\"%s\"\n", best_sentence);
    }
}

//connects words according to the context 
void PerformConceptConnector() 
{
    char input_line[MAX_STRING * 10];
    char words[10][MAX_STRING];
    long long indices[10];
    int word_count = 0;
    float center_vec[3000];
    long long a, b, c;
    int i;
    printf("\n");
    printf("--- The 'Codenames' AI (Concept Connector) ---");
    printf("\n");
    
    // Clear the input buffer
    int ch;
    while ((ch = getchar()) != '\n' && ch != EOF); 
    printf("\n");
    printf("Enter 2 to 5 words separated by spaces (e.g., web venom legs): ");
    
    if (fgets(input_line, sizeof(input_line), stdin) == NULL) 
    return;
    input_line[strcspn(input_line, "\n")] = 0;
    
    // Break the input line into individual words
    char *token = strtok(input_line, " ");
    while (token != NULL && word_count < 5) 
    {
        strcpy(words[word_count++], token);
        token = strtok(NULL, " ");
    }
    
    if (word_count < 2) 
    {
        printf("Error: Please enter at least 2 words to connect.\n");
        return;
    }
    
    //  Find the words in the vocabulary
    int valid_words = 0;
    for (i = 0; i < word_count; i++) 
    {
        indices[valid_words] = SearchVocab(words[i]);
        if (indices[valid_words] == -1) 
        {
            printf("Skipping '%s' (Not found in dictionary).\n", words[i]);
        } 
        else 
        {
            valid_words++;
        }
    }
    
    if (valid_words == 0)
     {
        printf("\n");
        printf("Error: None of your words were found in the dictionary.");
        return;
    }
    
    // Calculate the "Center of Mass" (Average Vector)
    for (a = 0; a < layer1_size; a++) 
    center_vec[a] = 0;
    
    for (i = 0; i < valid_words; i++) 
    {
        for (a = 0; a < layer1_size; a++)
         {
            center_vec[a] += syn0[a + indices[i] * layer1_size];
        }
    }
    
    // Normalize the center point mathematically
    float len = 0;
    for (a = 0; a < layer1_size; a++) 
    len += center_vec[a] * center_vec[a];
    len = sqrt(len);
    for (a = 0; a < layer1_size; a++)
     center_vec[a] /= len;
    
    //  Scan the dictionary for the closest matches
    float best_scores[5] = {-2.0, -2.0, -2.0, -2.0, -2.0};
    int best_indices[5] = {-1, -1, -1, -1, -1};
    
    for (b = 0; b < vocab_size; b++) 
    {
        
        int is_input = 0;
        for (i = 0; i < valid_words; i++) 
        {
            if (b == indices[i]) is_input = 1;
        }
        if (is_input)
         continue;
        
        
        float dist = 0;
        for (a = 0; a < layer1_size; a++)
         {
            dist += center_vec[a] * syn0[a + b * layer1_size];
        }
        
        // Keep track of the top 5 closest words
        for (i = 0; i < 5; i++)
         {
            if (dist > best_scores[i])
             {
                for (c = 4; c > i; c--) 
                {
                    best_scores[c] = best_scores[c-1];
                    best_indices[c] = best_indices[c-1];
                }
                best_scores[i] = dist;
                best_indices[i] = b;
                break;
            }
        }
    }
    
    // 4. Print the result
    printf("\n=> Top 'Codenames' Clues for your words:");
    printf("\n");
    for (i = 0; i < 5; i++) 
    {
        if (best_indices[i] != -1)
         {
            printf(" %d. %s (Similarity Score: %.4f)\n", i+1, vocab[best_indices[i]].word, best_scores[i]);
        }
    }
}
//an iterface that will appaea on the screen of linux
void InteractiveLoop() 
{
    int choice;
    
    while (1) 
    {
        printf("\n");
        printf("==============================================\n");
        printf("Choose Option:");
          printf("\n");
          printf("\n");
        printf("1. Similar Words (Input: 'good')");
          printf("\n");
          printf("\n");
        printf("2. Flexible Word Arithmetic (e.g. king - man + woman)");
          printf("\n");
          printf("\n");
        printf("3. Pair Similarity (Input: 'mango banana')");
              printf("\n");
          printf("\n");
        printf("4. Odd One Out (Input: 'apple car banana')");
          printf("\n");
          printf("\n");
        printf("5. Sentence Similarity (Input: 2 sentences)");
          printf("\n");
          printf("\n");
        printf("6. Word Clustering (Input: 2 centers + word list)");
          printf("\n");
          printf("\n");
           printf("7. Semantic Document Search (Requires 'document.txt')");
        printf("\n");
        printf("\n");
        
        printf("8. 'Codenames' Concept Connector (Input: 2-5 words)");
        printf("\n");
        printf("\n);
          
        printf("9. Exit");
          printf("\n");
        printf("Enter choice: ");
        
        scanf("%d", &choice);
        
        if (choice == 9) 
        {
            break;
        }
        if (choice>9)
        {
            printf("there is no feature");
            printf("\n");
        }
        
        if (choice == 1) 
        {
            PerformSimilarityCheck();
        }
        else if (choice == 2) 
        {
            PerformFlexibleArithmetic();
        }
        else if (choice == 3) 
        {
            PerformPairSimilarity();
        }
        else if (choice == 4) 
        {
            PerformOddOneOut();
        }
        else if (choice == 5)
        {
            PerformSentenceSimilarity();
        }
        else if (choice == 6)
        {
            PerformWordClustering();
        }
          else if (choice == 7) 
        {
            PerformSemanticSearch();
        }
        
        else if (choice == 8) 
        {
            PerformConceptConnector();
    }
}
}

void AllocateInitialMemory()
{
    size_t vocab_init_bytes;
    size_t hash_init_bytes;
    
    vocab_init_bytes = vocab_max_size * sizeof(struct vocab_word);
    vocab = (struct vocab_word *)calloc(1, vocab_init_bytes);
    
    if (vocab == NULL) 
    {
        printf("Initial vocab allocation failed");
        printf("\n");
        exit(1);
    }
    
    hash_init_bytes = vocab_hash_size * sizeof(int);
    vocab_hash = (int *)calloc(1, hash_init_bytes);
    
    if (vocab_hash == NULL) 
    {
        printf("Initial hash allocation failed");
        printf("\n");
        exit(1);
    }
}

int main() 
{
    int mode;
    
    printf("========================================");
    printf("\n");
    
    printf("\n");
    printf("     Word2Vec C Engine (Ultimate Version)   ");
    
    printf("\n");
    printf("========================================\n");
    printf("1. TRAIN New Model");
    printf("\n");
    
    printf("\n");
    printf("2. PLAY with Existing Model (vectors.txt)");
    printf("\n");
    printf("\n");
    printf("Select Mode: ");
    
    scanf("%d", &mode);
//calling the training function 
    if (mode == 1) 
    {
        AllocateInitialMemory();
        InitExpTable();
        LearnVocabFromTrainFile();
        InitNet();
        InitUnigramTable();
        TrainModel();
    } 
    //calling the model to deal with the user 
    if (mode == 2) 
    {
        LoadModel();
        InteractiveLoop();
    }
    
    return 0;
}
