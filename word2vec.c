//word2vec from scratch

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define TABLE_SIZE 1e8

long long layer1_size = 300;
long long min_count = 50;
int window = 5;
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

void InitExpTable()
{
    int i;
    float f;
    float numerator;
    float denominator;
    float exponent_val;
    
    expTable = (float *)malloc((EXP_TABLE_SIZE + 1) * sizeof(float));
    if (expTable == NULL) {
        printf("Memory allocation failed for ExpTable\n");
        exit(1);
    }

    i = 0;
    while (i < EXP_TABLE_SIZE) 
    {
        exponent_val = (i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP;
        f = exp(exponent_val);
        numerator = f;
        denominator = f + 1;
        expTable[i] = numerator / denominator;
        i++;
    }
}

static inline float GetSigmoid(float f)
{
    float result;
    int index;
    float formula_val;

    if (f > MAX_EXP)
    {
        return 1.0f;
    }
    if (f < -MAX_EXP) 
    {
        return 0.0f;
    }
    
    formula_val = (f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2);
    index = (int)formula_val;
    result = expTable[index];
    
    return result;
}

static inline void RunGradientDescent(long long input_idx, long long target_idx, int label, float alpha, float *neu1_error) 
{
    long long l1;
    long long l2;
    float f;
    float g;
    float pred;
    float product;
    float diff;
    int c;
    
    l1 = input_idx * layer1_size;
    l2 = target_idx * layer1_size;
    f = 0;
    
    c = 0;
    while (c < layer1_size)
        {
        product = syn0[l1 + c] * syn1neg[l2 + c];
        f = f + product;
        c++;
    }

    pred = GetSigmoid(f);
    
    diff = label - pred;
    g = diff * alpha;

    c = 0;
    while (c < layer1_size) 
    {
        product = g * syn1neg[l2 + c];
        neu1_error[c] = neu1_error[c] + product;
        c++;
    }

    c = 0;
    while (c < layer1_size) 
    {
        product = g * syn0[l1 + c];
        syn1neg[l2 + c] = syn1neg[l2 + c] + product;
        c++;
    }
}

void ReadWord(char *word, FILE *fin) 
{
    int a;
    int ch;
    
    a = 0;
    while (!feof(fin))
        {
        ch = fgetc(fin);
        
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

int GetWordHash(char *word) {
    unsigned long long a;
    unsigned long long hash;
    int len;
    
    hash = 0;
    len = strlen(word);
    
    a = 0;
    while (a < len) 
    {
        hash = hash * 257 + word[a];
        a++;
    }
    
    hash = hash % vocab_hash_size;
    return hash;
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

int AddWordToVocab(char *word) 
{
    unsigned int hash;
    int length;
    long long new_size;
    
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

void SortVocab() 
{
    int i;
    int a;
    unsigned int hash;
    unsigned long long size;
    long long remaining_words;
    
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
        if ((vocab[a].cn < min_count) && (a != 0)) {
            vocab_size--;
            free(vocab[a].word);
        } 
        else 
        {
            hash = GetWordHash(vocab[a].word);
            
            while (vocab_hash[hash] != -1) {
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

void ReduceVocab() 
{
    int a;
    int b;
    unsigned int hash;
    int i;
    
    b = 0;
    a = 0;
    
    while (a < vocab_size)
        {
        if (vocab[a].cn > min_reduce) {
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
    
    min_reduce++;
}

void LearnVocabFromTrainFile() 
{
    char word[MAX_STRING];
    FILE *fin;
    long long a;
    long long i;
    int index;
    int new_index;
    long long threshold;
    
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
    
    SortVocab();
    
    printf("\nVocab size: %lld\n", vocab_size);
    printf("Words in train file: %lld\n", train_words);
    
    fclose(fin);
}
//artificial neural network
void InitNet() 
{
    long long a;
    long long b;
    unsigned long long next_random;
    long long syn0_size;
    long long syn1_size;
    float random_val;
    float normalized_val;
    
    next_random = 1;
    
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
    
    a = 0;
    while (a < vocab_size) 
    {
        b = 0;
        while (b < layer1_size) {
            syn1neg[a * layer1_size + b] = 0;
            b++;
        }
        a++;
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
    
    power = 0.75;
    train_words_pow = 0;
    
    table = (int *)malloc(TABLE_SIZE * sizeof(int));
    if (table == NULL) 
    {
        printf("Memory allocation failed for Unigram Table\n");
        exit(1);
    }
    
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
    
    if (feof(fin)) {
        return -1;
    }
    
    index = SearchVocab(word);
    return index;
}

void TrainModel() 
{
    long long a;
    long long b;
    long long d;
    long long word;
    long long last_word;
    long long sentence_length;
    long long sentence_position;
    long long word_count;
    long long last_word_count;
    long long l1;
    long long c;
    long long target;
    long long label;
    long long diff_count;
    long long sen[MAX_SENTENCE_LENGTH + 1];
    long long input_offset;
    
    unsigned long long next_random;
    
    float f;
    float g;
    float ran;
    float prob_ratio;
    float *neu1;
    float progress;
    float new_neu1;
    float new_syn0;
    
    FILE *fi;
    FILE *fo;
    
    next_random = 1;
    sentence_length = 0;
    sentence_position = 0;
    word_count = 0;
    last_word_count = 0;
    
    fi = fopen("input.txt", "rb");
    if (fi == NULL) 
    {
        printf("Error opening input.txt for training\n");
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
        diff_count = word_count - last_word_count;
        
        if (diff_count > 10000) 
        {
            word_count_actual = word_count_actual + diff_count;
            last_word_count = word_count;
            
            progress = word_count_actual / (float)(train_words + 1) * 100;
            
            printf("Alpha: %f  Progress: %.2f%% \r", alpha, progress);
            fflush(stdout);
            
            alpha = 0.025 * (1 - word_count_actual / (float)(train_words + 1));
            
            if (alpha < 0.0001)
            {
                alpha = 0.0001;
            }
        }
        
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
                
                if (sample > 0)
                {
                    prob_ratio = sample * train_words;
                    ran = (sqrt(vocab[word].cn / prob_ratio) + 1) * prob_ratio / vocab[word].cn;
                    
                    next_random = next_random * 25214903917 + 11;
                    
                    if (ran < (next_random & 0xFFFF) / 65536.f) 
                    {
                        continue;
                    }
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
            if (sentence_length == 0) {
                break;
            }
        }
        
        word = sen[sentence_position];
        
        if (word == -1) 
        {
            continue;
        }
        
        c = 0;
        while (c < layer1_size) 
        {
            neu1[c] = 0;
            c++;
        }
        
        next_random = next_random * 25214903917 + 11;
        b = next_random % window;
        
        a = b;
        while (a < window * 2 + 1 - b) 
        {
            if (a == window) 
            {
                a++;
                continue;
            }
            
            c = sentence_position - window + a;
            
            if (c < 0) 
            {
                a++;
                continue;
            }
            
            if (c >= sentence_length)
            {
                a++;
                continue;
            }
            
            last_word = sen[c];
            l1 = last_word * layer1_size;
            
            d = 0;
            while (d < negative + 1) 
            {
                if (d == 0) {
                    target = word;
                    label = 1;
                } 
                else
                    {
                    next_random = next_random * 25214903917 + 11;
                    target = table[(next_random >> 16) % (int)TABLE_SIZE];
                    
                    if (target == 0) 
                    {
                        target = next_random % (vocab_size - 1) + 1;
                    }
                    
                    if (target == word) 
                       {
                        d++;
                        continue;
                    }
                    label = 0;
                }
                
                RunGradientDescent(last_word, target, label, alpha, neu1);
                d++;
            }
            
            c = 0;
            while (c < layer1_size) {
                input_offset = l1 + c;
                new_syn0 = syn0[input_offset] + neu1[c];
                syn0[input_offset] = new_syn0;
                c++;
            }
            a++;
        }
        
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
    
    printf("\nTraining complete. Saving vectors to vectors.txt...\n");
    fo = fopen("vectors.txt", "wb");
    if (fo == NULL)
    {
        printf("Error creating vectors.txt\n");
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

//training model loop
void LoadModel() 
{
    long long words;
    long long size;
    long long a;
    long long b;
    long long vocab_bytes;
    long long syn0_bytes;
    long long vocab_hash_bytes;
    float len;
    float val;
    float sq_sum;
    float normalized;
    char word[MAX_STRING];
    unsigned int hash;
    
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
        a = 0;
        fscanf(f, "%s", word);
        
        vocab[b].word = (char *)calloc(strlen(word) + 1, sizeof(char));
        strcpy(vocab[b].word, word);
        
        hash = GetWordHash(vocab[b].word);
        while (vocab_hash[hash] != -1) {
            hash = (hash + 1) % vocab_hash_size;
        }
        vocab_hash[hash] = b;
        
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
        b++;
    }
    fclose(f);
}



void InteractiveLoop()
{
    long long a;
    long long b;
    long long c;
    long long d;
    long long w1;
    long long w2;
    long long w3;
    long long bi[40];
    long long current_index;
    
    float dist;
    float len;
    float bestd[40];
    float vec[300];
    float dot_prod;
    float current_dist;
    
    char st1[MAX_STRING];
    char st2[MAX_STRING];
    char st3[MAX_STRING];
    int choice;
    
    while (1)
        {
        printf("\n");
        printf("==============================================\n");
        printf("Choose Option:\n");
        printf("1. Similar Words (Input: 'good')\n");
        printf("2. Linear Analogy (Input: 'king man woman')\n");
        printf("3. Exit\n");
        printf("Enter choice: ");
        
        scanf("%d", &choice);
        
        if (choice == 3) 
        {
            break;
        }
        
        if (choice == 1) 
        {
            printf("\nEnter word: ");
            scanf("%s", st1);
            
            b = SearchVocab(st1);
            
            if (b == -1) {
                printf("Out of dictionary word!\n");
                continue;
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
                if (c == b) {
                    c++;
                    continue;
                }
                
                dist = 0;
                a = 0;
                while (a < layer1_size)
                    {
                    dot_prod = syn0[a + b * layer1_size] * syn0[a + c * layer1_size];
                    dist = dist + dot_prod;
                    a++;
                }
                
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
        
        if (choice == 2) 
                       {
            printf("\nEnter 3 words (A B C) for formula A - B + C = ?\n");
            printf("Example: king man woman\n");
            printf("Input: ");
            
            scanf("%s %s %s", st1, st2, st3);
            
            w1 = SearchVocab(st1);
            w2 = SearchVocab(st2);
            w3 = SearchVocab(st3);
            
            if (w1 == -1)
            {
                printf("Word 1 is out of dictionary!\n");
                continue;
            }
            if (w2 == -1) 
            {
                printf("Word 2 is out of dictionary!\n");
                continue;
            }
            if (w3 == -1) 
            {
                printf("Word 3 is out of dictionary!\n");
                continue;
            }
            
            a = 0;
            while (a < layer1_size) 
            {
                vec[a] = syn0[a + w1*layer1_size] - syn0[a + w2*layer1_size] + syn0[a + w3*layer1_size];
                a++;
            }
            
            len = 0;
            a = 0;
            while (a < layer1_size) 
            {
                len = len + (vec[a] * vec[a]);
                a++;
            }
            
            len = sqrt(len);
            
            a = 0;
            while (a < layer1_size) 
            {
                vec[a] = vec[a] / len;
                a++;
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
                if (c == w1) { 
                    c++;
                    continue;
                }
                if (c == w2)
                { 
                    c++; 
                    continue;
                }
                if (c == w3)
                { 
                    c++;
                    continue;
                }
                
                dist = 0;
                a = 0;
                while (a < layer1_size)
                    {
                    dot_prod = vec[a] * syn0[a + c * layer1_size];
                    dist = dist + dot_prod;
                    a++;
                }
                
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
    }
}


//my main function(assumed)
int main() 
{
    int mode;
    size_t vocab_init_bytes;
    size_t hash_init_bytes;
    
    printf("========================================\n");
    printf("\n");
    printf("     Word2Vec C Engine (Teacher Ver)    \n");
        printf("\n");
    printf("========================================\n");
    printf("1. TRAIN New Model (Takes hours/days)\n");
        printf("\n");
    printf("2. PLAY with Existing Model (vectors.txt)\n");
        printf("\n");
    
    printf("Select Mode: ");
    
    scanf("%d", &mode);

    if (mode == 1)
    {
        vocab_init_bytes = vocab_max_size * sizeof(struct vocab_word);
        vocab = (struct vocab_word *)calloc(1, vocab_init_bytes);
        if (vocab == NULL)
        {
            printf("Initial vocab allocation failed\n");
            printf("\n");
            exit(1);
        }
        
        hash_init_bytes = vocab_hash_size * sizeof(int);
        vocab_hash = (int *)calloc(1, hash_init_bytes);
        if (vocab_hash == NULL) 
        {
            printf("Initial hash allocation failed\n");
            exit(1);
        }
        
        InitExpTable();
        LearnVocabFromTrainFile();
        InitNet();
        InitUnigramTable();
        TrainModel();
    } 
    
    if (mode == 2) 
    {
        LoadModel();
        InteractiveLoop();
    }
    
    return 0;
}
