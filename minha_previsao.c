#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <omp.h>

#define NUM_FEATURES 5
#define NUM_TREES 100
#define MAX_NODES_PER_TREE 15
#define MAX_LINE_LENGTH 256

/*
 * Cada árvore é uma árvore binária completa com 15 nós:
 *
 *              0
 *          /       \
 *         1         2
 *       /  \       /  \
 *      3    4     5    6
 *     / \  / \   / \  / \
 *    7  8 9  10 11 12 13 14
 *
 * Nós 0..6  : nós internos
 * Nós 7..14 : folhas
 */

#define NUM_INTERNAL_NODES 7
#define FIRST_LEAF 7


/* ================================================================
 * Estruturas de dados
 * ================================================================ */

typedef struct {
    int feature_index;   /* -1 indica folha */
    float threshold;
    float value;
    int left_child;
    int right_child;
} Node;

typedef struct {
    Node nodes[NUM_TREES * MAX_NODES_PER_TREE];
    int tree_roots[NUM_TREES];
} RandomForest;


/* ================================================================
 * Leitura dos dados meteorológicos
 * ================================================================ */

float* read_csv_data(const char* filename, int* out_num_samples)
{
    FILE* file = fopen(filename, "r");

    if (!file) {
        perror("Erro ao abrir o arquivo CSV");
        return NULL;
    }

    char line[MAX_LINE_LENGTH];
    int line_count = 0;

    /* Ignora o cabeçalho. */
    if (fgets(line, sizeof(line), file) == NULL) {
        fprintf(stderr, "Erro: arquivo CSV vazio.\n");
        fclose(file);
        return NULL;
    }

    /*
     * Primeira passagem:
     * determina quantas linhas válidas existem.
     */
    while (fgets(line, sizeof(line), file)) {

        float f0, f1, f2, f3, f4;

        if (sscanf(line, "%f,%f,%f,%f,%f",
                   &f0, &f1, &f2, &f3, &f4) == NUM_FEATURES) {
            line_count++;
        }
    }

    if (line_count == 0) {
        fprintf(stderr,
                "Erro: nenhuma amostra válida encontrada no CSV.\n");
        fclose(file);
        return NULL;
    }

    float* features =
        malloc((size_t)line_count * NUM_FEATURES * sizeof(float));

    if (!features) {
        fprintf(stderr, "Erro de alocação de memória.\n");
        fclose(file);
        return NULL;
    }

    /*
     * Segunda passagem:
     * carrega efetivamente os dados.
     */
    rewind(file);

    /* Ignora novamente o cabeçalho. */
    fgets(line, sizeof(line), file);

    int sample = 0;

    while (fgets(line, sizeof(line), file) &&
           sample < line_count) {

        float f0, f1, f2, f3, f4;

        if (sscanf(line, "%f,%f,%f,%f,%f",
                   &f0, &f1, &f2, &f3, &f4) == NUM_FEATURES) {

            features[sample * NUM_FEATURES + 0] = f0;
            features[sample * NUM_FEATURES + 1] = f1;
            features[sample * NUM_FEATURES + 2] = f2;
            features[sample * NUM_FEATURES + 3] = f3;
            features[sample * NUM_FEATURES + 4] = f4;

            sample++;
        }
    }

    fclose(file);

    *out_num_samples = sample;

    return features;
}


/* ================================================================
 * Determinação das faixas observadas no conjunto de dados
 * ================================================================ */

void compute_feature_ranges(
    const float* features,
    int num_samples,
    float min_values[NUM_FEATURES],
    float max_values[NUM_FEATURES])
{
    for (int f = 0; f < NUM_FEATURES; f++) {
        min_values[f] = FLT_MAX;
        max_values[f] = -FLT_MAX;
    }

    for (int i = 0; i < num_samples; i++) {

        for (int f = 0; f < NUM_FEATURES; f++) {

            float value =
                features[i * NUM_FEATURES + f];

            if (value < min_values[f])
                min_values[f] = value;

            if (value > max_values[f])
                max_values[f] = value;
        }
    }
}


/* ================================================================
 * Geração de números aleatórios
 * ================================================================ */

float random_float(float min_value, float max_value)
{
    float r = (float)rand() / (float)RAND_MAX;

    return min_value +
           r * (max_value - min_value);
}


/* ================================================================
 * Construção da floresta sintética
 * ================================================================ */

void build_random_forest_synthetic(
    RandomForest* forest,
    const float min_values[NUM_FEATURES],
    const float max_values[NUM_FEATURES])
{
    int node_counter = 0;

    for (int t = 0; t < NUM_TREES; t++) {

        int tree_start = node_counter;

        forest->tree_roots[t] = tree_start;

        /*
         * Cria os sete nós internos.
         */
        for (int n = 0; n < NUM_INTERNAL_NODES; n++) {

            int feature = rand() % NUM_FEATURES;

            float threshold =
                random_float(
                    min_values[feature],
                    max_values[feature]);

            int left =
                tree_start + (2 * n + 1);

            int right =
                tree_start + (2 * n + 2);

            forest->nodes[tree_start + n] =
                (Node){
                    feature,
                    threshold,
                    0.0f,
                    left,
                    right
                };
        }

        /*
         * Cria as oito folhas.
         *
         * Os valores são saídas sintéticas utilizadas
         * para compor a carga computacional do experimento.
         */
        for (int n = FIRST_LEAF;
             n < MAX_NODES_PER_TREE;
             n++) {

            float prediction =
                random_float(0.0f, 35.0f);

            forest->nodes[tree_start + n] =
                (Node){
                    -1,
                    0.0f,
                    prediction,
                    -1,
                    -1
                };
        }

        node_counter += MAX_NODES_PER_TREE;
    }
}


/* ================================================================
 * Inferência paralela com OpenMP
 * ================================================================ */

void predict_random_forest_omp(
    const float* features,
    const RandomForest* forest,
    float* predictions,
    int num_samples)
{
    #pragma omp parallel for schedule(static)

    for (int i = 0; i < num_samples; i++) {

        float total_prediction = 0.0f;

        const float* sample =
            &features[i * NUM_FEATURES];

        /*
         * Cada thread processa todas as árvores para
         * a amostra que lhe foi atribuída.
         */
        for (int t = 0; t < NUM_TREES; t++) {

            int current_node =
                forest->tree_roots[t];

            /*
             * Percorre a árvore até encontrar uma folha.
             */
            while (1) {

                Node node =
                    forest->nodes[current_node];

                if (node.feature_index == -1) {

                    total_prediction +=
                        node.value;

                    break;
                }

                float feature_value =
                    sample[node.feature_index];

                if (feature_value <= node.threshold) {

                    current_node =
                        node.left_child;

                } else {

                    current_node =
                        node.right_child;
                }
            }
        }

        /*
         * Resultado da floresta:
         * média das previsões das árvores.
         */
        predictions[i] =
            total_prediction / NUM_TREES;
    }
}


/* ================================================================
 * Interpretação textual dos resultados
 * ================================================================ */

const char* classificar_risco_chuva(float mm_previstos)
{
    if (mm_previstos < 1.0f)
        return "Tempo Seco / Estável";

    if (mm_previstos < 5.0f)
        return "Chuva Leve / Garoa";

    if (mm_previstos < 15.0f)
        return "Chuva Moderada";

    if (mm_previstos < 30.0f)
        return "ALERTA: Chuva Forte";

    return "PERIGO: Tempestade Severa";
}


/* ================================================================
 * Programa principal
 * ================================================================ */

int main(int argc, char* argv[])
{
    const char* csv_filename =
        "dados_pelotas.csv";

    /*
     * Permite especificar outro arquivo pela linha de comando.
     */
    if (argc > 1) {
        csv_filename = argv[1];
    }

    printf("=== RANDOM FOREST - INFERENCIA OPENMP ===\n\n");

    printf("Arquivo: %s\n", csv_filename);


    /* ------------------------------------------------------------
     * Leitura dos dados meteorológicos
     * ------------------------------------------------------------ */

    int num_samples = 0;

    float* features =
        read_csv_data(
            csv_filename,
            &num_samples);

    if (!features) {
        return EXIT_FAILURE;
    }

    printf("Amostras carregadas: %d\n",
           num_samples);


    /* ------------------------------------------------------------
     * Determinação das faixas observadas
     * ------------------------------------------------------------ */

    float min_values[NUM_FEATURES];
    float max_values[NUM_FEATURES];

    compute_feature_ranges(
        features,
        num_samples,
        min_values,
        max_values);


    printf("\nFaixas observadas:\n");

    for (int f = 0; f < NUM_FEATURES; f++) {

        printf(
            "Feature %d: %.2f a %.2f\n",
            f,
            min_values[f],
            max_values[f]);
    }


    /* ------------------------------------------------------------
     * Alocação das estruturas utilizadas na inferência
     * ------------------------------------------------------------ */

    float* predictions =
        malloc((size_t)num_samples *
               sizeof(float));

    RandomForest* forest =
        malloc(sizeof(RandomForest));

    if (!predictions || !forest) {

        fprintf(stderr,
                "Erro de alocação de memória.\n");

        free(features);
        free(predictions);
        free(forest);

        return EXIT_FAILURE;
    }


    /* ------------------------------------------------------------
     * Construção da floresta
     * ------------------------------------------------------------ */

    /*
     * Seed fixa para garantir que todas as execuções
     * utilizem exatamente a mesma floresta.
     */
    srand(42);

    build_random_forest_synthetic(
        forest,
        min_values,
        max_values);


    /* ------------------------------------------------------------
     * Inferência paralela
     * ------------------------------------------------------------ */

    int num_threads =
        omp_get_max_threads();

    double start =
        omp_get_wtime();

    predict_random_forest_omp(
        features,
        forest,
        predictions,
        num_samples);

    double end =
        omp_get_wtime();


    /* ------------------------------------------------------------
     * Resultados de desempenho
     * ------------------------------------------------------------ */

    printf(
        "\nThreads OpenMP: %d\n",
        num_threads);

    printf(
        "Tempo de inferencia: %.6f segundos\n",
        end - start);


    /* ------------------------------------------------------------
     * Interpretação dos primeiros resultados
     * ------------------------------------------------------------ */

    int print_limit =
        num_samples < 5
        ? num_samples
        : 5;

    printf(
        "\n--- PREVISÃO DE CHUVA PARA A PRÓXIMA HORA (PELOTAS) ---\n");

    for (int i = 0; i < print_limit; i++) {

        float mm = predictions[i];

        printf(
            "Hora [%d] -> "
            "Estimativa: %5.2f mm/h | "
            "Status: %s\n",
            i,
            mm,
            classificar_risco_chuva(mm));
    }


    /* ------------------------------------------------------------
     * Liberação de memória
     * ------------------------------------------------------------ */

    free(features);
    free(predictions);
    free(forest);

    return EXIT_SUCCESS;
}
