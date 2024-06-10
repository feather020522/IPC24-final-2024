<<<<<<< HEAD:metric_learning_seq.cpp
// #include <iostream>
// #include <cstdlib>
// #include <string>
// #include <utility>
// #include <cmath>
// #include <chrono>
// #include <iomanip>
// #include <algorithm>
=======
#include <iostream>
#include <cstdlib>
#include <string>
#include <utility>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <torch/torch.h>
>>>>>>> audrey_run_dp:test_code/metric_learning_seq.cpp

// #include "mnist/mnist_reader.hpp"

// using namespace std;

// #define RESULT_FILE "result.txt"
// #define MNIST_DATA_LOCATION "/tmp/dataset-nthu-ipc24/share/hw6/testcases/MNIST"
// #define WEIGHT_ROOT "/tmp/dataset-nthu-ipc24/share/hw6/cnn_weights"

// /* Model architecture:
//  *  Conv1: (Cin, Cout, H, W) = (1, 16, 5, 5)
//  *  Conv1: (Cin, Cout, H, W) = (16, 32, 5, 5)
//  *  FC1: 1568 * 512
//  *  FC2: 512 * 10
//  */

// #define IMAGE_SIZE 28
// #define C1 16
// #define C2 32
// #define K 5
// #define FC1 1568 // 32*7*7
// #define FC2 512
// #define OUT 10

// // num_pixel = 784

// void Padding2D(float *A, float *B, int n, int size)
// {
//     int p = (K - 1) / 2;
//     // #pragma acc data present(A, B)
//     // #pragma acc parallel loop collapse(3)
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < size + 2 * p; j++) {
//             for (int k = 0; k < size + 2 * p; k++) {
//                 if (j < p || j >= size + p || k < p || k >= size + p) {
//                     B[i * (size + 2 * p) * (size + 2 * p) + j * (size + 2 * p) + k] = 0;
//                 }
//                 else {
//                     // B[i][j][k] = A[i][j-2][k-2]
//                     B[i * (size + 2 * p) * (size + 2 * p) + j * (size + 2 * p) + k] = A[i * size * size + (j - p) * size + (k - p)];
//                 }
//             }
//         }
//     }
// }

// void Conv2D(float *A, float *B, float *C, float *D, int n, int cin, int cout, int size) {
   
//     int padded_size = size + K - 1;
//     // #pragma acc data present(A, B, C, D)
//     // #pragma acc parallel loop collapse(4)
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < cout; j++) {
//             for (int x = 0; x < size; x++) {
//                 for (int y = 0; y < size; y++) {
//                     // D[i][j][x][y] = C[j];
//                     int d_index = i * cout * size * size + j * size * size + x * size + y;
//                     D[d_index] = C[j];
//                     // V3
//                     // float tempsum = C[j];
//                     // #pragma acc loop reduction(+:tempsum) collapse(3)
//                     for (int k = 0; k < cin; k++) {
//                         for (int kx = 0; kx < K; kx++) {
//                             for (int ky = 0; ky < K; ky++) {
//                                 // D[i][j][x][y] += B[j][k][kx][ky] * A[i][k][x+kx][y+ky];
//                                 int a_idx = i * cin * padded_size * padded_size + k * padded_size * padded_size + (x + kx) * padded_size + (y + ky);
//                                 int b_idx = j * cin * K * K + k * K * K + kx * K + ky;
//                                 D[d_index] += B[b_idx] * A[a_idx];
//                                 // V3
//                                 // tempsum += B[b_idx] * A[a_idx];
//                             }
//                         }
//                     }
//                     // V3
//                     // D[d_index] = tempsum;
//                 }
//             }
//         }
//     }
// }

// void ReLU(float *A, int n) {
//     // #pragma acc data present(A)
//     // #pragma acc parallel loop
//     for (int i = 0; i < n; i++) {
//         A[i] = max(0.0f, A[i]);
//     }
// }

// void MaxPool2D(float *A, float *B, int n, int size) {
//     int pool_size = size / 2;
//     // #pragma acc data present(A, B)
//     // #pragma acc parallel loop collapse(3)
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < pool_size; j++) {
//             for (int k = 0; k < pool_size; k++) {
//                 // B[i][j][k] = max(A[i][2*j][2*k], A[i][2*j][2*k+1], A[i][2*j+1][2*k], A[i][2*j+1][2*k+1])
//                 float max1, max2;
//                 max1 = max(A[i * size * size + (2 * j) * size + (2 * k)], A[i * size * size + (2 * j) * size + (2 * k + 1)]);
//                 max2 = max(A[i * size * size + (2 * j + 1) * size + (2 * k)], A[i * size * size + (2 * j + 1) * size + (2 * k + 1)]);
//                 B[i * pool_size * pool_size + j * pool_size + k] = max(max1, max2);
//             }
//         }
//     }
// }

// /* https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
//  * D(n_i, cout_j) := sum_{k=0}^{Cin_k} ( B(cout_j, k) X A(cout_j, k) )+ C(cout_j), 
//  * where X  is the valid 2D cross-correlation operator.
//  *
//  * A := n*cin*size*size matrix
//  * B := cout*cin*K*K matrix
//  * C := cout matrix
//  * D := n*cout*size*size matrix
//  */

// /* TODO: Parallel the for loops
//  * HINT: 1. (a) copy array A, B, C to GPU device
//  *          (b) copy array D back to CPU
//  *       2. Parallel the loop in the called function using
//  *          (a) #pragma acc XXX
//  *          (b) CUDA kernel function
//  */
// void ConvolutionLayer(float *A, float *B, float *C, float *D, int n, int cin, int cout, int size) {
//     float *padded_input = new float[n * cin * (size + K - 1) * (size + K - 1)];
//     float *conv_output = new float[n * cout * size * size];

//     // #pragma acc data copyin(A[0:n*cin*size*size]) copyin(B[0:cout*cin*K*K]) copyin(C[0:cout])
//     // D: n * cout * pool_size * pool_size
//     // pool_size = size / 2
//     // #pragma acc data copyout(D[0:n*cout*size*size/4])
//     // #pragma acc data create(padded_input[0:n*cin*(size+K-1)*(size+K-1)])
//     // #pragma acc data create(conv_output[0:n*cout*size*size])
//     // {
//         Padding2D(A, padded_input, n * cin, size);
//         Conv2D(padded_input, B, C, conv_output, n, cin, cout, size);
//         ReLU(conv_output, n * cout * size * size);
//         MaxPool2D(conv_output, D, n * cout, size);
//     // }

//     delete[] padded_input;
//     delete[] conv_output;
// }

// /* https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
//  * D := AxB + C
//  *
//  * A := n*k matrix
//  * B := k*m matrix
//  * C := m*1 matrix
//  */

// /* TODO: Parallel the for loops
//  * HINT: 1. (a) copy array A, B, C to GPU device
//  *          (b) copy array D back to CPU
//  *       2. Parallel the loop using
//  *          (a) #pragma acc XXX
//  *          (b) CUDA kernel function
//  */
// void LinearLayer(float *A, float *B, float *C, float *D, int n, int k, int m) {
//     // #pragma acc data copyin(A[0:n*k]) copyin(B[0:k*m]) copyin(C[0:m])
//     // #pragma acc data copyout(D[0:n*m])
//     // #pragma acc kernels
//     // {
//     // #pragma acc loop independent collapse(2)
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < m; j++) {
//             float sum = C[j];
//             // #pragma acc parallel loop reduction(+:sum)
//             // #pragma acc loop independent reduction(+:sum)
//             for (int a = 0; a < k; a++) {
//                 // sum += A[i][a] * B[a][j]
//                 sum += A[i * k + a] * B[a * m + j];
//             }
//             D[i * m + j] = sum;
//         }
//     }

//     // }
// }

// /* https://pytorch.org/docs/stable/generated/torch.nn.Sigmoid.html
//  * A := sigmoid(A)
//  * A := n*m matrix
//  */

// /* TODO: Parallel the for loops */
// void Sigmoid(float *A, int n, int m) {
//     // #pragma acc data copy(A[0:n*m])
//     // #pragma acc parallel loop
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < m; j++) {
//             // Sigmoid(x) = 1/(1+exp(x))
//             A[i * m + j] = 1. / (1. + expf(-A[i * m + j]));
//         }
//     }
// }

// /* Argmax: Choose the index with the largest value
//  * A := n*m matrix (data type: float)
//  * D := n*1 matrix (data type: int)
//  */

// /* TODO: Parallel the for loops */
// void Argmax(float *A, int *D, int n, int m) {
//     // #pragma acc data copyin(A[0:n*m])
//     // #pragma acc parallel loop
//     for (int i = 0; i < n; i++) {
//         float mx = A[i * m];
//         int index = 0;
//         // #pragma acc loop independent reduction(max:mx) 
//         for (int j = 1; j < m; j++) {
//             if (mx < A[i * m + j]) {
//                 mx = A[i * m + j];
//                 index = j;
//             }
//         }

//         D[i] = index;
//     }
// }

// /* my_cnn: A simple neural network
//  * Input arrays:
//  *     training_images_flat: float[num_images * num_pixels]
//  *     conv1_weight:         float[1 * C1 * K * K]
//  *     conv1_bias:           float[C1]
//  *     conv2_weight:         float[C1 * C2 * K * K]
//  *     conv2_bias:           float[C2]
//  *     fc1_weight:           float[FC1 * FC2]
//  *     fc1_bias:             float[FC2]
//  *     fc2_weight:           float[FC2 * OUT]
//  *     fc2_bias:             float[OUT]
//  * Output array:
//  *     result:               int[num_images]
//  */
// void my_cnn(float *training_images_flat, int num_images,
//             float *conv1_weight, float *conv1_bias, float *conv2_weight, float *conv2_bias, 
//             float *fc1_weight, float *fc1_bias, float *fc2_weight, float *fc2_bias,
//             int *result) {

//     float *conv1_output = new float[num_images * C1 * IMAGE_SIZE / 2 * IMAGE_SIZE / 2];
//     float *conv2_output = new float[num_images * C2 * IMAGE_SIZE / 4 * IMAGE_SIZE / 4];
//     float *fc1_output = new float[num_images * FC2];
//     float *fc2_output = new float[num_images * OUT];

//     ConvolutionLayer(training_images_flat, conv1_weight, conv1_bias, conv1_output,
//                      num_images, 1, C1, IMAGE_SIZE);
//     ConvolutionLayer(conv1_output, conv2_weight, conv2_bias, conv2_output,
//                      num_images, C1, C2, IMAGE_SIZE / 2);
//     LinearLayer(conv2_output, fc1_weight, fc1_bias, fc1_output,
//                 num_images, FC1, FC2);
//     LinearLayer(fc1_output, fc2_weight, fc2_bias, fc2_output,
//                 num_images, FC2, OUT);
//     Argmax(fc2_output, result, num_images, OUT);

//     delete[] conv1_output;
//     delete[] conv2_output;
//     delete[] fc1_output;
//     delete[] fc2_output;
// }

<<<<<<< HEAD:metric_learning_seq.cpp
// /* Read neural network's weight from file (in binary format)
//  */
=======
/* Read neural network's weight from file (in binary format)
 */
>>>>>>> audrey_run_dp:test_code/metric_learning_seq.cpp
// void read_weight(float *array, string filename, int num_floats) {
//     string full_filename = string(WEIGHT_ROOT) + '/' + filename;
//     std::cout << "Reading file: " << full_filename << std::endl;

//     ifstream file(full_filename, ios::in | ios::binary);
//     if (!file) {
//         std::cerr << "error reading file: " << full_filename << std::endl;
//         exit(1);
//     }
//     file.read((char *)array, num_floats * sizeof(float));
// }

<<<<<<< HEAD:metric_learning_seq.cpp
// /* Write predicted result to file
//  */
=======
/* Write predicted result to file
 */
>>>>>>> audrey_run_dp:test_code/metric_learning_seq.cpp
// void write_predict(int *result, int n, char *filename) {
//     std::ofstream file(filename, std::ofstream::out);
//     for (int i = 0; i < n; i++) {
//         file << result[i] << '\n';
//     }
//     file.close();
// }

// /* Print an image
//  * Usage: print_img(training_images[i])
//  */
// void print_img(float *img) {
//     for (int i = 0; i < 28; i++) {
//         for (int j = 0; j < 28; j++) {
//             if (img[i * 28 + j] > 0.5) {
//                 std::cout << 'x';
//             }
//             else {
//                 std::cout << ' ';
//             }
//         }
//         std::cout << '\n';
//     }
//     std::cout << std::endl;
// }

// int main(int argc, char *argv[]) {
    
<<<<<<< HEAD:metric_learning_seq.cpp
//     auto read_start = std::chrono::steady_clock::now();
//     // std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
=======
    auto read_start = std::chrono::steady_clock::now();
    torch::optim::Adam optimizer
    // std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;
>>>>>>> audrey_run_dp:test_code/metric_learning_seq.cpp

//     /* Load MNIST data
//      */
//     // mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;
//     // if (argc == 1) {
//     //     dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(MNIST_DATA_LOCATION);
//     // }else {
//     //     dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(argv[1]);
//     // }

//     // std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
//     // // std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
//     // // std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
//     // // std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

//     // //int num_train_images = dataset.training_images.size();
//     // int num_train_images = 30000; // due to CT memory limit
//     // int num_pixels = dataset.training_images.front().size(); // should be 28*28 = 784

//     // /* Convert 60000 training images from [0, 255] to [0, 1]
//     //  * We will first generate another 2D array by `new`
//     //  */

//     // /* training_images_flat[i*num_pixels + j] == training_images[i][j]
//     //  * j-th pixel in i-th image
//     //  */
//     // float *training_images_flat = new float[num_train_images * num_pixels];

//     // float **training_images = new float *[num_train_images];
//     // for (int i = 0; i < num_train_images; i++) {
//     //     training_images[i] = training_images_flat + i * num_pixels;
//     // }

//     // for (int i = 0; i < num_train_images; i++) {
//     //     for (int j = 0; j < num_pixels; j++) {
//     //         training_images[i][j] = (float)(dataset.training_images[i][j]) / 255.0;
//     //     }
//     // }

//     /* Print first image */
//     // print_img(training_images[0]);

//     /* Load matrices' weight from binary file
//      * You can print the binary file by: `od -f weights/conv1_bias`
//      * https://stackoverflow.com/questions/36791622/how-to-print-float-value-from-binary-file-in-shell
//      */
//     // float *conv1_weight = new float[C1 * 1 * K * K];
//     // float *conv1_bias = new float[C1];
//     // float *conv2_weight = new float[C2 * C1 * K * K];
//     // float *conv2_bias = new float[C2];
//     // float *fc1_weight = new float[FC1 * FC2];
//     // float *fc1_bias = new float[FC2];
//     // float *fc2_weight = new float[FC2 * OUT];
//     // float *fc2_bias = new float[OUT];
//     // read_weight(conv1_weight, "conv1_weight", C1 * 1 * K * K);
//     // read_weight(conv1_bias, "conv1_bias", C1);
//     // read_weight(conv2_weight, "conv2_weight", C2 * C1 * K * K);
//     // read_weight(conv2_bias, "conv2_bias", C2);
//     // read_weight(fc1_weight, "fc1_weight", FC1 * FC2);
//     // read_weight(fc1_bias, "fc1_bias", OUT);
//     // read_weight(fc2_weight, "fc2_weight", FC2 * OUT);
//     // read_weight(fc2_bias, "fc2_bias", OUT);

//     // auto read_end = std::chrono::steady_clock::now();

//     // /* Inference */
//     // int *result = new int[num_train_images];
//     // my_cnn(training_images_flat, num_train_images,
//     //        conv1_weight, conv1_bias, conv2_weight, conv2_bias, 
//     //        fc1_weight, fc1_bias, fc2_weight, fc2_bias, result);
//     // auto inference_end = std::chrono::steady_clock::now();

//     // /* Calculate accuracy */
//     // int correct = 0;
//     // int total = 0;
//     // for (int i = 0; i < num_train_images; i++) {
//     //     if ((int)result[i] == (int)dataset.training_labels[i]) {
//     //         correct++;
//     //     }
//     //     total++;
//     // }
//     // std::cout << "\nInference accuracy: " << (double)correct / (double)total * 100.0 << "%\n";
//     // if (argc < 3) {
//     //     write_predict(result, num_train_images, RESULT_FILE);
//     // }else {
//     //     write_predict(result, num_train_images, argv[2]);
//     // }

//     // auto acc_end = std::chrono::steady_clock::now();

//     // std::cout << std::setprecision(5) << std::fixed;
//     // std::cout << "\n-----     STATS     -----\n";
//     // std::cout << "Time for reading MNIST data & weights: " << std::chrono::duration_cast<std::chrono::milliseconds>(read_end - read_start).count() << " m.s.\n";
//     // std::cout << "Time for inferencing                 : " << std::chrono::duration_cast<std::chrono::milliseconds>(inference_end - read_end).count() << " m.s.\n";
//     // std::cout << "Time for calculating accuracy        : " << std::chrono::duration_cast<std::chrono::milliseconds>(acc_end - inference_end).count() << " m.s.\n";
//     // std::cout << "----- END OF STATS  -----\n";

//     // delete[] result;
//     // delete[] conv1_weight;
//     // delete[] conv1_bias;
//     // delete[] conv2_weight;
//     // delete[] conv2_bias;
//     // delete[] fc1_weight;
//     // delete[] fc1_bias;
//     // delete[] fc2_weight;
//     // delete[] fc2_bias;
//     // delete[] training_images_flat;
//     // delete[] training_images;
//     return 0;
// }

#include <torch/torch.h>

__global__
void sobel() {
    // int x, y, i, v, u;
    // int R, G, B;
    // double val[MASK_N * 3] = {0.0};
    // int adjustX, adjustY, xBound, yBound;
    // // TOOD: no out of bound???
    // int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
    // int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride_x = blockDim.x * gridDim.x;
    // int stride_y = blockDim.y * gridDim.y;
    // for (y = y_idx; y < height; y += stride_y) {
    //     for (x = x_idx; x < width; x += stride_x) {
    //         for (i = 0; i < MASK_N; ++i) {
    //             adjustX = (MASK_X % 2) ? 1 : 0;
    //             adjustY = (MASK_Y % 2) ? 1 : 0;
    //             xBound = MASK_X / 2;
    //             yBound = MASK_Y / 2;

    //             val[i * 3 + 2] = 0.0;
    //             val[i * 3 + 1] = 0.0;
    //             val[i * 3] = 0.0;

    //             for (v = -yBound; v < yBound + adjustY; ++v) {
    //                 for (u = -xBound; u < xBound + adjustX; ++u) {
    //                     if ((x + u) >= 0 && (x + u) < width && y + v >= 0 && y + v < height) {
    //                         R = s[channels * (width * (y + v) + (x + u)) + 2];
    //                         G = s[channels * (width * (y + v) + (x + u)) + 1];
    //                         B = s[channels * (width * (y + v) + (x + u)) + 0];
    //                         val[i * 3 + 2] += R * mask[i][u + xBound][v + yBound];
    //                         val[i * 3 + 1] += G * mask[i][u + xBound][v + yBound];
    //                         val[i * 3 + 0] += B * mask[i][u + xBound][v + yBound];
    //                     }
    //                 }
    //             }
    //         }

    //         double totalR = 0.0;
    //         double totalG = 0.0;
    //         double totalB = 0.0;
    //         for (i = 0; i < MASK_N; ++i) {
    //             totalR += val[i * 3 + 2] * val[i * 3 + 2];
    //             totalG += val[i * 3 + 1] * val[i * 3 + 1];
    //             totalB += val[i * 3 + 0] * val[i * 3 + 0];
    //         }

    //         totalR = sqrt(totalR) / SCALE;
    //         totalG = sqrt(totalG) / SCALE;
    //         totalB = sqrt(totalB) / SCALE;
    //         const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
    //         const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
    //         const unsigned char cB = (totalB > 255.0) ? 255 : totalB;
    //         t[channels * (width * y + x) + 2] = cR;
    //         t[channels * (width * y + x) + 1] = cG;
    //         t[channels * (width * y + x) + 0] = cB;

    //     }
        
    // }
}

int main()
{
    torch::Tensor tensor = torch::zeros({2, 2});
    std::cout << tensor << std::endl;

    dim3 num_threads_per_block(32, 16, 1);
    dim3 num_blocks((width / num_threads_per_block.x ) + 1, (height / num_threads_per_block.y) + 1, 1);

    sobel<<<num_blocks, num_threads_per_block>>> ();

    return 0;
}

