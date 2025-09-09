#include <iostream>
#include <vector>

void cpu_prefix_sum(const std::vector<int>& input, std::vector<int>& output) {
    output[0] = 0;  // exclusive prefix sum starts with 0
    for (size_t i = 1; i < input.size(); ++i) {
        output[i] = output[i-1] + input[i-1];
    }
}

int main() {
    // Test case 2
    std::vector<int> input = {1, 10, 7, 2, 8, 0, 5, 6, 9, 3};
    std::vector<int> output(input.size());
    
    cpu_prefix_sum(input, output);
    
    std::cout << "Input: ";
    for (int x : input) std::cout << x << " ";
    std::cout << std::endl;
    
    std::cout << "Expected output: ";
    for (int x : output) std::cout << x << " ";
    std::cout << std::endl;
    
    return 0;
}
