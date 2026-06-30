#include"add.h"
#include <iostream>

int main() {
    int a = 3;
    int b = 5;
    int result = add(a, b);
    std::cout << "The sum of " << a << " and " << b << " is: " << result << std::endl;
    return 0;
}
