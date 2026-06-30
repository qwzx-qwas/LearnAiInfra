#include <iostream>
#include <string>

class Tensor {
public:
    explicit Tensor(const std::string& name) : name_(name) {
        std::cout << "创建 Tensor: " << name_ << std::endl;
    }

    ~Tensor() {
        std::cout << "销毁 Tensor: " << name_ << std::endl;
    }

private:
    std::string name_;

};

void run() {
    Tensor input("input");
    Tensor output("output");

    std::cout << "正在执行 run\n";
}

int main() {
    std::cout << "进入 main\n";
    run();
    std::cout << "离开 run\n";
} 
