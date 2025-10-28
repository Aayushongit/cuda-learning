#include <iostream>
#include <string>

namespace print{
	void print(std::string s){
		std::cout << s << std::endl;

	}
}

namespace{
	const double val = 3.14159;

	void mysecretname(){
		std::cout << "This was my name " << std::endl;

	}
}


int main(){
    // using namespace print;

    print::print("this is my name ");

    mysecretname();
    std::cout << val << std::endl;

    return 0;

}
